from bisect import bisect_right

import torch
import torch.nn as nn

from ..offload_config import OffloadConfig
from .memory_manager import ExpertMemoryManager
from .routing import LayerRoutingMap


class ColdExperts(nn.Module):
    """Reusable NPU buffer for one layer's cold expert weights."""

    def __init__(self, w13_template: torch.Tensor, w2_template: torch.Tensor,
                 num_experts: int, device: torch.device):
        super().__init__()
        self.num_experts = num_experts
        self.w13_weight = nn.Parameter(
            torch.empty((num_experts, *w13_template.shape),
                        dtype=w13_template.dtype,
                        device=device),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty((num_experts, *w2_template.shape),
                        dtype=w2_template.dtype,
                        device=device),
            requires_grad=False,
        )
        self.load_stream: torch.npu.Stream | None = None

    def load_from_cpu(self, w13_weight: torch.Tensor,
                      w2_weight: torch.Tensor) -> None:
        with torch.no_grad():
            cold_count = w13_weight.size(0)
            self.w13_weight[:cold_count].copy_(w13_weight, non_blocking=True)
            self.w2_weight[:cold_count].copy_(w2_weight, non_blocking=True)

    def wait(self) -> None:
        if self.load_stream is None:
            return
        torch.npu.current_stream().wait_stream(self.load_stream)


class OffloadExecutor(nn.Module):
    """Controls layer-ahead cold expert prefetch with reusable buffers."""

    def __init__(self, config: OffloadConfig):
        super().__init__()
        self.config = config
        self.offload_full_layer = config.mode == "layer_wise"
        self.offloaded_layer_ids = config.offloaded_layer_ids
        self.offloaded_layer_id_set = set(config.offloaded_layer_ids)
        self.memory_manager = ExpertMemoryManager(
            cpu_pin_memory=config.cpu_pin_memory)
        
        self.layers: dict[int, nn.Module] = {}
        self.cold_experts = nn.ModuleList()
        self.next_buffer_id = 0
        self.pending: dict[int, ColdExperts] = {}
        self.layer_cold_expert_ids: dict[int, list[int]] = {}
        self.routing_maps: dict[int, LayerRoutingMap] = {}
        self.active_cold_topk_ids: torch.Tensor | None = None
        self.active_cold_mask: torch.Tensor | None = None
        self.prefetch_stream: torch.npu.Stream | None = None

    def register_layer(self, layer) -> None:
        layer_id = int(layer.moe_instance_id)
        self.layers[layer_id] = layer

        cold_expert_ids = self.get_layer_cold_expert_ids(layer)
        self.layer_cold_expert_ids[layer_id] = cold_expert_ids
        if cold_expert_ids:
            self.routing_maps[layer_id] = LayerRoutingMap(
                layer, cold_expert_ids)
        self.memory_manager.register_layer(
            layer_id, layer.w13_weight.data, layer.w2_weight.data,
            cold_expert_ids)

    def get_layer_cold_expert_ids(self, layer) -> list[int]:
        if layer.moe_instance_id not in self.offloaded_layer_id_set:
            return []

        local_num_experts = int(layer.full_local_num_experts)
        first_cold_expert = (
            0 if self.offload_full_layer else self.config.num_hot_experts)
        return list(range(first_cold_expert, local_num_experts))

    def should_offload_layer(self, layer) -> bool:
        return layer.moe_instance_id in self.offloaded_layer_id_set

    def is_full_layer_offload(self, layer) -> bool:
        return self.offload_full_layer and self.should_offload_layer(layer)

    def should_offload_expert(self, layer, local_expert_id: int) -> bool:
        if local_expert_id < 0:
            return False
        if not self.should_offload_layer(layer):
            return False
        return (self.offload_full_layer
                or local_expert_id >= self.config.num_hot_experts)

    def load_weight(self, layer, weight_name: str, shard_id: str,
                    global_expert_id: int,
                    loaded_weight: torch.Tensor) -> bool:
        local_expert_id = layer.map_global_expert_id_to_full_local_expert_id(
            global_expert_id)
        if not self.should_offload_expert(layer, local_expert_id):
            return False

        cpu_weight_name = "w2_weight" if shard_id == "w2" else "w13_weight"
        return self.memory_manager.load_weight_shard(
            layer_id=layer.moe_instance_id,
            weight_name=cpu_weight_name,
            local_expert_id=local_expert_id,
            shard_id=shard_id,
            loaded_weight=loaded_weight,
            tp_rank=layer.tp_rank,
        )

    def process_layer_after_loading(self, layer, quant_method) -> None:
        if not self.should_offload_layer(layer):
            return
        self.memory_manager.process_layer_after_loading(layer.moe_instance_id,
                                                        quant_method)

    def prepare_cold_experts(self, layer,
                             topk_ids: torch.Tensor) -> ColdExperts | None:
        self.active_cold_topk_ids = None
        self.active_cold_mask = None

        if not self.should_offload_layer(layer):
            return None

        cold_experts = self.pending.pop(layer.moe_instance_id, None)
        if cold_experts is None:
            cold_experts = self.start_prefetch(layer)
        if cold_experts is None:
            return None

        if self.is_full_layer_offload(layer):
            return cold_experts

        self.active_cold_topk_ids, self.active_cold_mask = (
            self.build_cold_routing(layer, topk_ids))
        return cold_experts

    def prefetch_next_layers(self, layer) -> torch.npu.Stream | None:
        next_layer_id = self.next_layer_id(layer.moe_instance_id)
        if next_layer_id is None:
            return None
        next_layer = self.layers.get(next_layer_id)
        if next_layer is None or next_layer_id in self.pending:
            return None

        cold_experts = self.start_prefetch(next_layer)
        if cold_experts is None:
            return None
        self.pending[next_layer_id] = cold_experts
        return cold_experts.load_stream

    def next_layer_id(self, layer_id: int) -> int | None:
        next_index = bisect_right(self.offloaded_layer_ids, layer_id)
        if next_index >= len(self.offloaded_layer_ids):
            return None
        return self.offloaded_layer_ids[next_index]

    def start_prefetch(self, layer) -> ColdExperts | None:
        cold_expert_ids = self.layer_cold_expert_ids.get(
            layer.moe_instance_id, [])
        if not cold_expert_ids:
            return None

        weights = self.memory_manager.get_expert_weights(
            layer.moe_instance_id)

        self.maybe_create_cold_buffers(layer, weights)

        cold_experts = self.cold_experts[self.next_buffer_id]
        self.next_buffer_id = (
            self.next_buffer_id + 1) % len(self.cold_experts)

        if self.prefetch_stream is None:
            self.prefetch_stream = torch.npu.Stream()

        stream = self.prefetch_stream
        w13_weight, w2_weight = weights
        with torch.npu.stream(stream):
            cold_experts.load_from_cpu(w13_weight, w2_weight)
        cold_experts.load_stream = stream
        return cold_experts

    def maybe_create_cold_buffers(self,
                                  layer,
                                  weights: tuple[torch.Tensor,
                                                 torch.Tensor]) -> None:
        if len(self.cold_experts) != 0:
            return
        num_experts = max(
            len(cold_ids) for cold_ids in self.layer_cold_expert_ids.values())
        w13_weight, w2_weight = weights
        num_buffers = max(2, int(self.config.num_buffers))
        for _ in range(num_buffers):
            self.cold_experts.append(
                ColdExperts(
                    w13_template=w13_weight[0],
                    w2_template=w2_weight[0],
                    num_experts=num_experts,
                    device=layer.w13_weight.device,
                ))

    def build_cold_routing(self, layer, topk_ids: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor]:
        return self.routing_maps[layer.moe_instance_id].cold_routing(topk_ids)

    def summary(self) -> dict[str, object]:
        summary = self.memory_manager.summary()
        summary["mode"] = self.config.mode
        summary["num_layers"] = len(self.layers)
        summary["num_cold_expert_buffers"] = len(self.cold_experts)
        summary["pending_prefetch_layers"] = sorted(self.pending)
        return summary
