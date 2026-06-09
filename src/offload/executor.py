from bisect import bisect_right

import torch
import torch.nn as nn
import torch_npu

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

from ..offload_config import OffloadConfig
from .memory_manager import ExpertMemoryManager
from .routing import LayerRoutingMap


class ColdExperts(nn.Module):
    """Reusable NPU buffer for one layer's cold expert weights."""

    def __init__(self, templates: dict[str, torch.Tensor], num_experts: int,
                 device: torch.device, use_w8a8: bool):
        super().__init__()
        self.num_experts = num_experts
        for name, template in templates.items():
            tensor = torch.empty((num_experts, *template.shape[1:]),
                                 dtype=template.dtype,
                                 device=device)
            if use_w8a8 and name in ("w13_weight", "w2_weight"):
                tensor = torch_npu.npu_format_cast(
                    tensor, ACL_FORMAT_FRACTAL_NZ)
            self.register_parameter(
                name, nn.Parameter(tensor, requires_grad=False))
        if hasattr(self, "w13_weight_scale"):
            self.w13_weight_scale_fp32 = self.w13_weight_scale.data.to(
                torch.float32)
            self.w2_weight_scale_fp32 = self.w2_weight_scale.data.to(
                torch.float32)
        self.load_stream: torch.npu.Stream | None = None

    def load_from_cpu(self, weights: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, source in weights.items():
                target = getattr(self, name)
                target[:source.size(0)].copy_(source, non_blocking=True)
                if name in ("w13_weight_scale", "w2_weight_scale"):
                    fp32_target = getattr(self, f"{name}_fp32")
                    fp32_target[:source.size(0)].copy_(
                        source, non_blocking=True)

    def wait(self) -> None:
        if self.load_stream is None:
            return
        torch.npu.current_stream().wait_stream(self.load_stream)


PreparedColdExperts = tuple[ColdExperts, torch.Tensor | None,
                            torch.Tensor | None]


class OffloadExecutor(nn.Module):
    """Controls layer-ahead cold expert prefetch with reusable buffers."""

    def __init__(self, config: OffloadConfig):
        super().__init__()
        self.config = config
        self.offload_full_layer = config.offload_full_layers
        self.offloaded_layer_ids = set(config.offloaded_layer_ids)
        self.memory_manager = ExpertMemoryManager(
            cpu_pin_memory=config.cpu_pin_memory)

        self.layers: dict[int, nn.Module] = {}
        self.layer_ids: list[int] = []
        self.cold_experts = nn.ModuleList()
        self.next_buffer_id = 0
        self.pending: dict[int, ColdExperts] = {}
        self.layer_cold_expert_ids: dict[int, list[int]] = {}
        self.routing_maps: dict[int, LayerRoutingMap] = {}
        self.prefetch_stream: torch.npu.Stream | None = None

    def register_layer(self, layer) -> None:
        layer_id = int(layer.moe_instance_id)
        self.layers[layer_id] = layer

        cold_expert_ids = self.get_layer_cold_expert_ids(layer)
        self.layer_cold_expert_ids[layer_id] = cold_expert_ids
        if cold_expert_ids:
            self.layer_ids.append(layer_id)
            self.layer_ids.sort()
            self.routing_maps[layer_id] = LayerRoutingMap(
                layer, cold_expert_ids)
        self.memory_manager.register_layer(layer_id, layer, cold_expert_ids)

    def get_layer_cold_expert_ids(self, layer) -> list[int]:
        if not self.should_offload_layer(layer):
            return []
        local_num_experts = int(layer.full_local_num_experts)
        first_cold_expert = (
            0 if self.offload_full_layer else self.config.num_hot_experts)
        return list(range(first_cold_expert, local_num_experts))

    def should_offload_layer(self, layer) -> bool:
        return layer.moe_instance_id in self.offloaded_layer_ids

    def should_offload_expert(self, layer, local_expert_id: int) -> bool:
        return (self.should_offload_layer(layer) and local_expert_id >= 0
                and (self.offload_full_layer
                     or local_expert_id >= self.config.num_hot_experts))

    def load_weight(self, layer, param_name: str, shard_id: str,
                    global_expert_id: int,
                    loaded_weight: torch.Tensor) -> bool:
        local_expert_id = layer.map_global_expert_id_to_full_local_expert_id(
            global_expert_id)
        if not self.should_offload_expert(layer, local_expert_id):
            return False

        return self.memory_manager.load_weight_shard(
            layer_id=layer.moe_instance_id,
            param_name=param_name,
            local_expert_id=local_expert_id,
            shard_id=shard_id,
            loaded_weight=loaded_weight,
            tp_rank=layer.tp_rank,
        )

    def process_layer_after_loading(self, layer, quant_method) -> None:
        if layer.moe_instance_id not in self.memory_manager.layers:
            return
        self.memory_manager.process_layer_after_loading(layer.moe_instance_id,
                                                        quant_method)

    def prepare_cold_experts(self, layer,
                             topk_ids: torch.Tensor
                             ) -> PreparedColdExperts | None:
        cold_experts = self.pending.pop(layer.moe_instance_id, None)
        if cold_experts is None:
            cold_experts = self.start_prefetch(layer)
        if cold_experts is None:
            return None

        if self.offload_full_layer:
            return cold_experts, None, None

        cold_topk_ids, cold_mask = self.build_cold_routing(layer, topk_ids)
        return cold_experts, cold_topk_ids, cold_mask

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
        if not self.layer_ids:
            return None
        next_index = bisect_right(self.layer_ids, layer_id)
        if next_index >= len(self.layer_ids):
            next_index = 0
        return self.layer_ids[next_index]

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
        with torch.npu.stream(stream):
            cold_experts.load_from_cpu(weights)
        cold_experts.load_stream = stream
        return cold_experts

    def maybe_create_cold_buffers(self,
                                  layer,
                                  weights: dict[str, torch.Tensor]) -> None:
        if self.cold_experts:
            return
        num_experts = max(
            len(cold_ids) for cold_ids in self.layer_cold_expert_ids.values())
        for _ in range(self.config.num_buffers):
            self.cold_experts.append(
                ColdExperts(
                    templates=weights,
                    num_experts=num_experts,
                    device=layer.w13_weight.device,
                    use_w8a8=layer.uses_w8a8,
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
