from typing import NamedTuple

import torch
import torch.nn as nn
import torch_npu
from vllm.forward_context import get_forward_context

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

from ..moeload.history_mapping import HistoryExpertMap, ranked_experts_by_load
from ..moeload.profiler import ExpertLoadProfiler
from ..runtime_config import RuntimeConfig
from .memory_manager import ExpertMemoryManager
from .routing import ExpertPlacement, LayerRoutingMap


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
                    getattr(self, f"{name}_fp32")[:source.size(0)].copy_(
                        source, non_blocking=True)

    def wait(self) -> None:
        if self.load_stream is not None:
            torch.npu.current_stream().wait_stream(self.load_stream)


class PreparedColdExperts(NamedTuple):
    experts: ColdExperts
    topk_ids: torch.Tensor | None
    mask: torch.Tensor | None


def _layer_device(layer) -> torch.device:
    weight = getattr(layer, "w13_weight", None)
    if weight is not None:
        return weight.device

    weight_list = getattr(layer, "w13_weight_list", None)
    if weight_list:
        return weight_list[0].device

    return next(layer.parameters()).device


class ExoExecutor(nn.Module):
    """Runtime executor for CPU-to-NPU expert offload."""

    def __init__(self, config: RuntimeConfig, uses_w8a8: bool):
        super().__init__()
        self.config = config
        self.uses_w8a8 = uses_w8a8
        self.memory_manager = ExpertMemoryManager(config.cpu_pin_memory)
        self.profiler = ExpertLoadProfiler(
            config.load_history_path,
            metadata={
                "runtime_mode": config.runtime_mode,
                "runtime_layer_ids": list(config.runtime_layer_ids),
            },
        )
        self.history_mapping = (
            HistoryExpertMap(config.load_history_path)
            if config.enable_history_mapping else None)

        self.layers: dict[int, nn.Module] = {}
        self.layer_ids: list[int] = []
        self.placements: dict[int, ExpertPlacement] = {}
        self.routing_maps: dict[int, LayerRoutingMap] = {}
        self._resident_slot_to_global: dict[int, torch.Tensor] = {}
        self._cold_slot_to_global: dict[int, torch.Tensor] = {}
        self.cold_experts = nn.ModuleList()
        self.buffer_idx = 0
        self.buffer_status: list[int | None] = []
        self.next_layer_by_id: dict[int, int] = {}
        self.prefetch_stream: torch.npu.Stream | None = None

    @property
    def offload_full_layer(self) -> bool:
        return self.config.offload_layer_wise

    def should_manage_layer(self, layer) -> bool:
        return layer.moe_instance_id in set(self.config.runtime_layer_ids)

    def should_store_cpu_expert(self, layer, local_expert_id: int) -> bool:
        if not self.should_manage_layer(layer) or local_expert_id < 0:
            return False
        return self.offload_full_layer or local_expert_id in self.placements[
            layer.moe_instance_id].cold_slots

    def history_expert_map_for_layer(self, layer) -> torch.Tensor | None:
        if self.history_mapping is None:
            return None
        return self.history_mapping.expert_map_for_layer(layer)

    def init_layer_placement(self, layer) -> ExpertPlacement:
        num_resident = self._num_resident_experts(layer)
        resident_ids = self._history_resident_ids(layer, num_resident)
        if resident_ids is None:
            resident_ids = list(range(num_resident))

        resident_set = set(resident_ids)
        cold_ids = [
            expert_id for expert_id in range(layer.full_local_num_experts)
            if expert_id not in resident_set
        ]
        placement = ExpertPlacement(resident_ids, cold_ids)
        self.placements[int(layer.moe_instance_id)] = placement
        return placement

    def register_layer(self, layer) -> None:
        layer_id = int(layer.moe_instance_id)
        self.layers[layer_id] = layer
        self.layer_ids = sorted({*self.layer_ids, layer_id})
        self.profiler.register_layer(layer)

        if self.config.runtime_mode != "offload":
            return

        placement = self.placements[layer_id]
        self.routing_maps[layer_id] = LayerRoutingMap(layer, placement)
        self._resident_slot_to_global[layer_id] = self._slot_to_global(
            layer, placement.resident_expert_ids)
        self._cold_slot_to_global[layer_id] = self._slot_to_global(
            layer, placement.cold_expert_ids)
        self.memory_manager.register_layer(
            layer_id,
            layer,
            placement.cold_expert_ids,
        )

    def load_weight(self, layer, param_name: str, shard_id: str,
                    global_expert_id: int,
                    loaded_weight: torch.Tensor) -> bool:
        local_expert_id = layer.map_global_expert_id_to_full_local_expert_id(
            global_expert_id)
        if not self.should_store_cpu_expert(layer, local_expert_id):
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
        if self.config.runtime_mode != "offload":
            return
        self.memory_manager.process_layer_after_loading(layer.moe_instance_id,
                                                        quant_method)
        self._init_cold_buffers()
        torch.npu.empty_cache()

    def prepare_cold_experts(self, layer,
                             topk_ids: torch.Tensor) -> PreparedColdExperts:
        if not self.cold_experts:
            self._init_cold_buffers()
        if not self.cold_experts:
            raise RuntimeError("No cold expert buffer is available.")

        current_layer = int(layer.moe_instance_id)
        if current_layer == self.buffer_status[self.buffer_idx]:
            cold_expert = self.cold_experts[self.buffer_idx]
        else:
            cold_expert = self._start_prefetch(layer, self.buffer_idx)
        self.buffer_idx = (self.buffer_idx + 1) % self.config.num_buffers

        if self.offload_full_layer:
            return PreparedColdExperts(cold_expert, None, None)

        cache_maps = not get_forward_context().in_profile_run
        cold_topk_ids, cold_mask = self.routing_maps[
            current_layer].cold_routing(topk_ids, cache_maps)
        return PreparedColdExperts(cold_expert, cold_topk_ids, cold_mask)

    def prefetch_next_layers(self, layer) -> None:
        if len(self.layer_ids) <= 1:
            return
        next_layer_id = self.next_layer_by_id[int(layer.moe_instance_id)]
        if next_layer_id == self.buffer_status[self.buffer_idx]:
            return
        self._start_prefetch(self.layers[next_layer_id], self.buffer_idx)

    def record_expert_tokens(self, layer, group_list_type: int,
                             expert_tokens: torch.Tensor) -> None:
        self.profiler.record_expert_tokens(
            int(layer.moe_instance_id),
            expert_tokens,
            group_list_type,
        )

    def record_slot_expert_tokens(
        self,
        layer,
        group_list_type: int,
        expert_tokens: torch.Tensor,
        slot_to_global: torch.Tensor,
    ) -> None:
        self.profiler.record_slot_tokens(
            int(layer.moe_instance_id),
            expert_tokens,
            group_list_type,
            slot_to_global,
        )

    def resident_slot_to_global(self, layer) -> torch.Tensor:
        return self._resident_slot_to_global[int(layer.moe_instance_id)]

    def cold_slot_to_global(self, layer) -> torch.Tensor:
        return self._cold_slot_to_global[int(layer.moe_instance_id)]

    def save_load_history(self) -> None:
        self.profiler.save()

    def _init_cold_buffers(self) -> None:
        if self.cold_experts:
            return

        template_id = self.layer_ids[0]
        template_weights = self.memory_manager.get_expert_weights(template_id)
        max_cold_experts = max(
            len(placement.cold_expert_ids)
            for placement in self.placements.values())
        if max_cold_experts == 0:
            return

        self.buffer_status = [None] * self.config.num_buffers
        self.next_layer_by_id = {
            layer_id: self.layer_ids[(index + 1) % len(self.layer_ids)]
            for index, layer_id in enumerate(self.layer_ids)
        }
        for _ in range(self.config.num_buffers):
            self.cold_experts.append(
                ColdExperts(
                    templates=template_weights,
                    num_experts=max_cold_experts,
                    device=_layer_device(self.layers[template_id]),
                    use_w8a8=self.uses_w8a8,
                ))

    def _start_prefetch(self, layer, buffer_id: int) -> ColdExperts:
        layer_id = int(layer.moe_instance_id)
        cold_expert = self.cold_experts[buffer_id]
        weights = self.memory_manager.get_expert_weights(layer_id)

        stream = self._prefetch_stream()
        with torch.npu.stream(stream):
            cold_expert.load_from_cpu(weights)
        cold_expert.load_stream = stream
        self.buffer_status[buffer_id] = layer_id
        return cold_expert

    def _num_resident_experts(self, layer) -> int:
        if self.config.runtime_mode != "offload":
            return int(layer.full_local_num_experts)
        if self.offload_full_layer:
            return 0
        return min(int(self.config.num_hot_experts),
                   int(layer.full_local_num_experts))

    def _history_resident_ids(self, layer,
                              num_resident: int) -> list[int] | None:
        if num_resident <= 0 or self.history_mapping is None:
            return None

        load = self.history_mapping.global_load_for_layer(layer)
        if load is None or load.numel() != int(layer.global_num_experts):
            return None

        local_experts = self._local_experts(layer)
        local_by_global = {global_id: local_id for local_id, global_id in local_experts}
        ranked_global = ranked_experts_by_load(load, list(local_by_global))
        return [local_by_global[global_id] for global_id in ranked_global[:num_resident]]

    @staticmethod
    def _local_experts(layer) -> list[tuple[int, int]]:
        if layer.full_expert_map is None:
            return [
                (expert_id, expert_id)
                for expert_id in range(int(layer.global_num_experts))
            ]

        expert_map = layer.full_expert_map.detach().cpu()
        return [(int(local_id), int(global_id))
                for global_id, local_id in enumerate(expert_map.tolist())
                if local_id >= 0]

    @staticmethod
    def _slot_to_global(layer, local_expert_ids: list[int]) -> torch.Tensor:
        if layer.full_expert_map is None:
            return torch.tensor(local_expert_ids, dtype=torch.long)

        full_map = layer.full_expert_map.detach().cpu()
        local_to_global = {
            int(local_id): int(global_id)
            for global_id, local_id in enumerate(full_map.tolist())
            if local_id >= 0
        }
        return torch.tensor(
            [local_to_global.get(int(local_id), -1)
             for local_id in local_expert_ids],
            dtype=torch.long,
        )

    def _prefetch_stream(self) -> torch.npu.Stream:
        if self.prefetch_stream is None:
            self.prefetch_stream = torch.npu.Stream()
        return self.prefetch_stream
