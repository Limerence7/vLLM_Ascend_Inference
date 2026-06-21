from typing import NamedTuple

import torch
import torch.nn as nn
import torch_npu

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

from ..offload_config import OffloadConfig
from ..loadbalance import (DynamicExpertScheduler, DynamicLoadPolicy,
                           ExpertLoadStats, ExpertPlacement, ExpertSwap,
                           HistoryLoadPolicy)
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
        if self.load_stream is not None:
            torch.npu.current_stream().wait_stream(self.load_stream)


class PreparedColdExperts(NamedTuple):
    experts: ColdExperts
    topk_ids: torch.Tensor | None
    mask: torch.Tensor | None


class OffloadExecutor(nn.Module):
    """Controls layer-ahead cold expert prefetch with reusable buffers."""

    def __init__(self, config: OffloadConfig, uses_w8a8: bool):
        super().__init__()
        self.config = config
        self.offload_full_layer = config.offload_full_layers
        self.dynamic_load_balance = config.load_balance_mode == "dynamic"
        self.offloaded_layer_ids = set(config.offloaded_layer_ids)
        self.memory_manager = ExpertMemoryManager(
            cpu_pin_memory=config.cpu_pin_memory)

        self.layers: dict[int, nn.Module] = {}
        self.cold_experts = nn.ModuleList()
        self.uses_w8a8 = uses_w8a8
        self.layer_ids: list[int] = []
        self.next_layer_by_id: dict[int, int] = {}
        self.buffer_idx = 0
        self.buffer_status: list[int | None] = []
        self.placements: dict[int, ExpertPlacement] = {}
        self.routing_maps: dict[int, LayerRoutingMap] = {}
        self.prefetch_stream: torch.npu.Stream | None = None
        self.swap_stream: torch.npu.Stream | None = None
        self.load_stats = self._init_load_stats(config)
        history_stats = (
            self.load_stats
            if config.load_balance_mode in ("history", "dynamic") else None)
        self.history_policy = HistoryLoadPolicy(history_stats)
        self.dynamic_scheduler = self._init_dynamic_scheduler(config)

    def _init_dynamic_scheduler(
            self, config: OffloadConfig) -> DynamicExpertScheduler | None:
        if not self.dynamic_load_balance:
            return None

        policy = DynamicLoadPolicy(
            self.load_stats,
            config.dynamic_max_swaps,
            config.dynamic_min_swap_gain,
        )
        return DynamicExpertScheduler(
            policy,
            config.dynamic_update_interval,
            config.dynamic_cooldown_interval,
        )

    def _init_load_stats(self,
                         config: OffloadConfig) -> ExpertLoadStats | None:
        if config.load_stats_path or config.load_balance_mode != "none":
            return ExpertLoadStats(
                config.load_stats_path,
                metadata={
                    "offload_mode": config.mode,
                    "load_balance_mode": config.load_balance_mode,
                    "offloaded_layer_ids": list(config.offloaded_layer_ids),
                },
            )
        return None

    def init_cold_buffers(self) -> None:
        if self.cold_experts:
            return

        self.num_buffers = self.config.num_buffers
        template_id = self.layer_ids[0]
        template_weights = self.memory_manager.get_expert_weights(template_id)
        num_experts = max(
            len(placement.cold_expert_ids)
            for placement in self.placements.values()
        )

        self.buffer_idx = 0
        self.buffer_status = [None] * self.num_buffers
        self.next_layer_by_id = {
            layer_id: self.layer_ids[(index + 1) % len(self.layer_ids)]
            for index, layer_id in enumerate(self.layer_ids)
        }
        for _ in range(self.num_buffers):
            self.cold_experts.append(
                ColdExperts(
                    templates=template_weights,
                    num_experts=num_experts,
                    device=self.layers[template_id].w13_weight.device,
                    use_w8a8=self.uses_w8a8,
                ))

    def register_layer(self, layer) -> None:
        layer_id = int(layer.moe_instance_id)
        self.layers[layer_id] = layer

        placement = self.placements.get(layer_id)
        if placement is None:
            placement = self.init_layer_placement(layer)
        self.layer_ids.append(layer_id)
        self.layer_ids.sort()
        self.routing_maps[layer_id] = LayerRoutingMap(layer, placement)
        if self.load_stats is not None:
            self.load_stats.register_layer(layer)

        self.memory_manager.register_layer(
            layer_id,
            layer,
            self._stored_expert_ids(layer, placement),
        )

    def init_layer_placement(self, layer) -> ExpertPlacement:
        layer_id = layer.moe_instance_id
        placement = self.history_policy.placement_for_layer(
            layer, layer.resident_local_num_experts)
        self.placements[layer_id] = placement
        return placement

    def is_cold_expert(self, layer, local_expert_id: int) -> bool:
        placement = self.placements.get(layer.moe_instance_id)
        if placement is None:
            return False
        return local_expert_id in placement.cold_slots

    def should_offload_expert(self, layer, local_expert_id: int) -> bool:
        return (
            self.should_offload_layer(layer) and local_expert_id >= 0 and
            (self.offload_full_layer
             or self.is_cold_expert(layer, local_expert_id))
        )

    def should_store_cpu_expert(self, layer, local_expert_id: int) -> bool:
        if not self.should_offload_layer(layer) or local_expert_id < 0:
            return False
        return (
            self.dynamic_load_balance
            or self.offload_full_layer
            or self.is_cold_expert(layer, local_expert_id)
        )

    def should_offload_layer(self, layer) -> bool:
        return layer.moe_instance_id in self.offloaded_layer_ids

    def load_weight(self, layer, param_name: str, shard_id: str,
                    global_expert_id: int,
                    loaded_weight: torch.Tensor) -> bool:
        local_expert_id = layer.map_global_expert_id_to_full_local_expert_id(
            global_expert_id)
        if not self.should_store_cpu_expert(layer, local_expert_id):
            return False

        loaded = self.memory_manager.load_weight_shard(
            layer_id=layer.moe_instance_id,
            param_name=param_name,
            local_expert_id=local_expert_id,
            shard_id=shard_id,
            loaded_weight=loaded_weight,
            tp_rank=layer.tp_rank,
        )
        return loaded and self.should_offload_expert(layer, local_expert_id)

    def process_layer_after_loading(self, layer, quant_method) -> None:
        self.memory_manager.process_layer_after_loading(layer.moe_instance_id,
                                                        quant_method)

    def prepare_cold_experts(self, layer,
                             topk_ids: torch.Tensor) -> PreparedColdExperts:
        current_layer = layer.moe_instance_id
        if not self.cold_experts:
            self.init_cold_buffers()
        if self.dynamic_scheduler is not None and not self.offload_full_layer:
            self.dynamic_scheduler.maybe_update(self, layer)
        if current_layer == self.buffer_status[self.buffer_idx]:
            cold_expert = self.cold_experts[self.buffer_idx]
        else:
            cold_expert = self.start_prefetch(layer, self.buffer_idx)
        self.buffer_idx = (self.buffer_idx + 1) % self.num_buffers

        if self.offload_full_layer:
            return PreparedColdExperts(cold_expert, None, None)

        cold_topk_ids, cold_mask = self.routing_maps[
            layer.moe_instance_id].cold_routing(topk_ids)
        return PreparedColdExperts(cold_expert, cold_topk_ids, cold_mask)

    def prefetch_next_layers(self, layer) -> None:
        next_layer_id = self.next_layer_by_id[layer.moe_instance_id]
        if next_layer_id == self.buffer_status[self.buffer_idx]:
            return

        self.start_prefetch(self.layers[next_layer_id], self.buffer_idx)

    def start_prefetch(self, layer, buffer_id: int) -> ColdExperts:
        current_layer = layer.moe_instance_id
        cold_expert_ids = self.placements[current_layer].cold_expert_ids

        cold_expert = self.cold_experts[buffer_id]

        if self.prefetch_stream is None:
            self.prefetch_stream = torch.npu.Stream()
        stream = self.prefetch_stream
        with torch.npu.stream(stream):
            if self.dynamic_load_balance:
                with torch.no_grad():
                    self.memory_manager.copy_experts_to_module(
                        current_layer, cold_expert_ids, cold_expert)
            else:
                weights = self.memory_manager.get_expert_weights(current_layer)
                cold_expert.load_from_cpu(weights)
        cold_expert.load_stream = stream
        self.buffer_status[buffer_id] = current_layer
        return cold_expert

    def commit_layer_placement(
        self,
        layer,
        placement: ExpertPlacement,
        swaps: list[ExpertSwap] | None = None,
    ) -> None:
        layer_id = int(layer.moe_instance_id)
        self.placements[layer_id] = placement
        if self.dynamic_load_balance and swaps:
            self.routing_maps[layer_id].apply_placement(placement, swaps)
            layer.apply_expert_placement(placement, swaps)
            self._patch_loaded_cold_buffers(layer_id, swaps)
        else:
            self.routing_maps[layer_id] = LayerRoutingMap(layer, placement)
            layer.apply_expert_placement(placement)
            self._invalidate_layer_buffers(layer_id)

    def copy_swaps_to_resident(self, layer,
                               swaps: list[ExpertSwap]) -> None:
        layer_id = int(layer.moe_instance_id)
        expert_ids = [swap.swap_in for swap in swaps]
        target_slots = [swap.resident_slot for swap in swaps]
        if self.swap_stream is None:
            self.swap_stream = torch.npu.Stream()
        stream = self.swap_stream
        with torch.npu.stream(stream), torch.no_grad():
            self.memory_manager.copy_experts_to_module(
                layer_id, expert_ids, layer, target_slots)
        torch.npu.current_stream().wait_stream(stream)

    def _invalidate_layer_buffers(self, layer_id: int) -> None:
        self.buffer_status = [
            None if status == layer_id else status
            for status in self.buffer_status
        ]

    def _patch_loaded_cold_buffers(self, layer_id: int,
                                   swaps: list[ExpertSwap]) -> None:
        expert_ids = [swap.swap_out for swap in swaps]
        target_slots = [swap.cold_slot for swap in swaps]
        if self.prefetch_stream is None:
            self.prefetch_stream = torch.npu.Stream()
        stream = self.prefetch_stream
        for buffer_id, status in enumerate(self.buffer_status):
            if status != layer_id:
                continue
            cold_expert = self.cold_experts[buffer_id]
            if cold_expert.load_stream is not None:
                stream.wait_stream(cold_expert.load_stream)
            with torch.npu.stream(stream), torch.no_grad():
                self.memory_manager.copy_experts_to_module(
                    layer_id, expert_ids, cold_expert, target_slots)
            cold_expert.load_stream = stream

    def _stored_expert_ids(self, layer,
                           placement: ExpertPlacement) -> list[int]:
        if self.dynamic_load_balance:
            return list(range(layer.full_local_num_experts))
        return placement.cold_expert_ids

    def record_load(self, layer, topk_ids: torch.Tensor) -> None:
        if self.load_stats is not None:
            self.load_stats.record(layer.moe_instance_id, topk_ids)

    def save_load_stats(self) -> None:
        if self.load_stats is not None:
            self.load_stats.save()
