from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Set

import torch
import torch.nn as nn

from .config import ExpertWiseOffloadConfig


@dataclass(frozen=True)
class ExpertPlacement:
    global_expert_id: int
    local_expert_id: int
    device: str


class ExpertWiseExpertStore:
    """
    CPU expert store plus logical NPU cache state for one FusedMoE layer.

    By default this copies selected experts back into AscendFusedMoE's dense
    local expert slices. When compact_npu_cache is enabled, the dense local
    expert tensor is replaced by resident slots plus finite cache slots.
    """

    def __init__(
        self,
        *,
        layer_idx: Optional[int],
        config: ExpertWiseOffloadConfig,
    ):
        self.layer_idx = layer_idx
        self.config = config
        self.cpu_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        self.placements: Dict[int, ExpertPlacement] = {}
        self.loaded_experts: Set[int] = set()
        self.cache_lru: OrderedDict[int, None] = OrderedDict()
        self.copy_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        self.transfer_logs = 0
        self.compact_enabled = False
        self.global_to_slot: Dict[int, int] = {}
        self.resident_experts: Set[int] = set()
        self.cache_slots: OrderedDict[int, int] = OrderedDict()
        self.free_cache_slots: list[int] = []

    @property
    def enabled(self) -> bool:
        return bool(self.cpu_weights)

    @torch.no_grad()
    def init_cpu_store(
        self,
        fused_moe,
        selected_global_expert_ids: Iterable[int],
        local_expert_id_fn: Callable[[int], Optional[int]],
    ) -> None:
        for global_expert_id in sorted(set(selected_global_expert_ids)):
            local_expert_id = local_expert_id_fn(global_expert_id)
            if local_expert_id is None:
                continue

            self.cpu_weights[global_expert_id] = self._copy_expert_weights_to_cpu(
                fused_moe=fused_moe,
                local_expert_id=local_expert_id,
            )
            self.placements[global_expert_id] = ExpertPlacement(
                global_expert_id=global_expert_id,
                local_expert_id=local_expert_id,
                device="cpu",
            )

    @torch.no_grad()
    def maybe_compact_npu_weights(self, fused_moe) -> None:
        if not self.config.compact_npu_cache or not self.cpu_weights:
            return

        if fused_moe.expert_map is None:
            raise RuntimeError("compact_npu_cache requires expert parallel expert_map.")

        local_global_to_slot = self._local_global_to_slot(fused_moe.expert_map)
        resident_experts = [
            global_expert_id
            for global_expert_id, slot in sorted(
                local_global_to_slot.items(),
                key=lambda item: item[1],
            )
            if global_expert_id not in self.cpu_weights
        ]
        cache_capacity = self.config.npu_cache_capacity
        total_slots = len(resident_experts) + cache_capacity
        if total_slots <= 0:
            raise RuntimeError("compact_npu_cache needs at least one NPU slot.")

        old_w13 = fused_moe.w13_weight.data
        old_w2 = fused_moe.w2_weight.data

        new_w13 = torch.empty(
            (total_slots, *old_w13.shape[1:]),
            device=old_w13.device,
            dtype=old_w13.dtype,
        )
        new_w2 = torch.empty(
            (total_slots, *old_w2.shape[1:]),
            device=old_w2.device,
            dtype=old_w2.dtype,
        )

        new_w13_bias = None
        new_w2_bias = None
        if hasattr(fused_moe, "w13_bias"):
            old_w13_bias = fused_moe.w13_bias.data
            new_w13_bias = torch.empty(
                (total_slots, *old_w13_bias.shape[1:]),
                device=old_w13_bias.device,
                dtype=old_w13_bias.dtype,
            )
        if hasattr(fused_moe, "w2_bias"):
            old_w2_bias = fused_moe.w2_bias.data
            new_w2_bias = torch.empty(
                (total_slots, *old_w2_bias.shape[1:]),
                device=old_w2_bias.device,
                dtype=old_w2_bias.dtype,
            )

        new_expert_map = torch.full_like(fused_moe.expert_map, -1)
        self.global_to_slot.clear()
        self.resident_experts = set(resident_experts)

        for slot, global_expert_id in enumerate(resident_experts):
            old_slot = local_global_to_slot[global_expert_id]
            new_w13[slot].copy_(old_w13[old_slot])
            new_w2[slot].copy_(old_w2[old_slot])
            if new_w13_bias is not None:
                new_w13_bias[slot].copy_(old_w13_bias[old_slot])
            if new_w2_bias is not None:
                new_w2_bias[slot].copy_(old_w2_bias[old_slot])
            new_expert_map[global_expert_id] = slot
            self.global_to_slot[global_expert_id] = slot

        self.free_cache_slots = list(range(len(resident_experts), total_slots))
        self.cache_slots.clear()
        self.loaded_experts = set(resident_experts)

        fused_moe.w13_weight = nn.Parameter(new_w13, requires_grad=False)
        fused_moe.w2_weight = nn.Parameter(new_w2, requires_grad=False)
        if new_w13_bias is not None:
            fused_moe.w13_bias = nn.Parameter(new_w13_bias, requires_grad=False)
        if new_w2_bias is not None:
            fused_moe.w2_bias = nn.Parameter(new_w2_bias, requires_grad=False)

        fused_moe._expert_map = new_expert_map
        fused_moe.local_num_experts = total_slots
        fused_moe.moe_config.num_local_experts = total_slots
        self.compact_enabled = True
        print(
            "[Plugin] Expert-wise compact NPU cache initialized: "
            f"layer={self.layer_idx}, resident={len(resident_experts)}, "
            f"cache_capacity={cache_capacity}, total_slots={total_slots}"
        )

    @torch.no_grad()
    def restore_routed_experts(self, fused_moe, routed_global_expert_ids: Set[int]) -> None:
        if self.compact_enabled:
            self._restore_routed_experts_compact(fused_moe, routed_global_expert_ids)
            return

        for global_expert_id in sorted(routed_global_expert_ids):
            if global_expert_id not in self.cpu_weights:
                continue

            if (
                self.config.keep_loaded_on_npu
                and global_expert_id in self.loaded_experts
            ):
                self.cache_hits += 1
                self._touch_cached_expert(global_expert_id)
                continue

            self.cache_misses += 1
            self._copy_cpu_expert_to_npu(fused_moe, global_expert_id)

    def mark_loaded_experts_evicted_if_needed(self, fused_moe=None) -> None:
        if self.config.keep_loaded_on_npu:
            return

        if self.compact_enabled:
            if fused_moe is None:
                raise RuntimeError("compact cache eviction requires fused_moe.")
            for global_expert_id, slot in list(self.cache_slots.items()):
                fused_moe._expert_map[global_expert_id] = -1
                self.placements[global_expert_id] = ExpertPlacement(
                    global_expert_id=global_expert_id,
                    local_expert_id=slot,
                    device="cpu",
                )
                self.loaded_experts.discard(global_expert_id)
                self.global_to_slot.pop(global_expert_id, None)
                self.free_cache_slots.append(slot)
            self.cache_slots.clear()
            return

        for global_expert_id in list(self.loaded_experts):
            self._mark_expert_on_cpu(global_expert_id)
        self.loaded_experts.clear()
        self.cache_lru.clear()

    def offloaded_expert_ids(self) -> Set[int]:
        return set(self.cpu_weights)

    def is_compact(self) -> bool:
        return self.compact_enabled

    def summary(self) -> Dict[str, object]:
        return {
            "layer": self.layer_idx,
            "offloaded_experts": sorted(self.cpu_weights),
            "loaded_experts": sorted(self.loaded_experts),
            "placements": self.placements,
            "copy_count": self.copy_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "evictions": self.evictions,
            "compact_enabled": self.compact_enabled,
        }

    @staticmethod
    def _local_global_to_slot(expert_map: torch.Tensor) -> Dict[int, int]:
        expert_map_cpu = expert_map.detach().cpu()
        return {
            int(global_expert_id): int(local_slot)
            for global_expert_id, local_slot in enumerate(expert_map_cpu.tolist())
            if int(local_slot) >= 0
        }

    def _restore_routed_experts_compact(
        self,
        fused_moe,
        routed_global_expert_ids: Set[int],
    ) -> None:
        routed = set(routed_global_expert_ids)
        if len(routed) > self.config.npu_cache_capacity:
            raise RuntimeError(
                "Routed offloaded experts exceed compact NPU cache capacity: "
                f"routed={len(routed)}, capacity={self.config.npu_cache_capacity}, "
                f"layer={self.layer_idx}."
            )

        for global_expert_id in sorted(routed):
            if global_expert_id in self.cache_slots:
                self.cache_hits += 1
                self.cache_slots.move_to_end(global_expert_id)
                continue

            self.cache_misses += 1
            slot = self._acquire_cache_slot(
                fused_moe=fused_moe,
                protected_experts=routed,
            )
            self._copy_cpu_expert_to_slot(fused_moe, global_expert_id, slot)

    def _acquire_cache_slot(self, fused_moe, protected_experts: Set[int]) -> int:
        if self.free_cache_slots:
            return self.free_cache_slots.pop(0)

        for evicted_expert_id, slot in list(self.cache_slots.items()):
            if evicted_expert_id in protected_experts:
                continue

            self.cache_slots.pop(evicted_expert_id)
            fused_moe._expert_map[evicted_expert_id] = -1
            self.global_to_slot.pop(evicted_expert_id, None)
            self.loaded_experts.discard(evicted_expert_id)
            self.placements[evicted_expert_id] = ExpertPlacement(
                global_expert_id=evicted_expert_id,
                local_expert_id=slot,
                device="cpu",
            )
            self.evictions += 1
            return slot

        raise RuntimeError(
            "No evictable compact NPU cache slot is available for "
            f"layer={self.layer_idx}."
        )

    def _copy_cpu_expert_to_slot(
        self,
        fused_moe,
        global_expert_id: int,
        slot: int,
    ) -> None:
        cpu_weights = self.cpu_weights[global_expert_id]
        fused_moe.w13_weight.data[slot].copy_(
            cpu_weights["w13_weight"],
            non_blocking=True,
        )
        fused_moe.w2_weight.data[slot].copy_(
            cpu_weights["w2_weight"],
            non_blocking=True,
        )

        if "w13_bias" in cpu_weights:
            fused_moe.w13_bias.data[slot].copy_(
                cpu_weights["w13_bias"],
                non_blocking=True,
            )
        if "w2_bias" in cpu_weights:
            fused_moe.w2_bias.data[slot].copy_(
                cpu_weights["w2_bias"],
                non_blocking=True,
            )

        fused_moe._expert_map[global_expert_id] = slot
        self.global_to_slot[global_expert_id] = slot
        self.cache_slots[global_expert_id] = slot
        self.cache_slots.move_to_end(global_expert_id)
        self.loaded_experts.add(global_expert_id)
        self.copy_count += 1
        self.placements[global_expert_id] = ExpertPlacement(
            global_expert_id=global_expert_id,
            local_expert_id=slot,
            device="npu",
        )
        self._log_transfer_if_enabled(global_expert_id, slot)

    def _copy_expert_weights_to_cpu(
        self,
        *,
        fused_moe,
        local_expert_id: int,
    ) -> Dict[str, torch.Tensor]:
        weights = {
            "w13_weight": fused_moe.w13_weight.data[local_expert_id].detach(),
            "w2_weight": fused_moe.w2_weight.data[local_expert_id].detach(),
        }

        if hasattr(fused_moe, "w13_bias"):
            weights["w13_bias"] = fused_moe.w13_bias.data[local_expert_id].detach()
        if hasattr(fused_moe, "w2_bias"):
            weights["w2_bias"] = fused_moe.w2_bias.data[local_expert_id].detach()

        return {name: self._to_cpu_tensor(tensor) for name, tensor in weights.items()}

    def _to_cpu_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        cpu_tensor = tensor.cpu().contiguous()
        if not self.config.pin_cpu_memory:
            return cpu_tensor

        try:
            return cpu_tensor.pin_memory()
        except RuntimeError:
            return cpu_tensor

    def _copy_cpu_expert_to_npu(self, fused_moe, global_expert_id: int) -> None:
        placement = self.placements[global_expert_id]
        cpu_weights = self.cpu_weights[global_expert_id]
        local_expert_id = placement.local_expert_id

        fused_moe.w13_weight.data[local_expert_id].copy_(
            cpu_weights["w13_weight"],
            non_blocking=True,
        )
        fused_moe.w2_weight.data[local_expert_id].copy_(
            cpu_weights["w2_weight"],
            non_blocking=True,
        )

        if "w13_bias" in cpu_weights:
            fused_moe.w13_bias.data[local_expert_id].copy_(
                cpu_weights["w13_bias"],
                non_blocking=True,
            )
        if "w2_bias" in cpu_weights:
            fused_moe.w2_bias.data[local_expert_id].copy_(
                cpu_weights["w2_bias"],
                non_blocking=True,
            )

        self.copy_count += 1
        self.placements[global_expert_id] = ExpertPlacement(
            global_expert_id=global_expert_id,
            local_expert_id=local_expert_id,
            device="npu",
        )
        self._remember_loaded_expert(global_expert_id)
        self._log_transfer_if_enabled(global_expert_id, local_expert_id)

    def _remember_loaded_expert(self, global_expert_id: int) -> None:
        self.loaded_experts.add(global_expert_id)
        self.cache_lru[global_expert_id] = None
        self.cache_lru.move_to_end(global_expert_id)

        capacity = self.config.npu_cache_capacity
        if capacity <= 0:
            return

        while len(self.cache_lru) > capacity:
            evicted_expert_id, _ = self.cache_lru.popitem(last=False)
            if evicted_expert_id == global_expert_id:
                self.cache_lru[evicted_expert_id] = None
                break
            self._mark_expert_on_cpu(evicted_expert_id)
            self.evictions += 1

    def _touch_cached_expert(self, global_expert_id: int) -> None:
        if global_expert_id in self.cache_lru:
            self.cache_lru.move_to_end(global_expert_id)

    def _mark_expert_on_cpu(self, global_expert_id: int) -> None:
        placement = self.placements[global_expert_id]
        self.loaded_experts.discard(global_expert_id)
        if self.compact_enabled:
            self.global_to_slot.pop(global_expert_id, None)
        self.placements[global_expert_id] = ExpertPlacement(
            global_expert_id=global_expert_id,
            local_expert_id=placement.local_expert_id,
            device="cpu",
        )

    def _log_transfer_if_enabled(
        self,
        global_expert_id: int,
        local_expert_id: int,
    ) -> None:
        if not self.config.log_transfers:
            return
        if self.transfer_logs >= self.config.max_transfer_logs:
            return

        self.transfer_logs += 1
        suffix = ""
        if self.transfer_logs == self.config.max_transfer_logs:
            suffix = "; transfer log limit reached"

        print(
            "[Plugin] Expert-wise CPU->NPU copy: "
            f"layer={self.layer_idx}, "
            f"global_expert={global_expert_id}, "
            f"local_expert={local_expert_id}, "
            f"copy_count={self.copy_count}"
            f"{suffix}"
        )
