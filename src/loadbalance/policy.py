from dataclasses import dataclass, field

import torch

from .load_stats import ExpertLoadStats


@dataclass(frozen=True)
class ExpertPlacement:
    resident_expert_ids: list[int]
    cold_expert_ids: list[int]
    resident_slots: dict[int, int] = field(init=False)
    cold_slots: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "resident_slots", {
            expert_id: slot
            for slot, expert_id in enumerate(self.resident_expert_ids)
        })
        object.__setattr__(self, "cold_slots", {
            expert_id: slot
            for slot, expert_id in enumerate(self.cold_expert_ids)
        })


@dataclass(frozen=True)
class ExpertSwap:
    swap_in: int
    swap_out: int
    resident_slot: int
    cold_slot: int


class HistoryLoadPolicy:
    """Choose resident experts from persisted load stats."""

    def __init__(self, load_stats: ExpertLoadStats | None):
        self.load_stats = load_stats

    def placement_for_layer(
        self,
        layer,
        num_resident_experts: int,
        global_load: torch.Tensor | None = None,
    ) -> ExpertPlacement:
        local_num_experts = int(layer.full_local_num_experts)
        num_resident_experts = min(num_resident_experts, local_num_experts)

        resident_ids = self._history_resident_ids(
            layer, num_resident_experts, global_load)
        if resident_ids is None:
            resident_ids = list(range(num_resident_experts))

        resident_set = set(resident_ids)
        cold_ids = [
            expert_id for expert_id in range(local_num_experts)
            if expert_id not in resident_set
        ]
        return ExpertPlacement(resident_ids, cold_ids)

    def _history_resident_ids(
        self,
        layer,
        num_resident_experts: int,
        global_load: torch.Tensor | None = None,
    ) -> list[int] | None:
        if num_resident_experts <= 0:
            return None

        load = global_load
        if load is None:
            if self.load_stats is None:
                return None
            try:
                load = self.load_stats.get_layer_load(
                    int(layer.moe_instance_id))
            except KeyError:
                return None

        if load.numel() != int(layer.global_num_experts):
            return None

        local_experts = self._local_experts(layer)
        ranked = sorted(
            local_experts,
            key=lambda item: (-int(load[item[1]].item()), item[1]),
        )
        return [local_id for local_id, _ in ranked[:num_resident_experts]]

    @staticmethod
    def _local_experts(layer) -> list[tuple[int, int]]:
        if layer.full_expert_map is None:
            return [
                (expert_id, expert_id)
                for expert_id in range(int(layer.global_num_experts))
            ]

        expert_map = layer.full_expert_map.detach().cpu()
        local_experts: list[tuple[int, int]] = []
        for global_id, local_id in enumerate(expert_map.tolist()):
            if local_id >= 0:
                local_experts.append((int(local_id), int(global_id)))
        return local_experts


class DynamicLoadPolicy:
    """Choose small resident/cold swaps from recent rank-local load."""

    def __init__(self, load_stats: ExpertLoadStats | None, max_swaps: int,
                 min_swap_gain: int = 0):
        self.load_stats = load_stats
        self.max_swaps = max(0, int(max_swaps))
        self.min_swap_gain = max(1, int(min_swap_gain))
        self._last_loads: dict[int, torch.Tensor] = {}

    def plan_for_layer(self, layer,
                       placement: ExpertPlacement) -> list[ExpertSwap]:
        if self.load_stats is None or self.max_swaps == 0:
            return []

        layer_id = int(layer.moe_instance_id)
        try:
            current_load = self.load_stats.get_layer_load(layer_id)
        except KeyError:
            return []

        previous_load = self._last_loads.get(layer_id)
        self._last_loads[layer_id] = current_load
        if (previous_load is None
                or previous_load.numel() != current_load.numel()):
            return []

        window_load = current_load - previous_load
        local_load = self._to_local_load(layer, window_load)
        if int(local_load.sum().item()) == 0:
            return []

        desired_residents = sorted(
            range(int(local_load.numel())),
            key=lambda expert_id: (
                -int(local_load[expert_id].item()), expert_id),
        )[:len(placement.resident_expert_ids)]
        current_residents = set(placement.resident_expert_ids)
        desired_set = set(desired_residents)

        swap_in = [
            expert_id for expert_id in desired_residents
            if expert_id not in current_residents
        ]
        swap_out = sorted(
            (expert_id for expert_id in current_residents
             if expert_id not in desired_set),
            key=lambda expert_id: (
                int(local_load[expert_id].item()), expert_id),
        )

        swaps: list[ExpertSwap] = []
        for in_id, out_id in zip(swap_in, swap_out):
            gain = int(local_load[in_id].item()) - int(
                local_load[out_id].item())
            if gain < self.min_swap_gain:
                continue
            swaps.append(
                ExpertSwap(
                    swap_in=in_id,
                    swap_out=out_id,
                    resident_slot=placement.resident_slots[out_id],
                    cold_slot=placement.cold_slots[in_id],
                ))
            if len(swaps) >= self.max_swaps:
                break
        return swaps

    def apply_swaps(self, placement: ExpertPlacement,
                    swaps: list[ExpertSwap]) -> ExpertPlacement:
        if not swaps:
            return placement

        resident_ids = list(placement.resident_expert_ids)
        cold_ids = list(placement.cold_expert_ids)
        for swap in swaps:
            resident_ids[swap.resident_slot] = swap.swap_in
            cold_ids[swap.cold_slot] = swap.swap_out

        return ExpertPlacement(resident_ids, cold_ids)

    def _to_local_load(self, layer,
                       global_load: torch.Tensor) -> torch.Tensor:
        local_load = torch.zeros(int(layer.full_local_num_experts),
                                 dtype=torch.long,
                                 device="cpu")
        for local_id, global_id in HistoryLoadPolicy._local_experts(layer):
            local_load[local_id] = global_load[global_id]
        return local_load
