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


class HistoryLoadPolicy:
    """Choose resident experts from this rank's persisted load stats."""

    def __init__(self, load_stats: ExpertLoadStats | None):
        self.load_stats = load_stats

    def placement_for_layer(self, layer,
                            num_resident_experts: int) -> ExpertPlacement:
        local_num_experts = int(layer.full_local_num_experts)
        num_resident_experts = min(num_resident_experts, local_num_experts)

        resident_ids = self._history_resident_ids(layer,
                                                  num_resident_experts)
        if resident_ids is None:
            resident_ids = list(range(num_resident_experts))

        resident_set = set(resident_ids)
        cold_ids = [
            expert_id for expert_id in range(local_num_experts)
            if expert_id not in resident_set
        ]
        return ExpertPlacement(resident_ids, cold_ids)

    def _history_resident_ids(self, layer,
                              num_resident_experts: int) -> list[int] | None:
        if self.load_stats is None or num_resident_experts <= 0:
            return None

        try:
            load = self.load_stats.get_layer_load(int(layer.moe_instance_id))
        except KeyError:
            return None

        if load.numel() != int(layer.global_num_experts):
            return None

        local_experts = self._local_experts(layer)
        if len(local_experts) != int(layer.full_local_num_experts):
            return None

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
