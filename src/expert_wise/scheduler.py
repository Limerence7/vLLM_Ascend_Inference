from dataclasses import dataclass
from math import ceil
from typing import AbstractSet, Mapping, Optional, Set

from ..config import OffloadConfig


@dataclass(frozen=True)
class ExpertWisePlan:
    load_experts: AbstractSet[int]
    prefetch_next_experts: AbstractSet[int]
    prefetch_all_next: bool


class ExpertWiseScheduler:
    def __init__(self, config: OffloadConfig):
        self.config = config

    def should_offload_layer(self, layer_idx: Optional[int]) -> bool:
        if layer_idx is None:
            return False
        return layer_idx % self.config.offload_interval == 0

    def next_prefetch_layer(self, layer_idx: Optional[int]) -> Optional[int]:
        if layer_idx is None or self.config.prefetch_distance <= 0:
            return None

        if not self.should_offload_layer(layer_idx):
            return None

        return layer_idx + (
            self.config.prefetch_distance * self.config.offload_interval
        )

    def should_use_large_batch_plan(
        self,
        *,
        num_tokens: int,
        offloaded_count: int,
    ) -> bool:
        if not self.config.expert_wise.enable_large_batch_fast_path:
            return False

        if offloaded_count <= 0:
            return False

        threshold = max(
            1,
            ceil(offloaded_count * self.config.expert_wise.large_batch_active_ratio),
        )
        return num_tokens >= threshold

    def build_large_batch_plan(
        self,
        *,
        current_offloaded_experts: AbstractSet[int],
    ) -> ExpertWisePlan:
        return ExpertWisePlan(
            load_experts=current_offloaded_experts,
            prefetch_next_experts=frozenset(),
            prefetch_all_next=True,
        )

    def offloaded_experts(self, num_experts: int) -> Set[int]:
        validate_expert_partition(
            resident_experts=self.config.expert_wise.resident_experts,
            total_experts=num_experts,
            offload_multiple=self.config.expert_wise.offload_multiple,
        )
        resident = self.config.expert_wise.resident_experts
        return set(range(resident, num_experts))

    def offloaded_local_experts(
        self,
        *,
        local_global_to_slot: Mapping[int, int],
        total_experts: int,
    ) -> Set[int]:
        validate_expert_partition(
            resident_experts=self.config.expert_wise.resident_experts,
            total_experts=total_experts,
            offload_multiple=self.config.expert_wise.offload_multiple,
        )
        local_experts = [
            global_expert_id
            for global_expert_id, _ in sorted(
                local_global_to_slot.items(),
                key=lambda item: item[1],
            )
        ]
        if not local_experts:
            return set()

        resident = self.config.expert_wise.resident_experts
        offloaded_total = total_experts - resident
        local_offloaded_count = round(len(local_experts) * offloaded_total / total_experts)
        if local_offloaded_count <= 0:
            return set()
        if local_offloaded_count >= len(local_experts):
            return set(local_experts)
        return set(local_experts[-local_offloaded_count:])

    def build_plan(
        self,
        *,
        routed_experts: Set[int],
        current_offloaded_experts: Set[int],
    ) -> ExpertWisePlan:
        if self.config.expert_wise.on_demand_load:
            load_experts = routed_experts.intersection(current_offloaded_experts)
        else:
            load_experts = set(current_offloaded_experts)

        active_offloaded_count = len(
            routed_experts.intersection(current_offloaded_experts)
        )
        offloaded_count = len(current_offloaded_experts)
        large_batch = (
            offloaded_count > 0
            and active_offloaded_count >= max(
                1,
                ceil(
                    offloaded_count
                    * self.config.expert_wise.large_batch_active_ratio
                ),
            )
        )

        if large_batch:
            return ExpertWisePlan(
                load_experts=load_experts,
                prefetch_next_experts=set(),
                prefetch_all_next=True,
            )

        return ExpertWisePlan(
            load_experts=load_experts,
            prefetch_next_experts=(
                set(routed_experts)
                if self.config.expert_wise.enable_prediction
                else set()
            ),
            prefetch_all_next=False,
        )


def validate_expert_partition(
    *,
    resident_experts: int,
    total_experts: int,
    offload_multiple: int,
) -> None:
    if resident_experts > total_experts:
        raise ValueError(
            "ExpertWiseConfig.resident_experts cannot exceed total experts: "
            f"resident={resident_experts}, total={total_experts}."
        )

    offloaded_count = total_experts - resident_experts
    if offloaded_count % offload_multiple != 0:
        raise ValueError(
            "Expert-wise offloaded expert count must be a multiple of "
            f"{offload_multiple}: resident={resident_experts}, "
            f"total={total_experts}, offloaded={offloaded_count}."
        )
