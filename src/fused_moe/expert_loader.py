from typing import AbstractSet

from ..expert_wise import ExpertWiseExpertStore


class ExpertLoader:
    def __init__(self, store: ExpertWiseExpertStore):
        self.store = store

    def load_for_compute(self, fused_moe, expert_ids: AbstractSet[int]) -> None:
        if self.store.experts_ready(expert_ids):
            return
        self.store.restore_routed_experts(fused_moe, expert_ids)

    def prefetch(self, fused_moe, expert_ids: AbstractSet[int]) -> None:
        self.store.prefetch_experts(fused_moe, expert_ids)
