from typing import Optional, Set


class ExpertWisePrefetcher:
    def __init__(self, manager):
        self.manager = manager

    def schedule_next(self, layer_idx: Optional[int], activated_experts: Set[int], plan):
        self.manager.schedule_next_prefetch(
            layer_idx=layer_idx,
            activated_experts=activated_experts,
            plan=plan,
        )
