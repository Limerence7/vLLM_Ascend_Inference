from .policy import DynamicLoadPolicy


class DynamicExpertScheduler:
    """Synchronous dynamic resident expert scheduler."""

    def __init__(self, policy: DynamicLoadPolicy, update_interval: int):
        self.policy = policy
        self.update_interval = max(1, int(update_interval))
        self._steps: dict[int, int] = {}

    def maybe_update(self, executor, layer) -> bool:
        layer_id = int(layer.moe_instance_id)
        step = self._steps.get(layer_id, 0) + 1
        self._steps[layer_id] = step
        if step % self.update_interval != 0:
            return False

        placement = executor.placements[layer_id]
        swaps = self.policy.plan_for_layer(layer, placement)
        if not swaps:
            return False

        for swap in swaps:
            executor.copy_expert_to_resident(layer, swap.swap_in,
                                             swap.resident_slot)

        new_placement = self.policy.apply_swaps(placement, swaps)
        executor.commit_layer_placement(layer, new_placement)
        return True
