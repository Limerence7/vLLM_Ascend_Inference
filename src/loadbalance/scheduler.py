from .policy import DynamicLoadPolicy


class DynamicExpertScheduler:
    """Synchronous dynamic resident expert scheduler."""

    def __init__(self, policy: DynamicLoadPolicy, update_interval: int,
                 cooldown_interval: int = 0):
        self.policy = policy
        self.update_interval = max(1, int(update_interval))
        self.cooldown_interval = max(0, int(cooldown_interval))
        self._steps: dict[int, int] = {}
        self._cooldowns: dict[int, int] = {}

    def maybe_update(self, executor, layer) -> bool:
        layer_id = int(layer.moe_instance_id)
        step = self._steps.get(layer_id, 0) + 1
        self._steps[layer_id] = step
        if step % self.update_interval != 0:
            return False

        cooldown = self._cooldowns.get(layer_id, 0)
        if cooldown > 0:
            self._cooldowns[layer_id] = cooldown - 1
            return False

        placement = executor.placements[layer_id]
        swaps = self.policy.plan_for_layer(layer, placement)
        if not swaps:
            return False

        executor.copy_swaps_to_resident(layer, swaps)

        new_placement = self.policy.apply_swaps(placement, swaps)
        executor.commit_layer_placement(layer, new_placement, swaps)
        self._cooldowns[layer_id] = self.cooldown_interval
        return True
