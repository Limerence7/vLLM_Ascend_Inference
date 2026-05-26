from typing import Dict, Optional, Set


class SameExpertNextLayerPredictor:
    def __init__(self):
        self._predicted: Dict[int, Set[int]] = {}

    def record_activated(
        self,
        layer_idx: int,
        expert_ids: Set[int],
        target_layer_idx: Optional[int] = None,
    ) -> None:
        target = layer_idx + 1 if target_layer_idx is None else target_layer_idx
        self._predicted[target] = set(expert_ids)

    def predicted_for(self, layer_idx: int) -> Set[int]:
        return set(self._predicted.get(layer_idx, set()))

    def clear(self) -> None:
        self._predicted.clear()
