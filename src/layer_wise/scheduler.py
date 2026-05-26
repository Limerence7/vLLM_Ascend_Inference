from bisect import bisect_left
from typing import List, Optional, Sequence

from ..config import OffloadConfig


class LayerWiseScheduler:
    def __init__(self, config: OffloadConfig):
        self.config = config

    def select_layers(self, available_layers: Sequence[int]) -> List[int]:
        interval = self.config.offload_interval
        return [
            layer_idx
            for layer_idx in available_layers
            if layer_idx % interval == 0
        ]

    def next_prefetch_layer(
        self,
        selected_layers: Sequence[int],
        current_layer_idx: int,
    ) -> Optional[int]:
        if self.config.prefetch_distance <= 0 or not selected_layers:
            return None

        target_layer_idx = current_layer_idx + max(1, self.config.prefetch_distance)
        target = bisect_left(selected_layers, target_layer_idx)
        if target >= len(selected_layers):
            return None
        return selected_layers[target]
