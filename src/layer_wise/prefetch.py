from typing import Dict, Optional

from ..utils import ExpertBuffer, ExpertBufferPool, ExpertOffloadSlot


class LayerWisePrefetcher:
    def __init__(self, pool: ExpertBufferPool, slots: Dict[int, ExpertOffloadSlot]):
        self.pool = pool
        self.slots = slots

    def prefetch(
        self,
        layer_idx: Optional[int],
        avoid: Optional[ExpertBuffer] = None,
    ) -> bool:
        if layer_idx is None:
            return False

        slot = self.slots.get(layer_idx)
        if slot is None:
            return False

        return self.pool.prefetch(layer_idx, slot, avoid=avoid)
