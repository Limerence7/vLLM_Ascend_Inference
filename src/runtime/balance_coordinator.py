from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LayerBalanceCandidate:
    """A globally deterministic candidate for one MoE layer."""

    layer: Any
    plan: Any
    current_score: float
    target_score: float
    migrations: int
    context: Any = None

    @property
    def layer_id(self) -> int:
        return int(self.layer.moe_instance_id)

    @property
    def improvement(self) -> float:
        return self.current_score - self.target_score


class BalanceCoordinator:
    """Select a bounded set of profitable layers per rebalance cycle."""

    def __init__(self, max_layers: int, min_improvement: float):
        self.max_layers = int(max_layers)
        self.min_improvement = float(min_improvement)
        self._pending: dict[int, LayerBalanceCandidate] = {}

    def submit(self, candidate: LayerBalanceCandidate | None) -> None:
        if candidate is None:
            return
        self._pending[candidate.layer_id] = candidate

    def drain(self) -> list[LayerBalanceCandidate]:
        candidates = [
            candidate for candidate in self._pending.values()
            if candidate.improvement >= self.min_improvement
            and candidate.migrations > 0
        ]
        self._pending.clear()
        candidates.sort(key=lambda candidate: (
            -candidate.improvement,
            candidate.target_score,
            candidate.migrations,
            candidate.layer_id,
        ))
        if self.max_layers > 0:
            candidates = candidates[:self.max_layers]
        return candidates

    def clear(self) -> None:
        self._pending.clear()
