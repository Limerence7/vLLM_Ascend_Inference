from abc import ABC, abstractmethod
from typing import Callable, TypeVar


T = TypeVar("T")


class ExpertExecutionPlan(ABC):
    supports_resident_overlap = False

    @abstractmethod
    def run(self, load_fn: Callable[[], None], compute_fn: Callable[[], T]) -> T:
        raise NotImplementedError


class BarrierExecutor(ExpertExecutionPlan):
    def run(self, load_fn: Callable[[], None], compute_fn: Callable[[], T]) -> T:
        load_fn()
        return compute_fn()


class ResidentOverlapExecutor(BarrierExecutor):
    """
    Placeholder for second-stage resident/transfer overlap.

    True overlap requires splitting FusedMoE compute into resident-expert and
    transferred-expert sub-computations. AscendFusedMoE currently exposes a
    monolithic forward path here, so this executor intentionally falls back to
    barrier semantics while keeping the call site strategy-based.
    """

    supports_resident_overlap = False


def build_executor(overlap: bool) -> ExpertExecutionPlan:
    if overlap:
        return ResidentOverlapExecutor()
    return BarrierExecutor()
