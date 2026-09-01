from contextlib import contextmanager
from types import SimpleNamespace

import triton


@contextmanager
def _safe_ascend_target_probe():
    """Allow vLLM's model-inspection subprocess to import FLA on Ascend.

    The inspection subprocess does not own an NPU context.  Ascend Triton's
    ``get_current_target`` raises ``SystemError`` in that situation, while
    vLLM's FLA import-time probe only handles RuntimeError and AttributeError.
    Keep the workaround local to the imports that perform the probe and leave
    every successful device query untouched.
    """
    driver = triton.runtime.driver.active
    get_current_target = driver.get_current_target

    def get_current_target_or_ascend():
        try:
            return get_current_target()
        except SystemError:
            return SimpleNamespace(backend="ascend")

    driver.get_current_target = get_current_target_or_ascend
    try:
        yield
    finally:
        driver.get_current_target = get_current_target


with _safe_ascend_target_probe():
    import vllm.model_executor.models.qwen3_next as qwen3_next
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5MoeForCausalLM, Qwen3_5MoeForConditionalGeneration)

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import SharedFusedMoE

from .common import RuntimeModelMixin, make_dynamic_shared_moe


class RuntimeQwen3_5MoeForCausalLM(RuntimeModelMixin,
                                   Qwen3_5MoeForCausalLM):
    """Qwen3.5-MoE adapter; its sparse block is defined by qwen3_next."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self.prepare_runtime(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @staticmethod
    def runtime_moe_patch() -> None:
        qwen3_next.SharedFusedMoE = make_dynamic_shared_moe(SharedFusedMoE)


class RuntimeQwen3_5MoeForConditionalGeneration(
        RuntimeModelMixin, Qwen3_5MoeForConditionalGeneration):
    """Public Qwen3.5-MoE architecture used by vLLM model registry."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self.prepare_runtime(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @staticmethod
    def runtime_moe_patch() -> None:
        qwen3_next.SharedFusedMoE = make_dynamic_shared_moe(SharedFusedMoE)
