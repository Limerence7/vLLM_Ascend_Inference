import vllm.model_executor.models.qwen3_moe as qwen3_moe
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

from .common import RuntimeModelMixin, make_dynamic_shared_moe


class RuntimeQwen3MoeForCausalLM(RuntimeModelMixin,
                                 Qwen3MoeForCausalLM):
    """Qwen3 MoE adapter for the vLLM 0.18 SharedFusedMoE API."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self.prepare_runtime(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @staticmethod
    def runtime_moe_patch() -> None:
        qwen3_moe.SharedFusedMoE = make_dynamic_shared_moe(SharedFusedMoE)
