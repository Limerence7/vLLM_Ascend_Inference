import vllm.model_executor.models.mixtral as mixtral
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.models.mixtral import MixtralForCausalLM

from .common import RuntimeModelMixin, make_dynamic_moe


class RuntimeMixtralForCausalLM(RuntimeModelMixin, MixtralForCausalLM):
    """Mixtral adapter for the vLLM 0.18 FusedMoE API."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self.prepare_runtime(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @staticmethod
    def runtime_moe_patch() -> None:
        mixtral.FusedMoE = make_dynamic_moe(FusedMoE)
