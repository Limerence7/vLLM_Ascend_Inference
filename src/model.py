from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

from .offload_config import get_offload_config
from .layer.fused_moe import OffloadAscendFusedMoE


class OffloadQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    """Qwen3 MoE wrapper that installs the Ascend/offload MoE backend."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self.offload_config = get_offload_config()
        self.patch_moe_backend()

        super().__init__(vllm_config=vllm_config, prefix=prefix)

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hf_config = vllm_config.model_config.hf_text_config
        self.num_layers = self.hf_config.num_hidden_layers

    def patch_moe_backend(self) -> None:
        import vllm.model_executor.models.qwen3_moe as qwen3_moe

        qwen3_moe.FusedMoE = OffloadAscendFusedMoE
