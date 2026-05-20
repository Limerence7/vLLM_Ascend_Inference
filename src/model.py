from typing import Iterable, Optional, Tuple

import torch

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

from .fused_moe import ExpertWiseAscendFusedMoE
from .offload.config import get_offload_config
from .offload.layer_wise import LayerWiseOffloadController


class AscendQwen3MoeModel(Qwen3MoeForCausalLM):
    """
    Qwen3 MoE model wrapper used by the plugin.

    This class is the model-level integration point exposed to vLLM. It owns
    Qwen3 FusedMoE backend replacement, post-load setup dispatch, and model
    summary logging. Layer-wise buffers and expert-wise stores live in their
    offload/fused_moe modules.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self._patch_qwen3_moe_backend()
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.config = vllm_config.model_config.hf_text_config
        self.num_layers = self.config.num_hidden_layers

        self.runtime_offload_config = get_offload_config()
        self.offload_mode = self.runtime_offload_config.mode
        self.layer_wise_offload: Optional[LayerWiseOffloadController] = None

        self._log_model_summary("initializing")

    @staticmethod
    def _patch_qwen3_moe_backend() -> None:
        import vllm.model_executor.models.qwen3_moe as qwen3_moe

        qwen3_moe.FusedMoE = ExpertWiseAscendFusedMoE

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loaded_weights = super().load_weights(weights)
        self._setup_offload_after_weight_loading()
        return loaded_weights

    def _setup_offload_after_weight_loading(self) -> None:
        if self.offload_mode == "none":
            print("[Plugin] Qwen3 offload disabled by mode='none'.")
            return

        if self.offload_mode == "expert_wise":
            print("[Plugin] Qwen3 expert-wise offload is handled by FusedMoE.")
            return

        if self.offload_mode not in {"layer_wise", "auto"}:
            raise ValueError(f"Unsupported offload mode: {self.offload_mode}.")

        if self.offload_mode == "auto":
            print(
                "[Plugin] Qwen3 auto offload strategy is not implemented; "
                "falling back to layer-wise config."
            )

        if self.layer_wise_offload is None:
            self.layer_wise_offload = LayerWiseOffloadController(
                model=self,
                config=self.runtime_offload_config.layer_wise,
            )
        self.layer_wise_offload.setup_after_weight_loading()

    def _log_model_summary(self, phase: str) -> None:
        cfg = self.config
        attrs = {
            "num_layers": cfg.num_hidden_layers,
            "hidden_size": cfg.hidden_size,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "num_experts": cfg.num_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
            "moe_intermediate_size": cfg.moe_intermediate_size,
        }
        print(
            "[Plugin] AscendQwen3MoeModel "
            f"{phase}; model={attrs}; offload={self.runtime_offload_config}"
        )
