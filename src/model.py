from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

from .config import OffloadConfig, get_offload_config
from .expert_wise import ExpertWiseManager, validate_expert_partition
from .fused_moe import ExpertWiseAscendFusedMoE
from .layer_wise import LayerWiseManager


_ORIGINAL_QWEN3_FUSED_MOE = None


class OffloadModel(nn.Module):
    def __init__(self, model: nn.Module, offload_config: OffloadConfig):
        super().__init__()
        self.model = model
        self.offload_config = offload_config
        self.manager = self._build_manager()

    def _build_manager(self):
        if self.offload_config.mode == "layer_wise":
            return LayerWiseManager(self.model, self.offload_config)
        if self.offload_config.mode == "expert_wise":
            return ExpertWiseManager.activate(self.offload_config)
        if self.offload_config.mode == "none":
            return None
        raise ValueError(f"Unsupported offload mode: {self.offload_config.mode}.")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def load_weights(self, weights):
        loaded_weights = self.model.load_weights(weights)
        self.setup_after_weight_loading()
        return loaded_weights

    def setup_after_weight_loading(self) -> None:
        if hasattr(self.manager, "setup_after_weight_loading"):
            self.manager.setup_after_weight_loading()

    def offload_summary(self):
        if hasattr(self.manager, "summary"):
            return self.manager.summary()
        if self.offload_config.mode == "expert_wise":
            return ExpertWiseManager.active().summary()
        return {"mode": self.offload_config.mode}


class AscendQwen3MoeModel(Qwen3MoeForCausalLM):
    """
    Qwen3 MoE model wrapper used by the plugin.

    This class is the model-level integration point exposed to vLLM. It owns
    Qwen3 FusedMoE backend replacement, post-load setup dispatch, and model
    summary logging. Layer-wise buffers and expert-wise stores live in their
    offload/fused_moe modules.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        runtime_offload_config = get_offload_config()
        if runtime_offload_config.mode == "expert_wise":
            ExpertWiseManager.activate(runtime_offload_config)

        self._configure_qwen3_moe_backend(runtime_offload_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.config = vllm_config.model_config.hf_text_config
        self.num_layers = self.config.num_hidden_layers

        self.runtime_offload_config = runtime_offload_config
        self.offload_mode = self.runtime_offload_config.mode
        self.offload_manager = None
        self._validate_model_level_offload_config()

        self._log_model_summary("initializing")

    @classmethod
    def _configure_qwen3_moe_backend(cls, config: OffloadConfig) -> None:
        import vllm.model_executor.models.qwen3_moe as qwen3_moe

        global _ORIGINAL_QWEN3_FUSED_MOE
        if _ORIGINAL_QWEN3_FUSED_MOE is None:
            _ORIGINAL_QWEN3_FUSED_MOE = qwen3_moe.FusedMoE

        if config.mode == "expert_wise":
            qwen3_moe.FusedMoE = ExpertWiseAscendFusedMoE
            return

        qwen3_moe.FusedMoE = _ORIGINAL_QWEN3_FUSED_MOE

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

        if self.offload_mode != "layer_wise":
            raise ValueError(f"Unsupported offload mode: {self.offload_mode}.")

        if self.offload_manager is None:
            self.offload_manager = LayerWiseManager(
                model=self,
                config=self.runtime_offload_config,
            )
        self.offload_manager.setup_after_weight_loading()

    def offload_summary(self):
        if self.offload_mode == "expert_wise":
            return ExpertWiseManager.active().summary()
        if self.offload_manager is not None:
            return self.offload_manager.summary()
        return {"mode": self.offload_mode}

    def _validate_model_level_offload_config(self) -> None:
        if self.offload_mode != "expert_wise":
            return

        validate_expert_partition(
            resident_experts=self.runtime_offload_config.expert_wise.resident_experts,
            total_experts=int(self.config.num_experts),
            offload_multiple=self.runtime_offload_config.expert_wise.offload_multiple,
        )

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
