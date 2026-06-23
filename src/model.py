import re

import vllm.model_executor.models.qwen3_moe as qwen3_moe
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE

from .layer.fused_moe import RuntimeAscendFusedMoE
from .runtime_config import get_runtime_config


LAYER_ID_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")
RUNTIME_LAYER_IDS: set[int] = set()


def _layer_id_from_prefix(prefix: str) -> int:
    match = LAYER_ID_PATTERN.search(prefix)
    return int(match.group(1))


class DynamicFusedMoE(AscendFusedMoE):
    """Select the native or runtime MoE implementation for each layer."""

    def __new__(cls, *args, **kwargs):
        layer_id = _layer_id_from_prefix(kwargs.get("prefix", ""))
        moe_cls = (RuntimeAscendFusedMoE
                   if layer_id in RUNTIME_LAYER_IDS else AscendFusedMoE)
        moe_cls.moe_counter = layer_id - 1
        return moe_cls(*args, **kwargs)

    make_expert_params_mapping = AscendFusedMoE.make_expert_params_mapping


class RuntimeQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    """Qwen3 MoE wrapper that installs the runtime MoE backend."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hf_config = vllm_config.model_config.hf_text_config
        self.num_layers = self.hf_config.num_hidden_layers
        self.num_experts = self.hf_config.num_experts

        self.runtime_config = get_runtime_config()
        num_experts_per_partition = self.num_experts // self.tp_size
        self.runtime_config.prepare_for_model(
            num_layers=self.num_layers,
            num_experts=num_experts_per_partition,
        )
        self._patch_moe_backend()

        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def _patch_moe_backend(self) -> None:
        global RUNTIME_LAYER_IDS
        RUNTIME_LAYER_IDS = set(self.runtime_config.runtime_layer_ids)
        RuntimeAscendFusedMoE.reset_runtime()
        qwen3_moe.FusedMoE = DynamicFusedMoE
