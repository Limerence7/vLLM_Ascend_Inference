import re
from math import ceil
from pathlib import Path

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE

from .offload_config import get_offload_config
from .layer.fused_moe import OffloadAscendFusedMoE


LAYER_ID_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")
OFFLOADED_LAYER_IDS: set[int] = set()
USE_OFFLOAD_FOR_ALL_LAYERS = False


def _layer_id_from_prefix(prefix: str) -> int:
    match = LAYER_ID_PATTERN.search(prefix)
    if match is None:
        raise ValueError(f"Cannot parse MoE layer id from prefix: {prefix}")
    return int(match.group(1))


def _make_fused_moe(*args, **kwargs):
    layer_id = _layer_id_from_prefix(kwargs.get("prefix", ""))
    if USE_OFFLOAD_FOR_ALL_LAYERS or layer_id in OFFLOADED_LAYER_IDS:
        moe_cls = OffloadAscendFusedMoE
    else:
        moe_cls = AscendFusedMoE

    moe_cls.moe_counter = layer_id - 1
    return moe_cls(*args, **kwargs)


_make_fused_moe.make_expert_params_mapping = (
    AscendFusedMoE.make_expert_params_mapping)


class OffloadQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    """Qwen3 MoE wrapper that installs the Ascend/offload MoE backend."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hf_config = vllm_config.model_config.hf_text_config
        self.num_layers = self.hf_config.num_hidden_layers
        self.num_experts = self.hf_config.num_experts
        model_path = Path(vllm_config.model_config.model)
        self.uses_quantization = (
            vllm_config.quant_config is not None
            or (model_path / "quant_model_description.json").is_file())

        self.offload_config = get_offload_config()
        num_experts_per_partition = ceil(self.num_experts / self.tp_size)
        self.offload_config.prepare_for_model(
            num_layers=self.num_layers,
            num_experts=num_experts_per_partition,
        )
        self.patch_moe_backend()

        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def patch_moe_backend(self) -> None:
        import vllm.model_executor.models.qwen3_moe as qwen3_moe

        global OFFLOADED_LAYER_IDS, USE_OFFLOAD_FOR_ALL_LAYERS
        OFFLOADED_LAYER_IDS = set(self.offload_config.offloaded_layer_ids)
        USE_OFFLOAD_FOR_ALL_LAYERS = self.uses_quantization
        OffloadAscendFusedMoE.reset_runtime()
        qwen3_moe.FusedMoE = _make_fused_moe
