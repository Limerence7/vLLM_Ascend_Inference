import re
from collections.abc import Callable
from typing import TypeVar

from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE

from ..layer.fused_moe import (RuntimeAscendFusedMoE,
                               RuntimeAscendSharedFusedMoE)
from ..runtime_config import get_runtime_config


LAYER_ID_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")
ModelT = TypeVar("ModelT")
EXPERT_COUNT_FIELDS = ("num_experts", "num_local_experts")


def layer_id_from_prefix(prefix: str) -> int:
    match = LAYER_ID_PATTERN.search(prefix)
    if match is None:
        raise ValueError(f"Cannot determine MoE layer id from prefix {prefix!r}")
    return int(match.group(1))


def get_num_experts(hf_config: object) -> int:
    """Return the global expert count across supported MoE configs."""
    for field in EXPERT_COUNT_FIELDS:
        value = getattr(hf_config, field, None)
        if value is not None:
            return int(value)
    raise ValueError(
        "Cannot determine the model expert count: expected one of "
        f"{EXPERT_COUNT_FIELDS!r} in {type(hf_config).__name__}")


class RuntimeModelMixin:
    """Common model setup shared by vLLM 0.18 MoE architectures."""

    runtime_moe_patch: Callable[[], None]

    def prepare_runtime(self, vllm_config: VllmConfig) -> None:
        hf_config = vllm_config.model_config.hf_text_config
        runtime_config = get_runtime_config()
        ep_size = get_ep_group().world_size
        num_experts = get_num_experts(hf_config)
        if num_experts % ep_size:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by EP size "
                f"({ep_size})")

        runtime_config.prepare_for_model(
            num_layers=int(hf_config.num_hidden_layers),
            num_experts=num_experts // ep_size,
        )
        self.runtime_config = runtime_config
        RuntimeAscendFusedMoE.reset_runtime()
        self.runtime_moe_patch()


def make_dynamic_moe(native_cls, runtime_cls=RuntimeAscendFusedMoE):
    """Create a per-layer selector with the native weight mapping API."""

    class DynamicMoE(native_cls):
        def __new__(cls, *args, **kwargs):
            layer_id = layer_id_from_prefix(kwargs.get("prefix", ""))
            runtime_config = get_runtime_config()
            selected_cls = (runtime_cls
                            if layer_id in runtime_config.runtime_layer_ids
                            else native_cls)
            if issubclass(selected_cls, RuntimeAscendFusedMoE):
                RuntimeAscendFusedMoE.moe_counter = layer_id - 1
            else:
                # Ascend's native classes still use a construction counter as
                # their layer identity.  Keep it aligned when Runtime and
                # native layers are interleaved.
                AscendFusedMoE.moe_counter = layer_id - 1
            return selected_cls(*args, **kwargs)

        make_expert_params_mapping = native_cls.make_expert_params_mapping

    DynamicMoE.__name__ = f"Dynamic{native_cls.__name__}"
    return DynamicMoE


def make_dynamic_shared_moe(native_cls):
    return make_dynamic_moe(native_cls, RuntimeAscendSharedFusedMoE)
