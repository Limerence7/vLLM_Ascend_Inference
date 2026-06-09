from vllm import ModelRegistry

from .model import OffloadQwen3MoeForCausalLM
from .offload_config import OffloadConfig, set_offload_config


def register_plugin(config: OffloadConfig | None = None):
    if config is not None:
        set_offload_config(config)
    ModelRegistry.register_model("Qwen3MoeForCausalLM",
                                 OffloadQwen3MoeForCausalLM)
    return OffloadQwen3MoeForCausalLM
