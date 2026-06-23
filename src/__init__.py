from vllm import ModelRegistry

from .model import RuntimeQwen3MoeForCausalLM
from .runtime_config import RuntimeConfig, set_runtime_config


def register_plugin(config: RuntimeConfig | None = None):
    if config is not None:
        set_runtime_config(config)
    ModelRegistry.register_model("Qwen3MoeForCausalLM",
                                 RuntimeQwen3MoeForCausalLM)
    return RuntimeQwen3MoeForCausalLM
