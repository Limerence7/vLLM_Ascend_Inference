from vllm import ModelRegistry

from .model import RuntimeQwen3MoeForCausalLM
from .runtime_config import RuntimeConfig, set_runtime_config


def register_plugin(config: RuntimeConfig | None = None):
    runtime_config = set_runtime_config(config or RuntimeConfig())

    if runtime_config.enable_scheduler:
        from .scheduler import apply_scheduler_patch
        print("Applying request scheduler patch...")

        apply_scheduler_patch(runtime_config)
    ModelRegistry.register_model("Qwen3MoeForCausalLM",
                                 RuntimeQwen3MoeForCausalLM)
    return RuntimeQwen3MoeForCausalLM
