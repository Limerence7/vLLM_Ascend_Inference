from vllm import ModelRegistry

from .model import RuntimeQwen3MoeForCausalLM
from .runtime_config import RuntimeConfig, set_runtime_config


def register_plugin(config: RuntimeConfig | None = None):
    runtime_config = None
    if config is not None:
        runtime_config = set_runtime_config(config)
    if runtime_config is not None and runtime_config.enable_offline_scheduler:
        from .offline_scheduler import apply_offline_scheduler_patch
        print("Applying offline scheduler patch...")

        apply_offline_scheduler_patch(runtime_config.min_step_tokens)
    ModelRegistry.register_model("Qwen3MoeForCausalLM",
                                 RuntimeQwen3MoeForCausalLM)
    return RuntimeQwen3MoeForCausalLM
