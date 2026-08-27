from vllm import ModelRegistry

from .model import RuntimeQwen3MoeForCausalLM
from .runtime_config import RuntimeConfig, set_runtime_config


def register_plugin(config: RuntimeConfig | None = None):
    runtime_config = set_runtime_config(config or RuntimeConfig())

    if runtime_config.enable_offline_scheduler:
        from .offline_scheduler import apply_offline_scheduler_patch
        print("Applying offline scheduler patch...")

        apply_offline_scheduler_patch(
            runtime_config.scheduler_min_step_tokens,
            reorder_window=runtime_config.scheduler_reorder_window,
            policy=runtime_config.scheduler_policy,
        )
    ModelRegistry.register_model("Qwen3MoeForCausalLM",
                                 RuntimeQwen3MoeForCausalLM)
    return RuntimeQwen3MoeForCausalLM
