from vllm import ModelRegistry

from .models import RUNTIME_MODELS
from .runtime_config import RuntimeConfig, set_runtime_config


def register_plugin(config: RuntimeConfig | None = None):
    runtime_config = set_runtime_config(config or RuntimeConfig())

    if runtime_config.enable_scheduler:
        from .scheduler import apply_scheduler_patch
        print("Applying request scheduler patch...")

        apply_scheduler_patch(runtime_config)
    for architecture, model_cls in RUNTIME_MODELS.items():
        ModelRegistry.register_model(architecture, model_cls)
    return RUNTIME_MODELS["Qwen3MoeForCausalLM"]
