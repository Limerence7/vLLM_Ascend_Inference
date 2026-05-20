from typing import Optional

from .offload.config import OffloadConfig, configure_offload


def configure(config: OffloadConfig):
    return configure_offload(config)


def register_plugin(config: Optional[OffloadConfig] = None):
    if config is not None:
        configure_offload(config)

    from vllm import ModelRegistry

    try:
        from .model import AscendQwen3MoeModel
    except ImportError:
        from model import AscendQwen3MoeModel

    print("[Registry Logging] Registering MoE Plugin")
    ModelRegistry.register_model("Qwen3MoeForCausalLM", AscendQwen3MoeModel)
