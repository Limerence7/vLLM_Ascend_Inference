from typing import Optional

from .config import OffloadConfig, configure_offload
from .utils import install_worker_summary_rpc


def configure(config: OffloadConfig):
    return configure_offload(config)


def register_plugin(config: Optional[OffloadConfig] = None):
    if config is not None:
        configure_offload(config)

    install_worker_summary_rpc()

    from vllm import ModelRegistry

    from .model import AscendQwen3MoeModel

    print("[Registry Logging] Registering MoE Plugin")
    ModelRegistry.register_model("Qwen3MoeForCausalLM", AscendQwen3MoeModel)
