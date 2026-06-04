from typing import Optional

from vllm import ModelRegistry

from .model import OffloadQwen3MoeForCausalLM
from .offload_config import OffloadConfig, set_offload_config


def register_plugin(config: Optional[OffloadConfig] = None):
    if config is not None:
        set_offload_config(
            mode=config.mode,
            interval=config.interval,
            num_buffers=config.num_buffers,
            num_hot_experts=config.num_hot_experts,
            cpu_pin_memory=config.cpu_pin_memory,
            offloaded_layer_ids=config.offloaded_layer_ids,
        )
    print("[Registry Logging] Registering MoE Plugin")
    ModelRegistry.register_model("Qwen3MoeForCausalLM",
                                 OffloadQwen3MoeForCausalLM)
    return OffloadQwen3MoeForCausalLM
