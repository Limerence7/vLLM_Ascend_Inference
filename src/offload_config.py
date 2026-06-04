import copy
from dataclasses import dataclass, field
from typing import Optional


OFFLOAD_MODES = ("none", "layer_wise", "expert_wise")

@dataclass
class OffloadConfig:
    mode: str = "layer_wise"
    interval: int = 2
    num_buffers: int = 2
    num_hot_experts: int = 0
    cpu_pin_memory: bool = True
    offloaded_layer_ids: list[int] = field(default_factory=list)

    def validate(self) -> None:
        self.mode = self.mode.strip().lower()
        assert self.mode in OFFLOAD_MODES, f"{self.mode} is not supported for offloading."
        assert self.interval > 0, f"Offload Interval must be positive"
        assert self.num_buffers >= 2, "Offload needs at least two cold expert buffers."
        assert self.num_hot_experts >= 0, "num_hot_experts must be non-negative."

    def normalized_copy(self) -> "OffloadConfig":
        config = copy.deepcopy(self)
        config.validate()
        return config

    def prepare_for_model(self, num_layers: int,
                          num_experts: Optional[int] = None) -> None:
        self.validate()
        num_layers = int(num_layers)

        if self.mode == "none":
            layer_ids: list[int] = []
        else:
            layer_ids = self.offloaded_layer_ids or list(
                range(0, num_layers, self.interval))
            if num_experts is not None:
                assert self.num_hot_experts <= int(num_experts), (
                    "num_hot_experts cannot exceed num_experts.")

        self.offloaded_layer_ids = [
            layer_id for layer_id in sorted({int(i) for i in layer_ids})
            if 0 <= layer_id < num_layers
        ]


OFFLOAD_CONFIG = OffloadConfig()

def get_offload_config() -> OffloadConfig:
    global OFFLOAD_CONFIG
    return OFFLOAD_CONFIG

def set_offload_config(
        mode: str,
        interval,
        num_buffers,
        num_hot_experts,
        cpu_pin_memory,
        offloaded_layer_ids,
    ) -> OffloadConfig:
    global OFFLOAD_CONFIG
    OFFLOAD_CONFIG = OffloadConfig(
        mode=mode,
        interval=interval,
        num_buffers=num_buffers,
        num_hot_experts=num_hot_experts,
        cpu_pin_memory=cpu_pin_memory,
        offloaded_layer_ids=offloaded_layer_ids,
    )
    OFFLOAD_CONFIG.validate()
    
    return OFFLOAD_CONFIG
