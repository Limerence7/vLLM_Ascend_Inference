import copy
from dataclasses import dataclass, field
from typing import Optional

from .expertwise_config import ExpertWiseConfig
from .layerwise_config import LayerWiseConfig


OFFLOAD_MODES = ("none", "layer_wise", "expert_wise")


@dataclass
class OffloadConfig:
    mode: str = "layer_wise"
    overlap: bool = True
    prefetch_distance: int = 1
    offload_interval: int = 1
    layer_wise: LayerWiseConfig = field(default_factory=LayerWiseConfig)
    expert_wise: ExpertWiseConfig = field(default_factory=ExpertWiseConfig)

    def validate(self) -> None:
        self.mode = self.mode.strip().lower()
        if self.mode not in OFFLOAD_MODES:
            raise ValueError(
                "OffloadConfig.mode must be one of: "
                f"{', '.join(OFFLOAD_MODES)}."
            )
        if self.prefetch_distance < 0:
            raise ValueError("OffloadConfig.prefetch_distance must be >= 0.")
        if self.offload_interval <= 0:
            raise ValueError("OffloadConfig.offload_interval must be > 0.")

        self.layer_wise.validate()
        self.expert_wise.validate()

    def normalized_copy(self) -> "OffloadConfig":
        cfg = copy.deepcopy(self)
        cfg.validate()
        return cfg


DEFAULT_OFFLOAD_CONFIG = OffloadConfig()
_ACTIVE_OFFLOAD_CONFIG = DEFAULT_OFFLOAD_CONFIG.normalized_copy()


def configure_offload(config: Optional[OffloadConfig] = None) -> OffloadConfig:
    global _ACTIVE_OFFLOAD_CONFIG

    source = DEFAULT_OFFLOAD_CONFIG if config is None else config
    _ACTIVE_OFFLOAD_CONFIG = source.normalized_copy()
    return _ACTIVE_OFFLOAD_CONFIG.normalized_copy()


def get_offload_config() -> OffloadConfig:
    return _ACTIVE_OFFLOAD_CONFIG.normalized_copy()
