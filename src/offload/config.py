import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Set


OFFLOAD_MODES = ("none", "layer_wise", "expert_wise", "auto")
LAYER_POLICIES = ("every_n", "explicit", "ratio", "tail")


@dataclass
class ExpertOffloadConfig:
    """
    Layer-wise offload configuration.

    policy:
    - every_n: select layers N-1, 2N-1, ...
    - explicit: select explicit_layers.
    - ratio: uniformly select a ratio of available MoE layers.
    - tail: select the last tail_n MoE layers.
    """

    enabled: bool = True
    policy: str = "every_n"
    every_n: int = 16
    explicit_layers: Optional[list[int]] = None
    ratio: float = 0.0
    tail_n: int = 0
    num_buffers: int = 2
    pin_cpu_memory: bool = True
    prefetch: bool = True
    keep_owner_on_npu: bool = True

    def validate(self) -> None:
        self.policy = self.policy.strip().lower()
        if self.policy not in LAYER_POLICIES:
            raise ValueError(
                "ExpertOffloadConfig.policy must be one of: "
                f"{', '.join(LAYER_POLICIES)}."
            )

        if self.every_n <= 0:
            raise ValueError("ExpertOffloadConfig.every_n must be > 0.")
        if self.num_buffers <= 0:
            raise ValueError("ExpertOffloadConfig.num_buffers must be > 0.")
        if self.policy == "explicit" and not self.explicit_layers:
            raise ValueError("policy='explicit' requires explicit_layers.")
        if self.policy == "ratio" and not 0.0 < self.ratio <= 1.0:
            raise ValueError("policy='ratio' requires 0 < ratio <= 1.")
        if self.policy == "tail" and self.tail_n <= 0:
            raise ValueError("policy='tail' requires tail_n > 0.")

    def normalized_copy(self) -> "ExpertOffloadConfig":
        cfg = copy.copy(self)
        if cfg.explicit_layers is not None:
            cfg.explicit_layers = sorted(set(int(x) for x in cfg.explicit_layers))
        cfg.validate()
        return cfg


@dataclass
class ExpertWiseOffloadConfig:
    """
    Expert-wise offload configuration skeleton.

    offloaded_experts maps zero-based layer ids to:
    - None: offload all local experts in the layer.
    - Iterable[int]: offload only those expert ids.
    Missing layer ids mean no expert-wise offload for that layer.
    """

    offloaded_experts: Optional[Dict[int, Optional[Iterable[int]]]] = None
    npu_cache_capacity: int = 0
    pin_cpu_memory: bool = True
    prefetch: bool = True
    overlap: bool = True
    num_copy_streams: int = 1
    keep_loaded_on_npu: bool = True
    compact_npu_cache: bool = False
    log_transfers: bool = False
    max_transfer_logs: int = 32

    def validate(self) -> None:
        if self.npu_cache_capacity < 0:
            raise ValueError("npu_cache_capacity must be >= 0.")
        if self.compact_npu_cache and self.npu_cache_capacity <= 0:
            raise ValueError(
                "compact_npu_cache=True requires npu_cache_capacity > 0."
            )
        if self.num_copy_streams <= 0:
            raise ValueError("num_copy_streams must be > 0.")
        if self.max_transfer_logs < 0:
            raise ValueError("max_transfer_logs must be >= 0.")
        self.offloaded_experts = normalize_expert_selection(self.offloaded_experts)

    def normalized_copy(self) -> "ExpertWiseOffloadConfig":
        cfg = copy.deepcopy(self)
        cfg.validate()
        return cfg


@dataclass
class OffloadConfig:
    mode: str = "layer_wise"
    layer_wise: ExpertOffloadConfig = field(default_factory=ExpertOffloadConfig)
    expert_wise: ExpertWiseOffloadConfig = field(
        default_factory=ExpertWiseOffloadConfig
    )

    def validate(self) -> None:
        self.mode = self.mode.strip().lower()
        if self.mode not in OFFLOAD_MODES:
            raise ValueError(
                "OffloadConfig.mode must be one of: "
                f"{', '.join(OFFLOAD_MODES)}."
            )
        self.layer_wise.validate()
        self.expert_wise.validate()

    def normalized_copy(self) -> "OffloadConfig":
        cfg = copy.deepcopy(self)
        cfg.validate()
        return cfg


def normalize_expert_selection(
    selection: Optional[Dict[int, Optional[Iterable[int]]]],
) -> Optional[Dict[int, Optional[Set[int]]]]:
    if selection is None:
        return None

    normalized: Dict[int, Optional[Set[int]]] = {}
    for raw_layer_idx, raw_expert_ids in selection.items():
        layer_idx = int(raw_layer_idx)
        if layer_idx < 0:
            raise ValueError("Expert-wise layer indices must be >= 0.")

        if raw_expert_ids is None:
            normalized[layer_idx] = None
            continue

        expert_ids = {int(expert_id) for expert_id in raw_expert_ids}
        if any(expert_id < 0 for expert_id in expert_ids):
            raise ValueError("Expert-wise expert ids must be >= 0.")
        normalized[layer_idx] = expert_ids

    return dict(sorted(normalized.items()))


DEFAULT_EXPERT_OFFLOAD_CONFIG = ExpertOffloadConfig()
DEFAULT_OFFLOAD_CONFIG = OffloadConfig()
_ACTIVE_OFFLOAD_CONFIG = DEFAULT_OFFLOAD_CONFIG.normalized_copy()


def configure_offload(config: Optional[OffloadConfig] = None) -> OffloadConfig:
    """
    Set the process-local offload config used by plugin-created models.

    Call this before constructing vLLM.LLM. Passing None restores defaults.
    """

    global _ACTIVE_OFFLOAD_CONFIG
    source = DEFAULT_OFFLOAD_CONFIG if config is None else config
    _ACTIVE_OFFLOAD_CONFIG = source.normalized_copy()
    return _ACTIVE_OFFLOAD_CONFIG.normalized_copy()


def get_offload_config() -> OffloadConfig:
    return _ACTIVE_OFFLOAD_CONFIG.normalized_copy()
