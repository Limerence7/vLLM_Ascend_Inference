from .config import (
    DEFAULT_EXPERT_OFFLOAD_CONFIG,
    DEFAULT_OFFLOAD_CONFIG,
    ExpertOffloadConfig,
    ExpertWiseOffloadConfig,
    OFFLOAD_MODES,
    OffloadConfig,
    configure_offload,
    get_offload_config,
    normalize_expert_selection,
)

__all__ = [
    "DEFAULT_EXPERT_OFFLOAD_CONFIG",
    "DEFAULT_OFFLOAD_CONFIG",
    "ExpertBuffer",
    "ExpertBufferPool",
    "ExpertOffloadConfig",
    "ExpertOffloadSlot",
    "ExpertPlacement",
    "LayerWiseOffloadController",
    "ExpertWiseExpertStore",
    "ExpertWiseOffloadConfig",
    "OFFLOAD_MODES",
    "OffloadConfig",
    "configure_offload",
    "get_offload_config",
    "normalize_expert_selection",
]


def __getattr__(name):
    if name in {"ExpertBuffer", "ExpertBufferPool", "ExpertOffloadSlot"}:
        from .memory import ExpertBuffer, ExpertBufferPool, ExpertOffloadSlot

        return {
            "ExpertBuffer": ExpertBuffer,
            "ExpertBufferPool": ExpertBufferPool,
            "ExpertOffloadSlot": ExpertOffloadSlot,
        }[name]

    if name == "LayerWiseOffloadController":
        from .layer_wise import LayerWiseOffloadController

        return LayerWiseOffloadController

    if name in {"ExpertPlacement", "ExpertWiseExpertStore"}:
        from .expert_wise_memory import ExpertPlacement, ExpertWiseExpertStore

        return {
            "ExpertPlacement": ExpertPlacement,
            "ExpertWiseExpertStore": ExpertWiseExpertStore,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
