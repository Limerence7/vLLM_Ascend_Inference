from dataclasses import dataclass, field, replace


OFFLOAD_MODES = ("none", "manual", "auto")
LOAD_BALANCE_MODES = ("none", "history", "dynamic")


@dataclass
class OffloadConfig:
    mode: str = "manual"
    interval: int = 2
    num_buffers: int = 2
    num_hot_experts: int = 0
    cpu_pin_memory: bool = True
    offloaded_layer_ids: list[int] = field(default_factory=list)
    load_stats_path: str | None = None
    load_balance_mode: str = "none"
    dynamic_update_interval: int = 32
    dynamic_max_swaps: int = 2
    dynamic_min_swap_gain: int = 0
    dynamic_cooldown_interval: int = 0

    def validate(self) -> None:
        self.mode = self.mode.strip().lower()
        self.load_balance_mode = self.load_balance_mode.strip().lower()
        assert self.mode in OFFLOAD_MODES, (
            f"{self.mode} is not supported for offloading.")
        assert self.load_balance_mode in LOAD_BALANCE_MODES, (
            f"{self.load_balance_mode} is not supported for load balance.")
        assert self.interval > 0, "Offload Interval must be positive"
        # assert self.num_buffers >= 2, (
        #     "Offload needs at least two cold expert buffers.")
        assert self.num_hot_experts >= 0, "num_hot_experts must be non-negative."
        assert self.dynamic_update_interval > 0, (
            "dynamic_update_interval must be positive.")
        assert self.dynamic_max_swaps >= 0, (
            "dynamic_max_swaps must be non-negative.")
        assert self.dynamic_min_swap_gain >= 0, (
            "dynamic_min_swap_gain must be non-negative.")
        assert self.dynamic_cooldown_interval >= 0, (
            "dynamic_cooldown_interval must be non-negative.")

    @property
    def offload_full_layers(self) -> bool:
        return (self.num_hot_experts == 0)

    def prepare_for_model(self, num_layers: int, num_experts: int) -> None:
        self.validate()

        if self.mode == "none" or self.num_hot_experts >= num_experts:
            layer_ids: list[int] = []
        else:
            self.num_hot_experts = min(self.num_hot_experts, num_experts)
            no_offload = (self.num_hot_experts == num_experts)
            layer_ids = [] if no_offload else (
                self.offloaded_layer_ids
                or list(range(0, num_layers, self.interval)))

        self.offloaded_layer_ids = [
            layer_id for layer_id in sorted({int(i) for i in layer_ids})
            if 0 <= layer_id < num_layers
        ]

OFFLOAD_CONFIG = OffloadConfig()

def get_offload_config() -> OffloadConfig:
    return OFFLOAD_CONFIG

def set_offload_config(config: OffloadConfig) -> OffloadConfig:
    global OFFLOAD_CONFIG
    OFFLOAD_CONFIG = replace(
        config, offloaded_layer_ids=list(config.offloaded_layer_ids))
    OFFLOAD_CONFIG.validate()
    return OFFLOAD_CONFIG
