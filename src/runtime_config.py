from dataclasses import dataclass, field, replace


RUNTIME_MODES = ("profile", "offload", "balance")


@dataclass
class RuntimeConfig:
    runtime_mode: str = "profile"
    runtime_layer_ids: list[int] = field(default_factory=list)
    interval: int = 1
    cpu_pin_memory: bool = True

    load_history_path: str | None = None
    enable_history_mapping: bool = False

    num_hot_experts: int = 0
    num_buffers: int = 2

    pair_topology: list[tuple[int, int]] | None = None
    num_redundant_experts: int = 0
    imbalance_threshold: float = 0.2
    scheduler_interval: int = 16
    num_experts_per_update: int = 1

    def validate(self) -> None:
        self.runtime_mode = self.runtime_mode.strip().lower()
        assert self.runtime_mode in RUNTIME_MODES, (
            f"{self.runtime_mode} is not supported for runtime plugin.")
        assert self.interval > 0, 'interval must be a positive integer.'
        assert self.num_hot_experts >= 0, 'num_hot_experts must be non-negative.'
        if self.runtime_mode == "offload":
            assert self.num_buffers == 2, (
                'offload mode currently requires num_buffers to be 2.')
        else:
            assert self.num_buffers > 0, (
                'num_buffers must be a positive integer.')
        assert self.num_redundant_experts >= 0, (
            'num_redundant_experts must be non-negative.')
        assert self.scheduler_interval > 0, (
            'scheduler_interval must be a positive integer.')
        assert self.num_experts_per_update > 0, (
            'num_experts_per_update must be a positive integer.')

    @property
    def offload_layer_wise(self) -> bool:
        return self.runtime_mode == "offload" and self.num_hot_experts == 0

    @property
    def use_cpu_experts(self) -> bool:
        return self.runtime_mode in ("offload", "balance")

    def prepare_for_model(self, num_layers: int, num_experts: int) -> None:
        self.validate()

        layer_ids = (
            sorted(set(self.runtime_layer_ids))
            or list(range(0, num_layers, self.interval)))

        if self.runtime_mode == "offload" and self.num_hot_experts >= num_experts:
            layer_ids = []
        self.runtime_layer_ids = [
            layer_id for layer_id in layer_ids
            if 0 <= layer_id < num_layers
        ]


RUNTIME_CONFIG = RuntimeConfig()


def get_runtime_config() -> RuntimeConfig:
    return RUNTIME_CONFIG


def set_runtime_config(config: RuntimeConfig) -> RuntimeConfig:
    global RUNTIME_CONFIG
    RUNTIME_CONFIG = replace(
        config, runtime_layer_ids=list(config.runtime_layer_ids))
    RUNTIME_CONFIG.validate()
    return RUNTIME_CONFIG
