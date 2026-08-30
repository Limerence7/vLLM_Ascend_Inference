from dataclasses import dataclass, field, replace


RUNTIME_MODES = ("profile", "offload", "balance")


@dataclass
class RuntimeConfig:
    runtime_mode: str = "profile"
    runtime_layer_ids: list[int] = field(default_factory=list)
    interval: int = 1
    cpu_pin_memory: bool = True
    num_runtime_experts: int = 0

    load_history_path: str | None = None
    enable_history_mapping: bool = False
    enable_load_collection: bool = False
    load_collect_interval: int = 128
    rebalance_interval: int = 512
    policy_interval: int = 512
    imbalance_threshold: float = 1.5
    rebalance_max_layers: int = 1
    rebalance_min_improvement: float = 0.01

    num_buffers: int = 2

    enable_offline_scheduler: bool = False
    min_step_tokens: int = 4000
    scheduler_min_step_tokens: int | None = None
    scheduler_reorder_window: int = 64
    scheduler_policy: str = "expert"
    rebalance_min_step_tokens: int = 4096

    def validate(self) -> None:
        self.runtime_mode = self.runtime_mode.strip().lower()
        self.scheduler_policy = self.scheduler_policy.strip().lower()
        assert self.runtime_mode in RUNTIME_MODES, (
            f"{self.runtime_mode} is not supported for runtime plugin.")
        assert self.interval > 0, 'interval must be a positive integer.'
        assert self.num_buffers > 0, (
            'num_buffers must be a positive integer.')
        assert self.load_collect_interval > 0, (
            'load_collect_interval must be a positive integer.')
        if self.rebalance_interval is None:
            self.rebalance_interval = self.policy_interval
        assert self.rebalance_interval > 0, (
            'rebalance_interval must be a positive integer.')
        assert self.policy_interval > 0, (
            'policy_interval must be a positive integer.')
        assert self.imbalance_threshold >= 1.0, (
            'imbalance_threshold must be a peak/average ratio no smaller '
            'than 1.0.')
        assert self.rebalance_max_layers >= 0, (
            'rebalance_max_layers must be non-negative; 0 means unlimited.')
        assert self.rebalance_min_improvement >= 0, (
            'rebalance_min_improvement must be non-negative.')
        assert self.min_step_tokens >= 0, (
            'min_step_tokens must be a non-negative integer.')
        assert self.scheduler_policy in ("fifo", "throughput", "expert"), (
            'scheduler_policy must be one of fifo, throughput or expert.')
        assert self.scheduler_reorder_window > 0, (
            'scheduler_reorder_window must be a positive integer.')
        if self.scheduler_min_step_tokens is None:
            self.scheduler_min_step_tokens = self.min_step_tokens
        if self.rebalance_min_step_tokens is None:
            self.rebalance_min_step_tokens = self.min_step_tokens
        assert self.scheduler_min_step_tokens >= 0, (
            'scheduler_min_step_tokens must be a non-negative integer.')
        assert self.rebalance_min_step_tokens >= 0, (
            'rebalance_min_step_tokens must be a non-negative integer.')
        if self.runtime_mode == "profile":
            assert self.num_runtime_experts == 0, (
                'profile mode must keep num_runtime_experts at 0.')
        if self.runtime_mode == "offload":
            assert self.num_runtime_experts <= 0, (
                'offload mode expects num_runtime_experts <= 0.')

    @property
    def offload_count(self) -> int:
        return (
            abs(self.num_runtime_experts)
            if self.uses_cold_buffer else 0)

    @property
    def redundant_count(self) -> int:
        return (
            self.num_runtime_experts
            if self.runtime_mode == "balance" and self.num_runtime_experts > 0
            else 0)

    @property
    def uses_runtime_core(self) -> bool:
        return self.runtime_mode in ("offload", "balance")

    @property
    def uses_cold_buffer(self) -> bool:
        return (
            self.runtime_mode in ("offload", "balance")
            and self.num_runtime_experts < 0)

    @property
    def needs_load_collection(self) -> bool:
        return (
            self.runtime_mode == "profile"
            or self.enable_load_collection
            or self.runtime_mode == "balance")

    @property
    def needs_dynamic_rebalance(self) -> bool:
        return self.runtime_mode == "balance"

    def prepare_for_model(self, num_layers: int, num_experts: int) -> None:
        self.validate()

        layer_ids = (
            sorted(set(self.runtime_layer_ids))
            or list(range(0, num_layers, self.interval)))

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
        config,
        runtime_layer_ids=list(config.runtime_layer_ids),
    )
    RUNTIME_CONFIG.validate()
    return RUNTIME_CONFIG
