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
    policy_interval: int = 16
    imbalance_threshold: float = 0.2

    num_buffers: int = 2

    enable_offline_scheduler: bool = False
    min_step_tokens: int = 4000
    balance_audit_path: str | None = None
    balance_audit_interval: int = 1

    def validate(self) -> None:
        self.runtime_mode = self.runtime_mode.strip().lower()
        assert self.runtime_mode in RUNTIME_MODES, (
            f"{self.runtime_mode} is not supported for runtime plugin.")
        assert self.interval > 0, 'interval must be a positive integer.'
        assert self.num_buffers > 0, (
            'num_buffers must be a positive integer.')
        assert self.policy_interval > 0, (
            'policy_interval must be a positive integer.')
        assert self.min_step_tokens >= 0, (
            'min_step_tokens must be a non-negative integer.')
        assert self.balance_audit_interval > 0, (
            'balance_audit_interval must be a positive integer.')
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
