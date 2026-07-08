import torch

from ..moeload.profiler import ExpertLoadProfiler
from ..runtime_config import RuntimeConfig
from .exo_executor import ExoExecutor
from .lbvc_adaptor import LBVCAdaptor
from .memory_manager import ExpertMemoryManager


class RuntimeCore:
    """Coordinate profiler, policy adaptor and CPU-NPU transfer executor."""

    _executor: ExoExecutor | None = None
    _adaptor: LBVCAdaptor | None = None
    _profiler: ExpertLoadProfiler | None = None
    _memory_manager: ExpertMemoryManager | None = None

    @classmethod
    def reset(cls) -> None:
        cls._executor = None
        cls._adaptor = None
        cls._profiler = None
        cls._memory_manager = None

    def __init__(self, config: RuntimeConfig, uses_w8a8: bool):
        self.config = config
        if RuntimeCore._profiler is None:
            RuntimeCore._profiler = ExpertLoadProfiler(
                config.load_history_path,
                metadata={
                    "runtime_mode": config.runtime_mode,
                    "runtime_layer_ids": list(config.runtime_layer_ids),
                },
            )
        self.profiler = RuntimeCore._profiler

        if config.uses_runtime_core:
            if RuntimeCore._memory_manager is None:
                RuntimeCore._memory_manager = ExpertMemoryManager(
                    cpu_pin_memory=config.cpu_pin_memory,
                    share_cpu_experts=(
                        config.runtime_mode == "balance"
                        and config.share_all_cpu_experts),
                    shared_cpu_expert_dir=config.shared_cpu_expert_dir,
                    shared_cpu_expert_name=config.shared_cpu_expert_name,
                )
            self.memory_manager = RuntimeCore._memory_manager
            if RuntimeCore._executor is None:
                RuntimeCore._executor = ExoExecutor(
                    config, uses_w8a8, self.memory_manager, self.profiler)
            self.executor = RuntimeCore._executor
        else:
            self.memory_manager = None
            self.executor = None

        if config.runtime_mode == "balance":
            assert self.executor is not None
            if RuntimeCore._adaptor is None:
                RuntimeCore._adaptor = LBVCAdaptor(
                    config, self.executor, self.profiler)
            self.adaptor = RuntimeCore._adaptor
        else:
            self.adaptor = None

    def global_load_for_layer(self, layer) -> torch.Tensor | None:
        return (
            None if self.adaptor is None else
            self.adaptor.global_load_for_layer(layer))

    def initial_balance_maps(
        self,
        num_experts: int,
        ep_size: int,
        ep_rank: int,
        global_load: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        assert self.adaptor is not None
        return self.adaptor.initial_expert_maps(
            num_experts=num_experts,
            ep_size=ep_size,
            ep_rank=ep_rank,
            global_load=global_load,
        )

    def init_offload_placement(self, layer):
        assert self.executor is not None
        return self.executor.init_layer_placement(layer)

    def should_manage_layer(self, layer) -> bool:
        if self.executor is None:
            return False
        return self.executor.should_manage_layer(layer)

    @property
    def collect_load(self) -> bool:
        return bool(self.executor is not None and self.executor.collect_load)

    def register_layer(self, layer) -> None:
        if self.config.runtime_mode == "balance":
            assert self.adaptor is not None
            self.adaptor.register_layer(layer)
        elif self.config.runtime_mode == "offload":
            assert self.executor is not None
            self.executor.register_layer(layer)
        else:
            self.profiler.register_layer(layer)

    def load_weight(self, layer, param_name: str, shard_id: str,
                    expert_id: int, loaded_weight: torch.Tensor) -> bool:
        if self.config.runtime_mode == "balance":
            assert self.adaptor is not None
            return self.adaptor.load_weight(layer, param_name, shard_id,
                                            expert_id, loaded_weight)
        if self.config.runtime_mode == "offload":
            assert self.executor is not None
            return self.executor.load_weight(layer, param_name, shard_id,
                                             expert_id, loaded_weight)
        return False

    def process_layer_after_loading(self, layer, quant_method) -> None:
        if self.config.runtime_mode == "balance":
            assert self.adaptor is not None
            self.adaptor.process_layer_after_loading(layer, quant_method)
        elif self.config.runtime_mode == "offload":
            assert self.executor is not None
            self.executor.process_layer_after_loading(layer, quant_method)

    def prepare_cold_experts(self, layer, topk_ids: torch.Tensor):
        assert self.executor is not None
        return self.executor.prepare_cold_experts(layer, topk_ids)

    def prefetch_next_layers(self, layer) -> None:
        assert self.executor is not None
        self.executor.prefetch_next_layers(layer)

    def resident_slot_to_global(self, layer) -> torch.Tensor:
        assert self.executor is not None
        return self.executor.resident_slot_to_global(layer)

    def cold_slot_to_global(self, layer) -> torch.Tensor:
        assert self.executor is not None
        return self.executor.cold_slot_to_global(layer)

    def record_expert_tokens(self, layer, group_list_type: int,
                             expert_tokens: torch.Tensor) -> None:
        if self.config.runtime_mode == "balance":
            assert self.adaptor is not None
            self.adaptor.record_expert_tokens(layer, group_list_type,
                                              expert_tokens)
        elif self.config.runtime_mode == "offload":
            assert self.executor is not None
            self.executor.record_expert_tokens(layer, group_list_type,
                                               expert_tokens)
        else:
            self.profiler.record_expert_tokens(
                int(layer.moe_instance_id), expert_tokens, group_list_type)

    def record_slot_expert_tokens(
        self,
        layer,
        group_list_type: int,
        expert_tokens: torch.Tensor,
        slot_to_global: torch.Tensor,
    ) -> None:
        assert self.executor is not None
        self.executor.record_slot_expert_tokens(
            layer, group_list_type, expert_tokens, slot_to_global)

    def save_load_history(self) -> None:
        if self.adaptor is not None:
            self.adaptor.save_load_history()
        elif self.executor is not None:
            self.executor.save_load_history()
        else:
            self.profiler.save()
