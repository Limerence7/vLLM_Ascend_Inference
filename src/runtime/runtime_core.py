import torch

from ..moeload.policy import ExpertPolicy
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
    _policy: ExpertPolicy | None = None

    @property
    def executor(self) -> ExoExecutor | None:
        return RuntimeCore._executor

    @property
    def adaptor(self) -> LBVCAdaptor | None:
        return RuntimeCore._adaptor

    @property
    def profiler(self) -> ExpertLoadProfiler:
        assert RuntimeCore._profiler is not None
        return RuntimeCore._profiler

    @property
    def policy(self) -> ExpertPolicy:
        assert RuntimeCore._policy is not None
        return RuntimeCore._policy

    @classmethod
    def reset(cls) -> None:
        cls._executor = None
        cls._adaptor = None
        cls._profiler = None
        cls._memory_manager = None
        cls._policy = None

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
        if RuntimeCore._policy is None:
            RuntimeCore._policy = ExpertPolicy(config.load_history_path)

        if config.uses_runtime_core:
            if RuntimeCore._memory_manager is None:
                RuntimeCore._memory_manager = ExpertMemoryManager(
                    cpu_pin_memory=config.cpu_pin_memory,
                )
            if RuntimeCore._executor is None:
                assert RuntimeCore._memory_manager is not None
                RuntimeCore._executor = ExoExecutor(
                    config, uses_w8a8, RuntimeCore._memory_manager)

        if config.runtime_mode == "balance":
            assert self.executor is not None
            if RuntimeCore._adaptor is None:
                RuntimeCore._adaptor = LBVCAdaptor(
                    config, self.executor, self.profiler, self.policy)

    def global_load_for_layer(self, layer) -> torch.Tensor | None:
        return self.adaptor.global_load_for_layer(layer)

    def initial_balance_maps(
        self,
        num_experts: int,
        ep_size: int,
        ep_rank: int,
        global_load: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, torch.Tensor, list[int]]:
        return self.adaptor.initial_expert_maps(
            num_experts=num_experts,
            ep_size=ep_size,
            ep_rank=ep_rank,
            global_load=global_load,
        )

    def initial_offload_expert_map(self,
                                   layer,
                                   local_expert_map: list[int]
                                   ) -> tuple[list[int], int]:
        if self.config.offload_count > len(local_expert_map):
            raise ValueError(
                f"Layer {layer.moe_instance_id} has "
                f"{len(local_expert_map)} local expert slots, but "
                f"offload_count is {self.config.offload_count}.")
        hot_count = len(local_expert_map) - self.config.offload_count
        if self.config.runtime_mode == "offload":
            global_load = (
                self.policy.history_load(layer)
                if self.config.enable_history_mapping else None)
            return (self.policy.offload_expert_map(
                local_expert_map, hot_count, global_load), hot_count)
        global_load = (
            self.policy.history_load(layer)
            if self.config.enable_history_mapping else None)
        return (self.policy.offload_expert_map(
            local_expert_map, hot_count, global_load), hot_count)

    def should_manage_layer(self, layer) -> bool:
        if self.executor is None:
            return False
        return self.executor.should_manage_layer(layer)

    def uses_cold_buffer_for(self, layer) -> bool:
        return (self.config.uses_cold_buffer
                and self.should_manage_layer(layer))

    @property
    def collect_load(self) -> bool:
        return bool(self.executor is not None and self.executor.collect_load)

    def register_layer(self, layer, expert_map: list[int]) -> None:
        self.profiler.register_layer(layer, expert_map)
        if (self.executor is not None
                and not self.executor.should_manage_layer(layer)):
            return
        if self.config.runtime_mode in ("balance", "offload"):
            assert self.executor is not None
            self.executor.register_layer(layer, expert_map)

    def layout(self, layer) -> tuple[list[int], int]:
        assert self.executor is not None
        return self.executor.layout(layer)

    def load_weight(self, layer, param_name: str, shard_id: str,
                    expert_id: int, loaded_weight: torch.Tensor) -> bool:
        if (self.executor is not None
                and not self.executor.should_manage_layer(layer)):
            return False
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
        if (self.executor is not None
                and not self.executor.should_manage_layer(layer)):
            return
        if self.config.runtime_mode in ("balance", "offload"):
            assert self.executor is not None
            self.executor.process_layer_after_loading(layer, quant_method)

    def prepare_cold_experts(self, layer, topk_ids: torch.Tensor):
        assert self.executor is not None
        return self.executor.prepare_cold_experts(layer, topk_ids)

    def hot_routing(self, layer, topk_ids: torch.Tensor,
                    cold_mask: torch.Tensor) -> tuple[torch.Tensor,
                                                       torch.Tensor]:
        assert self.executor is not None
        return self.executor.hot_routing(layer, topk_ids, cold_mask)

    def prefetch_next_layers(self, layer) -> None:
        assert self.executor is not None
        self.executor.prefetch_next_layers(layer)

    def record_expert_tokens(self, layer, group_list_type: int,
                             expert_tokens: torch.Tensor) -> None:
        if (self.executor is not None
                and not self.executor.should_manage_layer(layer)):
            self.profiler.record_expert_tokens(
                int(layer.moe_instance_id), expert_tokens, group_list_type)
            return
        if self.config.runtime_mode == "balance":
            assert self.adaptor is not None
            self.adaptor.record_expert_tokens(layer, group_list_type,
                                              expert_tokens)
        elif self.config.runtime_mode == "offload":
            self.profiler.record_expert_tokens(
                int(layer.moe_instance_id), expert_tokens, group_list_type)
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
        self.profiler.record_slot_tokens(
            int(layer.moe_instance_id), expert_tokens, group_list_type,
            slot_to_global)

    def save_load_history(self) -> None:
        self.profiler.save()
