import torch
import torch.distributed as dist

from ..moeload.policy import ExpertPolicy
from ..runtime_config import RuntimeConfig
from .exo_executor import ExoExecutor


class LBVCAdaptor:
    """Create expert placement updates and delegate transfers to ExoExecutor."""

    def __init__(self, config: RuntimeConfig, executor: ExoExecutor):
        self.config = config
        self.executor = executor
        self.policy = ExpertPolicy(config.load_history_path)
        self.profiler = executor.profiler
        self.layers: dict[int, object] = {}
        self.steps: dict[int, int] = {}
        self.redundant_slots: dict[int, list[int]] = {}
        self.redundant_experts: dict[int, list[int]] = {}

    def initial_expert_maps(
        self,
        num_experts: int,
        ep_size: int,
        ep_rank: int,
        global_load: torch.Tensor | None = None,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        plan = self.policy.balance_plan(
            num_experts=num_experts,
            ep_size=ep_size,
            ep_rank=ep_rank,
            redundant_per_rank=self.config.redundant_count,
            global_load=global_load,
        )
        return (
            num_experts // ep_size + self.config.redundant_count,
            plan.expert_map,
            plan.log2phy,
        )

    def global_load_for_layer(self, layer) -> torch.Tensor | None:
        if not self.config.enable_history_mapping:
            return None
        return self.policy.history_load(layer)

    def register_layer(self, layer) -> None:
        layer_id = int(layer.moe_instance_id)
        redundant_count = int(self.config.redundant_count)

        self.layers[layer_id] = layer
        if self.config.uses_cold_buffer:
            slots = []
            experts = []
        else:
            slots = list(range(int(layer.full_local_num_experts)))
            experts = self._slot_experts(layer)

        self.redundant_slots[layer_id] = slots
        self.redundant_experts[layer_id] = experts
        self.executor.register_layer(layer)

    def load_weight(self, layer, param_name: str, shard_id: str,
                    global_expert_id: int,
                    loaded_weight: torch.Tensor) -> bool:
        return self.executor.load_weight(layer, param_name, shard_id,
                                         global_expert_id, loaded_weight)

    def process_layer_after_loading(self, layer, quant_method) -> None:
        self.executor.process_layer_after_loading(layer, quant_method)
        self._load_initial_redundant_experts(layer)
        torch.npu.empty_cache()

    def before_forward(self, layer) -> None:
        pass

    def record_expert_tokens(self, layer, group_list_type: int,
                             expert_tokens: torch.Tensor) -> None:
        self.profiler.record_expert_tokens(
            int(layer.moe_instance_id),
            expert_tokens,
            group_list_type,
        )
        if self._should_update(layer):
            self._update_global_experts(layer)

    def save_load_history(self) -> None:
        self.profiler.save()

    def _load_initial_redundant_experts(self, layer) -> None:
        layer_id = int(layer.moe_instance_id)
        experts = self.redundant_experts.get(layer_id, [])
        slots = self.redundant_slots.get(layer_id, [])
        if experts and slots:
            self.executor.load_experts_to_slots(layer, experts,
                                                slots[:len(experts)])

    def _update_global_experts(self, layer) -> None:
        layer_id = int(layer.moe_instance_id)
        slots = self.redundant_slots.get(layer_id, [])
        if not slots:
            return

        counts = self._global_expert_counts(layer)
        if counts is None:
            return
        if not self.policy.should_rebalance(
                counts, float(self.config.imbalance_threshold)):
            return

        target_slots = self.policy.global_balance_slots(
            counts=counts,
            num_experts=int(layer.logical_num_experts),
            ep_size=int(layer.ep_size),
            slots_per_rank=int(layer.full_local_num_experts),
        )
        target_experts = target_slots[int(layer.ep_rank)]
        current_experts = self.redundant_experts[layer_id]
        updates = [
            (slot, expert_id)
            for slot, expert_id in enumerate(target_experts)
            if slot >= len(current_experts)
            or current_experts[slot] != expert_id
        ]
        if not updates:
            return

        self.executor.load_experts_to_slots(
            layer,
            [expert_id for _, expert_id in updates],
            [slot for slot, _ in updates],
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        self.redundant_experts[layer_id] = list(target_experts)
        self._install_global_plan(layer, target_slots)
        self.profiler.update_layer_map(layer)

    def _should_update(self, layer) -> bool:
        layer_id = int(layer.moe_instance_id)
        step = self.steps.get(layer_id, 0) + 1
        self.steps[layer_id] = step
        return step % int(self.config.policy_interval) == 0

    def _global_expert_counts(self, layer) -> torch.Tensor | None:
        counts = self.profiler.get_layer_delta_load(int(layer.moe_instance_id))
        if counts.numel() < int(layer.logical_num_experts):
            return None

        counts = counts[:int(layer.logical_num_experts)]
        if int(counts.sum().item()) <= 0:
            return None
        if not dist.is_available() or not dist.is_initialized():
            return counts

        device_counts = counts.to(device=self._layer_device(layer),
                                  dtype=torch.long)
        dist.all_reduce(device_counts,
                        group=layer.moe_config.ep_group.device_group)
        return device_counts.cpu()

    @staticmethod
    def _current_local_experts(layer) -> list[int]:
        if layer.full_expert_map is None:
            return list(range(int(layer.logical_num_experts)))
        return [
            int(global_id)
            for global_id, local_id in enumerate(layer.full_expert_map.detach().cpu().tolist())
            if int(local_id) >= 0 and int(global_id) < int(layer.logical_num_experts)
        ]

    @staticmethod
    def _slot_experts(layer) -> list[int]:
        if layer.full_expert_map is None:
            return list(range(int(layer.full_local_num_experts)))

        experts = [-1] * int(layer.full_local_num_experts)
        for global_id, local_id in enumerate(layer.full_expert_map.detach().cpu().tolist()):
            if 0 <= int(local_id) < len(experts):
                experts[int(local_id)] = int(global_id)
        return experts

    @staticmethod
    def _layer_device(layer) -> torch.device:
        weight = getattr(layer, "w13_weight", None)
        if weight is not None:
            return weight.device

        weight_list = getattr(layer, "w13_weight_list", None)
        if weight_list:
            return weight_list[0].device

        return next(layer.parameters()).device

    def _install_global_plan(self, layer,
                             target_slots: list[list[int]]) -> None:
        expert_map = self.policy.maps_from_slots(
            target_slots=target_slots,
            num_experts=int(layer.logical_num_experts),
            global_num_experts=int(layer.global_num_experts),
        )[int(layer.ep_rank)]
        log2phy = self.policy.log2phy_from_slots(
            target_slots=target_slots,
            num_experts=int(layer.logical_num_experts),
            global_num_experts=int(layer.global_num_experts),
        )[int(layer.ep_rank)]

        if layer.full_expert_map is not None:
            layer.full_expert_map.copy_(expert_map)
        if layer._expert_map is not None:
            layer._expert_map.copy_(expert_map)
        if layer.log2phy is not None:
            layer.log2phy.copy_(log2phy.to(layer.log2phy.device))
        layer._resident_maps_by_device.clear()
