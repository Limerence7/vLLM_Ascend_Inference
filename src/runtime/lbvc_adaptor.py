import torch
import torch.distributed as dist

from ..moeload.policy import ExpertPolicy, ranked_experts_by_load
from ..moeload.profiler import ExpertLoadProfiler
from ..runtime_config import RuntimeConfig
from .exo_executor import ExoExecutor


class LBVCAdaptor:
    """Create expert placement updates and delegate transfers to ExoExecutor."""

    def __init__(
        self,
        config: RuntimeConfig,
        executor: ExoExecutor,
        profiler: ExpertLoadProfiler,
    ):
        self.config = config
        self.executor = executor
        self.policy = ExpertPolicy(config.load_history_path)
        self.profiler = profiler
        self.steps: dict[int, int] = {}
        self.slot_experts: dict[int, list[int]] = {}

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
        layer_id = layer.moe_instance_id
        self.executor.register_layer(layer)
        if self.config.uses_cold_buffer:
            self.slot_experts[layer_id] = (
                self.executor.resident_experts_for_layer(layer) +
                self.executor.cold_experts_for_layer(layer))
        else:
            self.slot_experts[layer_id] = self._slot_experts(layer)

    def load_weight(self, layer, param_name: str, shard_id: str,
                    global_expert_id: int,
                    loaded_weight: torch.Tensor) -> bool:
        self.executor.load_weight(layer, param_name, shard_id,
                                  global_expert_id, loaded_weight)
        return self._native_slot(layer, global_expert_id) < 0

    def process_layer_after_loading(self, layer, quant_method) -> None:
        self.executor.process_layer_after_loading(layer, quant_method)

    def record_expert_tokens(self, layer, group_list_type: int,
                             expert_tokens: torch.Tensor) -> None:
        self.profiler.record_expert_tokens(
            layer.moe_instance_id,
            expert_tokens,
            group_list_type,
        )
        if self._should_update(layer):
            self._update_global_experts(layer)

    def save_load_history(self) -> None:
        self.profiler.save()

    def _update_global_experts(self, layer) -> None:
        counts = self._global_expert_counts(layer)
        if counts is None:
            return
        if not self.policy.should_rebalance(
                counts, self.config.imbalance_threshold):
            return
        target_slots = self._target_slots(layer, counts)
        if self.config.uses_cold_buffer:
            self._update_cold_experts(layer, counts, target_slots)
        else:
            self._update_all_slots(layer, target_slots)

    def _update_all_slots(self, layer, target_slots: list[list[int]]) -> None:
        layer_id = layer.moe_instance_id
        target_experts = target_slots[layer.ep_rank]
        updates = self._slot_updates(self.slot_experts[layer_id],
                                     target_experts)
        if not updates:
            return

        self.executor.load_experts_to_slots(
            layer,
            [expert_id for _, expert_id in updates],
            [slot for slot, _ in updates],
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        self.slot_experts[layer_id] = list(target_experts)
        self._install_global_plan(layer, target_slots)
        self.profiler.update_layer_map(layer)

    def _update_cold_experts(
        self,
        layer,
        counts: torch.Tensor,
        target_slots: list[list[int]],
    ) -> None:
        current_cold = self.executor.cold_experts_for_layer(layer)
        if not current_cold:
            return

        resident = set(self.executor.resident_experts_for_layer(layer))
        target_cold = [
            expert_id for expert_id in target_slots[layer.ep_rank]
            if expert_id not in resident
        ]
        if len(target_cold) < len(current_cold):
            target_cold.extend(
                expert_id for expert_id in ranked_experts_by_load(
                    counts, list(range(layer.logical_num_experts)))
                if expert_id not in resident
                and expert_id not in target_cold)

        target_cold = target_cold[:len(current_cold)]
        if target_cold == current_cold:
            return

        self.executor.set_cold_experts(layer, target_cold)
        self.slot_experts[layer.moe_instance_id] = (
            self.executor.resident_experts_for_layer(layer) + target_cold)
        self.profiler.update_layer_map(layer)

    def _target_slots(self, layer, counts: torch.Tensor) -> list[list[int]]:
        return self.policy.global_balance_slots(
            counts=counts,
            num_experts=layer.logical_num_experts,
            ep_size=layer.ep_size,
            slots_per_rank=layer.full_local_num_experts,
        )

    @staticmethod
    def _slot_updates(current: list[int],
                      target: list[int]) -> list[tuple[int, int]]:
        return [
            (slot, expert_id)
            for slot, expert_id in enumerate(target)
            if slot >= len(current) or current[slot] != expert_id
        ]

    def _should_update(self, layer) -> bool:
        layer_id = layer.moe_instance_id
        step = self.steps.get(layer_id, 0) + 1
        self.steps[layer_id] = step
        return step % int(self.config.policy_interval) == 0

    def _global_expert_counts(self, layer) -> torch.Tensor | None:
        counts = self.profiler.get_layer_delta_load(layer.moe_instance_id)
        if counts.numel() < layer.logical_num_experts:
            return None

        counts = counts[:int(layer.logical_num_experts)]
        if int(counts.sum().item()) <= 0:
            return None
        if not dist.is_available() or not dist.is_initialized():
            return counts

        device = next(layer.parameters()).device
        device_counts = counts.to(device=device, dtype=torch.long)
        dist.all_reduce(device_counts,
                        group=layer.moe_config.ep_group.device_group)
        return device_counts.cpu()

    @staticmethod
    def _slot_experts(layer) -> list[int]:
        if layer.full_expert_map is None:
            return list(range(int(layer.full_local_num_experts)))

        experts = [-1] * layer.full_local_num_experts
        for global_id, local_id in enumerate(
                layer.full_expert_map.detach().cpu().tolist()):
            if 0 <= local_id < len(experts):
                experts[local_id] = global_id
        return experts

    @staticmethod
    def _native_slot(layer, global_expert_id: int) -> int:
        if layer._expert_map is None:
            return global_expert_id
        return int(layer._expert_map[global_expert_id].item())

    def _install_global_plan(self, layer,
                             target_slots: list[list[int]]) -> None:
        expert_map = self.policy.maps_from_slots(
            target_slots=target_slots,
            num_experts=layer.logical_num_experts,
            global_num_experts=layer.global_num_experts,
        )[layer.ep_rank]
        log2phy = self.policy.log2phy_from_slots(
            target_slots=target_slots,
            num_experts=layer.logical_num_experts,
            global_num_experts=layer.global_num_experts,
        )[layer.ep_rank]

        if layer.full_expert_map is not None:
            layer.full_expert_map.copy_(expert_map)
        if layer._expert_map is not None:
            layer._expert_map.copy_(expert_map)
        if layer.log2phy is not None:
            layer.log2phy.copy_(log2phy.to(layer.log2phy.device))
        layer._resident_maps_by_device.clear()
