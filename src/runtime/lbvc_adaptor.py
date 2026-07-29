import torch
import torch.distributed as dist

from ..moeload.balance_audit import BalanceAuditRecorder
from ..moeload.policy import ExpertPolicy
from ..moeload.profiler import ExpertLoadProfiler
from ..runtime_config import RuntimeConfig
from .exo_executor import ExoExecutor
from .transfer_planner import TransferPlanner


class LBVCAdaptor:
    """Balance expert placement and publish completed HCCL updates."""

    def __init__(
        self,
        config: RuntimeConfig,
        executor: ExoExecutor,
        profiler: ExpertLoadProfiler,
        policy: ExpertPolicy,
    ):
        self.config = config
        self.executor = executor
        self.policy = policy
        self.profiler = profiler
        self.transfer_planner = TransferPlanner()
        self.audit = BalanceAuditRecorder(config.balance_audit_path)
        self.steps: dict[int, int] = {}

    def initial_expert_maps(
        self,
        num_experts: int,
        ep_size: int,
        ep_rank: int,
        global_load: torch.Tensor | None = None,
    ) -> tuple[int, torch.Tensor, torch.Tensor, list[int]]:
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
            plan.slot_to_global,
        )

    def global_load_for_layer(self, layer) -> torch.Tensor | None:
        if not self.config.enable_history_mapping:
            return None
        return self.policy.history_load(layer)

    def load_weight(self, layer, param_name: str, shard_id: str,
                    global_expert_id: int,
                    loaded_weight: torch.Tensor) -> bool:
        self.executor.load_weight(layer, param_name, shard_id,
                                  global_expert_id, loaded_weight)
        return self._native_slot(layer, global_expert_id) < 0

    def record_expert_tokens(self, layer, group_list_type: int,
                             expert_tokens: torch.Tensor) -> None:
        self.profiler.record_expert_tokens(
            layer.moe_instance_id,
            expert_tokens,
            group_list_type,
        )
        if self._should_update(layer):
            self._update_global_experts(layer)

    def _update_global_experts(self, layer) -> None:
        counts = self._global_expert_counts(layer)
        if counts is None:
            return
        if not self.policy.should_rebalance(
                counts, self.config.imbalance_threshold):
            return
        if self.config.uses_cold_buffer:
            current_slots = self._all_rank_slots(
                self.executor.layout(layer)[0])
            hot_count = self.executor.layout(layer)[1]
            current_slots = [slots[:hot_count] for slots in current_slots]
            hot_experts = sorted({
                expert_id
                for rank_slots in current_slots
                for expert_id in rank_slots
            })
            target_slots = self.policy.global_balance_slots(
                counts=counts,
                num_experts=layer.logical_num_experts,
                ep_size=layer.ep_size,
                slots_per_rank=hot_count,
                expert_ids=hot_experts,
            )
        else:
            current_slots = self._all_rank_slots(
                self.executor.layout(layer)[0])
            target_slots = self.policy.global_balance_slots(
                counts=counts,
                num_experts=layer.logical_num_experts,
                ep_size=layer.ep_size,
                slots_per_rank=len(current_slots[0]),
            )

        plan = self.transfer_planner.plan(current_slots, target_slots)
        if not plan.copies:
            return
        self._record_balance_audit(layer, counts, plan.target_slots)
        self.transfer_planner.transfer(layer, plan)
        dist.barrier()

        local_slots = plan.target_slots[layer.ep_rank]
        self.executor.update_layout(layer, local_slots)
        if self.config.uses_cold_buffer:
            self._install_resident_map(layer, local_slots)
        else:
            self._install_global_plan(layer, plan.target_slots)
        self.profiler.update_layer_map(layer, local_slots)

    def _record_balance_audit(
        self,
        layer,
        counts: torch.Tensor,
        target_slots: list[list[int]],
    ) -> None:
        audit = getattr(self, "audit", None)
        if audit is None or not audit.enabled:
            return

        layer_id = layer.moe_instance_id
        step = self.steps.get(layer_id, 0)
        if step % int(self.config.balance_audit_interval) != 0:
            return

        audit.record_rebalance_plan(
            layer=layer,
            step=step,
            counts=counts,
            current_slots=self._all_rank_slots(
                self.executor.layout(layer)[0]),
            target_slots=target_slots,
        )

    def _should_update(self, layer) -> bool:
        layer_id = layer.moe_instance_id
        step = self.steps.get(layer_id, 0) + 1
        self.steps[layer_id] = step
        return step % int(self.config.policy_interval) == 0

    @staticmethod
    def _all_rank_slots(local_slots: list[int]) -> list[list[int]]:
        if not dist.is_available() or not dist.is_initialized():
            return [list(local_slots)]

        all_slots = [None] * dist.get_world_size()
        dist.all_gather_object(all_slots, list(local_slots))
        return [list(rank_slots) for rank_slots in all_slots]

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
    def _native_slot(layer, global_expert_id: int) -> int:
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

        layer._expert_map = expert_map
        if layer.log2phy is not None:
            layer.log2phy.copy_(log2phy.to(layer.log2phy.device))

    def _install_resident_map(self, layer, target_resident: list[int]) -> None:
        expert_map = torch.full((layer.global_num_experts, ),
                                -1,
                                dtype=torch.int32)
        for slot, expert_id in enumerate(target_resident):
            expert_map[int(expert_id)] = slot

        layer._expert_map = expert_map
