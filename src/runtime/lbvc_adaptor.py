from dataclasses import dataclass

import torch
import torch.distributed as dist

from ..moeload.policy import (ExpertPolicy, peak_to_average,
                              rank_loads_for_placement)
from ..moeload.profiler import ExpertLoadProfiler
from ..runtime_config import RuntimeConfig
from .exo_executor import ExoExecutor
from .exp_updator import ExpertUpdateTask, ExpertUpdator, HcclCopyTask
from .balance_coordinator import (BalanceCoordinator,
                                  LayerBalanceCandidate)


@dataclass(frozen=True)
class BalanceUpdatePlan:
    target_slots: list[list[int]]
    local_slots: list[int]
    expert_map: torch.Tensor
    log2phy: torch.Tensor | None
    task: ExpertUpdateTask


@dataclass(frozen=True)
class BalancePlanningContext:
    counts: torch.Tensor
    current_slots: list[list[int]]
    fixed_slots: list[list[int]]
    slots_per_rank: int
    expert_ids: list[int] | None


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
        self.exp_updator = ExpertUpdator()
        self.coordinator = BalanceCoordinator(
            max_layers=config.rebalance_max_layers,
            min_improvement=config.rebalance_min_improvement,
        )
        self._pending_layer_loads: dict[int, tuple[object, torch.Tensor]] = {}

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
                             expert_tokens: torch.Tensor, step: int) -> None:
        if step <= 0:
            return
        self.profiler.record_expert_tokens(
            layer.moe_instance_id,
            expert_tokens,
            group_list_type,
        )
        if step % int(self.config.rebalance_interval) == 0:
            self._queue_global_expert_update(layer)

    def record_slot_expert_tokens(
        self,
        layer,
        group_list_type: int,
        expert_tokens: torch.Tensor,
        slot_to_global: torch.Tensor,
        step: int,
    ) -> None:
        if step <= 0:
            return
        self.profiler.record_slot_tokens(
            layer.moe_instance_id,
            expert_tokens,
            group_list_type,
            slot_to_global,
        )
        if step % int(self.config.rebalance_interval) == 0:
            self._queue_global_expert_update(layer)

    def _queue_global_expert_update(self, layer) -> None:
        layer_id = int(layer.moe_instance_id)
        counts = self.profiler.get_layer_delta_load(layer_id)
        if counts.numel() >= int(layer.logical_num_experts):
            self._pending_layer_loads[layer_id] = (
                layer,
                counts[:int(layer.logical_num_experts)],
            )
        if self._is_rebalance_cycle_boundary(layer):
            self._plan_and_flush_updates()

    def _build_candidate(
        self,
        layer,
        counts: torch.Tensor,
        current_full_slots: list[list[int]],
    ) -> LayerBalanceCandidate | None:
        current_slots, fixed_slots, slots_per_rank, expert_ids = (
            self._current_balance_scope(layer, current_full_slots))
        current_rank_loads = rank_loads_for_placement(
            current_slots, counts, fixed_slots)
        if not self._should_rebalance(current_rank_loads):
            return None

        placement = self.policy.global_balance_plan(
            counts=counts,
            num_experts=layer.logical_num_experts,
            ep_size=layer.ep_size,
            slots_per_rank=slots_per_rank,
            expert_ids=expert_ids,
            fixed_slots_by_rank=fixed_slots,
            current_slots=current_slots,
            max_swap_passes=1,
        )
        plan = self._build_update_plan(
            layer, current_slots, placement.slots)
        if not plan.task.copies:
            return None

        return LayerBalanceCandidate(
            layer=layer,
            plan=plan,
            current_score=peak_to_average(current_rank_loads),
            target_score=placement.peak_to_average,
            migrations=len(plan.task.copies),
            context=BalancePlanningContext(
                counts=counts,
                current_slots=current_slots,
                fixed_slots=fixed_slots,
                slots_per_rank=slots_per_rank,
                expert_ids=expert_ids,
            ),
        )

    def _plan_and_flush_updates(self) -> None:
        reduced = self._reduce_pending_layer_loads()
        layouts = self._gather_layer_layouts(
            [layer for layer, _ in reduced])
        for layer, counts in reduced:
            if int(counts.sum().item()) > 0:
                self.coordinator.submit(
                    self._build_candidate(
                        layer,
                        counts,
                        layouts[int(layer.moe_instance_id)],
                    ))
        self._flush_updates()

    def _reduce_pending_layer_loads(
        self,
    ) -> list[tuple[object, torch.Tensor]]:
        pending = [
            self._pending_layer_loads[layer_id]
            for layer_id in sorted(self._pending_layer_loads)
        ]
        self._pending_layer_loads.clear()
        if not pending:
            return []

        reduced: list[tuple[object, torch.Tensor]] = []
        groups: dict[int, list[tuple[object, torch.Tensor]]] = {}
        for layer, counts in pending:
            groups.setdefault(int(counts.numel()), []).append(
                (layer, counts))

        for group in groups.values():
            layers = [layer for layer, _ in group]
            stacked = torch.stack([counts for _, counts in group])
            if dist.is_available() and dist.is_initialized():
                device = next(layers[0].parameters()).device
                stacked = stacked.to(device=device, dtype=torch.long)
                dist.all_reduce(
                    stacked,
                    group=layers[0].moe_config.ep_group.device_group,
                )
                stacked = stacked.cpu()
            reduced.extend(zip(layers, stacked.unbind(0)))
        return reduced

    def _gather_layer_layouts(
        self,
        layers: list[object],
    ) -> dict[int, list[list[int]]]:
        if not layers:
            return {}
        local_layouts = {
            int(layer.moe_instance_id): self.executor.layout(layer)[0]
            for layer in layers
        }
        if not dist.is_available() or not dist.is_initialized():
            return {
                layer_id: [slots]
                for layer_id, slots in local_layouts.items()
            }

        first_layer = layers[0]
        group = getattr(first_layer.moe_config.ep_group, "cpu_group", None)
        world_size = dist.get_world_size(group)
        all_layouts = [None] * world_size
        dist.all_gather_object(all_layouts, local_layouts, group=group)
        return {
            layer_id: [
                list(rank_layouts[layer_id])
                for rank_layouts in all_layouts
            ]
            for layer_id in local_layouts
        }

    def _flush_updates(self) -> None:
        candidates = self.coordinator.drain()
        if not candidates:
            return

        candidates = [
            self._refine_candidate(candidate)
            for candidate in candidates
        ]
        candidates = [
            candidate for candidate in candidates
            if candidate.migrations > 0
            and candidate.improvement >= (
                self.config.rebalance_min_improvement)
        ]
        if not candidates:
            return

        self.exp_updator.transfer_many([
            (candidate.layer, candidate.plan.task)
            for candidate in candidates
        ])
        if dist.is_available() and dist.is_initialized():
            first_layer = candidates[0].layer
            dist.barrier(group=getattr(first_layer.moe_config.ep_group,
                                       "cpu_group", None))

        for candidate in candidates:
            layer = candidate.layer
            plan = candidate.plan
            self.executor.update_layout(layer, plan.local_slots)
            self._install_plan(layer, plan)
            self.profiler.update_layer_map(
                layer, self.executor.layout(layer)[0])

    def _refine_candidate(
        self,
        candidate: LayerBalanceCandidate,
    ) -> LayerBalanceCandidate:
        context = candidate.context
        assert isinstance(context, BalancePlanningContext)
        layer = candidate.layer
        placement = self.policy.global_balance_plan(
            counts=context.counts,
            num_experts=layer.logical_num_experts,
            ep_size=layer.ep_size,
            slots_per_rank=context.slots_per_rank,
            expert_ids=context.expert_ids,
            fixed_slots_by_rank=context.fixed_slots,
            current_slots=context.current_slots,
        )
        plan = self._build_update_plan(
            layer, context.current_slots, placement.slots)
        return LayerBalanceCandidate(
            layer=layer,
            plan=plan,
            current_score=candidate.current_score,
            target_score=placement.peak_to_average,
            migrations=len(plan.task.copies),
            context=context,
        )

    def _current_balance_scope(
        self,
        layer,
        current_slots: list[list[int]],
    ) -> tuple[list[list[int]], list[list[int]], int, list[int] | None]:
        _, hot_count = self.executor.layout(layer)
        if not self.config.uses_cold_buffer:
            fixed_slots = [[] for _ in current_slots]
            return current_slots, fixed_slots, len(current_slots[0]), None

        resident_slots = [slots[:hot_count] for slots in current_slots]
        fixed_slots = [slots[hot_count:] for slots in current_slots]
        resident_experts = sorted({
            expert_id
            for rank_slots in resident_slots
            for expert_id in rank_slots
        })
        return resident_slots, fixed_slots, hot_count, resident_experts

    def _build_update_plan(
        self,
        layer,
        current_slots: list[list[int]],
        target_slots: list[list[int]],
    ) -> BalanceUpdatePlan:
        task = self._make_update_task(current_slots, target_slots)
        local_slots = task.target_slots[layer.ep_rank]
        if self.config.uses_cold_buffer:
            expert_map = self._resident_expert_map(
                layer.global_num_experts, local_slots)
            log2phy = None
        else:
            expert_map = self.policy.maps_from_slots(
                target_slots=task.target_slots,
                num_experts=layer.logical_num_experts,
                global_num_experts=layer.global_num_experts,
            )[layer.ep_rank]
            log2phy = None
            if self.config.redundant_count > 0:
                log2phy = self.policy.log2phy_from_slots(
                    target_slots=task.target_slots,
                    num_experts=layer.logical_num_experts,
                    global_num_experts=layer.global_num_experts,
                )[layer.ep_rank]
        return BalanceUpdatePlan(
            target_slots=task.target_slots,
            local_slots=local_slots,
            expert_map=expert_map,
            log2phy=log2phy,
            task=task,
        )

    def _should_rebalance(
        self,
        rank_loads: torch.Tensor,
    ) -> bool:
        if int(rank_loads.sum().item()) < int(
                self.config.rebalance_min_step_tokens):
            return False

        max_load = float(rank_loads.max().item())
        mean_load = float(rank_loads.mean().item())
        if max_load <= 0 or mean_load <= 0:
            return False
        ratio = max_load / mean_load
        return ratio > float(self.config.imbalance_threshold)

    @staticmethod
    def _rank_loads(
        slots_by_rank: list[list[int]],
        counts: torch.Tensor,
    ) -> torch.Tensor:
        return rank_loads_for_placement(slots_by_rank, counts)

    def _is_rebalance_cycle_boundary(self, layer) -> bool:
        registered_layers = sorted(self.executor.layers)
        return bool(registered_layers) and int(
            layer.moe_instance_id) == registered_layers[-1]

    def _make_update_task(
        self,
        current_slots: list[list[int]],
        target_slots: list[list[int]],
    ) -> ExpertUpdateTask:
        aligned = [
            self._align_slots(current, target)
            for current, target in zip(current_slots, target_slots)
        ]
        locations: dict[int, list[tuple[int, int]]] = {}
        for rank, rank_slots in enumerate(current_slots):
            for slot, expert_id in enumerate(rank_slots):
                locations.setdefault(int(expert_id), []).append((rank, slot))
        copies = [
            HcclCopyTask(
                *self._source_location(locations[int(expert_id)], rank),
                rank,
                slot,
                int(expert_id),
            )
            for rank, rank_slots in enumerate(aligned)
            for slot, expert_id in enumerate(rank_slots)
            if current_slots[rank][slot] != expert_id
        ]
        return ExpertUpdateTask(aligned, copies)

    @staticmethod
    def _source_location(
        locations: list[tuple[int, int]],
        target_rank: int,
    ) -> tuple[int, int]:
        for rank, slot in locations:
            if rank == target_rank:
                return rank, slot
        return locations[0]

    @staticmethod
    def _align_slots(current: list[int], target: list[int]) -> list[int]:
        remaining: dict[int, int] = {}
        for expert_id in target:
            remaining[expert_id] = remaining.get(expert_id, 0) + 1

        aligned: list[int | None] = []
        for expert_id in current:
            if remaining.get(expert_id, 0) > 0:
                aligned.append(expert_id)
                remaining[expert_id] -= 1
            else:
                aligned.append(None)

        incoming = iter(
            expert_id for expert_id, count in remaining.items()
            for _ in range(count))
        return [
            expert_id if expert_id is not None else next(incoming)
            for expert_id in aligned
        ]

    @staticmethod
    def _native_slot(layer, global_expert_id: int) -> int:
        return int(layer._expert_map[global_expert_id].item())

    def _install_plan(self, layer, plan: BalanceUpdatePlan) -> None:
        layer._expert_map = plan.expert_map.to(layer._expert_map.device,
                                               non_blocking=True)
        if layer.log2phy is not None and plan.log2phy is not None:
            layer.log2phy.copy_(plan.log2phy.to(layer.log2phy.device))

    @staticmethod
    def _resident_expert_map(
        global_num_experts: int,
        target_resident: list[int],
    ) -> torch.Tensor:
        expert_map = torch.full((global_num_experts, ),
                                -1,
                                dtype=torch.int32)
        for slot, expert_id in enumerate(target_resident):
            expert_map[int(expert_id)] = slot

        return expert_map
