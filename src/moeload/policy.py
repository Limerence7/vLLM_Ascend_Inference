import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


DEFAULT_MAX_SWAP_PASSES = 32
MAX_REPLICA_CANDIDATES = 16


class LoadHistory:
    """Load profiler records and expose one global load vector per layer."""

    _cpu_group = None

    def __init__(self, history_path: str | None):
        self.history_path = history_path
        self._loads: dict[int, torch.Tensor | None] = {}

    def global_load_for_layer(self, layer) -> torch.Tensor | None:
        layer_id = int(layer.moe_instance_id)
        if layer_id not in self._loads:
            self._loads[layer_id] = self._broadcast_load(layer)
        return self._loads[layer_id]

    def _broadcast_load(self, layer) -> torch.Tensor | None:
        load = self._build_load(layer) if self._is_rank0() else None
        if not dist.is_available() or not dist.is_initialized():
            return load
        if dist.get_world_size() == 1:
            return load

        payload: list[torch.Tensor | None] = [load]
        dist.broadcast_object_list(payload, src=0, group=self._get_cpu_group())
        return payload[0]

    def _build_load(self, layer) -> torch.Tensor | None:
        if not self.history_path:
            return None

        layer_id = int(layer.moe_instance_id)
        num_experts = int(
            getattr(layer, "logical_num_experts", layer.global_num_experts))
        total = torch.zeros(num_experts, dtype=torch.long)
        found = False

        for rank in range(int(layer.ep_size)):
            rank_load = self._read_rank_load(rank, layer_id, num_experts)
            if rank_load is None:
                continue
            total += rank_load
            found = True

        return total if found else None

    def _read_rank_load(
        self,
        rank: int,
        layer_id: int,
        num_experts: int,
    ) -> torch.Tensor | None:
        path = self._rank_file_path(rank)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as file:
            payload: dict[str, Any] = json.load(file)

        layer_record = payload.get("layers", {}).get(str(layer_id))
        if not isinstance(layer_record, dict):
            return None

        load = torch.zeros(num_experts, dtype=torch.long)
        for record in layer_record.get("experts", []):
            expert_id = int(record["expert_id"])
            if 0 <= expert_id < num_experts:
                load[expert_id] += int(record["activated_tokens"])
        return load

    def _rank_file_path(self, rank: int) -> Path:
        assert self.history_path is not None
        directory = Path(self.history_path)
        return directory / f"{directory.name}_rank{rank}.json"

    @staticmethod
    def _is_rank0() -> bool:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True

    @classmethod
    def _get_cpu_group(cls):
        if cls._cpu_group is None:
            cls._cpu_group = dist.new_group(backend="gloo")
        return cls._cpu_group


@dataclass(frozen=True)
class BalancePlan:
    expert_map: torch.Tensor
    log2phy: torch.Tensor
    slot_to_global: list[int]
    redundant_experts: list[int]


@dataclass(frozen=True)
class GlobalPlacementPlan:
    """One capacity-constrained placement and its predicted rank loads."""

    slots: list[list[int]]
    rank_loads: torch.Tensor

    @property
    def peak_to_average(self) -> float:
        return peak_to_average(self.rank_loads)


class ExpertPolicy:
    """Expert placement policy for offload and balance modes."""

    def __init__(self, history_path: str | None = None):
        self.history = LoadHistory(history_path)

    def history_load(self, layer) -> torch.Tensor | None:
        return self.history.global_load_for_layer(layer)

    def offload_expert_map(
        self,
        local_expert_map: list[int],
        num_resident: int,
        global_load: torch.Tensor | None,
    ) -> list[int]:
        """Return one hot-first slot-to-global map for an offload layer."""
        hot_experts = ranked_experts_by_load(
            global_load, local_expert_map)[:num_resident]
        hot_set = set(hot_experts)
        cold_experts = [
            expert_id for expert_id in local_expert_map
            if expert_id not in hot_set
        ]
        return [*hot_experts, *cold_experts]

    def balance_plan(
        self,
        num_experts: int,
        ep_size: int,
        ep_rank: int,
        redundant_per_rank: int,
        global_load: torch.Tensor | None,
    ) -> BalancePlan:
        if num_experts % ep_size != 0:
            raise ValueError("balance mode requires evenly split experts.")

        base_slots = num_experts // ep_size
        slots_per_rank = base_slots + redundant_per_rank
        global_num_experts = num_experts + ep_size * redundant_per_rank

        if global_load is None:
            target_slots = self._native_slots(num_experts, ep_size)
            self._append_redundant_slots(target_slots, redundant_per_rank)
        else:
            target_slots = self.global_balance_slots(
                counts=global_load,
                num_experts=num_experts,
                ep_size=ep_size,
                slots_per_rank=slots_per_rank,
            )

        expert_maps = self.maps_from_slots(
            target_slots, num_experts, global_num_experts)
        log2phy = self.log2phy_from_slots(
            target_slots, num_experts, global_num_experts)
        expert_map = expert_maps[ep_rank]
        return BalancePlan(
            expert_map=expert_map,
            log2phy=log2phy[ep_rank],
            slot_to_global=list(target_slots[ep_rank]),
            redundant_experts=redundant_experts_from_map(
                expert_map, base_slots),
        )

    def global_balance_slots(
        self,
        counts: torch.Tensor,
        num_experts: int,
        ep_size: int,
        slots_per_rank: int,
        expert_ids: list[int] | None = None,
        fixed_slots_by_rank: list[list[int]] | None = None,
        current_slots: list[list[int]] | None = None,
        max_swap_passes: int = DEFAULT_MAX_SWAP_PASSES,
    ) -> list[list[int]]:
        return self.global_balance_plan(
            counts=counts,
            num_experts=num_experts,
            ep_size=ep_size,
            slots_per_rank=slots_per_rank,
            expert_ids=expert_ids,
            fixed_slots_by_rank=fixed_slots_by_rank,
            current_slots=current_slots,
            max_swap_passes=max_swap_passes,
        ).slots

    def global_balance_plan(
        self,
        counts: torch.Tensor,
        num_experts: int,
        ep_size: int,
        slots_per_rank: int,
        expert_ids: list[int] | None = None,
        fixed_slots_by_rank: list[list[int]] | None = None,
        current_slots: list[list[int]] | None = None,
        max_swap_passes: int = DEFAULT_MAX_SWAP_PASSES,
    ) -> GlobalPlacementPlan:
        """Place movable experts after accounting for fixed rank load.

        ``fixed_slots_by_rank`` is intentionally separate from the returned
        placement.  Cold-buffer experts are immutable, but their load is part
        of the objective used to place resident experts.
        """
        if expert_ids is None:
            expert_ids = list(range(num_experts))
        expert_ids = [int(expert_id) for expert_id in expert_ids]
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError("candidate expert ids must be unique.")
        if ep_size * slots_per_rank < len(expert_ids):
            raise ValueError("not enough slots to place every expert.")
        if slots_per_rank > len(expert_ids):
            raise ValueError(
                "slots_per_rank cannot exceed the number of candidate experts.")

        fixed_slots = _normalize_fixed_slots(fixed_slots_by_rank, ep_size)
        fixed_experts = {
            int(expert_id)
            for rank_slots in fixed_slots
            for expert_id in rank_slots
        }
        overlap = fixed_experts.intersection(expert_ids)
        if overlap:
            raise ValueError(
                "fixed and movable expert scopes must be disjoint; overlap: "
                f"{sorted(overlap)}")

        load = _load_tensor(counts, num_experts)
        slots: list[list[int]] = [[] for _ in range(ep_size)]
        rank_loads = _rank_loads_for_slots(fixed_slots, load).tolist()
        unique_capacities = _balanced_unique_capacities(
            len(expert_ids), slots_per_rank, rank_loads)
        ranked = ranked_experts_by_load(load, expert_ids)
        expert_ranks: dict[int, list[int]] = {}

        for expert_id in ranked:
            rank = _best_unique_rank(slots, rank_loads, unique_capacities,
                                     float(load[expert_id]))
            slots[rank].append(expert_id)
            rank_loads[rank] += float(load[expert_id])
            expert_ranks.setdefault(int(expert_id), []).append(rank)

        while any(len(rank_slots) < slots_per_rank for rank_slots in slots):
            rank, expert_id, rank_loads = _best_replica_placement(
                slots=slots,
                rank_loads=rank_loads,
                expert_ranks=expert_ranks,
                ranked_experts=ranked,
                load=load,
                slots_per_rank=slots_per_rank,
            )
            slots[rank].append(expert_id)
            expert_ranks.setdefault(int(expert_id), []).append(rank)

        rank_load_tensor = rank_loads_for_placement(
            slots, load, fixed_slots)
        slots, rank_load_tensor = _improve_with_pair_swaps(
            slots, rank_load_tensor, load, max_swap_passes)

        if _valid_current_slots(current_slots, expert_ids, ep_size,
                                slots_per_rank):
            assert current_slots is not None
            stable_slots = [list(rank_slots) for rank_slots in current_slots]
            stable_loads = rank_loads_for_placement(
                stable_slots, load, fixed_slots)
            stable_slots, stable_loads = _improve_with_pair_swaps(
                stable_slots, stable_loads, load, max_swap_passes)
            if _placement_choice_key(
                    stable_loads, stable_slots, current_slots) < (
                        _placement_choice_key(
                            rank_load_tensor, slots, current_slots)):
                slots = stable_slots
                rank_load_tensor = stable_loads

        return GlobalPlacementPlan(slots=slots, rank_loads=rank_load_tensor)

    @staticmethod
    def maps_from_slots(
        target_slots: list[list[int]],
        num_experts: int,
        global_num_experts: int,
    ) -> torch.Tensor:
        expert_maps = torch.full((len(target_slots), global_num_experts),
                                 -1,
                                 dtype=torch.int32)
        for rank, rank_slots in enumerate(target_slots):
            for slot, expert_id in enumerate(rank_slots):
                expert_maps[rank, expert_id] = slot
        return expert_maps

    @staticmethod
    def log2phy_from_slots(
        target_slots: list[list[int]],
        num_experts: int,
        global_num_experts: int | None = None,
    ) -> torch.Tensor:
        global_num_experts = global_num_experts or num_experts
        slots_per_rank = len(target_slots[0])
        log2phy = torch.zeros((len(target_slots), global_num_experts),
                              dtype=torch.int32)
        physical_slots = _physical_slots_by_expert(target_slots,
                                                   num_experts)

        for rank, rank_slots in enumerate(target_slots):
            local_map = {expert_id: slot
                         for slot, expert_id in enumerate(rank_slots)}
            for expert_id in range(num_experts):
                if expert_id in local_map:
                    slot = rank * slots_per_rank + local_map[expert_id]
                else:
                    slots = physical_slots[expert_id]
                    slot = slots[(rank + expert_id) % len(slots)]
                log2phy[rank, expert_id] = slot
        return log2phy

    def should_rebalance(self, counts: torch.Tensor,
                         imbalance_threshold: float) -> bool:
        if counts.numel() == 0:
            return False
        return peak_to_average(counts) > float(imbalance_threshold)

    @staticmethod
    def _native_slots(num_experts: int, ep_size: int) -> list[list[int]]:
        base_slots = num_experts // ep_size
        return [
            list(range(rank * base_slots, (rank + 1) * base_slots))
            for rank in range(ep_size)
        ]

    @staticmethod
    def _append_redundant_slots(
        target_slots: list[list[int]],
        redundant_per_rank: int,
    ) -> None:
        if redundant_per_rank <= 0:
            return

        num_experts = sum(len(rank_slots) for rank_slots in target_slots)
        cursor = 0
        for rank_slots in target_slots:
            for _ in range(redundant_per_rank):
                while cursor % num_experts in rank_slots:
                    cursor += 1
                rank_slots.append(cursor % num_experts)
                cursor += 1

def redundant_experts_from_map(expert_map: torch.Tensor,
                               local_count: int) -> list[int]:
    return [
        expert_id
        for expert_id, slot in enumerate(expert_map.detach().cpu().tolist())
        if slot >= local_count
    ]


def ranked_experts_by_load(
    global_load: torch.Tensor | list[int] | None,
    expert_ids: list[int],
) -> list[int]:
    if global_load is None:
        return expert_ids

    values = (global_load.detach().cpu().tolist()
              if isinstance(global_load, torch.Tensor) else global_load)
    return sorted(expert_ids, key=lambda expert_id:
                  (-int(values[expert_id]), expert_id))


def _load_tensor(counts: torch.Tensor, num_experts: int) -> torch.Tensor:
    return counts.detach().cpu()[:num_experts].to(torch.float32)


def peak_to_average(rank_loads: torch.Tensor | list[float]) -> float:
    values = torch.as_tensor(rank_loads, dtype=torch.float64)
    if values.numel() == 0:
        return 1.0
    mean_load = float(values.mean().item())
    if mean_load <= 0:
        return 1.0
    return float(values.max().item()) / mean_load


def rank_loads_for_placement(
    movable_slots_by_rank: list[list[int]],
    counts: torch.Tensor,
    fixed_slots_by_rank: list[list[int]] | None = None,
) -> torch.Tensor:
    """Estimate rank load, sharing an expert evenly across its replicas."""
    fixed_slots = _normalize_fixed_slots(
        fixed_slots_by_rank, len(movable_slots_by_rank))
    return _rank_loads_for_slots([
        [*movable, *fixed]
        for movable, fixed in zip(movable_slots_by_rank, fixed_slots)
    ], counts)


def _rank_loads_for_slots(
    slots_by_rank: list[list[int]],
    counts: torch.Tensor,
) -> torch.Tensor:
    load = counts.detach().cpu().to(torch.float64)
    replicas: dict[int, int] = {}
    for rank_slots in slots_by_rank:
        for expert_id in rank_slots:
            expert_id = int(expert_id)
            replicas[expert_id] = replicas.get(expert_id, 0) + 1

    rank_loads = torch.zeros(len(slots_by_rank), dtype=torch.float64)
    for rank, rank_slots in enumerate(slots_by_rank):
        for expert_id in rank_slots:
            expert_id = int(expert_id)
            if 0 <= expert_id < load.numel():
                rank_loads[rank] += load[expert_id] / replicas[expert_id]
    return rank_loads.to(torch.float32)


def _normalize_fixed_slots(
    fixed_slots_by_rank: list[list[int]] | None,
    ep_size: int,
) -> list[list[int]]:
    if fixed_slots_by_rank is None:
        return [[] for _ in range(ep_size)]
    if len(fixed_slots_by_rank) != ep_size:
        raise ValueError(
            "fixed_slots_by_rank must contain one entry per EP rank.")
    return [list(map(int, rank_slots))
            for rank_slots in fixed_slots_by_rank]


def _best_unique_rank(
    slots: list[list[int]],
    rank_loads: list[float],
    unique_capacities: list[int],
    expert_load: float,
) -> int:
    candidates = [
        rank for rank, rank_slots in enumerate(slots)
        if len(rank_slots) < unique_capacities[rank]
    ]
    if not candidates:
        raise ValueError("unable to place expert: all ranks are full.")
    return min(candidates, key=lambda rank: (
        rank_loads[rank] + expert_load,
        len(slots[rank]),
        rank,
    ))


def _balanced_unique_capacities(
    num_unique_experts: int,
    slots_per_rank: int,
    base_rank_loads: list[float],
) -> list[int]:
    ep_size = len(base_rank_loads)
    base, remainder = divmod(num_unique_experts, ep_size)
    if base + (1 if remainder else 0) > slots_per_rank:
        raise ValueError("not enough per-rank capacity for unique experts.")
    capacities = [base] * ep_size
    lightest_ranks = sorted(
        range(ep_size), key=lambda rank: (base_rank_loads[rank], rank))
    for rank in lightest_ranks[:remainder]:
        capacities[rank] += 1
    return capacities


def _improve_with_pair_swaps(
    slots: list[list[int]],
    rank_loads: torch.Tensor,
    load: torch.Tensor,
    max_passes: int,
) -> tuple[list[list[int]], torch.Tensor]:
    """Reach a pair-swap local optimum without changing replica counts."""
    slots = [list(rank_slots) for rank_slots in slots]
    values = rank_loads.detach().cpu().to(torch.float64).tolist()
    replicas: dict[int, int] = {}
    for rank_slots in slots:
        for expert_id in rank_slots:
            expert_id = int(expert_id)
            replicas[expert_id] = replicas.get(expert_id, 0) + 1

    contributions = {
        expert_id: float(load[expert_id]) / replica_count
        for expert_id, replica_count in replicas.items()
    }
    max_passes = min(
        max(0, int(max_passes)),
        sum(len(rank_slots) for rank_slots in slots),
    )
    for _ in range(max_passes):
        best: tuple[tuple[float, float, float], int, int, int, int, float,
                    float] | None = None
        current_score = _balance_objective(values)
        mean_load = sum(values) / len(values)
        peak_load = max(values)
        hot_ranks = [
            rank for rank, rank_load in enumerate(values)
            if abs(rank_load - peak_load) <= 1e-9
        ]
        for left_rank in hot_ranks:
            for right_rank in range(len(slots)):
                if left_rank == right_rank:
                    continue
                unaffected = [
                    value for rank, value in enumerate(values)
                    if rank not in (left_rank, right_rank)
                ]
                other_max = max(unaffected) if unaffected else float("-inf")
                other_min = min(unaffected) if unaffected else float("inf")
                for left_slot, left_expert in enumerate(slots[left_rank]):
                    for right_slot, right_expert in enumerate(
                            slots[right_rank]):
                        if left_expert == right_expert:
                            continue
                        if (right_expert in slots[left_rank]
                                or left_expert in slots[right_rank]):
                            continue
                        new_left_load = values[left_rank] + (
                            contributions[right_expert]
                            - contributions[left_expert])
                        new_right_load = values[right_rank] + (
                            contributions[left_expert]
                            - contributions[right_expert])
                        new_max = max(
                            other_max, new_left_load, new_right_load)
                        new_min = min(
                            other_min, new_left_load, new_right_load)
                        squared_error = (
                            current_score[2]
                            - (values[left_rank] - mean_load)**2
                            - (values[right_rank] - mean_load)**2
                            + (new_left_load - mean_load)**2
                            + (new_right_load - mean_load)**2
                        )
                        score = (
                            new_max,
                            new_max - new_min,
                            squared_error,
                        )
                        if score >= current_score:
                            continue
                        candidate = (score, left_rank, right_rank, left_slot,
                                     right_slot, new_left_load,
                                     new_right_load)
                        if best is None or candidate[:5] < best[:5]:
                            best = candidate
        if best is None:
            break
        (_, left_rank, right_rank, left_slot, right_slot, new_left_load,
         new_right_load) = best
        values[left_rank] = new_left_load
        values[right_rank] = new_right_load
        slots[left_rank][left_slot], slots[right_rank][right_slot] = (
            slots[right_rank][right_slot], slots[left_rank][left_slot])

    return slots, torch.tensor(values, dtype=torch.float32)


def _balance_objective(
    rank_loads: list[float] | torch.Tensor,
) -> tuple[float, float, float]:
    values = (rank_loads.detach().cpu().to(torch.float64).tolist()
              if isinstance(rank_loads, torch.Tensor) else rank_loads)
    if not values:
        return (0.0, 0.0, 0.0)
    max_load = max(values)
    min_load = min(values)
    mean_load = sum(values) / len(values)
    squared_error = sum((value - mean_load)**2 for value in values)
    return (max_load, max_load - min_load, squared_error)


def _valid_current_slots(
    current_slots: list[list[int]] | None,
    expert_ids: list[int],
    ep_size: int,
    slots_per_rank: int,
) -> bool:
    if current_slots is None or len(current_slots) != ep_size:
        return False
    if any(len(rank_slots) != slots_per_rank
           or len(rank_slots) != len(set(rank_slots))
           for rank_slots in current_slots):
        return False
    flattened = [int(expert_id)
                 for rank_slots in current_slots
                 for expert_id in rank_slots]
    candidates = set(expert_ids)
    return set(flattened).issubset(candidates) and candidates.issubset(
        flattened)


def _placement_choice_key(
    rank_loads: torch.Tensor,
    target_slots: list[list[int]],
    current_slots: list[list[int]],
) -> tuple[float, float, float, int]:
    migrations = 0
    for current, target in zip(current_slots, target_slots):
        remaining = list(current)
        for expert_id in target:
            if expert_id in remaining:
                remaining.remove(expert_id)
            else:
                migrations += 1
    return (*_balance_objective(rank_loads), migrations)


def _best_replica_placement(
    slots: list[list[int]],
    rank_loads: list[float],
    expert_ranks: dict[int, list[int]],
    ranked_experts: list[int],
    load: torch.Tensor,
    slots_per_rank: int,
) -> tuple[int, int, list[float]]:
    best: tuple[tuple[float, float, float, int, int], int, int, list[float]]
    best = ((float("inf"), float("inf"), float("inf"), -1, -1), -1, -1, [])
    ranked_by_unit_load = sorted(
        ranked_experts,
        key=lambda expert_id: (
            -float(load[expert_id])
            / max(1, len(expert_ranks.get(int(expert_id), []))),
            int(expert_id),
        ),
    )
    evaluated = 0
    for expert_id in ranked_by_unit_load:
        holders = expert_ranks.get(int(expert_id), [])
        candidate_ranks = [
            rank for rank, rank_slots in enumerate(slots)
            if len(rank_slots) < slots_per_rank and expert_id not in rank_slots
        ]
        if not candidate_ranks:
            continue
        evaluated += 1
        rank = min(candidate_ranks, key=lambda candidate: (
            rank_loads[candidate], len(slots[candidate]), candidate))
        projected = _rank_loads_after_replica(
            rank_loads, holders, rank, float(load[expert_id]))
        score = _rank_load_score(projected, len(slots[rank]), expert_id)
        if score < best[0]:
            best = (score, rank, int(expert_id), projected)
        if evaluated >= MAX_REPLICA_CANDIDATES:
            break
    if best[1] < 0:
        raise ValueError("unable to place a replica without duplicate rank slots.")
    return best[1], best[2], best[3]


def _rank_loads_after_replica(
    rank_loads: list[float],
    current_holders: list[int],
    target_rank: int,
    expert_load: float,
) -> list[float]:
    old_replicas = len(current_holders)
    if old_replicas <= 0:
        projected = list(rank_loads)
        projected[target_rank] += expert_load
        return projected

    old_share = expert_load / old_replicas
    new_share = expert_load / (old_replicas + 1)
    projected = list(rank_loads)
    for rank in current_holders:
        projected[rank] -= old_share
        projected[rank] += new_share
    projected[target_rank] += new_share
    return projected


def _rank_load_score(
    rank_loads: list[float],
    target_slot_count: int,
    expert_id: int,
) -> tuple[float, float, float, int, int]:
    max_load, spread, squared_error = _balance_objective(rank_loads)
    return (max_load, spread, squared_error, target_slot_count, expert_id)


def _physical_slots_by_expert(
    target_slots: list[list[int]],
    num_experts: int,
) -> list[list[int]]:
    slots_per_rank = len(target_slots[0])
    physical_slots: list[list[int]] = [[] for _ in range(num_experts)]
    for rank, rank_slots in enumerate(target_slots):
        for slot, expert_id in enumerate(rank_slots):
            physical_slots[expert_id].append(rank * slots_per_rank + slot)
    return physical_slots
