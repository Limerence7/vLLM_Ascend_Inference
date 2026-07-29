import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


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
    ) -> list[list[int]]:
        if expert_ids is None:
            expert_ids = list(range(num_experts))
        if ep_size * slots_per_rank < len(expert_ids):
            raise ValueError("not enough slots to place every expert.")

        load = _load_tensor(counts, num_experts)
        slots: list[list[int]] = [[] for _ in range(ep_size)]
        rank_loads = [0.0] * ep_size
        ranked = ranked_experts_by_load(load, expert_ids)

        for expert_id in ranked:
            rank = _lightest_rank(slots, rank_loads, slots_per_rank)
            slots[rank].append(expert_id)
            rank_loads[rank] += float(load[expert_id])

        replica_index = 0
        while any(len(rank_slots) < slots_per_rank for rank_slots in slots):
            expert_id = ranked[replica_index % len(ranked)]
            rank = _lightest_rank(
                slots, rank_loads, slots_per_rank, exclude=expert_id)
            slots[rank].append(expert_id)
            rank_loads[rank] += float(load[expert_id])
            replica_index += 1

        return slots

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
        mean = float(counts.float().mean().item())
        return mean > 0 and (
            float(counts.max().item() - counts.min().item()) / mean
            >= imbalance_threshold)

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


def _lightest_rank(
    slots: list[list[int]],
    rank_loads: list[float],
    slots_per_rank: int,
    exclude: int | None = None,
) -> int:
    candidates = [
        rank for rank, rank_slots in enumerate(slots)
        if len(rank_slots) < slots_per_rank
        and (exclude is None or exclude not in rank_slots)
    ]
    if not candidates:
        candidates = [
            rank for rank, rank_slots in enumerate(slots)
            if len(rank_slots) < slots_per_rank
        ]
    return min(candidates, key=lambda rank:
               (rank_loads[rank], len(slots[rank]), rank))


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
