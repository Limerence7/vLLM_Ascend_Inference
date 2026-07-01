import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class HistoryLayerLoad:
    global_load: torch.Tensor


class LoadHistory:
    """Read and broadcast saved profiler output."""

    _cpu_group = None

    def __init__(self, history_path: str | None):
        self.history_path = history_path
        self._loads: dict[int, HistoryLayerLoad | None] = {}

    def global_load_for_layer(self, layer) -> torch.Tensor | None:
        layer_id = int(layer.moe_instance_id)
        if layer_id not in self._loads:
            self._loads[layer_id] = self._broadcast_load(layer)
        load = self._loads[layer_id]
        return None if load is None else load.global_load

    def _broadcast_load(self, layer) -> HistoryLayerLoad | None:
        load = self._build_load(layer) if self._is_rank0() else None
        if not dist.is_available() or not dist.is_initialized():
            return load
        if dist.get_world_size() == 1:
            return load

        payload: list[HistoryLayerLoad | None] = [load]
        dist.broadcast_object_list(payload, src=0, group=self._get_cpu_group())
        return payload[0]

    def _build_load(self, layer) -> HistoryLayerLoad | None:
        if not self.history_path:
            return None

        num_experts = int(
            getattr(layer, "logical_num_experts", layer.global_num_experts))
        total = torch.zeros(num_experts, dtype=torch.long)
        found = False
        for rank in range(int(layer.ep_size)):
            counts = self._read_rank_load(rank, int(layer.moe_instance_id))
            if counts is None or len(counts) != num_experts:
                continue
            found = True
            total += torch.tensor(counts, dtype=torch.long)
        return HistoryLayerLoad(total) if found else None

    def _read_rank_load(self, rank: int, layer_id: int) -> list[int] | None:
        path = self._rank_file_path(rank)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as file:
            payload: dict[str, Any] = json.load(file)
        counts = payload.get("layers", {}).get(str(layer_id))
        return counts if isinstance(counts, list) else None

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
    redundant_experts: list[int]


class ExpertPolicy:
    """Pure expert-placement policy for offload and balance modes."""

    def __init__(self, history_path: str | None = None):
        self.history = LoadHistory(history_path)

    def history_load(self, layer) -> torch.Tensor | None:
        return self.history.global_load_for_layer(layer)

    def offload_resident_ids(
        self,
        layer,
        num_resident: int,
        global_load: torch.Tensor | None,
    ) -> list[int]:
        if num_resident <= 0:
            return []

        local_experts = self._local_experts(layer)
        local_by_global = {
            global_id: local_id
            for local_id, global_id in local_experts
        }
        ranked_global = ranked_experts_by_load(global_load,
                                               list(local_by_global))
        return [
            local_by_global[global_id]
            for global_id in ranked_global[:num_resident]
        ]

    def balance_plan(
        self,
        num_experts: int,
        ep_size: int,
        ep_rank: int,
        redundant_per_rank: int,
        global_load: torch.Tensor | None,
    ) -> BalancePlan:
        if num_experts % ep_size != 0:
            raise ValueError(
                "balance mode requires experts evenly split by EP.")

        base_experts_per_rank = num_experts // ep_size
        local_slots = base_experts_per_rank + redundant_per_rank
        global_num_experts = num_experts + ep_size * redundant_per_rank

        if global_load is None:
            all_maps = self._base_maps(
                ep_size, num_experts, base_experts_per_rank,
                global_num_experts)
            if redundant_per_rank > 0:
                self._place_redundant_experts(all_maps, num_experts,
                                              base_experts_per_rank,
                                              redundant_per_rank, global_load)
        else:
            target_slots = self.global_balance_slots(
                counts=global_load,
                num_experts=num_experts,
                ep_size=ep_size,
                slots_per_rank=local_slots,
            )
            all_maps = self.maps_from_slots(
                target_slots=target_slots,
                num_experts=num_experts,
                global_num_experts=global_num_experts,
            )

        log2phy = self._build_log2phy(all_maps, num_experts, local_slots)
        expert_map = all_maps[ep_rank]
        return BalancePlan(
            expert_map=expert_map,
            log2phy=log2phy[ep_rank],
            redundant_experts=redundant_experts_from_map(
                expert_map, base_experts_per_rank),
        )

    def global_balance_slots(
        self,
        counts: torch.Tensor,
        num_experts: int,
        ep_size: int,
        slots_per_rank: int,
    ) -> list[list[int]]:
        load_values = self._load_values(counts, num_experts)
        targets: list[list[int]] = [[] for _ in range(ep_size)]
        rank_loads = [0.0 for _ in range(ep_size)]

        for expert_id in ranked_experts_by_load(load_values,
                                                list(range(num_experts))):
            rank = self._lightest_rank(targets, rank_loads, slots_per_rank)
            targets[rank].append(expert_id)
            rank_loads[rank] += float(load_values[expert_id].item())

        ranked = ranked_experts_by_load(load_values, list(range(num_experts)))
        replica_index = 0
        while any(len(slots) < slots_per_rank for slots in targets):
            expert_id = ranked[replica_index % len(ranked)]
            rank = self._lightest_rank(
                targets,
                rank_loads,
                slots_per_rank,
                exclude_expert=expert_id,
            )
            targets[rank].append(expert_id)
            rank_loads[rank] += float(load_values[expert_id].item())
            replica_index += 1

        return targets

    @staticmethod
    def maps_from_slots(
        target_slots: list[list[int]],
        num_experts: int,
        global_num_experts: int,
    ) -> torch.Tensor:
        all_maps = torch.full((len(target_slots), global_num_experts),
                              -1,
                              dtype=torch.int32)
        for rank, experts in enumerate(target_slots):
            for slot, expert_id in enumerate(experts):
                if expert_id < num_experts:
                    all_maps[rank, expert_id] = slot
        return all_maps

    @staticmethod
    def log2phy_from_slots(
        target_slots: list[list[int]],
        num_experts: int,
        global_num_experts: int | None = None,
    ) -> torch.Tensor:
        global_num_experts = global_num_experts or num_experts
        return ExpertPolicy._build_log2phy(
            ExpertPolicy.maps_from_slots(
                target_slots,
                num_experts,
                global_num_experts,
            ),
            num_experts,
            len(target_slots[0]),
        )

    def should_rebalance(self, counts: torch.Tensor,
                         imbalance_threshold: float) -> bool:
        if counts.numel() == 0:
            return False
        mean = float(counts.float().mean().item())
        if mean <= 0:
            return False
        spread = float(counts.max().item() - counts.min().item()) / mean
        return spread >= imbalance_threshold

    def replacement_candidates(
        self,
        counts: torch.Tensor,
        current_experts: list[int],
        num_experts: int,
    ) -> list[int]:
        current = set(current_experts)
        candidates = [
            expert_id for expert_id in range(num_experts)
            if expert_id not in current
        ]
        return ranked_experts_by_load(counts, candidates)

    @staticmethod
    def least_loaded_slot(
        counts: torch.Tensor,
        current_experts: list[int],
        used_slots: set[int],
    ) -> int:
        available = [
            index for index in range(len(current_experts))
            if index not in used_slots
        ]
        if not available:
            return -1
        loads = [int(counts[expert_id].item()) for expert_id in current_experts]
        return min(available, key=loads.__getitem__)

    @staticmethod
    def _lightest_rank(
        targets: list[list[int]],
        rank_loads: list[float],
        slots_per_rank: int,
        exclude_expert: int | None = None,
    ) -> int:
        ranks = [
            rank for rank, experts in enumerate(targets)
            if len(experts) < slots_per_rank
            and (exclude_expert is None or exclude_expert not in experts)
        ]
        if not ranks:
            ranks = [
                rank for rank, experts in enumerate(targets)
                if len(experts) < slots_per_rank
            ]
        return min(ranks, key=lambda rank:
                   (rank_loads[rank], len(targets[rank]), rank))

    @staticmethod
    def _base_maps(ep_size: int, num_experts: int,
                   base_experts_per_rank: int,
                   global_num_experts: int) -> torch.Tensor:
        maps = torch.full((ep_size, global_num_experts),
                          -1,
                          dtype=torch.int32)
        for rank in range(ep_size):
            start = rank * base_experts_per_rank
            end = start + base_experts_per_rank
            maps[rank, start:end] = torch.arange(base_experts_per_rank,
                                                 dtype=torch.int32)
        return maps

    def _place_redundant_experts(
        self,
        all_maps: torch.Tensor,
        num_experts: int,
        base_experts_per_rank: int,
        redundant_per_rank: int,
        global_load: torch.Tensor | None,
    ) -> None:
        replica_counts = torch.ones(num_experts, dtype=torch.long)
        load_values = self._load_values(global_load, num_experts)
        for rank in range(all_maps.size(0)):
            for offset in range(redundant_per_rank):
                candidates = [
                    expert_id for expert_id in range(num_experts)
                    if int(all_maps[rank, expert_id].item()) < 0
                ]
                if not candidates:
                    continue
                expert_id = self._select_redundant_expert(
                    candidates, replica_counts, load_values,
                    rank * redundant_per_rank + offset,
                    global_load is not None)
                all_maps[rank, expert_id] = base_experts_per_rank + offset
                replica_counts[expert_id] += 1

    @staticmethod
    def _select_redundant_expert(
        candidates: list[int],
        replica_counts: torch.Tensor,
        load_values: torch.Tensor,
        seed: int,
        has_history: bool,
    ) -> int:
        if has_history:
            return min(
                candidates,
                key=lambda expert_id:
                (-float(load_values[expert_id].item()) /
                 max(int(replica_counts[expert_id].item()), 1), expert_id),
            )
        num_experts = int(load_values.numel())
        return min(candidates,
                   key=lambda expert_id: ((expert_id - seed) % num_experts,
                                          expert_id))

    @staticmethod
    def _build_log2phy(all_maps: torch.Tensor, num_experts: int,
                       local_slots: int) -> torch.Tensor:
        ep_size, global_num_experts = all_maps.shape
        log2phy = torch.zeros((ep_size, global_num_experts), dtype=torch.int32)
        expert_slots: dict[int, list[int]] = {}

        for rank in range(ep_size):
            for expert_id in range(num_experts):
                local_slot = int(all_maps[rank, expert_id].item())
                if local_slot >= 0:
                    expert_slots.setdefault(expert_id, []).append(
                        rank * local_slots + local_slot)

        for rank in range(ep_size):
            for expert_id in range(num_experts):
                local_slot = int(all_maps[rank, expert_id].item())
                if local_slot >= 0:
                    log2phy[rank, expert_id] = rank * local_slots + local_slot
                else:
                    slots = expert_slots.get(expert_id, [0])
                    log2phy[rank, expert_id] = slots[
                        (rank + expert_id) % len(slots)]
        return log2phy

    @staticmethod
    def _load_values(global_load: torch.Tensor | None,
                     num_experts: int) -> torch.Tensor:
        if global_load is None:
            return torch.zeros(num_experts, dtype=torch.float32)
        return global_load.detach().cpu()[:num_experts].to(torch.float32)

    @staticmethod
    def _local_experts(layer) -> list[tuple[int, int]]:
        if layer.full_expert_map is None:
            return [
                (expert_id, expert_id)
                for expert_id in range(int(layer.global_num_experts))
            ]

        expert_map = layer.full_expert_map.detach().cpu()
        return [(int(local_id), int(global_id))
                for global_id, local_id in enumerate(expert_map.tolist())
                if local_id >= 0]


def redundant_experts_from_map(expert_map: torch.Tensor | None,
                               local_count: int) -> list[int]:
    if expert_map is None:
        return []
    return [
        int(global_id)
        for global_id, local_id in enumerate(expert_map.detach().cpu().tolist())
        if int(local_id) >= local_count
    ]


def ranked_experts_by_load(
    global_load: torch.Tensor | list[int] | None,
    expert_ids: list[int],
) -> list[int]:
    if global_load is None:
        return list(expert_ids)

    if isinstance(global_load, torch.Tensor):
        load_values = global_load.detach().cpu().tolist()
    else:
        load_values = global_load

    return sorted(
        expert_ids,
        key=lambda expert_id: (
            -int(load_values[expert_id])
            if 0 <= expert_id < len(load_values) else 0,
            expert_id,
        ),
    )
