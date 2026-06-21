import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class HistoryLayerPlan:
    expert_maps: list[list[int]]
    global_load: list[int]


class HistoryExpertMapCoordinator:
    """Build and share history-based expert maps from rank 0."""

    def __init__(self, load_stats_path: str | None):
        self.load_stats_path = load_stats_path
        self._plans: dict[int, HistoryLayerPlan | None] = {}

    def expert_map_for_layer(self, layer) -> torch.Tensor | None:
        plan = self._plan_for_layer(layer)
        if plan is None:
            return None

        rank = int(layer.ep_rank)
        if rank >= len(plan.expert_maps):
            return None
        return torch.tensor(plan.expert_maps[rank], dtype=torch.int32)

    def global_load_for_layer(self, layer) -> torch.Tensor | None:
        plan = self._plan_for_layer(layer)
        if plan is None:
            return None
        return torch.tensor(plan.global_load, dtype=torch.long)

    def _plan_for_layer(self, layer) -> HistoryLayerPlan | None:
        layer_id = int(layer.moe_instance_id)
        if layer_id not in self._plans:
            self._plans[layer_id] = self._broadcast_plan(layer)
        return self._plans[layer_id]

    def _broadcast_plan(self, layer) -> HistoryLayerPlan | None:
        plan = self._build_plan(layer) if self._is_rank0() else None
        if not dist.is_available() or not dist.is_initialized():
            return plan

        payload: list[HistoryLayerPlan | None] = [plan]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def _build_plan(self, layer) -> HistoryLayerPlan | None:
        if not self.load_stats_path:
            return None

        global_load = self._read_global_load(
            layer_id=int(layer.moe_instance_id),
            num_experts=int(layer.global_num_experts),
            num_ranks=int(layer.ep_size),
        )
        if global_load is None:
            return None

        expert_maps = self._balance_experts(
            global_load=global_load,
            num_ranks=int(layer.ep_size),
        )
        return HistoryLayerPlan(expert_maps, global_load)

    def _read_global_load(self, layer_id: int, num_experts: int,
                          num_ranks: int) -> list[int] | None:
        total = [0] * num_experts
        found = False
        for rank in range(num_ranks):
            counts = self._read_rank_load(rank, layer_id)
            if counts is None or len(counts) != num_experts:
                continue
            found = True
            total = [left + int(right) for left, right in zip(total, counts)]
        return total if found else None

    def _read_rank_load(self, rank: int, layer_id: int) -> list[int] | None:
        path = self._rank_file_path(rank)
        if path is None or not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as file:
            payload: dict[str, Any] = json.load(file)
        counts = payload.get("layers", {}).get(str(layer_id))
        return counts if isinstance(counts, list) else None

    def _rank_file_path(self, rank: int) -> Path | None:
        if not self.load_stats_path:
            return None

        directory = Path(self.load_stats_path)
        return directory / f"{directory.name}_rank{rank}.json"

    @staticmethod
    def _balance_experts(global_load: list[int],
                         num_ranks: int) -> list[list[int]]:
        num_experts = len(global_load)
        base_capacity = num_experts // num_ranks
        extra = num_experts % num_ranks
        capacities = [
            base_capacity + (1 if rank < extra else 0)
            for rank in range(num_ranks)
        ]

        rank_loads = [0] * num_ranks
        rank_experts: list[list[int]] = [[] for _ in range(num_ranks)]
        ranked_experts = sorted(
            range(num_experts),
            key=lambda expert_id: (-int(global_load[expert_id]), expert_id),
        )

        for expert_id in ranked_experts:
            rank = min(
                (rank for rank in range(num_ranks)
                 if len(rank_experts[rank]) < capacities[rank]),
                key=lambda rank: (rank_loads[rank], len(rank_experts[rank]),
                                  rank),
            )
            rank_experts[rank].append(expert_id)
            rank_loads[rank] += int(global_load[expert_id])

        expert_maps: list[list[int]] = []
        for experts in rank_experts:
            expert_map = [-1] * num_experts
            for local_id, global_id in enumerate(experts):
                expert_map[global_id] = local_id
            expert_maps.append(expert_map)
        return expert_maps

    @staticmethod
    def _is_rank0() -> bool:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True
