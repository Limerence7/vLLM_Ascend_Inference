from dataclasses import dataclass

import torch

from ..moeload.history_mapping import ranked_experts_by_load
from ..runtime_config import RuntimeConfig


@dataclass(frozen=True)
class TokenSplitPlan:
    enabled: bool = False
    expert_map: torch.Tensor | None = None


class LBVCAdaptor:
    """Load-balance-via-CPU adaptor.

    Initial implementation keeps the native local expert path and exposes the
    scheduling surface used by the runtime layer. Redundant expert replacement
    can be added here without touching offload execution.
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._steps: dict[int, int] = {}
        self.local_slots: dict[int, list[int]] = {}
        self.redundant_slots: dict[int, list[int]] = {}
        self.peer_expert_ids: dict[int, list[int]] = {}

    def register_layer(self, layer) -> None:
        layer_id = int(layer.moe_instance_id)
        local_count = int(layer.logical_num_experts) // int(layer.ep_size)
        redundant_count = int(self.config.num_redundant_experts)
        self.local_slots[layer_id] = list(range(local_count))
        self.redundant_slots[layer_id] = list(
            range(local_count, local_count + redundant_count))
        self.peer_expert_ids[layer_id] = self._redundant_experts_from_map(
            layer.full_expert_map, local_count)

    def maybe_update(self, layer) -> bool:
        layer_id = int(layer.moe_instance_id)
        step = self._steps.get(layer_id, 0) + 1
        self._steps[layer_id] = step
        return step % self.config.scheduler_interval == 0

    def token_split_plan(self, layer, expert_tokens: torch.Tensor | None
                         ) -> TokenSplitPlan:
        return TokenSplitPlan(False)

    def initial_expert_maps(
        self,
        num_experts: int,
        ep_size: int,
        ep_rank: int,
        global_load: torch.Tensor | None = None,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        if num_experts % ep_size != 0:
            raise ValueError(
                "balance mode currently requires experts evenly split by EP.")

        base_experts_per_rank = num_experts // ep_size
        redundant_per_rank = int(self.config.num_redundant_experts)
        local_slots = base_experts_per_rank + redundant_per_rank
        global_redundant = ep_size * redundant_per_rank
        global_num_experts = num_experts + global_redundant

        all_maps = torch.full((ep_size, global_num_experts),
                              -1,
                              dtype=torch.int32)
        for rank in range(ep_size):
            start = rank * base_experts_per_rank
            end = start + base_experts_per_rank
            all_maps[rank, start:end] = torch.arange(
                base_experts_per_rank, dtype=torch.int32)

            peer_rank = self._peer_rank(rank, ep_size)
            if peer_rank is None or redundant_per_rank == 0:
                continue

            peer_start = peer_rank * base_experts_per_rank
            peer_end = peer_start + base_experts_per_rank
            peer_experts = list(range(peer_start, peer_end))
            redundant_experts = ranked_experts_by_load(
                global_load, peer_experts)[:redundant_per_rank]
            for offset, expert_id in enumerate(redundant_experts):
                all_maps[rank, expert_id] = base_experts_per_rank + offset

        log2phy = self._build_log2phy(all_maps, num_experts, local_slots)

        return local_slots, all_maps[ep_rank], log2phy[ep_rank]

    @staticmethod
    def _build_log2phy(all_maps: torch.Tensor, num_experts: int,
                       local_slots: int) -> torch.Tensor:
        ep_size, global_num_experts = all_maps.shape
        log2phy = torch.zeros((ep_size, global_num_experts), dtype=torch.int32)
        owner_slots: dict[int, int] = {}

        for rank in range(ep_size):
            for expert_id in range(num_experts):
                local_slot = int(all_maps[rank, expert_id].item())
                if local_slot >= 0 and expert_id not in owner_slots:
                    owner_slots[expert_id] = rank * local_slots + local_slot

        for rank in range(ep_size):
            for expert_id in range(num_experts):
                local_slot = int(all_maps[rank, expert_id].item())
                if local_slot >= 0:
                    log2phy[rank, expert_id] = rank * local_slots + local_slot
                else:
                    log2phy[rank, expert_id] = owner_slots[expert_id]
        return log2phy

    @staticmethod
    def _redundant_experts_from_map(expert_map: torch.Tensor | None,
                                    local_count: int) -> list[int]:
        if expert_map is None:
            return []
        return [
            int(global_id)
            for global_id, local_id in enumerate(expert_map.detach().cpu().tolist())
            if int(local_id) >= local_count
        ]

    def _peer_experts(self, layer) -> list[int]:
        peer_rank = self._peer_rank(int(layer.ep_rank), int(layer.ep_size))
        if peer_rank is None:
            return []

        global_num_experts = int(layer.global_num_experts)
        ep_size = int(layer.ep_size)
        experts_per_rank = global_num_experts // ep_size
        start = peer_rank * experts_per_rank
        end = start + experts_per_rank
        return list(range(start, min(end, global_num_experts)))

    def _peer_rank(self, rank: int, ep_size: int) -> int | None:
        if self.config.pair_topology:
            for left, right in self.config.pair_topology:
                if rank == left:
                    return right
                if rank == right:
                    return left
            return None

        peer_rank = rank + 1 if rank % 2 == 0 else rank - 1
        return peer_rank if 0 <= peer_rank < ep_size else None
