from dataclasses import dataclass

import torch
import torch.distributed as dist

from ..moeload.history_mapping import HistoryExpertMap, ranked_experts_by_load
from ..moeload.profiler import ExpertLoadProfiler
from ..runtime_config import RuntimeConfig
from .memory_manager import ExpertMemoryManager


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
        self.memory_manager = ExpertMemoryManager(config.cpu_pin_memory)
        self.profiler = ExpertLoadProfiler(
            config.load_history_path,
            metadata={
                "runtime_mode": config.runtime_mode,
                "runtime_layer_ids": list(config.runtime_layer_ids),
            },
        )
        self.history_mapping = (
            HistoryExpertMap(config.load_history_path)
            if config.enable_history_mapping else None)
        self._steps: dict[int, int] = {}
        self.local_slots: dict[int, list[int]] = {}
        self.redundant_slots: dict[int, list[int]] = {}
        self.peer_expert_ids: dict[int, list[int]] = {}
        self.layers: dict[int, object] = {}

    def register_layer(self, layer) -> None:
        layer_id = int(layer.moe_instance_id)
        local_count = int(layer.logical_num_experts) // int(layer.ep_size)
        redundant_count = int(self.config.num_redundant_experts)
        peer_experts = self._peer_experts(layer)
        self.layers[layer_id] = layer
        self.local_slots[layer_id] = list(range(local_count))
        self.redundant_slots[layer_id] = list(
            range(local_count, local_count + redundant_count))
        self.peer_expert_ids[layer_id] = self._redundant_experts_from_map(
            layer.full_expert_map, local_count)
        self.memory_manager.register_layer(layer_id, layer, peer_experts)
        self.profiler.register_layer(layer)

    def load_weight(self, layer, param_name: str, shard_id: str,
                    global_expert_id: int,
                    loaded_weight: torch.Tensor) -> bool:
        layer_id = int(layer.moe_instance_id)
        weights = self.memory_manager.layers.get(layer_id)
        if weights is None or global_expert_id not in weights.expert_id_to_slot:
            return False

        weights.load_shard(param_name, int(global_expert_id), shard_id,
                           loaded_weight, int(layer.tp_rank))
        return False

    def process_layer_after_loading(self, layer, quant_method) -> None:
        self.memory_manager.process_layer_after_loading(
            int(layer.moe_instance_id), quant_method)
        self._load_initial_redundant_experts(layer)
        torch.npu.empty_cache()

    def record_expert_tokens(self, layer, group_list_type: int,
                             expert_tokens: torch.Tensor) -> None:
        self.profiler.record_expert_tokens(
            int(layer.moe_instance_id),
            expert_tokens,
            group_list_type,
        )
        if self._should_update(layer):
            self._update_redundant_experts(layer, group_list_type,
                                           expert_tokens)

    def save_load_history(self) -> None:
        self.profiler.save()

    def global_load_for_layer(self, layer) -> torch.Tensor | None:
        if self.history_mapping is None:
            return None
        return self.history_mapping.global_load_for_layer(layer)

    def _should_update(self, layer) -> bool:
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

    def _update_redundant_experts(self, layer, group_list_type: int,
                                  expert_tokens: torch.Tensor) -> None:
        layer_id = int(layer.moe_instance_id)
        redundant_slots = self.redundant_slots.get(layer_id, [])
        peer_experts = self._peer_experts(layer)
        if not redundant_slots:
            return

        local_load = self._local_load(group_list_type, expert_tokens)
        pair_load = self._pair_load(layer, local_load)
        peer_counts = self._global_expert_counts(layer)
        if not peer_experts:
            return
        if pair_load is None:
            return

        candidates = self._ranked_new_peer_experts(
            peer_experts, self.peer_expert_ids[layer_id], peer_counts)
        if not candidates:
            return

        max_updates = min(int(self.config.num_experts_per_update),
                          len(redundant_slots), len(candidates))
        used_slots: set[int] = set()
        for candidate in candidates[:max_updates]:
            slot_index = self._least_loaded_redundant_slot(
                layer, peer_counts, used_slots)
            used_slots.add(slot_index)
            target_slot = redundant_slots[slot_index]
            old_expert = self.peer_expert_ids[layer_id][slot_index]

            self.memory_manager.copy_experts_to_module(
                layer_id,
                [candidate],
                layer,
                [target_slot],
            )
            self.peer_expert_ids[layer_id][slot_index] = candidate
            self._set_expert_map(layer, old_expert, -1)
            self._set_log2phy_to_owner(layer, old_expert)
            self._set_expert_map(layer, candidate, target_slot)
            self._set_log2phy_to_local_slot(layer, candidate, target_slot)
        self.profiler.update_layer_map(layer)

    def _load_initial_redundant_experts(self, layer) -> None:
        layer_id = int(layer.moe_instance_id)
        experts = self.peer_expert_ids.get(layer_id, [])
        slots = self.redundant_slots.get(layer_id, [])
        if experts and slots:
            self.memory_manager.copy_experts_to_module(
                layer_id,
                experts,
                layer,
                slots[:len(experts)],
            )

    def _set_expert_map(self, layer, expert_id: int, slot: int) -> None:
        if layer.full_expert_map is not None:
            layer.full_expert_map[int(expert_id)] = int(slot)
        if layer._expert_map is not None:
            layer._expert_map[int(expert_id)] = int(slot)
        for device_map in layer._resident_maps_by_device.values():
            device_map[int(expert_id)] = int(slot)

    def _set_log2phy_to_local_slot(self, layer, expert_id: int,
                                   slot: int) -> None:
        if layer.log2phy is None:
            return
        physical_slot = int(layer.ep_rank) * int(layer.full_local_num_experts)
        physical_slot += int(slot)
        layer.log2phy[int(expert_id)] = physical_slot

    def _set_log2phy_to_owner(self, layer, expert_id: int) -> None:
        if layer.log2phy is None:
            return
        experts_per_rank = int(layer.logical_num_experts) // int(layer.ep_size)
        owner_rank = int(expert_id) // experts_per_rank
        owner_slot = int(expert_id) - owner_rank * experts_per_rank
        physical_slot = owner_rank * int(layer.full_local_num_experts)
        physical_slot += owner_slot
        layer.log2phy[int(expert_id)] = physical_slot

    def _local_load(self, group_list_type: int,
                    expert_tokens: torch.Tensor) -> int:
        counts = self._to_counts(expert_tokens, group_list_type)
        return int(counts.sum().item())

    def _pair_load(self, layer, local_load: int) -> tuple[int, int] | None:
        peer_rank = self._peer_rank(int(layer.ep_rank), int(layer.ep_size))
        if not dist.is_available() or not dist.is_initialized():
            return None

        loads = torch.zeros(int(layer.ep_size),
                            dtype=torch.long,
                            device=layer.w13_weight.device)
        own = torch.tensor([local_load],
                           dtype=torch.long,
                           device=layer.w13_weight.device)
        dist.all_gather_into_tensor(
            loads, own, group=layer.moe_config.ep_group.device_group)
        if peer_rank is None:
            return None
        return int(loads[int(layer.ep_rank)].item()), int(loads[peer_rank].item())

    def _global_expert_counts(self, layer) -> torch.Tensor | None:
        counts = self.profiler.get_layer_load(int(layer.moe_instance_id))
        if counts.numel() < int(layer.logical_num_experts):
            return None

        if not dist.is_available() or not dist.is_initialized():
            return counts[:int(layer.logical_num_experts)]

        device_counts = counts.to(device=layer.w13_weight.device,
                                  dtype=torch.long)
        dist.all_reduce(device_counts,
                        group=layer.moe_config.ep_group.device_group)
        return device_counts.cpu()[:int(layer.logical_num_experts)]

    @staticmethod
    def _ranked_new_peer_experts(peer_experts: list[int],
                                 current_experts: list[int],
                                 counts: torch.Tensor | None) -> list[int]:
        current = set(current_experts)
        candidates = [expert for expert in peer_experts if expert not in current]
        return ranked_experts_by_load(counts, candidates)

    def _least_loaded_redundant_slot(self, layer, counts: torch.Tensor | None,
                                     used_slots: set[int]) -> int:
        current = self.peer_expert_ids[int(layer.moe_instance_id)]
        available = [
            index for index in range(len(current))
            if index not in used_slots
        ]
        if counts is None:
            return available[0]
        loads = [int(counts[expert_id].item()) for expert_id in current]
        return min(available, key=loads.__getitem__)

    @staticmethod
    def _to_counts(expert_tokens: torch.Tensor,
                   group_list_type: int) -> torch.Tensor:
        if group_list_type == 1:
            return expert_tokens.detach().to(device="cpu", dtype=torch.long)
        expert_tokens = expert_tokens.detach().to(device="cpu",
                                                  dtype=torch.long)
        return torch.cat([expert_tokens[:1],
                          expert_tokens[1:] - expert_tokens[:-1]])

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

        ep_size = int(layer.ep_size)
        experts_per_rank = int(layer.logical_num_experts) // ep_size
        start = peer_rank * experts_per_rank
        return list(range(start, start + experts_per_rank))

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
