from dataclasses import dataclass

import torch
import torch.distributed as dist
from vllm_ascend.utils import npu_stream_switch

from ..moeload.history_mapping import HistoryExpertMap, ranked_experts_by_load
from ..moeload.profiler import ExpertLoadProfiler
from ..runtime_config import RuntimeConfig
from .memory_manager import ExpertMemoryManager


@dataclass
class PendingExpertUpdate:
    old_expert: int
    new_expert: int
    target_slot: int
    slot_index: int
    event: object | None


class LBVCAdaptor:
    """Load-balance-via-CPU adaptor."""

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
        self.redundant_slots: dict[int, list[int]] = {}
        self.peer_expert_ids: dict[int, list[int]] = {}
        self.layers: dict[int, object] = {}
        self.pending_updates: dict[int, list[PendingExpertUpdate]] = {}
        self._copy_stream = None

    def register_layer(self, layer) -> None:
        layer_id = int(layer.moe_instance_id)
        local_count = int(layer.logical_num_experts) // int(layer.ep_size)
        redundant_count = int(self.config.num_redundant_experts)
        peer_experts = self._peer_experts(layer)
        self.layers[layer_id] = layer
        self.redundant_slots[layer_id] = list(
            range(local_count, local_count + redundant_count))
        self.peer_expert_ids[layer_id] = self._redundant_experts_from_map(
            layer.full_expert_map, local_count)
        self.pending_updates[layer_id] = []
        self.memory_manager.register_layer(layer_id, layer, peer_experts)
        self.profiler.register_layer(layer)

    def load_weight(self, layer, param_name: str, shard_id: str,
                    global_expert_id: int,
                    loaded_weight: torch.Tensor) -> bool:
        layer_id = int(layer.moe_instance_id)
        weights = self.memory_manager.layers.get(layer_id)
        if weights is None or global_expert_id not in weights.expert_id_to_slot:
            return False

        return weights.load_shard(param_name, int(global_expert_id), shard_id,
                                  loaded_weight, int(layer.tp_rank))

    def process_layer_after_loading(self, layer, quant_method) -> None:
        self.memory_manager.process_layer_after_loading(
            int(layer.moe_instance_id), quant_method)
        self._load_initial_redundant_experts(layer)
        torch.npu.empty_cache()

    def record_expert_tokens(self, layer, group_list_type: int,
                             expert_tokens: torch.Tensor) -> None:
        self.apply_ready_updates(layer)
        self.profiler.record_expert_tokens(
            int(layer.moe_instance_id),
            expert_tokens,
            group_list_type,
        )
        if self._should_update(layer):
            self._update_redundant_experts(layer)

    def save_load_history(self) -> None:
        self.profiler.save()

    def before_forward(self, layer) -> None:
        self.apply_ready_updates(layer)

    def global_load_for_layer(self, layer) -> torch.Tensor | None:
        if self.history_mapping is None:
            return None
        return self.history_mapping.global_load_for_layer(layer)

    def _should_update(self, layer) -> bool:
        layer_id = int(layer.moe_instance_id)
        step = self._steps.get(layer_id, 0) + 1
        self._steps[layer_id] = step
        return step % self.config.scheduler_interval == 0

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

        all_maps = self._build_base_maps(ep_size, num_experts,
                                         base_experts_per_rank,
                                         global_num_experts)
        if redundant_per_rank > 0:
            self._place_initial_redundant_experts(
                all_maps=all_maps,
                num_experts=num_experts,
                base_experts_per_rank=base_experts_per_rank,
                redundant_per_rank=redundant_per_rank,
                global_load=global_load,
            )

        log2phy = self._build_log2phy(all_maps, num_experts, local_slots)

        return local_slots, all_maps[ep_rank], log2phy[ep_rank]

    @staticmethod
    def _build_base_maps(ep_size: int, num_experts: int,
                         base_experts_per_rank: int,
                         global_num_experts: int) -> torch.Tensor:
        all_maps = torch.full((ep_size, global_num_experts),
                              -1,
                              dtype=torch.int32)
        for rank in range(ep_size):
            start = rank * base_experts_per_rank
            end = start + base_experts_per_rank
            all_maps[rank, start:end] = torch.arange(
                base_experts_per_rank, dtype=torch.int32)
        return all_maps

    def _place_initial_redundant_experts(
        self,
        all_maps: torch.Tensor,
        num_experts: int,
        base_experts_per_rank: int,
        redundant_per_rank: int,
        global_load: torch.Tensor | None,
    ) -> None:
        replica_counts = torch.ones(num_experts, dtype=torch.long)
        load_values = self._load_values(global_load, num_experts)
        has_history = global_load is not None

        for rank in range(all_maps.size(0)):
            peer_rank = self._peer_rank(rank, int(all_maps.size(0)))
            if peer_rank is None:
                continue
            peer_start = peer_rank * base_experts_per_rank
            peer_end = peer_start + base_experts_per_rank
            for offset in range(redundant_per_rank):
                candidates = [
                    expert_id for expert_id in range(peer_start, peer_end)
                    if int(all_maps[rank, expert_id].item()) < 0
                ]
                if not candidates:
                    continue
                seed = rank * redundant_per_rank + offset
                expert_id = self._select_initial_redundant_expert(
                    candidates, replica_counts, load_values, seed,
                    has_history)
                all_maps[rank, expert_id] = base_experts_per_rank + offset
                replica_counts[expert_id] += 1

    @staticmethod
    def _select_initial_redundant_expert(
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

    def _update_redundant_experts(self, layer) -> None:
        self.apply_ready_updates(layer)
        layer_id = int(layer.moe_instance_id)
        redundant_slots = self.redundant_slots.get(layer_id, [])
        peer_experts = self._peer_experts(layer)
        if not peer_experts:
            return
        if not redundant_slots:
            return

        peer_counts = self._global_expert_counts(layer)
        if peer_counts is None or not self._pair_is_imbalanced(
                layer, peer_counts):
            return

        candidates = self._ranked_new_peer_experts(
            peer_experts, self.peer_expert_ids[layer_id], peer_counts)
        if not candidates:
            return

        pending_slot_indexes = {
            update.slot_index
            for update in self.pending_updates.get(layer_id, [])
        }
        max_updates = min(int(self.config.num_experts_per_update),
                          len(redundant_slots), len(candidates))
        used_slots: set[int] = set()
        for candidate in candidates[:max_updates]:
            slot_index = self._least_loaded_redundant_slot(
                layer, peer_counts, used_slots | pending_slot_indexes)
            if slot_index < 0:
                return
            used_slots.add(slot_index)
            target_slot = redundant_slots[slot_index]
            old_expert = self.peer_expert_ids[layer_id][slot_index]

            event = self._copy_expert_to_slot(layer_id, candidate, layer,
                                              target_slot)
            self.pending_updates[layer_id].append(
                PendingExpertUpdate(
                    old_expert=old_expert,
                    new_expert=candidate,
                    target_slot=target_slot,
                    slot_index=slot_index,
                    event=event,
                ))

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

    def apply_ready_updates(self, layer) -> None:
        layer_id = int(layer.moe_instance_id)
        pending = self.pending_updates.get(layer_id, [])
        if not pending:
            return

        ready: list[PendingExpertUpdate] = []
        waiting: list[PendingExpertUpdate] = []
        for update in pending:
            if update.event is None or update.event.query():
                ready.append(update)
            else:
                waiting.append(update)

        if not ready:
            return

        for update in ready:
            self.peer_expert_ids[layer_id][update.slot_index] = (
                update.new_expert)
            self._set_expert_map(layer, update.old_expert, -1)
            self._set_log2phy_to_owner(layer, update.old_expert)
            self._set_expert_map(layer, update.new_expert,
                                 update.target_slot)
            self._set_log2phy_to_local_slot(layer, update.new_expert,
                                            update.target_slot)
        self.pending_updates[layer_id] = waiting
        self.profiler.update_layer_map(layer)

    def _copy_expert_to_slot(self, layer_id: int, expert_id: int, layer,
                             target_slot: int):
        if self._copy_stream is None:
            self._copy_stream = torch.npu.Stream()

        current_stream = torch.npu.current_stream()
        self._copy_stream.wait_stream(current_stream)
        with npu_stream_switch(self._copy_stream):
            self.memory_manager.copy_experts_to_module(
                layer_id, [expert_id], layer, [target_slot])
            event = torch.npu.Event()
            event.record(self._copy_stream)
        return event

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

    def _global_expert_counts(self, layer) -> torch.Tensor | None:
        counts = self.profiler.get_layer_load(int(layer.moe_instance_id))
        if counts.numel() < int(layer.logical_num_experts):
            return None

        if not dist.is_available() or not dist.is_initialized():
            return counts[:int(layer.logical_num_experts)]

        device_counts = counts.to(device=self._layer_device(layer),
                                  dtype=torch.long)
        dist.all_reduce(device_counts,
                        group=layer.moe_config.ep_group.device_group)
        return device_counts.cpu()[:int(layer.logical_num_experts)]

    @staticmethod
    def _layer_device(layer) -> torch.device:
        weight = getattr(layer, "w13_weight", None)
        if weight is not None:
            return weight.device

        weight_list = getattr(layer, "w13_weight_list", None)
        if weight_list:
            return weight_list[0].device

        return next(layer.parameters()).device

    def _pair_is_imbalanced(self, layer, counts: torch.Tensor) -> bool:
        peer_rank = self._peer_rank(int(layer.ep_rank), int(layer.ep_size))
        if peer_rank is None:
            return False

        experts_per_rank = int(layer.logical_num_experts) // int(layer.ep_size)
        local_load = self._rank_load(counts, int(layer.ep_rank),
                                     experts_per_rank)
        peer_load = self._rank_load(counts, peer_rank, experts_per_rank)
        heavier_load = max(local_load, peer_load, 1)
        imbalance = abs(local_load - peer_load) / heavier_load
        return imbalance >= float(self.config.imbalance_threshold)

    @staticmethod
    def _rank_load(counts: torch.Tensor, rank: int,
                   experts_per_rank: int) -> int:
        start = rank * experts_per_rank
        end = start + experts_per_rank
        return int(counts[start:end].sum().item())

    @staticmethod
    def _ranked_new_peer_experts(peer_experts: list[int],
                                 current_experts: list[int],
                                 counts: torch.Tensor) -> list[int]:
        current = set(current_experts)
        candidates = [expert for expert in peer_experts if expert not in current]
        return ranked_experts_by_load(counts, candidates)

    def _least_loaded_redundant_slot(self, layer, counts: torch.Tensor,
                                     used_slots: set[int]) -> int:
        current = self.peer_expert_ids[int(layer.moe_instance_id)]
        available = [
            index for index in range(len(current))
            if index not in used_slots
        ]
        if not available:
            return -1
        loads = [int(counts[expert_id].item()) for expert_id in current]
        return min(available, key=loads.__getitem__)

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
