import torch
import torch.nn as nn
from ..runtime_config import RuntimeConfig
from .memory_manager import (ColdExperts, CombinedExperts, ExpertMemoryManager,
                             PreparedCombinedExperts)


class ExoExecutor(nn.Module):
    """Execute all CPU-to-NPU expert transfers for runtime modes."""

    def __init__(
        self,
        config: RuntimeConfig,
        uses_w8a8: bool,
        memory_manager: ExpertMemoryManager | None,
    ):
        super().__init__()
        self.config = config
        self.uses_w8a8 = uses_w8a8
        self.memory_manager = memory_manager
        self.layers: dict[int, nn.Module] = {}
        self.expert_maps: dict[int, list[int]] = {}
        self.full_expert_maps: dict[int, torch.Tensor] = {}
        self.slot_to_global_maps: dict[int, torch.Tensor] = {}
        self.cold_experts = nn.ModuleList()
        self.buffer_idx = 0
        self.buffer_status: list[int | None] = []
        self.prefetch_stream: torch.npu.Stream | None = None

    def should_manage_layer(self, layer) -> bool:
        return layer.moe_instance_id in self.config.runtime_layer_ids

    def should_store_cpu_expert(self, layer, global_expert_id: int) -> bool:
        if not self.should_manage_layer(layer) or global_expert_id < 0:
            return False
        if not self.config.uses_cold_buffer:
            return False
        layer_id = layer.moe_instance_id
        return (layer_id in self.expert_maps
                and global_expert_id in self._cold_experts(layer_id))

    def register_layer(self, layer, expert_map: list[int]) -> None:
        layer_id = layer.moe_instance_id
        if self.config.uses_cold_buffer:
            cold_count = self.config.offload_count
            if cold_count > len(expert_map):
                raise ValueError(
                    f"Cannot offload {cold_count} experts from "
                    f"{len(expert_map)} slots.")

        self.layers[layer_id] = layer
        self.expert_maps[layer_id] = list(expert_map)
        if self.config.uses_cold_buffer:
            assert self.memory_manager is not None
            self.memory_manager.register_layer(
                layer_id, layer,
                self._cold_experts(layer_id))

    def load_weight(self, layer, param_name: str, shard_id: str,
                    global_expert_id: int,
                    loaded_weight: torch.Tensor) -> bool:
        if not self.should_store_cpu_expert(layer, global_expert_id):
            return False

        assert self.memory_manager is not None
        return self.memory_manager.load_weight_shard(
            layer_id=layer.moe_instance_id,
            param_name=param_name,
            global_expert_id=global_expert_id,
            shard_id=shard_id,
            loaded_weight=loaded_weight,
            tp_rank=layer.tp_rank,
        )

    def process_layer_after_loading(self, layer, quant_method) -> None:
        if self.memory_manager is None:
            return

        layer_id = layer.moe_instance_id
        if layer_id not in self.memory_manager.layers:
            return

        self.memory_manager.process_layer_after_loading(layer_id, quant_method)
        if self.config.uses_cold_buffer:
            self._init_cold_buffers()
        torch.npu.synchronize()
        torch.npu.empty_cache()

    def prepare_combined_experts(self, layer,
                                 device: torch.device) -> PreparedCombinedExperts:
        cold_expert = self._prepare_cold_buffer(layer)
        layer_id = layer.moe_instance_id
        expert_ids = self.expert_maps[layer_id]
        return PreparedCombinedExperts(
            CombinedExperts(layer, cold_expert, len(expert_ids)),
            self._full_expert_map(layer, device),
            self._slot_to_global(layer),
        )

    def prefetch_next_layers(self, layer) -> None:
        if self.config.num_buffers <= 1 or len(self.layers) <= 1:
            return
        layer_ids = sorted(self.layers)
        layer_index = layer_ids.index(layer.moe_instance_id)
        next_layer_id = layer_ids[(layer_index + 1) % len(layer_ids)]
        if next_layer_id == self.buffer_status[self.buffer_idx]:
            return
        self._start_prefetch(self.layers[next_layer_id], self.buffer_idx)

    def layout(self, layer) -> tuple[list[int], int]:
        expert_map = self.expert_maps[layer.moe_instance_id]
        return list(expert_map), self._hot_count(len(expert_map))

    def update_layout(self, layer, expert_ids: list[int]) -> None:
        layer_id = layer.moe_instance_id
        cold_experts = self._cold_experts(layer_id)
        self.expert_maps[layer_id] = [*expert_ids, *cold_experts]
        self.full_expert_maps.pop(layer_id, None)
        self.slot_to_global_maps.pop(layer_id, None)

    def _hot_count(self, num_slots: int) -> int:
        return (max(0, num_slots - self.config.offload_count)
                if self.config.uses_cold_buffer else num_slots)

    def _hot_experts(self, layer_id: int) -> list[int]:
        expert_map = self.expert_maps[layer_id]
        return expert_map[:self._hot_count(len(expert_map))]

    def _cold_experts(self, layer_id: int) -> list[int]:
        expert_map = self.expert_maps[layer_id]
        return expert_map[self._hot_count(len(expert_map)):]

    def _full_expert_map(self, layer, device: torch.device) -> torch.Tensor:
        layer_id = layer.moe_instance_id
        expert_map = self.full_expert_maps.get(layer_id)
        if expert_map is None:
            expert_map = self._build_routing_map(layer,
                                                 self.expert_maps[layer_id],
                                                 device)
            self.full_expert_maps[layer_id] = expert_map
        if expert_map.device != device:
            raise RuntimeError(
                f"Layer {layer_id} is bound to {expert_map.device}, "
                f"but received {device}.")
        return expert_map

    def _slot_to_global(self, layer) -> torch.Tensor:
        layer_id = layer.moe_instance_id
        slot_to_global = self.slot_to_global_maps.get(layer_id)
        if slot_to_global is None:
            slot_to_global = torch.tensor(
                self.expert_maps[layer_id], dtype=torch.long)
            self.slot_to_global_maps[layer_id] = slot_to_global
        return slot_to_global

    @staticmethod
    def _build_routing_map(layer, expert_ids: list[int],
                           device: torch.device) -> torch.Tensor:
        expert_map = torch.full((int(layer.global_num_experts), ),
                                -1,
                                dtype=torch.long)
        for slot, expert_id in enumerate(expert_ids):
            expert_map[int(expert_id)] = slot
        return expert_map.to(device, non_blocking=True)

    def _init_cold_buffers(self) -> None:
        if self.cold_experts:
            return
        layer_ids = sorted(self.layers)
        template_id = layer_ids[0]
        template_weights = self.memory_manager.get_expert_weights(template_id)
        max_cold_experts = max(
            len(self._cold_experts(layer_id)) for layer_id in layer_ids)

        self.buffer_status = [None] * self.config.num_buffers
        for _ in range(self.config.num_buffers):
            self.cold_experts.append(
                ColdExperts(
                    templates=template_weights,
                    num_experts=max_cold_experts,
                    device=_layer_device(self.layers[template_id]),
                    use_w8a8=self.uses_w8a8,
                ))

    def _prepare_cold_buffer(self, layer) -> ColdExperts:
        current_layer = layer.moe_instance_id
        if current_layer == self.buffer_status[self.buffer_idx]:
            cold_expert = self.cold_experts[self.buffer_idx]
        else:
            cold_expert = self._start_prefetch(layer, self.buffer_idx)
        self.buffer_idx = (self.buffer_idx + 1) % self.config.num_buffers
        return cold_expert

    def _start_prefetch(self, layer, buffer_id: int) -> ColdExperts:
        layer_id = layer.moe_instance_id
        cold_expert = self.cold_experts[buffer_id]
        weights = self.memory_manager.get_expert_weights(
            layer_id,
        )

        stream = self._prefetch_stream()
        with torch.npu.stream(stream):
            cold_expert.load_from_cpu(weights)
        cold_expert.load_stream = stream
        self.buffer_status[buffer_id] = layer_id
        return cold_expert

    def _prefetch_stream(self) -> torch.npu.Stream:
        if self.prefetch_stream is None:
            self.prefetch_stream = torch.npu.Stream()
        return self.prefetch_stream

def _layer_device(layer) -> torch.device:
    for parameter in layer.parameters():
        return parameter.device
    return torch.device(f"npu:{torch.npu.current_device()}")
