import torch
import torch.nn as nn
from ..layer.routing import map_expert_ids
from ..runtime_config import RuntimeConfig
from .memory_manager import ColdExperts, ExpertMemoryManager, PreparedColdExperts


class ExoExecutor(nn.Module):
    """Execute all CPU-to-NPU expert transfers for runtime modes."""

    def __init__(
        self,
        config: RuntimeConfig,
        uses_w8a8: bool,
        memory_manager: ExpertMemoryManager,
    ):
        super().__init__()
        self.config = config
        self.uses_w8a8 = uses_w8a8
        self.memory_manager = memory_manager
        self.layers: dict[int, nn.Module] = {}
        self.expert_maps: dict[int, list[int]] = {}
        self.routing_maps: dict[
            int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.cold_experts = nn.ModuleList()
        self.buffer_idx = 0
        self.buffer_status: list[int | None] = []
        self.prefetch_stream: torch.npu.Stream | None = None

    def should_manage_layer(self, layer) -> bool:
        return layer.moe_instance_id in self.config.runtime_layer_ids

    @property
    def collect_load(self) -> bool:
        return (self.config.runtime_mode == "balance"
                or bool(self.config.load_history_path))

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
            self.memory_manager.register_layer(
                layer_id, layer,
                self._cold_experts(layer_id))

    def load_weight(self, layer, param_name: str, shard_id: str,
                    global_expert_id: int,
                    loaded_weight: torch.Tensor) -> bool:
        if not self.should_store_cpu_expert(layer, global_expert_id):
            return False

        return self.memory_manager.load_weight_shard(
            layer_id=layer.moe_instance_id,
            param_name=param_name,
            global_expert_id=global_expert_id,
            shard_id=shard_id,
            loaded_weight=loaded_weight,
            tp_rank=layer.tp_rank,
        )

    def process_layer_after_loading(self, layer, quant_method) -> None:
        layer_id = layer.moe_instance_id
        if layer_id not in self.memory_manager.layers:
            return

        self.memory_manager.process_layer_after_loading(layer_id, quant_method)
        if self.config.uses_cold_buffer:
            self._init_cold_buffers()
        torch.npu.synchronize()
        torch.npu.empty_cache()

    def prepare_cold_experts(self, layer,
                             topk_ids: torch.Tensor) -> PreparedColdExperts:
        current_layer = layer.moe_instance_id
        if current_layer == self.buffer_status[self.buffer_idx]:
            cold_expert = self.cold_experts[self.buffer_idx]
        else:
            cold_expert = self._start_prefetch(layer, self.buffer_idx)
        self.buffer_idx = (self.buffer_idx + 1) % self.config.num_buffers

        _, cold_map = self._routing_maps(layer, topk_ids.device)
        cold_topk_ids, cold_mask = map_expert_ids(topk_ids, cold_map)
        cold_topk_ids = cold_topk_ids.masked_fill(~cold_mask, 0).to(
            topk_ids.dtype)
        return PreparedColdExperts(cold_expert, cold_topk_ids, cold_mask)

    def hot_routing(self, layer, topk_ids: torch.Tensor,
                    cold_mask: torch.Tensor) -> tuple[torch.Tensor,
                                                       torch.Tensor]:
        hot_map, _ = self._routing_maps(layer, topk_ids.device)
        hot_ids, hot_mask = map_expert_ids(topk_ids, hot_map)
        hot_mask &= ~cold_mask
        return hot_ids.masked_fill(~hot_mask, 0).to(topk_ids.dtype), hot_mask

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
        self.routing_maps.pop(layer_id, None)

    def _hot_count(self, num_slots: int) -> int:
        return (max(0, num_slots - self.config.offload_count)
                if self.config.uses_cold_buffer else num_slots)

    def _hot_experts(self, layer_id: int) -> list[int]:
        expert_map = self.expert_maps[layer_id]
        return expert_map[:self._hot_count(len(expert_map))]

    def _cold_experts(self, layer_id: int) -> list[int]:
        expert_map = self.expert_maps[layer_id]
        return expert_map[self._hot_count(len(expert_map)):]

    def _routing_maps(
        self,
        layer,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_id = layer.moe_instance_id
        maps = self.routing_maps.get(layer_id)
        if maps is None:
            maps = (
                self._build_routing_map(
                    layer, self._hot_experts(layer_id), device),
                self._build_routing_map(
                    layer, self._cold_experts(layer_id), device),
            )
            self.routing_maps[layer_id] = maps
        if maps[0].device != device:
            raise RuntimeError(
                f"Layer {layer_id} is bound to {maps[0].device}, "
                f"but received {device}.")
        return maps

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
