import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


LOAD_HISTORY_VERSION = 2


class ExpertLoadProfiler:
    """Track per-layer expert hits using global expert ids."""

    def __init__(self,
                 path: str | None = None,
                 metadata: dict[str, Any] | None = None):
        self.path = path
        self.rank = self._get_rank()
        
        self._layers: dict[int, int] = {}
        self._counts: dict[int, torch.Tensor] = {}
        self._last_load_snapshot: dict[int, torch.Tensor] = {}
        self._local_to_global: dict[int, torch.Tensor] = {}
        self._local_map_values: dict[int, tuple[int, ...]] = {}
        self._device_slot_maps: dict[
            tuple[int, str, tuple[int, ...]], torch.Tensor] = {}
        self._metadata = dict(metadata or {})
        self._metadata["rank"] = self.rank

    @property
    def output_path(self) -> str | None:
        path = self._rank_file_path(self.path)
        return None if path is None else str(path)
    
    def _get_rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    def register_layer(self, layer,
                       slot_to_global: list[int] | torch.Tensor | None = None
                       ) -> None:
        layer_id = layer.moe_instance_id
        num_experts = layer.global_num_experts
        self._layers[layer_id] = num_experts
        self.update_layer_map(layer, slot_to_global)
        self._counts.setdefault(
            layer_id,
            torch.zeros(num_experts, dtype=torch.long, device="cpu"),
        )
        self._last_load_snapshot.setdefault(
            layer_id,
            torch.zeros(num_experts, dtype=torch.long, device="cpu"),
        )

    def update_layer_map(
        self,
        layer,
        slot_to_global: list[int] | torch.Tensor | None = None,
    ) -> None:
        layer_id = layer.moe_instance_id
        if slot_to_global is None:
            slot_to_global = self._slot_to_global_from_lookup(layer)
        cpu_slot_map = torch.as_tensor(
            slot_to_global, dtype=torch.long, device="cpu").clone()
        self._local_to_global[layer_id] = cpu_slot_map
        self._local_map_values[layer_id] = tuple(
            int(value) for value in cpu_slot_map.tolist())

    def record_expert_tokens(
        self,
        layer_id: int,
        expert_tokens: torch.Tensor,
        group_list_type: int,
    ) -> None:
        self.record_slot_tokens(
            layer_id,
            expert_tokens,
            group_list_type,
            self._local_to_global[layer_id],
            self._local_map_values[layer_id],
        )

    def record_slot_tokens(
        self,
        layer_id: int,
        expert_tokens: torch.Tensor,
        group_list_type: int,
        slot_to_global: torch.Tensor,
        map_values: tuple[int, ...] | None = None,
    ) -> None:
        if expert_tokens.numel() == 0:
            return

        num_experts = self._layers[layer_id]
        local_counts = self._to_counts(
            expert_tokens, group_list_type).to(dtype=torch.long)
        device = local_counts.device
        counts = self._counts[layer_id]
        if counts.device != device:
            # Keep counters next to the fused MoE output. Moving the small
            # counter to CPU for every layer forces the NPU stream to
            # synchronize and destroys CPU/NPU transfer overlap.
            counts = counts.to(device=device, non_blocking=True)
            self._counts[layer_id] = counts

        cpu_slot_map = None
        if map_values is None:
            local_map = self._local_to_global.get(layer_id)
            if slot_to_global is local_map:
                map_values = self._local_map_values[layer_id]
            else:
                cpu_slot_map = slot_to_global.to(device="cpu",
                                                 dtype=torch.long)
                map_values = tuple(
                    int(value) for value in cpu_slot_map.tolist())
        map_key = (layer_id, str(device), map_values)
        device_slot_map = self._device_slot_maps.get(map_key)
        if device_slot_map is None:
            if cpu_slot_map is None:
                cpu_slot_map = torch.as_tensor(
                    map_values, dtype=torch.long, device="cpu")
            device_slot_map = cpu_slot_map.to(device=device,
                                              non_blocking=True)
            self._device_slot_maps[map_key] = device_slot_map

        size = min(local_counts.numel(), slot_to_global.numel())
        global_ids = device_slot_map[:size]
        valid = (global_ids >= 0) & (global_ids < num_experts)
        counts.index_add_(0, global_ids[valid], local_counts[:size][valid])

    def get_layer_load(self, layer_id: int) -> torch.Tensor:
        return self._counts[layer_id].detach().cpu()

    def get_layer_delta_load(self, layer_id: int) -> torch.Tensor:
        current = self._counts[layer_id].detach().cpu()
        previous = self._last_load_snapshot[layer_id]
        delta = current - previous
        self._last_load_snapshot[layer_id] = current.clone()
        return delta

    def save(self) -> None:
        target_path = self._rank_file_path(self.path)
        if target_path is None:
            return

        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": LOAD_HISTORY_VERSION,
            "metadata": self._metadata,
            "layers": {
                str(layer_id): self._layer_record(layer_id, counts)
                for layer_id, counts in sorted(self._counts.items())
            },
        }
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, target_path)

    def _layer_record(
        self,
        layer_id: int,
        counts: torch.Tensor,
    ) -> dict[str, Any]:
        return {
            "num_experts": self._layers[layer_id],
            "experts": [
                {
                    "expert_id": expert_id,
                    "activated_tokens": int(tokens),
                }
                for expert_id, tokens in enumerate(counts.tolist())
                if int(tokens) > 0
            ],
        }

    @staticmethod
    def _to_counts(expert_tokens: torch.Tensor,
                   group_list_type: int) -> torch.Tensor:
        if group_list_type == 1:
            return expert_tokens.detach()
        else:
            return torch.cat([
                expert_tokens[:1],
                expert_tokens[1:] - expert_tokens[:-1],
            ]).detach()

    @staticmethod
    def _slot_to_global_from_lookup(layer) -> torch.Tensor:
        lookup = layer._expert_map.detach().to(device="cpu",
                                               dtype=torch.long)
        local_num_experts = int(torch.sum(lookup >= 0).item())
        local_to_global = torch.full((local_num_experts, ),
                                     -1,
                                     dtype=torch.long,
                                     device="cpu")
        for global_id, local_id in enumerate(lookup.tolist()):
            if local_id >= 0:
                local_to_global[local_id] = global_id
        return local_to_global

    def _rank_file_path(self, path: str | None) -> Path | None:
        if path is None:
            return None
        directory = Path(path)
        rank_name = f"{directory.name}_rank{self.rank}"
        return directory / f"{rank_name}.json"
