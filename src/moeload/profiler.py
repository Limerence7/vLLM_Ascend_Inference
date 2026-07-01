import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


LOAD_HISTORY_VERSION = 1


class ExpertLoadProfiler:
    """Track per-layer expert hits using global expert ids."""

    def __init__(self,
                 path: str | None = None,
                 metadata: dict[str, Any] | None = None):
        self.path = path
        self.rank = self._get_rank()
        
        self._layers: dict[int, int] = {}
        self._counts: dict[int, torch.Tensor] = {}
        self._local_to_global: dict[int, torch.Tensor] = {}
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

    def register_layer(self, layer) -> None:
        layer_id = layer.moe_instance_id
        num_experts = layer.global_num_experts
        self._layers[layer_id] = num_experts
        self._local_to_global[layer_id] = self._build_local_to_global(layer)
        self._counts.setdefault(
            layer_id,
            torch.zeros(num_experts, dtype=torch.long, device="cpu"),
        )

    def update_layer_map(self, layer) -> None:
        layer_id = layer.moe_instance_id
        self._local_to_global[layer_id] = (
            self._build_local_to_global(layer))

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
        )

    def record_slot_tokens(
        self,
        layer_id: int,
        expert_tokens: torch.Tensor,
        group_list_type: int,
        slot_to_global: torch.Tensor,
    ) -> None:
        if expert_tokens.numel() == 0:
            return

        num_experts = self._layers[layer_id]
        local_counts = self._to_counts(expert_tokens, group_list_type)
        local_counts = local_counts.to(device="cpu", dtype=torch.long)
        global_counts = torch.zeros(num_experts, dtype=torch.long, device="cpu")

        slot_to_global = slot_to_global.to(device="cpu", dtype=torch.long)
        size = min(local_counts.numel(), slot_to_global.numel())
        global_ids = slot_to_global[:size]
        valid = (global_ids >= 0) & (global_ids < num_experts)
        global_counts.index_add_(0, global_ids[valid],
                                 local_counts[:size][valid])
        self._counts[layer_id].add_(global_counts)

    def get_layer_load(self, layer_id: int) -> torch.Tensor:
        return self._counts[layer_id].detach().cpu()

    def save(self) -> None:
        target_path = self._rank_file_path(self.path)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": LOAD_HISTORY_VERSION,
            "metadata": self._metadata,
            "layers": {
                str(layer_id): counts.tolist()
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
    def _build_local_to_global(layer) -> torch.Tensor:
        if layer.full_expert_map is None:
            return torch.arange(layer.global_num_experts, dtype=torch.long)

        full_map = layer.full_expert_map.detach().to(device="cpu",
                                                     dtype=torch.long)
        local_num_experts = int(torch.sum(full_map >= 0).item())
        local_to_global = torch.full((local_num_experts, ),
                                     -1,
                                     dtype=torch.long,
                                     device="cpu")
        for global_id, local_id in enumerate(full_map.tolist()):
            if local_id >= 0:
                local_to_global[local_id] = global_id
        return local_to_global

    def _rank_file_path(self, path: str | None) -> Path | None:
        directory = Path(path)
        rank_name = f"{directory.name}_rank{self.rank}"
        return directory / f"{rank_name}.json"
