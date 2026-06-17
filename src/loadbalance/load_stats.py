import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


LOAD_STATS_VERSION = 1


class ExpertLoadStats:
    """Track per-layer expert hits using global expert ids."""

    def __init__(self,
                 path: str | None = None,
                 metadata: dict[str, Any] | None = None):
        self.path = path
        self._layers: dict[int, int] = {}
        self._counts: dict[int, torch.Tensor] = {}
        self._global_to_local: dict[int, torch.Tensor | None] = {}
        self._metadata: dict[str, Any] = dict(metadata or {})
        self._metadata["rank"] = self._rank()
        load_path = self._rank_file_path(path) if path else None
        if load_path is not None and load_path.exists():
            self.load(str(load_path))
            self._metadata.update(metadata or {})
            self._metadata["rank"] = self._rank()

    @property
    def output_path(self) -> str | None:
        path = self._rank_file_path(self.path)
        return None if path is None else str(path)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        layers = payload.get("layers", {})
        if not isinstance(layers, dict):
            raise ValueError("Load stats file field 'layers' must be a mapping.")

        self._metadata = dict(payload.get("metadata", {}))
        for layer_id_text, counts in layers.items():
            layer_id = int(layer_id_text)
            if not isinstance(counts, list):
                raise ValueError(
                    f"Load stats for layer {layer_id} must be a list.")
            tensor = torch.tensor(counts, dtype=torch.long, device="cpu")
            self._layers[layer_id] = int(tensor.numel())
            self._counts[layer_id] = tensor

    def register_layer(self, layer) -> None:
        layer_id = int(layer.moe_instance_id)
        num_experts = int(layer.global_num_experts)
        self._layers[layer_id] = num_experts
        global_to_local = self._build_global_to_local(layer)
        self._global_to_local[layer_id] = global_to_local
        self._record_layer_metadata(layer, global_to_local)

        counts = self._counts.get(layer_id)
        if counts is None:
            self._counts[layer_id] = torch.zeros(num_experts,
                                                 dtype=torch.long,
                                                 device="cpu")
        elif counts.numel() != num_experts:
            raise ValueError(
                f"Load stats for layer {layer_id} has {counts.numel()} "
                f"experts, expected {num_experts}.")

    def record(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        if layer_id not in self._layers or topk_ids.numel() == 0:
            return

        num_experts = self._layers[layer_id]
        expert_ids = topk_ids.detach().reshape(-1).to(device="cpu",
                                                      dtype=torch.long)
        valid = (expert_ids >= 0) & (expert_ids < num_experts)

        global_to_local = self._global_to_local.get(layer_id)
        if global_to_local is not None:
            safe_ids = expert_ids.clamp(min=0, max=num_experts - 1)
            valid = valid & (global_to_local[safe_ids] >= 0)

        expert_ids = expert_ids[valid]
        if expert_ids.numel() == 0:
            return

        batch_counts = torch.bincount(expert_ids, minlength=num_experts)
        self._counts[layer_id].add_(batch_counts)

    def get_layer_load(self, layer_id: int) -> torch.Tensor:
        return self._counts[int(layer_id)].detach().cpu().clone()

    def save(self, path: str | None = None) -> None:
        target_path = self._rank_file_path(path or self.path)
        if target_path is None:
            return

        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": LOAD_STATS_VERSION,
            "metadata": self._metadata,
            "layers": {
                str(layer_id): counts.tolist()
                for layer_id, counts in sorted(self._snapshot().items())
            },
        }
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, target_path)

    def _snapshot(self) -> dict[int, torch.Tensor]:
        return {
            layer_id: counts.detach().cpu().clone()
            for layer_id, counts in self._counts.items()
        }

    @staticmethod
    def _build_global_to_local(layer) -> torch.Tensor | None:
        if layer.full_expert_map is None:
            return None
        return layer.full_expert_map.detach().to(device="cpu",
                                                 dtype=torch.long)

    def _record_layer_metadata(self, layer,
                               global_to_local: torch.Tensor | None) -> None:
        layer_id = int(layer.moe_instance_id)
        if global_to_local is None:
            local_global_ids = list(range(int(layer.global_num_experts)))
        else:
            local_global_ids = torch.nonzero(
                global_to_local >= 0, as_tuple=False).flatten().tolist()
        layers = self._metadata.setdefault("layers", {})
        layers[str(layer_id)] = {
            "global_num_experts": int(layer.global_num_experts),
            "local_num_experts": int(layer.full_local_num_experts),
            "local_global_expert_ids": [int(i) for i in local_global_ids],
        }

    @staticmethod
    def _rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @classmethod
    def _rank_file_path(cls, path: str | None) -> Path | None:
        if not path:
            return None

        directory = Path(path)
        rank = cls._rank()
        rank_name = f"{directory.name}_rank{rank}"
        return directory / f"{rank_name}.json"
