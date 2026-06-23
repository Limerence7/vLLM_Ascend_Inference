from dataclasses import dataclass, field

import torch

from ..layer.routing import map_expert_ids


@dataclass(frozen=True)
class ExpertPlacement:
    resident_expert_ids: list[int]
    cold_expert_ids: list[int]
    resident_slots: dict[int, int] = field(init=False)
    cold_slots: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "resident_slots", {
            expert_id: slot
            for slot, expert_id in enumerate(self.resident_expert_ids)
        })
        object.__setattr__(self, "cold_slots", {
            expert_id: slot
            for slot, expert_id in enumerate(self.cold_expert_ids)
        })


class LayerRoutingMap:
    """Map router-selected global expert ids to cold-buffer slots."""

    def __init__(self, layer, placement: ExpertPlacement):
        self.local_num_experts = int(layer.full_local_num_experts)
        self.global_to_local = self._build_global_to_local(layer)
        self.local_to_cold_slot = self._build_local_to_cold_slot(placement)
        self._device_maps: dict[torch.device,
                                tuple[torch.Tensor, torch.Tensor]] = {}

    def cold_routing(self, topk_ids: torch.Tensor,
                     cache_maps: bool = True) -> tuple[torch.Tensor,
                                                       torch.Tensor]:
        global_to_local, local_to_cold_slot = self._maps_for(
            topk_ids.device, cache_maps)
        local_ids, is_local = map_expert_ids(topk_ids, global_to_local)
        cold_slots = self._lookup_local_ids(local_ids, local_to_cold_slot)

        cold_mask = is_local & (cold_slots >= 0)
        cold_topk_ids = cold_slots.masked_fill(~cold_mask, 0)
        return cold_topk_ids.to(topk_ids.dtype), cold_mask

    def _maps_for(self, device: torch.device,
                  cache_maps: bool) -> tuple[torch.Tensor, torch.Tensor]:
        maps = self._device_maps.get(device)
        if maps is None:
            maps = (
                self.global_to_local.to(device=device, non_blocking=True),
                self.local_to_cold_slot.to(device=device, non_blocking=True),
            )
            if cache_maps:
                self._device_maps[device] = maps
        return maps

    def _lookup_local_ids(self, local_ids: torch.Tensor,
                          local_to_cold_slot: torch.Tensor) -> torch.Tensor:
        safe_ids = local_ids.clamp(min=0, max=max(0, self.local_num_experts - 1))
        return local_to_cold_slot[safe_ids]

    @staticmethod
    def _build_global_to_local(layer) -> torch.Tensor:
        if layer.full_expert_map is None:
            return torch.arange(layer.global_num_experts, dtype=torch.long)
        return layer.full_expert_map.detach().to(device="cpu",
                                                 dtype=torch.long)

    def _build_local_to_cold_slot(
            self, placement: ExpertPlacement) -> torch.Tensor:
        local_to_cold = torch.full((self.local_num_experts, ),
                                   -1,
                                   dtype=torch.long)
        for expert_id, slot in placement.cold_slots.items():
            local_to_cold[expert_id] = slot
        return local_to_cold
