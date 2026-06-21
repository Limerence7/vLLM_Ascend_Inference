import torch

from ..layer.routing import map_expert_ids
from ..loadbalance import ExpertPlacement, ExpertSwap


class LayerRoutingMap:
    """Map router-selected global expert ids to resident cold-buffer ids."""

    def __init__(self, layer, placement: ExpertPlacement):
        self.local_num_experts = int(layer.full_local_num_experts)
        self.global_to_local = self._build_global_to_local(layer)
        self.local_to_cold_expert = self._build_local_to_cold_expert(
            placement)
        self._device_maps: dict[torch.device,
                                tuple[torch.Tensor, torch.Tensor]] = {}

    def cold_routing(self, topk_ids: torch.Tensor) -> tuple[torch.Tensor,
                                                           torch.Tensor]:
        global_to_local, local_to_cold_expert = self._maps_for(
            topk_ids.device)

        local_ids, is_local = map_expert_ids(topk_ids, global_to_local)
        cold_expert_ids = self._lookup_local_ids(local_ids,
                                                 local_to_cold_expert)

        cold_mask = is_local & (cold_expert_ids >= 0)
        cold_topk_ids = cold_expert_ids.masked_fill(~cold_mask, 0)
        return cold_topk_ids.to(topk_ids.dtype), cold_mask

    def apply_placement(self, placement: ExpertPlacement,
                        swaps: list[ExpertSwap] | None = None) -> None:
        if not swaps:
            self.local_to_cold_expert = self._build_local_to_cold_expert(
                placement)
            self._device_maps.clear()
            return

        for swap in swaps:
            self._set_cold_slot(swap.swap_in, -1)
            self._set_cold_slot(swap.swap_out, swap.cold_slot)

    def _set_cold_slot(self, local_expert_id: int, cold_slot: int) -> None:
        self.local_to_cold_expert[local_expert_id] = cold_slot
        for _, local_to_cold_expert in self._device_maps.values():
            local_to_cold_expert[local_expert_id] = cold_slot

    def _maps_for(self, device: torch.device) -> tuple[torch.Tensor,
                                                       torch.Tensor]:
        maps = self._device_maps.get(device)
        if maps is None:
            maps = (
                self.global_to_local.to(device=device, non_blocking=True),
                self.local_to_cold_expert.to(device=device,
                                             non_blocking=True),
            )
            self._device_maps[device] = maps
        return maps

    def _lookup_local_ids(self, local_ids: torch.Tensor,
                          local_to_cold_expert: torch.Tensor) -> torch.Tensor:
        safe_ids = local_ids.clamp(min=0, max=self.local_num_experts - 1)
        return local_to_cold_expert[safe_ids]

    def _build_global_to_local(self, layer) -> torch.Tensor:
        if layer.full_expert_map is None:
            return torch.arange(layer.global_num_experts, dtype=torch.long)
        return layer.full_expert_map.detach().to(device="cpu",
                                                 dtype=torch.long)

    def _build_local_to_cold_expert(
            self, placement: ExpertPlacement) -> torch.Tensor:
        local_to_cold = torch.full((self.local_num_experts, ),
                                   -1,
                                   dtype=torch.long)
        for expert_id, slot in placement.cold_slots.items():
            local_to_cold[expert_id] = slot
        return local_to_cold
