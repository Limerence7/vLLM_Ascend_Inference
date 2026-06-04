import torch


class LayerRoutingMap:
    """Map router-selected global expert ids to resident cold-buffer ids."""

    def __init__(self, layer, cold_expert_ids: list[int]):
        self.local_num_experts = int(layer.full_local_num_experts)
        self.global_to_local = self._build_global_to_local(layer)
        self.local_to_cold_expert = self._build_local_to_cold_expert(
            cold_expert_ids)
        self._device_maps: dict[torch.device,
                                tuple[torch.Tensor, torch.Tensor]] = {}

    def cold_routing(self, topk_ids: torch.Tensor) -> tuple[torch.Tensor,
                                                           torch.Tensor]:
        global_to_local, local_to_cold_expert = self._maps_for(
            topk_ids.device)

        local_ids, valid_global = self._lookup_global_ids(
            topk_ids, global_to_local)
        cold_expert_ids = self._lookup_local_ids(local_ids,
                                                 local_to_cold_expert)

        cold_mask = valid_global & (cold_expert_ids >= 0)
        fallback = torch.zeros_like(cold_expert_ids)
        cold_topk_ids = torch.where(cold_mask, cold_expert_ids, fallback)
        return cold_topk_ids.to(topk_ids.dtype), cold_mask

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

    def _lookup_global_ids(self, topk_ids: torch.Tensor,
                           global_to_local: torch.Tensor) -> tuple[
                               torch.Tensor, torch.Tensor]:
        global_ids = topk_ids.long()
        valid = (global_ids >= 0) & (global_ids < global_to_local.numel())
        safe_ids = global_ids.clamp(min=0, max=global_to_local.numel() - 1)
        local_ids = global_to_local[safe_ids]
        valid &= (local_ids >= 0) & (local_ids < self.local_num_experts)
        return local_ids, valid

    def _lookup_local_ids(self, local_ids: torch.Tensor,
                          local_to_cold_expert: torch.Tensor) -> torch.Tensor:
        safe_ids = local_ids.clamp(min=0, max=self.local_num_experts - 1)
        return local_to_cold_expert[safe_ids]

    def _build_global_to_local(self, layer) -> torch.Tensor:
        if layer.full_expert_map is None:
            return torch.arange(layer.global_num_experts, dtype=torch.long)
        return layer.full_expert_map.detach().to(device="cpu",
                                                 dtype=torch.long)

    def _build_local_to_cold_expert(self,
                                    cold_expert_ids: list[int]) -> torch.Tensor:
        local_to_cold = torch.full((self.local_num_experts, ),
                                   -1,
                                   dtype=torch.long)
        expert_ids = torch.tensor(cold_expert_ids, dtype=torch.long)
        local_to_cold[expert_ids] = torch.arange(len(cold_expert_ids),
                                                 dtype=torch.long)
        return local_to_cold
