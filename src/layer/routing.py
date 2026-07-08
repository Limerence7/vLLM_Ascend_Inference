from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch


@dataclass(frozen=True)
class MoERoutingView:
    """A compact token-row view for one side of split MoE execution."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    row_indices: torch.Tensor


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
        self.global_num_experts = int(layer.global_num_experts)
        self.global_to_local = self._build_global_to_local(layer)
        self.local_to_cold_slot = self._build_local_to_cold_slot(placement)
        self.global_to_cold_slot: torch.Tensor | None = None
        self._device_maps: dict[torch.device,
                                tuple[torch.Tensor, torch.Tensor,
                                      torch.Tensor | None]] = {}

    def cold_routing(self, topk_ids: torch.Tensor,
                     cache_maps: bool = True) -> tuple[torch.Tensor,
                                                       torch.Tensor]:
        global_to_local, local_to_cold_slot, global_to_cold_slot = self._maps_for(
            topk_ids.device, cache_maps)
        if global_to_cold_slot is not None:
            cold_slots = global_to_cold_slot[topk_ids.long()]
            cold_mask = cold_slots >= 0
            return cold_slots.masked_fill(~cold_mask, 0).to(
                topk_ids.dtype), cold_mask

        local_ids, is_local = map_expert_ids(topk_ids, global_to_local)
        cold_slots = self._lookup_local_ids(local_ids, local_to_cold_slot)

        cold_mask = is_local & (cold_slots >= 0)
        cold_topk_ids = cold_slots.masked_fill(~cold_mask, 0)
        return cold_topk_ids.to(topk_ids.dtype), cold_mask

    def update_global_cold_experts(self, expert_ids: list[int]) -> None:
        global_to_cold = torch.full((self.global_num_experts, ),
                                    -1,
                                    dtype=torch.long)
        for slot, expert_id in enumerate(expert_ids):
            if 0 <= expert_id < self.global_num_experts:
                global_to_cold[expert_id] = slot
        self.global_to_cold_slot = global_to_cold
        self._device_maps.clear()

    def _maps_for(self, device: torch.device,
                  cache_maps: bool) -> tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor | None]:
        maps = self._device_maps.get(device)
        if maps is None:
            global_to_cold_slot = (
                None if self.global_to_cold_slot is None else
                self.global_to_cold_slot.to(device=device, non_blocking=True))
            maps = (
                self.global_to_local.to(device=device, non_blocking=True),
                self.local_to_cold_slot.to(device=device, non_blocking=True),
                global_to_cold_slot,
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


def map_expert_ids(expert_ids: torch.Tensor,
                   expert_map: torch.Tensor) -> tuple[torch.Tensor,
                                                     torch.Tensor]:
    """Map valid router expert ids and mark experts absent from this rank."""

    mapped_ids = expert_map[expert_ids.long()]
    return mapped_ids, mapped_ids >= 0


def build_routing_view(hidden_states: torch.Tensor, topk_ids: torch.Tensor,
                       topk_weights: torch.Tensor,
                       row_mask: torch.Tensor) -> MoERoutingView | None:
    """Select token rows that have work on the hot or cold expert side."""

    row_indices = torch.nonzero(row_mask, as_tuple=False).flatten()
    if row_indices.numel() == 0:
        return None

    return MoERoutingView(
        hidden_states=hidden_states.index_select(0, row_indices),
        topk_ids=topk_ids.index_select(0, row_indices),
        topk_weights=topk_weights.index_select(0, row_indices),
        row_indices=row_indices,
    )


def add_routing_output(output: torch.Tensor, routing: MoERoutingView,
                       routing_output: torch.Tensor) -> None:
    """Accumulate a compact MoE result back into the full token output."""

    output.index_add_(0, routing.row_indices, routing_output)


def select_optional_rows(tensor: torch.Tensor | None,
                         row_indices: torch.Tensor,
                         num_rows: int) -> torch.Tensor | None:
    """Apply the same token-row selection to optional token-aligned tensors."""

    if tensor is None:
        return None
    if tensor.dim() > 0 and tensor.size(0) == num_rows:
        return tensor.index_select(0, row_indices)
    return tensor


@contextmanager
def dispatch_with_local_experts(moe_comm_method,
                                num_local_experts: int) -> Iterator[None]:
    """Match Ascend dispatcher metadata to the weight tensor in this call."""

    token_dispatcher = moe_comm_method.token_dispatcher
    fields = tuple(name for name in ("num_experts_local", "num_local_experts")
                   if hasattr(token_dispatcher, name))
    old_values = {name: getattr(token_dispatcher, name) for name in fields}

    for name in fields:
        setattr(token_dispatcher, name, int(num_local_experts))
    try:
        yield
    finally:
        for name, value in old_values.items():
            setattr(token_dispatcher, name, value)
