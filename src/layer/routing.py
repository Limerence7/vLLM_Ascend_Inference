from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MoERoutingView:
    """A compact token-row view for one side of split MoE execution."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    row_indices: torch.Tensor


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
