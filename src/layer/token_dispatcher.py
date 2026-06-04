from contextlib import contextmanager
from typing import Iterator


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
