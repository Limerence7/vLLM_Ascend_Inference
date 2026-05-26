from typing import Any, Optional


def worker_offload_summary(worker: Any) -> Optional[dict]:
    get_model = getattr(worker, "get_model", None)
    if callable(get_model):
        model = get_model()
        summary_fn = getattr(model, "offload_summary", None)
        if callable(summary_fn):
            return summary_fn()

    model_runner = getattr(worker, "model_runner", None)
    if model_runner is not None:
        return find_offload_summary(model_runner)

    return None


def install_worker_summary_rpc() -> None:
    try:
        from vllm_ascend.worker.worker import NPUWorker
    except Exception:
        return

    if not hasattr(NPUWorker, "plugin_offload_summary"):
        NPUWorker.plugin_offload_summary = worker_offload_summary


def find_offload_summary(root: Any) -> Optional[dict]:
    visited = set()
    stack = [root]
    attr_names = (
        "model",
        "model_runner",
        "driver_worker",
        "model_executor",
        "llm_engine",
    )

    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in visited:
            continue
        visited.add(obj_id)

        summary_fn = getattr(obj, "offload_summary", None)
        if callable(summary_fn):
            return summary_fn()

        for attr_name in attr_names:
            child = getattr(obj, attr_name, None)
            if child is not None:
                stack.append(child)

    return None


def print_offload_summary(root: Any) -> None:
    summary = find_offload_summary(root)
    if summary is not None:
        print(f"[Plugin] Offload summary: {summary}")
        return

    collective_rpc = getattr(root, "collective_rpc", None)
    if callable(collective_rpc):
        try:
            worker_summaries = collective_rpc("plugin_offload_summary")
        except Exception:
            worker_summaries = None

        if worker_summaries is not None:
            available = [item for item in worker_summaries if item is not None]
            if available:
                print(
                    "[Plugin] Worker offload summaries: "
                    f"{[_compact_summary(item) for item in available]}"
                )
                return

        try:
            worker_summaries = collective_rpc(worker_offload_summary)
        except Exception as exc:
            print(f"[Plugin] Worker offload summary is unavailable: {exc}")
            return

        available = [item for item in worker_summaries if item is not None]
        if available:
            print(
                "[Plugin] Worker offload summaries: "
                f"{[_compact_summary(item) for item in available]}"
            )
            return

        print("[Plugin] Offload summary is unavailable from this LLM object.")
        return

    print("[Plugin] Offload summary is unavailable from this LLM object.")


def _compact_summary(summary: dict) -> dict:
    if summary.get("mode") != "expert_wise":
        return summary

    layers = summary.get("layers", {})
    if not isinstance(layers, dict):
        return summary

    layer_items = [
        item for item in layers.values() if isinstance(item, dict)
    ]
    compact_sizing_items = [
        item.get("compact_sizing")
        for item in layer_items
        if isinstance(item.get("compact_sizing"), dict)
    ]
    compact_sizing_examples = compact_sizing_items[:3]
    return {
        "mode": "expert_wise",
        "num_layers": len(layer_items),
        "total_cpu_store_bytes": summary.get("total_cpu_store_bytes", 0),
        "total_copy_count": summary.get("total_copy_count", 0),
        "total_prefetch_count": summary.get("total_prefetch_count", 0),
        "total_prefetch_wait_count": summary.get(
            "total_prefetch_wait_count",
            0,
        ),
        "total_prefetch_capacity_skips": summary.get(
            "total_prefetch_capacity_skips",
            0,
        ),
        "total_chunked_compact_forward_count": summary.get(
            "total_chunked_compact_forward_count",
            0,
        ),
        "total_chunked_compact_piece_count": summary.get(
            "total_chunked_compact_piece_count",
            0,
        ),
        "total_chunked_compact_token_count": summary.get(
            "total_chunked_compact_token_count",
            0,
        ),
        "total_chunked_compact_full_token_count": summary.get(
            "total_chunked_compact_full_token_count",
            0,
        ),
        "total_no_shrink_skipped_layers": summary.get(
            "total_no_shrink_skipped_layers",
            0,
        ),
        "total_capacity_unsafe_skipped_layers": summary.get(
            "total_capacity_unsafe_skipped_layers",
            0,
        ),
        "total_routing_select_count": sum(
            item.get("routing_select_count", 0) for item in layer_items
        ),
        "total_routing_large_batch_fast_path_count": sum(
            item.get("routing_large_batch_fast_path_count", 0)
            for item in layer_items
        ),
        "compact_enabled_layers": sum(
            1 for item in layer_items if item.get("compact_enabled")
        ),
        "offloaded_layers": sum(
            1 for item in layer_items if item.get("offloaded_experts")
        ),
        "compact_sizing_examples": compact_sizing_examples,
    }
