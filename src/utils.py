class RuntimeWorkerExtension:
    """Worker RPC methods for runtime utilities."""

    def save_load_history(self) -> dict[str, object]:
        from .runtime.runtime_core import RuntimeCore

        profiler = RuntimeCore._profiler
        if profiler is None:
            return {
                "saved": False,
                "reason": "runtime core is not initialized",
            }

        profiler.save()
        num_layers = len(profiler._local_to_global)

        return {
            "saved": profiler.path is not None,
            "load_history_path": profiler.path,
            "output_path": profiler.output_path,
            "num_layers": num_layers,
        }
