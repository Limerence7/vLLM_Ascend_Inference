class RuntimeWorkerExtension:
    """Worker RPC methods for runtime utilities."""

    def save_load_history(self) -> dict[str, object]:
        from .layer.fused_moe import RuntimeAscendFusedMoE

        executor = RuntimeAscendFusedMoE.executor
        if executor is None:
            return {
                "saved": False,
                "reason": "runtime executor is not initialized",
            }

        executor.save_load_history()
        return {
            "saved": executor.profiler.path is not None,
            "load_history_path": executor.config.load_history_path,
            "output_path": executor.profiler.output_path,
            "num_layers": len(executor.layers),
        }
