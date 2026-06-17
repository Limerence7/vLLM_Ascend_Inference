class OffloadWorkerExtension:
    """Worker RPC methods for offload runtime utilities."""

    def save_load_stats(self) -> dict[str, object]:
        from .layer.fused_moe import OffloadAscendFusedMoE

        executor = OffloadAscendFusedMoE.executor
        if executor is None:
            return {
                "saved": False,
                "reason": "offload executor is not initialized",
            }

        executor.save_load_stats()
        return {
            "saved": executor.load_stats is not None,
            "load_stats_path": executor.config.load_stats_path,
            "output_path": (
                None if executor.load_stats is None else
                executor.load_stats.output_path),
            "num_layers": len(executor.layers),
        }
