class RuntimeWorkerExtension:
    """Worker RPC methods for runtime utilities."""

    def save_load_history(self) -> dict[str, object]:
        from .layer.fused_moe import RuntimeAscendFusedMoE

        recorder = (RuntimeAscendFusedMoE.lbvc_adaptor
                    or RuntimeAscendFusedMoE.executor
                    or RuntimeAscendFusedMoE.load_profiler)
        if recorder is None:
            return {
                "saved": False,
                "reason": "runtime executor is not initialized",
            }

        if hasattr(recorder, "save_load_history"):
            recorder.save_load_history()
            profiler = recorder.profiler
            num_layers = len(recorder.layers)
        else:
            recorder.save()
            profiler = recorder
            num_layers = len(recorder._local_to_global)

        return {
            "saved": profiler.path is not None,
            "load_history_path": profiler.path,
            "output_path": profiler.output_path,
            "num_layers": num_layers,
        }
