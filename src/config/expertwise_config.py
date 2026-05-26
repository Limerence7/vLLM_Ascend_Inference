from dataclasses import dataclass


@dataclass
class ExpertWiseConfig:
    resident_experts: int = 0
    partition_scope: str = "global"
    offload_multiple: int = 4
    enable_prediction: bool = True
    on_demand_load: bool = True
    npu_cache_capacity: int = 0
    pin_cpu_memory: bool = True
    keep_loaded_on_npu: bool = True
    compact_npu_cache: bool = False
    skip_no_shrink_compact: bool = False
    enable_large_batch_fast_path: bool = True
    enable_single_select_forward: bool = True
    enable_chunked_compact_forward: bool = True
    enable_compact_chunk_token_filter: bool = False
    large_batch_active_ratio: float = 0.8
    log_transfers: bool = False
    max_transfer_logs: int = 32

    def validate(self) -> None:
        self.partition_scope = self.partition_scope.strip().lower()
        if self.partition_scope not in {"global", "local_rank"}:
            raise ValueError(
                "ExpertWiseConfig.partition_scope must be 'global' or "
                "'local_rank'."
            )
        if self.resident_experts < 0:
            raise ValueError("ExpertWiseConfig.resident_experts must be >= 0.")
        if self.offload_multiple <= 0:
            raise ValueError("ExpertWiseConfig.offload_multiple must be > 0.")
        if self.npu_cache_capacity < 0:
            raise ValueError("ExpertWiseConfig.npu_cache_capacity must be >= 0.")
        if self.compact_npu_cache and self.npu_cache_capacity <= 0:
            raise ValueError(
                "ExpertWiseConfig.compact_npu_cache=True requires "
                "npu_cache_capacity > 0."
            )
        if not 0.0 < self.large_batch_active_ratio <= 1.0:
            raise ValueError(
                "ExpertWiseConfig.large_batch_active_ratio must satisfy "
                "0 < ratio <= 1."
            )
        if self.max_transfer_logs < 0:
            raise ValueError("ExpertWiseConfig.max_transfer_logs must be >= 0.")
