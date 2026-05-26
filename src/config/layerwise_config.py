from dataclasses import dataclass


@dataclass
class LayerWiseConfig:
    cpu_cache_size: int = 2
    async_prefetch: bool = True

    def validate(self) -> None:
        if self.cpu_cache_size <= 0:
            raise ValueError("LayerWiseConfig.cpu_cache_size must be > 0.")
