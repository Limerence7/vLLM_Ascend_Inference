from src.config import OffloadConfig
from src.layer_wise.scheduler import LayerWiseScheduler


def test_layer_wise_prefetch_distance_is_model_layer_distance():
    config = OffloadConfig(
        mode="layer_wise",
        prefetch_distance=16,
        offload_interval=16,
    ).normalized_copy()
    scheduler = LayerWiseScheduler(config)
    selected_layers = [0, 16, 32, 48, 64, 80]

    assert scheduler.next_prefetch_layer(selected_layers, 0) == 16
    assert scheduler.next_prefetch_layer(selected_layers, 16) == 32


def test_layer_wise_prefetch_uses_next_selected_layer_for_short_distance():
    config = OffloadConfig(
        mode="layer_wise",
        prefetch_distance=1,
        offload_interval=16,
    ).normalized_copy()
    scheduler = LayerWiseScheduler(config)
    selected_layers = [0, 16, 32]

    assert scheduler.next_prefetch_layer(selected_layers, 0) == 16


def test_layer_wise_prefetch_returns_none_past_last_selected_layer():
    config = OffloadConfig(
        mode="layer_wise",
        prefetch_distance=16,
        offload_interval=16,
    ).normalized_copy()
    scheduler = LayerWiseScheduler(config)
    selected_layers = [0, 16, 32]

    assert scheduler.next_prefetch_layer(selected_layers, 32) is None
