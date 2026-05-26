import torch

from src.config import ExpertWiseConfig, OffloadConfig
from src.expert_wise.offload import ExpertWiseExpertStore
from src.expert_wise.manager import ExpertWiseManager
from src.expert_wise.scheduler import ExpertWisePlan, ExpertWiseScheduler
from src.fused_moe.fused_moe import ExpertWiseAscendFusedMoE
from src.utils.summary import _compact_summary


class FakeExpertLayer:
    def __init__(self):
        self.prefetched_all = 0
        self.prefetched = []

    def prefetch_all_offloaded_experts(self):
        self.prefetched_all += 1

    def prefetch_experts(self, expert_ids):
        self.prefetched.append(set(expert_ids))


def test_expert_wise_prefetch_follows_offload_interval():
    config = OffloadConfig(
        mode="expert_wise",
        prefetch_distance=1,
        offload_interval=4,
        expert_wise=ExpertWiseConfig(enable_prediction=True),
    ).normalized_copy()
    manager = ExpertWiseManager(config)
    layer4 = FakeExpertLayer()
    manager.register_layer(4, layer4)

    manager.schedule_next_prefetch(
        layer_idx=0,
        activated_experts={3, 7},
        plan=ExpertWisePlan(
            load_experts=set(),
            prefetch_next_experts={3, 7},
            prefetch_all_next=False,
        ),
    )

    assert layer4.prefetched == [{3, 7}]


def test_prediction_fallback_uses_next_offload_layer():
    config = OffloadConfig(
        mode="expert_wise",
        prefetch_distance=1,
        offload_interval=4,
        expert_wise=ExpertWiseConfig(enable_prediction=True),
    ).normalized_copy()
    manager = ExpertWiseManager(config)
    layer4 = FakeExpertLayer()
    manager.register_layer(4, layer4)

    manager.schedule_next_prefetch(
        layer_idx=0,
        activated_experts={2, 5},
        plan=ExpertWisePlan(
            load_experts=set(),
            prefetch_next_experts=set(),
            prefetch_all_next=False,
        ),
    )

    assert layer4.prefetched == [{2, 5}]


def test_large_batch_all_prefetch_does_not_depend_on_prediction_flag():
    config = OffloadConfig(
        mode="expert_wise",
        prefetch_distance=1,
        offload_interval=4,
        expert_wise=ExpertWiseConfig(enable_prediction=False),
    ).normalized_copy()
    manager = ExpertWiseManager(config)
    layer4 = FakeExpertLayer()
    manager.register_layer(4, layer4)

    manager.schedule_next_prefetch(
        layer_idx=0,
        activated_experts={0, 1, 2, 3},
        plan=ExpertWisePlan(
            load_experts=set(),
            prefetch_next_experts=set(),
            prefetch_all_next=True,
        ),
    )

    assert layer4.prefetched_all == 1


def test_large_batch_threshold_uses_ceiling():
    config = OffloadConfig(
        mode="expert_wise",
        expert_wise=ExpertWiseConfig(
            large_batch_active_ratio=0.8,
            enable_prediction=True,
        ),
    ).normalized_copy()
    scheduler = ExpertWiseScheduler(config)

    just_below = scheduler.build_plan(
        routed_experts={0, 1},
        current_offloaded_experts={0, 1, 2},
    )
    at_threshold = scheduler.build_plan(
        routed_experts={0, 1, 2},
        current_offloaded_experts={0, 1, 2},
    )

    assert not just_below.prefetch_all_next
    assert at_threshold.prefetch_all_next


def test_large_batch_plan_can_be_selected_without_routing():
    config = OffloadConfig(
        mode="expert_wise",
        expert_wise=ExpertWiseConfig(
            large_batch_active_ratio=0.8,
            on_demand_load=True,
        ),
    ).normalized_copy()
    scheduler = ExpertWiseScheduler(config)
    offloaded = {4, 5, 6, 7}

    assert scheduler.should_use_large_batch_plan(
        num_tokens=4,
        offloaded_count=len(offloaded),
    )

    plan = scheduler.build_large_batch_plan(current_offloaded_experts=offloaded)
    assert plan.load_experts == offloaded
    assert plan.prefetch_all_next


def test_large_batch_fast_path_can_be_disabled():
    config = OffloadConfig(
        mode="expert_wise",
        expert_wise=ExpertWiseConfig(
            enable_large_batch_fast_path=False,
            large_batch_active_ratio=0.8,
        ),
    ).normalized_copy()
    scheduler = ExpertWiseScheduler(config)

    assert not scheduler.should_use_large_batch_plan(
        num_tokens=1024,
        offloaded_count=4,
    )


def test_local_rank_partition_spreads_offloaded_experts_per_rank():
    config = OffloadConfig(
        mode="expert_wise",
        expert_wise=ExpertWiseConfig(
            resident_experts=96,
            partition_scope="local_rank",
            offload_multiple=4,
        ),
    ).normalized_copy()
    scheduler = ExpertWiseScheduler(config)

    rank0 = {global_expert_id: slot for slot, global_expert_id in enumerate(range(32))}
    rank3 = {
        global_expert_id: slot
        for slot, global_expert_id in enumerate(range(96, 128))
    }

    assert scheduler.offloaded_local_experts(
        local_global_to_slot=rank0,
        total_experts=128,
    ) == set(range(24, 32))
    assert scheduler.offloaded_local_experts(
        local_global_to_slot=rank3,
        total_experts=128,
    ) == set(range(120, 128))


def test_global_partition_keeps_consecutive_global_suffix():
    config = OffloadConfig(
        mode="expert_wise",
        expert_wise=ExpertWiseConfig(
            resident_experts=96,
            partition_scope="global",
            offload_multiple=4,
        ),
    ).normalized_copy()
    scheduler = ExpertWiseScheduler(config)

    assert scheduler.offloaded_experts(128) == set(range(96, 128))


def test_prefetch_wait_deduplicates_shared_event(monkeypatch):
    store = ExpertWiseExpertStore.__new__(ExpertWiseExpertStore)
    event = object()
    waits = []

    class FakeStream:
        def wait_event(self, waited_event):
            waits.append(waited_event)

    class FakeNpu:
        @staticmethod
        def current_stream():
            return FakeStream()

    monkeypatch.setattr("src.expert_wise.offload.torch_npu.npu", FakeNpu)
    store.prefetch_events = {1: event, 2: event, 3: object()}
    store.prefetch_wait_count = 0

    store.wait_for_prefetch({1, 2})

    assert waits == [event]
    assert store.prefetch_wait_count == 1
    assert store.prefetch_events.keys() == {3}


def test_compact_sizing_records_shrink_and_restore_bounds():
    layer = ExpertWiseAscendFusedMoE.__new__(ExpertWiseAscendFusedMoE)
    layer.expert_wise_config = ExpertWiseConfig(npu_cache_capacity=2)
    layer.expert_wise_compact_sizing = {}

    layer._record_compact_sizing(
        resident_local_count=12,
        offloaded_local_count=4,
        original_local_slots=16,
    )

    assert layer.expert_wise_compact_sizing == {
        "resident_local": 12,
        "offloaded_local": 4,
        "cache_capacity": 2,
        "original_local_slots": 16,
        "compact_slots": 14,
        "max_cache_capacity_for_shrink": 3,
        "min_cache_capacity_for_full_restore": 4,
    }


def test_compact_worker_summary_includes_sizing_examples():
    summary = _compact_summary(
        {
            "mode": "expert_wise",
            "layers": {
                0: {
                    "compact_sizing": {
                        "resident_local": 12,
                        "offloaded_local": 4,
                    },
                },
                1: {
                    "compact_sizing": {
                        "resident_local": 13,
                        "offloaded_local": 3,
                    },
                },
            },
        }
    )

    assert summary["compact_sizing_examples"] == [
        {"resident_local": 12, "offloaded_local": 4},
        {"resident_local": 13, "offloaded_local": 3},
    ]


def test_chunk_expert_ids_respects_cache_capacity():
    chunks = ExpertWiseAscendFusedMoE._chunk_expert_ids([10, 11, 12, 13, 14], 2)

    assert chunks == [{10, 11}, {12, 13}, {14}]


def test_chunk_compact_forward_merges_resident_with_first_offloaded_chunk():
    chunks = ExpertWiseAscendFusedMoE._chunk_compact_forward_experts(
        resident_experts={0, 1},
        offloaded_experts=[10, 11, 12, 13],
        chunk_size=3,
    )

    assert chunks == [{0, 1, 10, 11, 12}, {13}]


def test_compact_chunk_token_selection_slices_token_aligned_tensors():
    row_mask = torch.tensor([True, False, True])
    hidden_states = torch.arange(12).reshape(3, 4)
    topk_weights = torch.arange(6).reshape(3, 2)
    topk_ids = torch.arange(6).reshape(3, 2)
    mc2_mask = torch.tensor([1, 0, 1])
    pertoken_scale = torch.tensor([3, 4, 5])

    selected = ExpertWiseAscendFusedMoE._select_compact_chunk_tokens(
        row_mask=row_mask,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        mc2_mask=mc2_mask,
        pertoken_scale=pertoken_scale,
    )

    assert torch.equal(selected[0], hidden_states[[0, 2]])
    assert torch.equal(selected[1], topk_weights[[0, 2]])
    assert torch.equal(selected[2], topk_ids[[0, 2]])
    assert torch.equal(selected[3], mc2_mask[[0, 2]])
    assert torch.equal(selected[4], pertoken_scale[[0, 2]])


def test_optional_token_selection_preserves_non_token_tensors():
    row_mask = torch.tensor([True, False, True])
    non_token_tensor = torch.ones(2, 2)

    selected = ExpertWiseAscendFusedMoE._select_optional_token_tensor(
        non_token_tensor,
        row_mask,
    )

    assert selected is non_token_tensor


def test_single_select_allows_chunked_compact_over_capacity():
    layer = ExpertWiseAscendFusedMoE.__new__(ExpertWiseAscendFusedMoE)
    layer.expert_wise_config = ExpertWiseConfig(
        npu_cache_capacity=2,
        enable_single_select_forward=True,
        enable_chunked_compact_forward=True,
    )
    layer.multistream_overlap_gate = False
    layer.dynamic_eplb = False

    class FakeQuantType:
        name = "NONE"

    class FakeStore:
        @staticmethod
        def is_compact():
            return True

        @staticmethod
        def offloaded_expert_ids():
            return {10, 11, 12, 13}

    layer.quant_type = FakeQuantType()
    layer.expert_store = FakeStore()

    assert layer._can_use_single_select_forward()
