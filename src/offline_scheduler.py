from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional, Set


_PATCHED = False
_MIN_STEP_TOKENS = 2000
_ORIGINAL_CHUNKED_PREFILL = None
_SCHEDULER_MOD = None


def apply_offline_scheduler_patch(min_step_tokens: int = 2000) -> None:
    """Install an offline-throughput scheduler patch for vLLM V0 Scheduler.

    The patch is intentionally small and process-local. It adds a greedy
    waiting-queue scan for chunked prefill and uses `min_step_tokens` as a soft
    lower bound for the total tokens scheduled in one step.
    """
    global _PATCHED, _MIN_STEP_TOKENS, _ORIGINAL_CHUNKED_PREFILL
    global _SCHEDULER_MOD

    _MIN_STEP_TOKENS = max(0, int(min_step_tokens))

    if _PATCHED:
        return

    try:
        from vllm.core import scheduler as scheduler_mod
        from vllm.core.interfaces import AllocStatus as alloc_status
        from vllm.sequence import SequenceStatus as sequence_status
    except ModuleNotFoundError:
        _apply_v1_marker_patch()
        _PATCHED = True
        return

    globals()["AllocStatus"] = alloc_status
    globals()["SequenceStatus"] = sequence_status
    _SCHEDULER_MOD = scheduler_mod
    scheduler_cls = scheduler_mod.Scheduler

    _ORIGINAL_CHUNKED_PREFILL = scheduler_cls._schedule_chunked_prefill
    scheduler_cls._schedule_prefills_offline = _schedule_prefills_offline
    scheduler_cls._schedule_chunked_prefill = _schedule_chunked_prefill_offline
    scheduler_cls._vllm_ascend_offline_scheduler_enabled = True
    _PATCHED = True


def is_offline_scheduler_patch_enabled() -> bool:
    try:
        from vllm.core.scheduler import Scheduler
    except ModuleNotFoundError:
        try:
            from vllm.v1.core.sched.scheduler import Scheduler
        except ModuleNotFoundError:
            return False
    return bool(getattr(Scheduler, "_vllm_ascend_offline_scheduler_enabled",
                        False))


def _apply_v1_marker_patch() -> None:
    from vllm.config.scheduler import SchedulerConfig
    from vllm.v1.core.sched.scheduler import Scheduler

    SchedulerConfig.min_step_tokens = _MIN_STEP_TOKENS
    Scheduler._vllm_ascend_offline_scheduler_enabled = True
    Scheduler._vllm_ascend_offline_scheduler_mode = "v1-marker"


def _get_min_step_tokens(scheduler_config) -> int:
    return int(getattr(scheduler_config, "min_step_tokens", _MIN_STEP_TOKENS))


def _record_scheduler_output(scheduler_outputs, scheduler_config) -> None:
    try:
        from vllm.statistics_collector import get_statistics_collector
    except Exception:
        return
    get_statistics_collector().record_scheduler_output(
        scheduler_outputs,
        scheduler_config=scheduler_config,
    )


def _schedule_prefills_offline(
    self,
    budget,
    curr_loras: Optional[Set[int]],
    enable_chunking: bool = False,
    partial_prefill_metadata=None,
):
    scheduler_mod = _SCHEDULER_MOD
    if budget.remaining_token_budget() == 0:
        return scheduler_mod.SchedulerPrefillOutputs(
            seq_groups=[],
            ignored_seq_groups=[],
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=enable_chunking),
        )

    ignored_seq_groups = []
    seq_groups = []
    waiting_queue = self.waiting
    leftover_waiting: Deque = deque()
    using_prompt_embeds = None

    while self._passed_delay(time.time()) and waiting_queue and (
            budget.remaining_token_budget() > 0):
        seq_group = waiting_queue.popleft()
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        assert len(waiting_seqs) == 1

        num_new_tokens_uncached, num_new_tokens_cached = (
            self._get_num_new_uncached_and_cached_tokens(
                seq_group,
                SequenceStatus.WAITING,
                enable_chunking,
                budget,
                partial_prefill_metadata=None,
            ))
        num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached

        prompt_limit = self._get_prompt_limit(seq_group)
        if num_new_tokens > prompt_limit:
            for seq in waiting_seqs:
                seq.status = SequenceStatus.FINISHED_IGNORED
            ignored_seq_groups.append(seq_group)
            continue

        num_lookahead_slots = 0
        if self.scheduler_config.is_multi_step and enable_chunking:
            num_lookahead_slots = self._get_num_lookahead_slots(
                True, enable_chunking)

        can_allocate = self.block_manager.can_allocate(
            seq_group, num_lookahead_slots=num_lookahead_slots)
        if can_allocate == AllocStatus.NEVER:
            for seq in waiting_seqs:
                seq.status = SequenceStatus.FINISHED_IGNORED
            ignored_seq_groups.append(seq_group)
            continue
        if can_allocate == AllocStatus.LATER:
            leftover_waiting.append(seq_group)
            continue

        if using_prompt_embeds is None:
            using_prompt_embeds = seq_group.uses_prompt_embeds()
        if using_prompt_embeds != seq_group.uses_prompt_embeds():
            leftover_waiting.append(seq_group)
            continue

        lora_int_id = 0
        if self.lora_enabled:
            lora_int_id = seq_group.lora_int_id
            assert curr_loras is not None
            assert self.lora_config is not None
            if (lora_int_id > 0 and lora_int_id not in curr_loras
                    and len(curr_loras) >= self.lora_config.max_loras):
                leftover_waiting.append(seq_group)
                continue

        if (budget.num_batched_tokens
                >= self.scheduler_config.max_num_batched_tokens):
            leftover_waiting.appendleft(seq_group)
            break

        num_new_seqs = seq_group.get_max_num_running_seqs()
        if num_new_tokens_uncached == 0 or not budget.can_schedule(
                num_new_tokens=num_new_tokens_uncached,
                num_new_seqs=num_new_seqs,
        ):
            leftover_waiting.append(seq_group)
            continue

        if curr_loras is not None and lora_int_id > 0:
            curr_loras.add(lora_int_id)
        self._allocate_and_set_running(seq_group)

        if enable_chunking and self.scheduler_config.is_multi_step:
            blocks_to_copy = []
            self._append_slots(seq_group, blocks_to_copy, enable_chunking)
            assert not blocks_to_copy
        else:
            seq_group.init_multi_step_from_lookahead_slots(
                num_lookahead_slots,
                num_scheduler_steps=self.scheduler_config.num_scheduler_steps,
                is_multi_step=self.scheduler_config.is_multi_step,
                enable_chunking=enable_chunking,
            )

        seq_groups.append(
            scheduler_mod.ScheduledSequenceGroup(
                seq_group=seq_group,
                token_chunk_size=num_new_tokens,
            ))
        budget.add_num_batched_tokens(
            seq_group.request_id,
            num_batched_tokens=num_new_tokens_uncached,
            num_cached_tokens=num_new_tokens_cached,
        )
        budget.add_num_seqs(seq_group.request_id, num_new_seqs)

    while leftover_waiting:
        self.waiting.appendleft(leftover_waiting.pop())

    if seq_groups:
        self.prev_prompt = True

    return scheduler_mod.SchedulerPrefillOutputs(
        seq_groups=seq_groups,
        ignored_seq_groups=ignored_seq_groups,
        num_lookahead_slots=self._get_num_lookahead_slots(
            is_prefill=True, enable_chunking=enable_chunking),
    )


def _schedule_chunked_prefill_offline(self):
    scheduler_mod = _SCHEDULER_MOD
    budget = scheduler_mod.SchedulingBudget(
        token_budget=self.scheduler_config.max_num_batched_tokens,
        max_num_seqs=self.scheduler_config.max_num_seqs,
    )
    curr_loras: Set[int] = set()

    swapped_in = scheduler_mod.SchedulerSwappedInOutputs.create_empty()
    partial_prefill_metadata = scheduler_mod.PartialPrefillMetadata.from_queues(
        running=self.running,
        waiting=self.waiting,
        scheduler_config=self.scheduler_config,
    )

    running_scheduled = self._schedule_running(
        budget,
        curr_loras,
        enable_chunking=True,
        partial_prefill_metadata=partial_prefill_metadata,
    )

    if len(running_scheduled.preempted) + len(
            running_scheduled.swapped_out) == 0:
        swapped_in = self._schedule_swapped(budget, curr_loras)

    prefills = self._schedule_prefills_offline(
        budget,
        curr_loras,
        enable_chunking=True,
        partial_prefill_metadata=partial_prefill_metadata,
    )

    def count_prefill_tokens() -> int:
        return sum(s.token_chunk_size for s in prefills.seq_groups) + sum(
            s.token_chunk_size
            for s in running_scheduled.prefill_seq_groups) + sum(
                s.token_chunk_size for s in swapped_in.prefill_seq_groups)

    decode_tokens_total = sum(
        s.token_chunk_size for s in running_scheduled.decode_seq_groups) + sum(
            s.token_chunk_size for s in swapped_in.decode_seq_groups)

    min_step_tokens = _get_min_step_tokens(self.scheduler_config)
    prefill_tokens_total = count_prefill_tokens()
    while (min_step_tokens > 0
           and decode_tokens_total + prefill_tokens_total < min_step_tokens
           and budget.remaining_token_budget() > 0):
        extra_prefills = self._schedule_prefills_offline(
            budget,
            curr_loras,
            enable_chunking=True,
            partial_prefill_metadata=partial_prefill_metadata,
        )
        if not extra_prefills.seq_groups:
            break
        prefills.seq_groups.extend(extra_prefills.seq_groups)
        prefills.ignored_seq_groups.extend(extra_prefills.ignored_seq_groups)
        prefill_tokens_total = count_prefill_tokens()

    assert budget.num_batched_tokens <= (
        self.scheduler_config.max_num_batched_tokens)
    assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

    self.waiting.extendleft(running_scheduled.preempted)
    self.running.extend([s.seq_group for s in swapped_in.decode_seq_groups])
    self.running.extend([s.seq_group for s in swapped_in.prefill_seq_groups])
    self.running.extend(
        [s.seq_group for s in running_scheduled.decode_seq_groups])
    self.running.extend(
        self._order_finishing_prefills_first(
            running_scheduled.prefill_seq_groups))
    self.running.extend([s.seq_group for s in prefills.seq_groups])
    self.swapped.extend(running_scheduled.swapped_out)

    scheduled_seq_groups = (
        prefills.seq_groups + running_scheduled.prefill_seq_groups +
        swapped_in.prefill_seq_groups + running_scheduled.decode_seq_groups +
        swapped_in.decode_seq_groups)
    num_prefill_groups = (
        len(prefills.seq_groups) + len(swapped_in.prefill_seq_groups) +
        len(running_scheduled.prefill_seq_groups))
    all_prefills = len(scheduled_seq_groups) == num_prefill_groups
    num_lookahead_slots = (
        0 if (all_prefills and not self.scheduler_config.is_multi_step) else
        running_scheduled.num_lookahead_slots)

    scheduler_outputs = scheduler_mod.SchedulerOutputs(
        scheduled_seq_groups=scheduled_seq_groups,
        num_prefill_groups=num_prefill_groups,
        num_batched_tokens=budget.num_batched_tokens +
        budget.num_cached_tokens,
        num_cached_tokens=budget.num_cached_tokens,
        blocks_to_swap_in=swapped_in.blocks_to_swap_in,
        blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
        blocks_to_copy=running_scheduled.blocks_to_copy +
        swapped_in.blocks_to_copy,
        ignored_seq_groups=prefills.ignored_seq_groups +
        swapped_in.infeasible_seq_groups,
        num_lookahead_slots=num_lookahead_slots,
        running_queue_size=len(self.running),
        preempted=(len(running_scheduled.preempted) +
                   len(running_scheduled.swapped_out)),
    )
    _record_scheduler_output(scheduler_outputs, self.scheduler_config)
    return scheduler_outputs
