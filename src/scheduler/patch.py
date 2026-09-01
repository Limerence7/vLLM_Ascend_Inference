from __future__ import annotations

from .core import (begin_worker_activation_batch, clear_request_activation,
                   configure_worker_activation,
                   derive_scheduler_parameters,
                   discard_worker_activation_requests,
                   finish_worker_activation_batch,
                   record_request_activation, reorder_requests)


_PATCHED = False
_ORIGINAL_SCHEDULE = None
_ORIGINAL_UPDATE_FROM_OUTPUT = None
_ORIGINAL_PREPARE_INPUTS = None
_ORIGINAL_SAMPLE_TOKENS = None
_RUNTIME_CONFIG = None
_ACTIVATION_OUTPUT_ATTR = "_vllm_ascend_request_activations"


def apply_scheduler_patch(runtime_config) -> None:
    """Install the request scheduler on the current vLLM V1 Scheduler."""

    global _PATCHED, _ORIGINAL_SCHEDULE, _ORIGINAL_UPDATE_FROM_OUTPUT
    global _ORIGINAL_PREPARE_INPUTS, _ORIGINAL_SAMPLE_TOKENS, _RUNTIME_CONFIG
    _RUNTIME_CONFIG = runtime_config
    configure_worker_activation(runtime_config)
    if _PATCHED:
        return

    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    _ORIGINAL_SCHEDULE = Scheduler.schedule
    _ORIGINAL_UPDATE_FROM_OUTPUT = Scheduler.update_from_output
    Scheduler.schedule = _schedule
    Scheduler.update_from_output = _update_from_output
    Scheduler._vllm_ascend_scheduler_enabled = True

    _ORIGINAL_PREPARE_INPUTS = NPUModelRunner._prepare_inputs
    _ORIGINAL_SAMPLE_TOKENS = NPUModelRunner.sample_tokens
    NPUModelRunner._prepare_inputs = _prepare_inputs
    NPUModelRunner.sample_tokens = _sample_tokens
    _PATCHED = True


def is_scheduler_enabled() -> bool:
    from vllm.v1.core.sched.scheduler import Scheduler

    return bool(getattr(Scheduler, "_vllm_ascend_scheduler_enabled", False))


def _schedule(self):
    parameters = _parameters(self.scheduler_config)
    policy = getattr(getattr(self, "policy", None), "value", "fcfs")
    if parameters.policy != "fifo" and policy == "fcfs":
        _reorder_running(self, parameters)
        _reorder_waiting(self, parameters)
    return _ORIGINAL_SCHEDULE(self)


def _update_from_output(self, scheduler_output, model_runner_output):
    payload = getattr(model_runner_output, _ACTIVATION_OUTPUT_ATTR, None) or {}
    parameters = _parameters(self.scheduler_config)
    for request_id, activation in payload.items():
        record_request_activation(
            request_id,
            activation,
            max_experts=parameters.max_profile_experts,
        )

    outputs = _ORIGINAL_UPDATE_FROM_OUTPUT(
        self, scheduler_output, model_runner_output)
    finished = set(scheduler_output.finished_req_ids)
    finished.update(request_id for request_id in payload
                    if request_id not in self.requests)
    for request_id in finished:
        clear_request_activation(request_id)
    return outputs


def _prepare_inputs(self, scheduler_output, *args, **kwargs):
    prepared = _ORIGINAL_PREPARE_INPUTS(
        self, scheduler_output, *args, **kwargs)
    discard_worker_activation_requests(scheduler_output.finished_req_ids)
    # vLLM-Ascend 0.18 returns total_num_scheduled_tokens at index 2;
    # per-request counts are the second _prepare_inputs argument.
    token_counts = args[0] if args else kwargs["num_scheduled_tokens"]
    begin_worker_activation_batch(self.input_batch.req_ids, token_counts)
    return prepared


def _sample_tokens(self, *args, **kwargs):
    output = _ORIGINAL_SAMPLE_TOKENS(self, *args, **kwargs)
    payload = finish_worker_activation_batch()
    target = getattr(
        output,
        "_model_runner_output",
        getattr(output, "model_runner_output", output),
    )
    if payload and target is not None:
        setattr(target, _ACTIVATION_OUTPUT_ATTR, payload)
    return output


def _parameters(scheduler_config):
    cached = getattr(scheduler_config,
                     "_vllm_ascend_scheduler_parameters", None)
    if cached is None:
        cached = derive_scheduler_parameters(_RUNTIME_CONFIG, scheduler_config)
        setattr(scheduler_config, "_vllm_ascend_scheduler_parameters", cached)
    return cached


def _reorder_running(scheduler, parameters) -> None:
    window = min(len(scheduler.running), parameters.reorder_window)
    if window <= 1:
        return
    scheduler.running[:window] = reorder_requests(
        list(scheduler.running[:window]), parameters, waiting=False)


def _reorder_waiting(scheduler, parameters) -> None:
    window = min(len(scheduler.waiting), parameters.reorder_window)
    if window <= 1:
        return
    candidates = []
    for index, request in enumerate(scheduler.waiting):
        if index >= window:
            break
        candidates.append(request)
    ordered = reorder_requests(candidates, parameters, waiting=True)
    scheduler.waiting.remove_requests(candidates)
    for request in reversed(ordered):
        scheduler.waiting.prepend_request(request)
