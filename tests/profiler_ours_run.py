import src
import json
import torch
import torch_npu

from pathlib import Path
from src.runtime_config import RuntimeConfig
from vllm import LLM, SamplingParams


LOAD_HISTORY_PATH = (
    Path(__file__).resolve().parents[1]
    / "load_records"
    / "only_run_load_history"
)
AUDIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "load_records"
    / "only_run_audit_path"
)

TEST_CONFIG = {
    "model_path": "/workspace/models/Qwen3-30B-A3B",
    "batch_size": 768,
    "max_length": 2048,
    "max_new_tokens": 512,
    "world_size": 4,
    "utilization": 0.85,
}

RUNTIME_CONFIG = RuntimeConfig(
    runtime_mode="offload",
    interval=16,
    num_buffers=2,
    num_runtime_experts=-2,
    cpu_pin_memory=True,
    runtime_layer_ids=[],
    load_history_path=str(LOAD_HISTORY_PATH),
    enable_offline_scheduler=False,
    enable_history_mapping=False,
)

def load_contents_from_jsonl(jsonl_path, tokenizer, batch_size, max_length):
    text = ""
    num_seqs = 0
    contents = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            conversation = data.get('conversation', None)
            if conversation is None:
                continue
            for conver in conversation:
                human = conver.get("human", "")
                assistant = conver.get("assistant", "")
                text += (human + assistant)
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
            )
            if len(encoded["input_ids"]) >= max_length:
                contents.append(text)
                text = ""
                num_seqs += 1
            if num_seqs >= batch_size:
                break
        
    return contents


def build_prompts(batch_size: int, max_length: int, tokenizer) -> list[str]:
    jsonl_path = "/workspace/Huawei/datasets/computer_en_26k.jsonl"
    combined_list = load_contents_from_jsonl(
        jsonl_path, 
        tokenizer,
        batch_size,
        max_length
    )

    encoded = tokenizer(
        combined_list,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=False,
    )

    batch_user_inputs = tokenizer.batch_decode(
        encoded["input_ids"],
        skip_special_tokens=True,
    )

    return batch_user_inputs


def save_load_history(llm: LLM) -> None:
    if not RUNTIME_CONFIG.load_history_path:
        print("Skip load stats save.")
        return

    save_results = llm.collective_rpc(
        "save_load_history",
        timeout=120,
    )
    if not all(result.get("saved") for result in save_results):
        raise RuntimeError(f"Failed to save load history: {save_results}")
    print(f"Load history worker results: {save_results}")
    print(f"Load history saved under: {LOAD_HISTORY_PATH}")


def shutdown_llm(llm: LLM) -> None:
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    src.register_plugin(RUNTIME_CONFIG)

    print("Loading model...")
    print(f"Runtime config: {RUNTIME_CONFIG}")

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=TEST_CONFIG["max_new_tokens"],
    )

    llm = LLM(
        model=TEST_CONFIG["model_path"],
        tensor_parallel_size=TEST_CONFIG["world_size"],
        enable_expert_parallel=True,
        trust_remote_code=True,
        gpu_memory_utilization=TEST_CONFIG["utilization"],
        max_model_len=TEST_CONFIG["max_length"] + TEST_CONFIG["max_new_tokens"],
        dtype="bfloat16",
        # quantization='ascend',
        enforce_eager=True,
        worker_extension_cls="src.utils.RuntimeWorkerExtension",
    )

    batch_user_inputs = build_prompts(
        TEST_CONFIG["batch_size"], 
        TEST_CONFIG["max_length"],
        llm.get_tokenizer(),
    )
    
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=True,
        data_simplification=False,
        record_op_args=False
    )

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config
    ) as prof:
        outputs = llm.generate(batch_user_inputs, sampling_params)
        prof.step()
    prof.export_chrome_trace("./trace.json")
    
    save_load_history(llm)
    shutdown_llm(llm)

    for output in outputs:
        print(f"Prompt:\n{output.prompt}\n")
        print(f"Response:\n{output.outputs[0].text}\n")
        break
