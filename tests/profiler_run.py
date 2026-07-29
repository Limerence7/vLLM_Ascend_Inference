
import sys
import os
import gc
import json
import pandas as pd
import csv
import time
import torch
import torch_npu
from pathlib import Path
from vllm import LLM, SamplingParams

ROOT_DIR = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT_DIR / "load_records" / "vllm_profile"

Inference_Config = {
    "Qwen3-30B-A3B": {
        "model_path": "/workspace/models/Qwen3-30B-A3B",
        "batch_size": 256,
        "max_length": 2048,
        "max_new_tokens": 128,
        "world_size": 4,
        "utilization": 0.85,
    },
    "Qwen3-235B-A22B": {
        "model_path": "/workspace/models/Qwen3-235B-A22B",
        "batch_size": 512,
        "max_length": 32,
        "max_new_tokens": 32,
        "world_size": 8,
        "utilization": 0.9,
    },
    "Qwen3-235B-A22B-W8A8": {
        "model_path": "/workspace/models/Qwen3-235B-A22B-W8A8",
        "batch_size": 512,
        "max_length": 32,
        "max_new_tokens": 5120,
        "world_size": 8,
        "utilization": 0.80,
    },
}

current_config = Inference_Config["Qwen3-30B-A3B"]

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

def run_warmup(llm, batch_user_inputs, sampling_params):
    warmup_prompts = batch_user_inputs[:min(4, len(batch_user_inputs))]
    llm.generate(
        warmup_prompts, 
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    print("Warmup completed.")
    
def run_profile(llm, batch_user_inputs, sampling_params):
    print(f"Starting worker profiling: {PROFILE_DIR}")
    llm.start_profile()

    try:
        outputs = llm.generate(
            batch_user_inputs,
            sampling_params,
            use_tqdm=True,
        )
    finally:
        torch.npu.synchronize()
        # 必须执行 stop_profile
        llm.stop_profile()

    print("Worker profiling completed.")
    return outputs

def shutdown_llm(llm):
    try:
        llm.llm_engine.engine_core.shutdown()
    finally:
        del llm
        gc.collect()

if __name__ == "__main__":
    
    print("Loading model...")

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=current_config["max_new_tokens"]
    )

    llm = LLM(
        model=current_config["model_path"],
        tensor_parallel_size=current_config["world_size"],
        enable_expert_parallel=True,
        trust_remote_code=True,
        gpu_memory_utilization=current_config["utilization"],
        max_model_len=current_config["max_length"] + current_config["max_new_tokens"],
        dtype="bfloat16",
        # quantization='ascend',
        enforce_eager=True,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": str(PROFILE_DIR),

            "torch_profiler_record_shapes": False,
            "torch_profiler_with_memory": False,
            "torch_profiler_with_stack": False,
            "torch_profiler_with_flops": False,
            "torch_profiler_use_gzip": False,
        },
    )

    tokenizer = llm.get_tokenizer()
    jsonl_path = "/workspace/Huawei/datasets/computer_en_26k.jsonl"
    combined_list = load_contents_from_jsonl(
        jsonl_path, 
        tokenizer,
        current_config["batch_size"],
        current_config["max_length"]
    )

    encoded = tokenizer(
        combined_list,
        add_special_tokens=False,
        truncation=True,
        max_length=current_config["max_length"],
        padding=False,
        return_attention_mask=False,
    )

    batch_user_inputs = tokenizer.batch_decode(
        encoded["input_ids"],
        skip_special_tokens=True,
    )
    try:
        run_warmup(llm, batch_user_inputs, sampling_params)
        
        outputs = run_profile(llm, batch_user_inputs, sampling_params)
        for output in outputs:
            prompt = output.prompt
            response = output.outputs[0].text
            print(f"Response is:\n {response}\n")
            break
    except Exception as e:  
        print(f"Error during profiling: {e}")

