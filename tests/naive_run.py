import sys
import os
import json
import pandas as pd
import csv
import random
import math

from vllm import LLM, SamplingParams

Inference_Config = {
    "Qwen3-30B-A3B": {
        "model_path": "/workspace/models/Qwen3-30B-A3B",
        "batch_size": 1024,
        "max_length": 2048,
        "max_new_tokens": 128,
        "world_size": 4,
        "utilization": 0.85,
    },
    "Qwen3-235B-A22B": {
        "model_path": "/workspace/models/Qwen3-235B-A22B",
        "batch_size": 512,
        "max_length": 1024,
        "max_new_tokens": 512,
        "world_size": 8,
        "utilization": 0.98,
    },
    "Qwen3-235B-A22B-W8A8": {
        "model_path": "/workspace/models/Qwen3-235B-A22B-W8A8",
        "batch_size": 512,
        "max_length": 1024,
        "max_new_tokens": 128,
        "world_size": 4,
        "utilization": 0.98,
    },
}

current_config = Inference_Config["Qwen3-235B-A22B-W8A8"]

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
        # dtype="bfloat16",
        quantization='ascend',
        enforce_eager=True,
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

    outputs = llm.generate(batch_user_inputs, sampling_params)

    for output in outputs:
        prompt = output.prompt
        response = output.outputs[0].text
        print(f"Response is:\n {response}\n")
        break

