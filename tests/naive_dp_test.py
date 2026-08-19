import contextlib
import gc
import os
import json
import torch

from time import sleep
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (  # noqa E402
    destroy_distributed_environment, destroy_model_parallel)
from vllm.utils.network_utils import get_open_port

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

Inference_Config = {
    "Qwen3-30B-A3B": {
        "model_path": "/workspace/models/Qwen3-30B-A3B",
        "batch_size": 1024,
        "max_length": 2048,
        "max_new_tokens": 512,
        "world_size": 2,
        "utilization": 0.85,
    },
    "Qwen3-235B-A22B": {
        "model_path": "/workspace/models/Qwen3-235B-A22B",
        "batch_size": 512,
        "max_length": 512,
        "max_new_tokens": 512,
        "world_size": 8,
        "utilization": 0.98,
    },
    "Qwen3-235B-A22B-W8A8": {
        "model_path": "/workspace/models/Qwen3-235B-A22B-W8A8",
        "batch_size": 256,
        "max_length": 1024,
        "max_new_tokens": 128,
        "world_size": 4,
        "utilization": 0.98,
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

def cleanup_env_and_memory():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

def main(
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    model_path,
    max_length,
    max_new_tokens,
    tp_size,
    utilization,
    prompts,
):
    # DP only support on V1 engine
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # with DP, each rank should process different prompts.
    # usually all the DP ranks process a full dataset,
    # and each rank processes a different part of the dataset.
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    # Distribute prompts into even groups.
    def start(rank):
        return rank * floor + min(rank, remainder)

    prompts = prompts[start(global_dp_rank):start(global_dp_rank + 1)]
    if len(prompts) == 0:
        # if any rank has no prompts to process,
        # we need to set a placeholder prompt
        prompts = ["Placeholder"]
    print(f"DP rank {global_dp_rank} needs to process {len(prompts)} prompts")

    sampling_params = SamplingParams(temperature=0.7,
                                     top_p=0.9,
                                     max_tokens=max_new_tokens)

    # Create an LLM.
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        max_model_len=max_length+max_new_tokens,
        gpu_memory_utilization=utilization,
        dtype="bfloat16",
        enforce_eager=True,
        enable_expert_parallel=True,
        trust_remote_code=True,
        quantization=None,
    )
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"DP rank {global_dp_rank}, Prompt: {prompt!r}, "
              f"Generated text: {generated_text!r}")
        break

    # Give engines time to pause their processing loops before exiting.
    sleep(5)
    del llm
    cleanup_env_and_memory()

if __name__ == "__main__":

    dp_size = 2
    node_size = 1
    node_rank = 0
    dp_per_node = dp_size // node_size
    
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()
    
    tokenizer = AutoTokenizer.from_pretrained(current_config["model_path"])
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

    prompts = tokenizer.batch_decode(
        encoded["input_ids"],
        skip_special_tokens=True,
    )

    from multiprocessing import Process
    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
            range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)):
        proc = Process(
            target=main,
            args=(
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                current_config["model_path"],
                current_config["max_length"],
                current_config["max_new_tokens"],
                current_config["world_size"],
                current_config["utilization"], 
                prompts,
            ),
        )
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join(timeout=900)
        if proc.exitcode is None:
            print(
                f"Killing process {proc.pid} that didn't stop within 15 minutes."
            )
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)