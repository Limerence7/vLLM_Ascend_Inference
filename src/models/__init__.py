RUNTIME_MODELS = {
    "Qwen3MoeForCausalLM":
        "src.models.qwen3_moe:RuntimeQwen3MoeForCausalLM",
    "MixtralForCausalLM":
        "src.models.mixtral:RuntimeMixtralForCausalLM",
    "Qwen3_5MoeForCausalLM":
        "src.models.qwen3_5:RuntimeQwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration":
        "src.models.qwen3_5:RuntimeQwen3_5MoeForConditionalGeneration",
}

__all__ = ["RUNTIME_MODELS"]
