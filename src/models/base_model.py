"""
Load and configure base language models for reasoning and PRM training.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from ..config import ModelConfig


def load_base_model(config: ModelConfig, device_map: str = "auto"):
    """Load a base causal LM with optional quantization and LoRA."""
    quant_config = None
    if config.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quant_config,
        device_map=device_map,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=None if quant_config else torch.float16,
    )

    if getattr(model.config, "pad_token_id", None) is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
        if eos_token_id is not None:
            model.config.pad_token_id = eos_token_id
            if hasattr(model, "generation_config"):
                model.generation_config.pad_token_id = eos_token_id

    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)

    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


def load_tokenizer(config: ModelConfig):
    """Load tokenizer with proper padding configuration."""
    tokenizer_name = config.tokenizer_name or config.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=config.trust_remote_code
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"
    return tokenizer
