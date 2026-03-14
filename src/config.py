"""
Configuration for PRM training and evaluation experiments.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_name: str = "microsoft/phi-2"
    tokenizer_name: Optional[str] = None
    max_length: int = 1024
    use_4bit: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )


@dataclass
class PRMConfig:
    """Process Reward Model configuration."""
    reward_model_name: str = "microsoft/phi-2"
    step_separator: str = "\n\n"
    positive_label: int = 1
    negative_label: int = 0
    reward_aggregation: str = "min"  # min, mean, last, product
    num_reward_classes: int = 2
    hidden_size: Optional[int] = None  # Auto-detected from model if None
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = True
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    seed: int = 42
    wandb_project: str = "prm-reasoning-stability"


@dataclass
class DataConfig:
    dataset_name: str = "gsm8k"
    train_split: str = "train"
    test_split: str = "test"
    num_reasoning_steps: int = 8
    max_samples: Optional[int] = None
    num_paraphrases: int = 5  # prompts per question for consistency testing


@dataclass
class EvalConfig:
    benchmarks: list = field(
        default_factory=lambda: ["gsm8k", "math", "arc"]
    )
    num_samples_per_prompt: int = 10
    temperature: float = 0.7
    top_p: float = 0.95
    num_beams: int = 1
    consistency_metric: str = "agreement_rate"
    best_of_n: int = 8  # for best-of-n with PRM scoring
