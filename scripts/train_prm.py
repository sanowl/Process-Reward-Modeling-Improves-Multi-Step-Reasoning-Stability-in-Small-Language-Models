"""
Script to train the Process Reward Model on reasoning traces.

Usage:
    python scripts/train_prm.py \
        --model_name Qwen/Qwen2.5-1.5B \
        --output_dir outputs/prm \
        --num_epochs 3 \
        --batch_size 4
"""

import argparse
import os
import sys
import random
import torch
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ModelConfig, PRMConfig, TrainingConfig, DataConfig
from src.models.base_model import load_tokenizer
from src.models.prm import ProcessRewardModel, StepLabelDataset, PRMTrainer
from src.data.dataset_loader import load_prm_training_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train Process Reward Model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--output_dir", type=str, default="./outputs/prm")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prepare_training_data(raw_data, tokenizer, max_length=1024):
    """Convert PRM800K data into step-labeled training format."""
    traces = []

    for example in raw_data:
        if "label" not in example or "completions" not in example:
            continue

        completions = example["completions"]
        if not isinstance(completions, list):
            continue

        steps = []
        labels = []

        for completion in completions:
            if isinstance(completion, dict):
                step_text = completion.get("text", "")
                rating = completion.get("rating", None)
                if step_text and rating is not None:
                    steps.append(step_text)
                    labels.append(1 if rating > 0 else 0)

        if steps and labels:
            traces.append({
                "steps": steps,
                "step_labels": labels,
            })

    return traces


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Seed everything for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Config
    prm_config = PRMConfig(reward_model_name=args.model_name)
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    # Load data
    print("Loading PRM training data...")
    raw_data = load_prm_training_data(max_samples=args.max_samples)

    model_config = ModelConfig(model_name=args.model_name)
    tokenizer = load_tokenizer(model_config)

    print("Preparing training examples...")
    traces = prepare_training_data(raw_data, tokenizer, args.max_length)
    print(f"Prepared {len(traces)} training traces")

    if not traces:
        raise ValueError(
            f"No valid training traces found from {len(raw_data)} raw examples. "
            "Check that the PRM800K data format matches expectations."
        )

    # Split train/eval
    split_idx = int(0.9 * len(traces))
    train_traces = traces[:split_idx]
    eval_traces = traces[split_idx:]

    train_dataset = StepLabelDataset(train_traces, tokenizer, max_length=args.max_length)
    eval_dataset = StepLabelDataset(eval_traces, tokenizer, max_length=args.max_length)

    # Initialize and train PRM
    print("Initializing PRM...")
    prm = ProcessRewardModel(prm_config)

    trainer = PRMTrainer(prm, train_dataset, eval_dataset, training_config)
    print("Starting training...")
    trained_model = trainer.train()

    # Save
    save_path = os.path.join(args.output_dir, "prm_model.pt")
    torch.save(trained_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model_name": args.model_name,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_train_traces": len(train_traces),
            "num_eval_traces": len(eval_traces),
        }, f, indent=2)


if __name__ == "__main__":
    main()
