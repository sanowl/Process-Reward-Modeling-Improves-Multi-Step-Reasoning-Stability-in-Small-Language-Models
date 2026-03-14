"""
Dataset loading utilities for GSM8K, MATH, and ARC benchmarks.
"""

import re
from datasets import load_dataset
from typing import Optional


def load_benchmark_dataset(benchmark: str, split: str = "test", max_samples: Optional[int] = None):
    """Load a benchmark dataset with standardized format."""
    loaders = {
        "gsm8k": _load_gsm8k,
        "math": _load_math,
        "arc": _load_arc,
    }
    if benchmark not in loaders:
        raise ValueError(f"Unknown benchmark: {benchmark}. Choose from {list(loaders.keys())}")
    dataset = loaders[benchmark](split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def load_prm_training_data(max_samples: Optional[int] = None):
    """Load PRM800K-style training data for process reward model."""
    dataset = load_dataset("openai/prm800k", "phase2_train", split="train", trust_remote_code=True)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def _load_gsm8k(split: str):
    """Load GSM8K with standardized fields."""
    dataset = load_dataset("openai/gsm8k", "main", split=split, trust_remote_code=True)

    def _process(example):
        answer_text = example["answer"]
        # Extract the final numeric answer after ####
        parts = answer_text.split("####")
        if len(parts) >= 2:
            final_answer = parts[-1].strip()
            reasoning = parts[0].strip()
        else:
            final_answer = ""
            reasoning = answer_text.strip()
        return {
            "question": example["question"],
            "reasoning": reasoning,
            "answer": final_answer,
            "benchmark": "gsm8k",
        }

    return dataset.map(_process)


def _extract_boxed(text: str) -> Optional[str]:
    """Extract content from \\boxed{...}, handling nested braces."""
    idx = text.find("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1] if depth == 0 else None


def _load_math(split: str):
    """Load MATH dataset with standardized fields."""
    dataset = load_dataset("hendrycks/competition_math", split=split, trust_remote_code=True)

    def _process(example):
        solution = example["solution"]
        # Extract boxed answer with nested brace support
        final_answer = _extract_boxed(solution)
        if final_answer is None:
            final_answer = solution.split("\n")[-1]
        return {
            "question": example["problem"],
            "reasoning": solution,
            "answer": final_answer,
            "benchmark": "math",
            "level": example.get("level", ""),
            "type": example.get("type", ""),
        }

    return dataset.map(_process)


def _load_arc(split: str):
    """Load ARC-Challenge with standardized fields."""
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split, trust_remote_code=True)

    def _process(example):
        choices = example["choices"]
        labels = choices["label"]
        texts = choices["text"]
        choice_str = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
        return {
            "question": f"{example['question']}\n\nChoices:\n{choice_str}",
            "reasoning": "",
            "answer": example["answerKey"],
            "benchmark": "arc",
        }

    return dataset.map(_process)
