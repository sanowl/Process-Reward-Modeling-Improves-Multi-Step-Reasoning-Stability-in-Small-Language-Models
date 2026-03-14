"""
Benchmark evaluation: accuracy on GSM8K, MATH, and ARC.
"""

import torch
import numpy as np
from collections import Counter
from typing import List, Dict, Optional
from tqdm import tqdm

from ..data.reasoning_traces import (
    generate_reasoning_traces,
    extract_final_answer,
    _answers_match,
)


class BenchmarkEvaluator:
    """Evaluate model accuracy and reasoning quality on benchmarks."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate_accuracy(
        self,
        dataset,
        max_samples: Optional[int] = None,
        num_samples: int = 1,
        temperature: float = 0.0,
    ) -> Dict:
        """
        Evaluate accuracy on a benchmark dataset.

        Args:
            dataset: HF dataset with 'question' and 'answer' fields
            max_samples: limit evaluation to N samples
            num_samples: number of generations per question (for pass@k)
            temperature: sampling temperature (0 = greedy)
        """
        questions = dataset["question"]
        gold_answers = dataset["answer"]

        if max_samples:
            questions = questions[:max_samples]
            gold_answers = gold_answers[:max_samples]

        correct = 0
        total = 0
        results = []

        for q, gold in tqdm(zip(questions, gold_answers), total=len(questions)):
            traces = generate_reasoning_traces(
                self.model,
                self.tokenizer,
                [q],
                num_samples=num_samples,
                temperature=temperature,
                device=self.device,
            )[0]

            # Majority vote if multiple samples
            answers = [t["final_answer"] for t in traces]
            most_common = Counter(answers).most_common(1)[0][0]

            is_correct = _answers_match(most_common, gold)
            correct += int(is_correct)
            total += 1

            results.append({
                "question": q,
                "gold_answer": gold,
                "predicted_answer": most_common,
                "all_answers": answers,
                "correct": is_correct,
            })

        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "results": results,
        }

    def evaluate_with_prm(
        self,
        dataset,
        prm,
        prm_tokenizer,
        num_samples: int = 8,
        max_samples: Optional[int] = None,
        aggregation: str = "min",
    ) -> Dict:
        """
        Evaluate using PRM-guided best-of-N selection.
        Generate N traces, score with PRM, select highest-scored.
        """
        from .best_of_n import BestOfNSelector

        selector = BestOfNSelector(prm, prm_tokenizer, aggregation=aggregation)

        questions = dataset["question"]
        gold_answers = dataset["answer"]

        if max_samples:
            questions = questions[:max_samples]
            gold_answers = gold_answers[:max_samples]

        correct = 0
        total = 0
        results = []

        for q, gold in tqdm(zip(questions, gold_answers), total=len(questions)):
            traces = generate_reasoning_traces(
                self.model,
                self.tokenizer,
                [q],
                num_samples=num_samples,
                temperature=0.7,
                device=self.device,
            )[0]

            trace_texts = [t["full_text"] for t in traces]
            best_idx, scores = selector.select_best(trace_texts)
            best_answer = traces[best_idx]["final_answer"]

            is_correct = _answers_match(best_answer, gold)
            correct += int(is_correct)
            total += 1

            results.append({
                "question": q,
                "gold_answer": gold,
                "predicted_answer": best_answer,
                "scores": scores,
                "correct": is_correct,
            })

        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "results": results,
            "method": f"best_of_{num_samples}_prm",
        }
