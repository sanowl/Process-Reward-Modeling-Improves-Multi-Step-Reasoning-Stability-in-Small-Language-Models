"""
Evaluate reasoning consistency across paraphrased prompts.
This is the core metric for the paper's research question.
"""

import re
import numpy as np
from typing import List, Dict
from collections import Counter
from itertools import combinations


class ConsistencyEvaluator:
    """
    Measures how consistently a model arrives at the same answer
    when given semantically equivalent prompts.
    """

    def compute_agreement_rate(self, answers: List[str]) -> float:
        """
        Fraction of answer pairs that agree.
        For n answers, compute pairwise agreement over C(n,2) pairs.
        """
        if len(answers) < 2:
            return 1.0

        n_pairs = 0
        n_agree = 0

        for a1, a2 in combinations(answers, 2):
            n_pairs += 1
            if self._normalize(a1) == self._normalize(a2):
                n_agree += 1

        return n_agree / n_pairs if n_pairs > 0 else 1.0

    def compute_majority_fraction(self, answers: List[str]) -> float:
        """Fraction of answers matching the most common answer."""
        if not answers:
            return 0.0
        normalized = [self._normalize(a) for a in answers]
        counter = Counter(normalized)
        most_common_count = counter.most_common(1)[0][1]
        return most_common_count / len(answers)

    def compute_entropy(self, answers: List[str]) -> float:
        """Shannon entropy of the answer distribution (lower = more consistent)."""
        if not answers:
            return 0.0
        normalized = [self._normalize(a) for a in answers]
        counter = Counter(normalized)
        total = len(normalized)
        probs = [count / total for count in counter.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    def compute_consistency_metrics(
        self, per_question_answers: List[List[str]]
    ) -> Dict[str, float]:
        """
        Compute all consistency metrics over a set of questions.

        Args:
            per_question_answers: For each question, a list of answers
                from different paraphrased prompts.

        Returns:
            Dictionary of aggregate consistency metrics.
        """
        if not per_question_answers:
            return {
                "mean_agreement_rate": 0.0,
                "std_agreement_rate": 0.0,
                "mean_majority_fraction": 0.0,
                "std_majority_fraction": 0.0,
                "mean_entropy": 0.0,
                "std_entropy": 0.0,
                "perfect_consistency_rate": 0.0,
                "num_questions": 0,
            }

        agreement_rates = []
        majority_fracs = []
        entropies = []

        for answers in per_question_answers:
            agreement_rates.append(self.compute_agreement_rate(answers))
            majority_fracs.append(self.compute_majority_fraction(answers))
            entropies.append(self.compute_entropy(answers))

        return {
            "mean_agreement_rate": float(np.mean(agreement_rates)),
            "std_agreement_rate": float(np.std(agreement_rates)),
            "mean_majority_fraction": float(np.mean(majority_fracs)),
            "std_majority_fraction": float(np.std(majority_fracs)),
            "mean_entropy": float(np.mean(entropies)),
            "std_entropy": float(np.std(entropies)),
            "perfect_consistency_rate": float(np.mean(
                [1.0 if ar == 1.0 else 0.0 for ar in agreement_rates]
            )),
            "num_questions": len(per_question_answers),
        }

    def _normalize(self, answer: str) -> str:
        """Normalize answer for comparison."""
        s = answer.lower().strip()
        s = re.sub(r"[,$%]", "", s)
        s = re.sub(r"\s+", " ", s)
        try:
            return str(float(s))
        except ValueError:
            return s
