from typing import List, Tuple
import torch


class BestOfNSelector:
    """Select the best reasoning trace from N candidates using PRM."""

    def __init__(self, prm, tokenizer, aggregation: str = "min", device: str = "cuda"):
        self.prm = prm
        self.tokenizer = tokenizer
        self.aggregation = aggregation
        self.device = device

    def select_best(self, traces: List[str]) -> Tuple[int, List[float]]:
        """
        Score each trace and return the index of the best one.

        Returns:
            (best_index, list_of_scores)
        """
        if not traces:
            return 0, []

        scores = []
        for trace in traces:
            step_scores = self.prm.score_steps(
                trace, self.tokenizer, device=self.device
            )
            agg_score = self.prm.aggregate_score(step_scores, method=self.aggregation)
            scores.append(agg_score)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return best_idx, scores

    def select_best_batch(
        self, all_traces: List[List[str]]
    ) -> List[Tuple[int, List[float]]]:
        """Select best trace for each question from a batch."""
        return [self.select_best(traces) for traces in all_traces]
