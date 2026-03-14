"""
Main experiment: Does PRM improve reasoning consistency across paraphrased prompts?

Protocol:
1. For each question, generate N paraphrased prompts
2. For each prompt, generate K reasoning traces
3. Baseline: majority vote per prompt
4. PRM: best-of-K selection per prompt using PRM scores
5. Compare consistency of final answers across prompts
"""

import json
import os
import torch
from collections import Counter
from typing import Dict
from tqdm import tqdm

from ..config import ModelConfig, PRMConfig, EvalConfig, DataConfig
from ..models.base_model import load_base_model, load_tokenizer
from ..models.prm import ProcessRewardModel
from ..data.dataset_loader import load_benchmark_dataset
from ..data.paraphraser import generate_paraphrases
from ..data.reasoning_traces import generate_reasoning_traces, extract_final_answer
from ..evaluation.consistency import ConsistencyEvaluator
from ..evaluation.best_of_n import BestOfNSelector


def run_consistency_experiment(
    model_config: ModelConfig = None,
    prm_config: PRMConfig = None,
    eval_config: EvalConfig = None,
    data_config: DataConfig = None,
    prm_checkpoint: str = None,
    output_dir: str = "./results/consistency",
) -> Dict:
    """Run the full consistency experiment."""
    model_config = model_config or ModelConfig()
    prm_config = prm_config or PRMConfig()
    eval_config = eval_config or EvalConfig()
    data_config = data_config or DataConfig()

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    print("Loading base model...")
    model = load_base_model(model_config)
    tokenizer = load_tokenizer(model_config)

    prm = None
    prm_tokenizer = None
    selector = None

    if prm_checkpoint:
        print("Loading PRM...")
        prm = ProcessRewardModel(prm_config)
        prm.load_state_dict(
            torch.load(prm_checkpoint, map_location=device, weights_only=True)
        )
        prm.to(device)
        prm.eval()
        prm_tokenizer = load_tokenizer(
            ModelConfig(
                model_name=prm_config.reward_model_name,
                trust_remote_code=prm_config.trust_remote_code,
            )
        )
        selector = BestOfNSelector(
            prm,
            prm_tokenizer,
            aggregation=prm_config.reward_aggregation,
            device=device,
        )

    evaluator = ConsistencyEvaluator()
    all_results = {}

    for benchmark in eval_config.benchmarks:
        print(f"\n{'='*60}")
        print(f"Evaluating consistency on {benchmark.upper()}")
        print(f"{'='*60}")

        dataset = load_benchmark_dataset(
            benchmark, "test", max_samples=data_config.max_samples
        )

        baseline_answers_per_q = []
        prm_answers_per_q = []
        detailed_results = []

        for idx in tqdm(range(len(dataset)), desc=benchmark):
            example = dataset[idx]
            question = example["question"]
            gold = example["answer"]

            # Generate paraphrased prompts
            prompts = generate_paraphrases(
                question, benchmark, data_config.num_paraphrases, seed=idx
            )

            baseline_answers = []
            prm_answers = []

            for prompt in prompts:
                # Generate multiple traces per prompt
                traces = generate_reasoning_traces(
                    model, tokenizer, [prompt],
                    num_samples=eval_config.best_of_n,
                    temperature=eval_config.temperature,
                    device=device,
                )[0]

                # Baseline: majority vote
                answers = [t["final_answer"] for t in traces]
                majority = Counter(answers).most_common(1)[0][0]
                baseline_answers.append(majority)

                # PRM: best-of-N
                if selector:
                    try:
                        trace_texts = [t["full_text"] for t in traces]
                        best_idx, scores = selector.select_best(trace_texts)
                        prm_answers.append(traces[best_idx]["final_answer"])
                    except Exception as e:
                        print(f"  Warning: PRM scoring failed for question {idx}: {e}")
                        prm_answers.append(majority)

            baseline_answers_per_q.append(baseline_answers)
            if prm_answers:
                prm_answers_per_q.append(prm_answers)

            detailed_results.append({
                "question": question,
                "gold_answer": gold,
                "baseline_answers": baseline_answers,
                "prm_answers": prm_answers,
            })

        # Compute consistency metrics
        baseline_metrics = evaluator.compute_consistency_metrics(baseline_answers_per_q)
        baseline_metrics["method"] = "majority_vote"

        result = {
            "benchmark": benchmark,
            "baseline": baseline_metrics,
            "detailed": detailed_results,
        }

        if prm_answers_per_q:
            prm_metrics = evaluator.compute_consistency_metrics(prm_answers_per_q)
            prm_metrics["method"] = "prm_best_of_n"
            prm_metrics["aggregation"] = prm_config.reward_aggregation
            result["prm"] = prm_metrics

            # Improvement (negative entropy_delta = improvement)
            result["improvement"] = {
                "agreement_rate_delta": (
                    prm_metrics["mean_agreement_rate"]
                    - baseline_metrics["mean_agreement_rate"]
                ),
                "entropy_delta": (
                    prm_metrics["mean_entropy"] - baseline_metrics["mean_entropy"]
                ),
            }

        all_results[benchmark] = result

        # Save per-benchmark results
        with open(os.path.join(output_dir, f"{benchmark}_consistency.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)

    # Save summary
    summary = {
        bm: {
            "baseline_agreement": r["baseline"]["mean_agreement_rate"],
            "prm_agreement": r.get("prm", {}).get("mean_agreement_rate", None),
            "improvement": r.get("improvement", {}).get("agreement_rate_delta", None),
        }
        for bm, r in all_results.items()
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("CONSISTENCY EXPERIMENT SUMMARY")
    print("=" * 60)
    for bm, s in summary.items():
        print(f"\n{bm.upper()}:")
        print(f"  Baseline agreement rate: {s['baseline_agreement']:.4f}")
        if s["prm_agreement"] is not None:
            print(f"  PRM agreement rate:      {s['prm_agreement']:.4f}")
            print(f"  Improvement:             {s['improvement']:+.4f}")

    return all_results
