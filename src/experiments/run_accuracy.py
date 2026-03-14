"""
Accuracy experiments: Compare baseline vs PRM-guided accuracy on benchmarks.
"""

import json
import os
import torch
from typing import Dict

from ..config import ModelConfig, PRMConfig, EvalConfig, DataConfig
from ..models.base_model import load_base_model, load_tokenizer
from ..models.prm import ProcessRewardModel
from ..data.dataset_loader import load_benchmark_dataset
from ..evaluation.benchmarks import BenchmarkEvaluator


def run_accuracy_experiment(
    model_config: ModelConfig = None,
    prm_config: PRMConfig = None,
    eval_config: EvalConfig = None,
    data_config: DataConfig = None,
    prm_checkpoint: str = None,
    output_dir: str = "./results/accuracy",
) -> Dict:
    """Run accuracy evaluation with and without PRM."""
    model_config = model_config or ModelConfig()
    prm_config = prm_config or PRMConfig()
    eval_config = eval_config or EvalConfig()
    data_config = data_config or DataConfig()

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading base model...")
    model = load_base_model(model_config)
    tokenizer = load_tokenizer(model_config)
    bench_eval = BenchmarkEvaluator(model, tokenizer, device=device)

    prm = None
    prm_tokenizer = None

    if prm_checkpoint:
        print("Loading PRM...")
        prm = ProcessRewardModel(prm_config)
        prm.load_state_dict(
            torch.load(prm_checkpoint, map_location=device, weights_only=True)
        )
        prm.to(device)
        prm.eval()
        prm_tokenizer = load_tokenizer(
            ModelConfig(model_name=prm_config.reward_model_name)
        )

    all_results = {}

    for benchmark in eval_config.benchmarks:
        print(f"\n{'='*60}")
        print(f"Evaluating accuracy on {benchmark.upper()}")
        print(f"{'='*60}")

        dataset = load_benchmark_dataset(
            benchmark, "test", max_samples=data_config.max_samples
        )

        # Baseline: greedy decoding
        print("Running baseline (greedy)...")
        baseline = bench_eval.evaluate_accuracy(
            dataset, temperature=0.0
        )

        # Baseline: majority vote with sampling
        print("Running majority vote...")
        majority = bench_eval.evaluate_accuracy(
            dataset,
            num_samples=eval_config.best_of_n,
            temperature=eval_config.temperature,
        )

        result = {
            "benchmark": benchmark,
            "greedy": {"accuracy": baseline["accuracy"], "total": baseline["total"]},
            "majority_vote": {"accuracy": majority["accuracy"], "total": majority["total"]},
        }

        # PRM best-of-N
        if prm:
            print("Running PRM best-of-N...")
            prm_result = bench_eval.evaluate_with_prm(
                dataset, prm, prm_tokenizer,
                num_samples=eval_config.best_of_n,
            )
            result["prm_best_of_n"] = {
                "accuracy": prm_result["accuracy"],
                "total": prm_result["total"],
            }

        all_results[benchmark] = result

        with open(os.path.join(output_dir, f"{benchmark}_accuracy.json"), "w") as f:
            json.dump(result, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("ACCURACY EXPERIMENT SUMMARY")
    print("=" * 60)
    for bm, r in all_results.items():
        print(f"\n{bm.upper()}:")
        print(f"  Greedy:        {r['greedy']['accuracy']:.4f}")
        print(f"  Majority Vote: {r['majority_vote']['accuracy']:.4f}")
        if "prm_best_of_n" in r:
            print(f"  PRM Best-of-N: {r['prm_best_of_n']['accuracy']:.4f}")

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results
