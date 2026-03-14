"""
Run a study sweep across models, benchmarks, ablations, and PRM aggregations.

Examples:
    python scripts/run_study.py \
        --model_names Qwen/Qwen2.5-1.5B TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --prm_checkpoint_template outputs/{model_slug}/prm_model.pt \
        --best_of_n_values 4 8 16 \
        --num_paraphrases_values 3 5 8 \
        --reward_aggregations min mean last product
"""

import argparse
import json
import re
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DataConfig, EvalConfig, ModelConfig, PRMConfig
from src.experiments.run_accuracy import run_accuracy_experiment
from src.experiments.run_consistency import run_consistency_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run a PRM study sweep")
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=["Qwen/Qwen2.5-1.5B"],
        help="Base model names to evaluate.",
    )
    parser.add_argument(
        "--prm_model_name",
        type=str,
        default=None,
        help="Single PRM backbone to use for all runs. Defaults to each base model.",
    )
    parser.add_argument(
        "--prm_checkpoint",
        type=str,
        default=None,
        help="Single PRM checkpoint path to use for all runs.",
    )
    parser.add_argument(
        "--prm_checkpoint_template",
        type=str,
        default=None,
        help="Template for per-model PRM checkpoints. Supports {model_name} and {model_slug}.",
    )
    parser.add_argument(
        "--prm_checkpoint_map",
        nargs="*",
        default=[],
        help="Explicit per-model PRM checkpoint mapping: model_name=path",
    )
    parser.add_argument("--output_dir", type=str, default="./study_results")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math", "arc"])
    parser.add_argument(
        "--best_of_n_values",
        nargs="+",
        type=int,
        default=[8],
    )
    parser.add_argument(
        "--num_paraphrases_values",
        nargs="+",
        type=int,
        default=[5],
    )
    parser.add_argument(
        "--reward_aggregations",
        nargs="+",
        choices=["min", "mean", "last", "product"],
        default=["min"],
    )
    parser.add_argument(
        "--experiment",
        choices=["consistency", "accuracy", "all"],
        default="all",
    )
    parser.add_argument(
        "--require_prm",
        action="store_true",
        help="Fail if a requested model does not have a PRM checkpoint.",
    )
    return parser.parse_args()


def slugify_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)


def parse_checkpoint_map(entries):
    checkpoint_map = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --prm_checkpoint_map entry '{entry}'. Expected model_name=path."
            )
        model_name, checkpoint = entry.split("=", 1)
        checkpoint_map[model_name] = checkpoint
    return checkpoint_map


def resolve_checkpoint(model_name: str, args, checkpoint_map):
    if model_name in checkpoint_map:
        return checkpoint_map[model_name]
    if args.prm_checkpoint_template:
        return args.prm_checkpoint_template.format(
            model_name=model_name,
            model_slug=slugify_model_name(model_name),
        )
    return args.prm_checkpoint


def summarize_consistency(results):
    summary = {}
    for benchmark, benchmark_result in results.items():
        row = {
            "baseline_agreement": benchmark_result["baseline"]["mean_agreement_rate"],
            "baseline_entropy": benchmark_result["baseline"]["mean_entropy"],
        }
        if "prm" in benchmark_result:
            row["prm_agreement"] = benchmark_result["prm"]["mean_agreement_rate"]
            row["prm_entropy"] = benchmark_result["prm"]["mean_entropy"]
            row["agreement_delta"] = benchmark_result["improvement"][
                "agreement_rate_delta"
            ]
        summary[benchmark] = row
    return summary


def summarize_accuracy(results):
    summary = {}
    for benchmark, benchmark_result in results.items():
        row = {
            "greedy_accuracy": benchmark_result["greedy"]["accuracy"],
            "majority_vote_accuracy": benchmark_result["majority_vote"]["accuracy"],
        }
        if "prm_best_of_n" in benchmark_result:
            row["prm_best_of_n_accuracy"] = benchmark_result["prm_best_of_n"]["accuracy"]
            row["aggregation"] = benchmark_result["prm_best_of_n"]["aggregation"]
        summary[benchmark] = row
    return summary


def write_manifest(output_dir: Path, manifest):
    manifest_path = output_dir / "study_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_map = parse_checkpoint_map(args.prm_checkpoint_map)

    manifest = []

    for model_name in args.model_names:
        model_slug = slugify_model_name(model_name)
        prm_model_name = args.prm_model_name or model_name
        prm_checkpoint = resolve_checkpoint(model_name, args, checkpoint_map)
        prm_available = bool(prm_checkpoint)
        if prm_available and not Path(prm_checkpoint).exists():
            if args.require_prm:
                raise FileNotFoundError(
                    f"Configured PRM checkpoint for model '{model_name}' does not exist: "
                    f"{prm_checkpoint}"
                )
            print(
                f"Warning: skipping PRM runs for '{model_name}' because checkpoint was not found: "
                f"{prm_checkpoint}"
            )
            prm_checkpoint = None
            prm_available = False

        if args.require_prm and not prm_available:
            raise FileNotFoundError(
                f"No PRM checkpoint configured for model '{model_name}'."
            )

        aggregations = (
            args.reward_aggregations if prm_available else [args.reward_aggregations[0]]
        )

        for best_of_n, num_paraphrases, aggregation in product(
            args.best_of_n_values,
            args.num_paraphrases_values,
            aggregations,
        ):
            combo_dir = output_dir / model_slug / (
                f"best_of_n_{best_of_n}__paraphrases_{num_paraphrases}"
                f"__aggregation_{aggregation}"
            )
            combo_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 80)
            print(f"Model: {model_name}")
            print(
                f"best_of_n={best_of_n} | num_paraphrases={num_paraphrases} "
                f"| aggregation={aggregation}"
            )
            print("=" * 80)

            model_config = ModelConfig(model_name=model_name)
            prm_config = PRMConfig(
                reward_model_name=prm_model_name,
                reward_aggregation=aggregation,
            )
            eval_config = EvalConfig(
                benchmarks=args.benchmarks,
                best_of_n=best_of_n,
            )
            data_config = DataConfig(
                num_paraphrases=num_paraphrases,
                max_samples=args.max_samples,
            )

            combo_result = {
                "model_name": model_name,
                "prm_model_name": prm_model_name,
                "prm_checkpoint": prm_checkpoint,
                "best_of_n": best_of_n,
                "num_paraphrases": num_paraphrases,
                "reward_aggregation": aggregation,
                "benchmarks": args.benchmarks,
                "max_samples": args.max_samples,
                "output_dir": str(combo_dir),
            }

            if args.experiment in ("consistency", "all"):
                consistency_results = run_consistency_experiment(
                    model_config=model_config,
                    prm_config=prm_config,
                    eval_config=eval_config,
                    data_config=data_config,
                    prm_checkpoint=prm_checkpoint,
                    output_dir=str(combo_dir / "consistency"),
                )
                combo_result["consistency"] = summarize_consistency(consistency_results)

            if args.experiment in ("accuracy", "all"):
                accuracy_results = run_accuracy_experiment(
                    model_config=model_config,
                    prm_config=prm_config,
                    eval_config=eval_config,
                    data_config=data_config,
                    prm_checkpoint=prm_checkpoint,
                    output_dir=str(combo_dir / "accuracy"),
                )
                combo_result["accuracy"] = summarize_accuracy(accuracy_results)

            manifest.append(combo_result)
            write_manifest(output_dir, manifest)

    print("\nStudy complete. Manifest saved to:")
    print(output_dir / "study_manifest.json")


if __name__ == "__main__":
    main()
