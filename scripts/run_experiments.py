"""
Run all experiments for the paper.

Usage:
    python scripts/run_experiments.py \
        --model_name microsoft/phi-2 \
        --prm_checkpoint outputs/prm/prm_model.pt \
        --output_dir results \
        --max_samples 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ModelConfig, PRMConfig, EvalConfig, DataConfig
from src.experiments.run_consistency import run_consistency_experiment
from src.experiments.run_accuracy import run_accuracy_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run PRM experiments")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--prm_model_name", type=str, default=None,
                        help="Model name for PRM (defaults to --model_name)")
    parser.add_argument("--prm_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_paraphrases", type=int, default=5)
    parser.add_argument("--best_of_n", type=int, default=8)
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math", "arc"])
    parser.add_argument("--experiment", choices=["consistency", "accuracy", "all"], default="all")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate checkpoint exists before starting expensive GPU work
    if args.prm_checkpoint:
        checkpoint_path = Path(args.prm_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"PRM checkpoint not found: {args.prm_checkpoint}")

    prm_model_name = args.prm_model_name or args.model_name

    model_config = ModelConfig(model_name=args.model_name)
    prm_config = PRMConfig(reward_model_name=prm_model_name)
    eval_config = EvalConfig(
        benchmarks=args.benchmarks,
        best_of_n=args.best_of_n,
    )
    data_config = DataConfig(
        num_paraphrases=args.num_paraphrases,
        max_samples=args.max_samples,
    )

    if args.experiment in ("consistency", "all"):
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: REASONING CONSISTENCY")
        print("=" * 70)
        run_consistency_experiment(
            model_config=model_config,
            prm_config=prm_config,
            eval_config=eval_config,
            data_config=data_config,
            prm_checkpoint=args.prm_checkpoint,
            output_dir=f"{args.output_dir}/consistency",
        )

    if args.experiment in ("accuracy", "all"):
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: ACCURACY")
        print("=" * 70)
        run_accuracy_experiment(
            model_config=model_config,
            prm_config=prm_config,
            eval_config=eval_config,
            data_config=data_config,
            prm_checkpoint=args.prm_checkpoint,
            output_dir=f"{args.output_dir}/accuracy",
        )

    print("\nAll experiments complete! Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()
