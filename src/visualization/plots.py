"""
Publication-quality plots for PRM reasoning stability experiments.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
})


def plot_consistency_comparison(
    results: Dict,
    save_path: Optional[str] = None,
):
    """
    Bar chart comparing consistency (agreement rate) across benchmarks
    for baseline vs PRM.
    """
    benchmarks = list(results.keys())
    baseline_scores = [results[b]["baseline"]["mean_agreement_rate"] for b in benchmarks]
    baseline_stds = [results[b]["baseline"]["std_agreement_rate"] for b in benchmarks]

    has_prm = all("prm" in results[b] for b in benchmarks)

    x = np.arange(len(benchmarks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(
        x - width / 2, baseline_scores, width,
        yerr=baseline_stds, label="Majority Vote (Baseline)",
        color="#4C72B0", capsize=4, edgecolor="white",
    )

    if has_prm:
        prm_scores = [results[b]["prm"]["mean_agreement_rate"] for b in benchmarks]
        prm_stds = [results[b]["prm"]["std_agreement_rate"] for b in benchmarks]
        bars2 = ax.bar(
            x + width / 2, prm_scores, width,
            yerr=prm_stds, label="PRM Best-of-N",
            color="#DD8452", capsize=4, edgecolor="white",
        )

    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Agreement Rate")
    ax.set_title("Reasoning Consistency Across Paraphrased Prompts")
    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in benchmarks])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_accuracy_comparison(
    results: Dict,
    save_path: Optional[str] = None,
):
    """Bar chart comparing accuracy across methods and benchmarks."""
    benchmarks = list(results.keys())
    methods = ["greedy", "majority_vote"]
    if any("prm_best_of_n" in results[b] for b in benchmarks):
        methods.append("prm_best_of_n")

    method_labels = {
        "greedy": "Greedy",
        "majority_vote": "Majority Vote",
        "prm_best_of_n": "PRM Best-of-N",
    }
    colors = ["#4C72B0", "#55A868", "#DD8452"]

    x = np.arange(len(benchmarks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, method in enumerate(methods):
        scores = [results[b].get(method, {}).get("accuracy", 0) for b in benchmarks]
        ax.bar(
            x + i * width - width, scores, width,
            label=method_labels[method], color=colors[i],
            edgecolor="white",
        )

    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy: Baseline vs PRM-Guided Selection")
    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in benchmarks])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_step_scores(
    step_scores: List[float],
    step_labels: Optional[List[str]] = None,
    title: str = "PRM Step Scores",
    save_path: Optional[str] = None,
):
    """Visualize PRM scores for each reasoning step."""
    n = len(step_scores)
    x = range(1, n + 1)
    labels = step_labels or [f"Step {i}" for i in x]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), 4))

    colors = ["#55A868" if s >= 0.5 else "#C44E52" for s in step_scores]
    bars = ax.bar(x, step_scores, color=colors, edgecolor="white")

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold")
    ax.set_xlabel("Reasoning Step")
    ax.set_ylabel("PRM Score (P(correct))")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_consistency_by_difficulty(
    results: Dict,
    difficulty_bins: List[str] = None,
    save_path: Optional[str] = None,
):
    """Plot consistency as a function of problem difficulty."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, color, label in [
        ("baseline", "#4C72B0", "Majority Vote"),
        ("prm", "#DD8452", "PRM Best-of-N"),
    ]:
        if method not in results:
            continue
        bins = difficulty_bins or list(results[method].keys())
        scores = [results[method][b]["mean_agreement_rate"] for b in bins]
        ax.plot(bins, scores, "o-", color=color, label=label, linewidth=2, markersize=8)

    ax.set_xlabel("Problem Difficulty")
    ax.set_ylabel("Agreement Rate")
    ax.set_title("Consistency vs Problem Difficulty")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
