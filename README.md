# Process Reward Modeling Improves Multi-Step Reasoning Stability in Small Language Models

## Research Question
Do Process Reward Models (PRMs) improve reasoning consistency across different prompt phrasings in small language models?

## Key Idea
Small LMs often give different answers to the same question when phrased differently. We train a **Process Reward Model** that scores each reasoning step, then use it to select the most reliable reasoning trace. This improves both accuracy and consistency.

## Method

1. **Generate reasoning traces** from a small LM (e.g., Phi-2, Qwen-1.5B)
2. **Train a PRM** to score individual reasoning steps (correct vs. incorrect)
3. **Best-of-N selection**: Generate N traces, score with PRM, pick the highest-scored
4. **Measure consistency**: Paraphrase each question K ways, check if answers agree

## Benchmarks
- **GSM8K** — Grade school math word problems
- **MATH** — Competition mathematics
- **ARC** — Science reasoning (multiple choice)

## Quick Start

### Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sanowl/Process-Reward-Modeling-Improves-Multi-Step-Reasoning-Stability-in-Small-Language-Models/blob/main/notebooks/PRM_Reasoning_Stability.ipynb)

### Local Setup
```bash
pip install -r requirements.txt

# Train PRM
python scripts/train_prm.py --model_name microsoft/phi-2 --output_dir outputs/prm

# Run experiments
python scripts/run_experiments.py \
    --model_name microsoft/phi-2 \
    --prm_checkpoint outputs/prm/prm_model.pt \
    --benchmarks gsm8k math arc
```

## Project Structure
```
├── notebooks/
│   └── PRM_Reasoning_Stability.ipynb  # Colab notebook (start here)
├── src/
│   ├── config.py                      # All configuration dataclasses
│   ├── data/
│   │   ├── dataset_loader.py          # Load GSM8K, MATH, ARC
│   │   ├── reasoning_traces.py        # Generate & parse reasoning traces
│   │   └── paraphraser.py             # Generate prompt paraphrases
│   ├── models/
│   │   ├── base_model.py              # Load base LM with LoRA/4-bit
│   │   └── prm.py                     # Process Reward Model + Trainer
│   ├── evaluation/
│   │   ├── consistency.py             # Consistency metrics (agreement rate, entropy)
│   │   ├── benchmarks.py              # Accuracy evaluation
│   │   └── best_of_n.py              # PRM-guided best-of-N selection
│   ├── experiments/
│   │   ├── run_consistency.py         # Main consistency experiment
│   │   └── run_accuracy.py            # Accuracy experiment
│   └── visualization/
│       └── plots.py                   # Publication-quality plots
├── scripts/
│   ├── train_prm.py                   # PRM training script
│   └── run_experiments.py             # Run all experiments
├── requirements.txt
└── README.md
```

## Metrics
- **Agreement Rate**: Fraction of paraphrase pairs giving the same answer
- **Majority Fraction**: How often answers match the mode
- **Answer Entropy**: Shannon entropy of the answer distribution (lower = more consistent)
- **Accuracy**: Standard correctness on each benchmark
