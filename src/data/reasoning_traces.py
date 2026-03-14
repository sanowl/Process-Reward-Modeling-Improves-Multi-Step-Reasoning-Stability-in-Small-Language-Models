"""
Generate and parse multi-step reasoning traces from language models.
"""

import re
import warnings
import torch
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


STEP_PROMPT_TEMPLATE = """Solve the following problem step by step. Show each reasoning step clearly, separated by blank lines.

Problem: {question}

Solution:
Let me solve this step by step.

"""


def generate_reasoning_traces(
    model,
    tokenizer,
    questions: List[str],
    num_samples: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    device: str = "cuda",
) -> List[List[Dict]]:
    """
    Generate multiple reasoning traces per question.

    Returns:
        List of lists, where each inner list contains trace dicts
        with keys: 'full_text', 'steps', 'final_answer'
    """
    all_traces = []

    for question in questions:
        question_traces = []
        prompt = STEP_PROMPT_TEMPLATE.format(question=question)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_len = inputs["input_ids"].shape[1]
        if input_len >= 1024:
            warnings.warn(
                f"Prompt has {input_len} tokens, truncated to 1024. "
                "Question content may be incomplete."
            )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        do_sample = temperature > 0
        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        for _ in range(num_samples):
            # Clone inputs to prevent any in-place mutation across samples
            sample_inputs = {k: v.clone() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **sample_inputs,
                    **generate_kwargs,
                )

            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            steps = parse_reasoning_steps(generated)
            final_answer = extract_final_answer(generated)

            question_traces.append({
                "full_text": generated,
                "steps": steps,
                "final_answer": final_answer,
                "prompt": prompt,
            })

        all_traces.append(question_traces)

    return all_traces


def parse_reasoning_steps(text: str, separator: str = "\n\n") -> List[str]:
    """Parse a reasoning trace into individual steps."""
    steps = [s.strip() for s in text.split(separator) if s.strip()]

    # If we got very few steps, try splitting on numbered patterns
    if len(steps) <= 1:
        numbered = re.split(r'\n(?=(?:Step\s+)?\d+[\.\):])', text)
        numbered = [s.strip() for s in numbered if s.strip()]
        if len(numbered) > len(steps):
            steps = numbered

    return steps


def extract_final_answer(text: str) -> str:
    """Extract the final answer from a reasoning trace."""
    # Try common patterns (ordered by specificity)
    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)",
        r"####\s*(.+?)(?:\n|$)",
        r"(?:therefore|thus|so|hence)[,\s]+the\s+(?:final\s+)?answer\s+is\s+(.+?)(?:\.|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Try boxed answer
    from .dataset_loader import _extract_boxed
    boxed = _extract_boxed(text)
    if boxed:
        return boxed

    # Fallback: last line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def label_reasoning_steps(
    steps: List[str],
    gold_answer: str,
    final_answer: str,
) -> List[Tuple[str, int]]:
    """
    Create step-level labels for PRM training.
    Heuristic: if the final answer is correct, label all steps as positive.
    If incorrect, label all steps as negative (since we cannot pinpoint
    which step introduced the error without more information).
    """
    is_correct = _answers_match(final_answer, gold_answer)

    if is_correct:
        return [(step, 1) for step in steps]
    else:
        # Label all steps as negative for incorrect traces
        return [(step, 0) for step in steps]


def _answers_match(predicted: str, gold: str) -> bool:
    """Check if two answers match (with normalization)."""
    def normalize(s):
        s = s.lower().strip()
        s = re.sub(r"[,$%]", "", s)
        s = re.sub(r"\s+", " ", s)
        try:
            return str(float(s))
        except ValueError:
            return s

    return normalize(predicted) == normalize(gold)
