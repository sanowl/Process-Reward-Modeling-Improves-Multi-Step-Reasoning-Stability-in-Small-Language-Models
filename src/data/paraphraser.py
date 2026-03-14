"""
Generate paraphrased versions of questions to test reasoning consistency.
"""

from typing import List, Dict
import random


# Templates for systematically varying question phrasing
PARAPHRASE_TEMPLATES = {
    "gsm8k": [
        "Solve the following math problem:\n{question}",
        "Please work through this math word problem step by step:\n{question}",
        "Think carefully and solve:\n{question}",
        "Here is a math problem. Show your work:\n{question}",
        "Can you figure out the answer to this?\n{question}",
        "I need help solving this problem. Walk me through it:\n{question}",
        "Work out the solution to the following:\n{question}",
        "Calculate the answer:\n{question}",
    ],
    "math": [
        "Solve the following mathematics problem:\n{question}",
        "Please solve this problem, showing all steps:\n{question}",
        "Find the answer to this math problem:\n{question}",
        "Work through this problem carefully:\n{question}",
        "Here's a math problem to solve:\n{question}",
        "Determine the solution:\n{question}",
        "Solve step by step:\n{question}",
        "Can you solve this? Show your reasoning:\n{question}",
    ],
    "arc": [
        "Answer the following science question:\n{question}",
        "Select the correct answer:\n{question}",
        "Think about this and choose the best answer:\n{question}",
        "Which answer is correct?\n{question}",
        "Reason through this question and pick the right option:\n{question}",
        "Here's a science question. What's the answer?\n{question}",
        "Consider this carefully and answer:\n{question}",
        "Analyze and answer:\n{question}",
    ],
}


def generate_paraphrases(
    question: str,
    benchmark: str = "gsm8k",
    num_paraphrases: int = 5,
    seed: int = 42,
) -> List[str]:
    """Generate paraphrased versions of a question using prompt templates."""
    templates = PARAPHRASE_TEMPLATES.get(benchmark, PARAPHRASE_TEMPLATES["gsm8k"])
    rng = random.Random(seed)
    selected = rng.sample(templates, min(num_paraphrases, len(templates)))
    return [t.format(question=question) for t in selected]


def generate_paraphrase_dataset(
    dataset,
    benchmark: str = "gsm8k",
    num_paraphrases: int = 5,
) -> List[Dict]:
    """Generate a full paraphrase dataset for consistency evaluation."""
    paraphrase_data = []

    for idx, example in enumerate(dataset):
        question = example["question"]
        answer = example["answer"]
        paraphrases = generate_paraphrases(
            question, benchmark, num_paraphrases, seed=idx
        )

        paraphrase_data.append({
            "original_question": question,
            "answer": answer,
            "paraphrased_prompts": paraphrases,
            "benchmark": benchmark,
            "example_idx": idx,
        })

    return paraphrase_data
