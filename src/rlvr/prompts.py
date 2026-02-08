from typing import Dict, Any, List


def build_prompt(ex: Dict[str, Any]) -> str:
    t = ex.get("task_type")

    if t == "numeric":
        return (
            "Solve the problem step by step through reasoning.\n"
            "You must Put reasoning in <think>...</think>.\n"
            "You must Put the final answer in <answer>...</answer> as a single integer.\n\n"
            f"Problem: {ex['problem']}\n"
        )

    if t == "mcq":
        choices: List[Dict[str, str]] = ex.get("choices", [])
        choice_lines = "\n".join([f"{c['label']}. {c['text']}" for c in choices])
        return (
            "Choose the correct option through step by step reasoning before answering.\n"
            "You must Put reasoning in <think>...</think>.\n"
            "You must Put the final answer in <answer>...</answer> as a single letter: A, B, C, or D.\n\n"
            f"Question: {ex['problem']}\n"
            f"Options:\n{choice_lines}\n"
        )

    if t == "short_text":
        return (
            "You are given a Context containing facts and rules.\n"
            "You are also given an Observation.\n\n"
            "Task: Write ONE missing fact (a single sentence) that, if added to the Context,\n"
            "would allow the Observation to be proven OR disproven through doing step by step reasoning.\n\n"
            f"Context:\n{ex['context']}\n\n"
            f"Observation:\n{ex['observation']}\n\n"
            "You Must Put reasoning in <think>...</think>.\n"
            "You Must Put the final answer in <answer>...</answer> as a single sentence.\n"
        )

    raise ValueError(f"Unknown task_type: {t}")
  