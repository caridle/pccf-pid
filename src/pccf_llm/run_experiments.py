"""
PCCF-LLM Experiments: Counterfactual knowledge, rule-shift, and long-context
contradiction experiments with precision-modulated attention.

Strategy: Modify attention scaling (scaling * pi) on HuggingFace models.
Lower pi = flatter attention = less reliance on strong priors.

Usage:
    python -m pccf_llm.run_experiments --exp counterfactual
    python -m pccf_llm.run_experiments --exp all
"""

import argparse
import time
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Optional

from .experiment_data import (
    COUNTERFACTUAL_TESTS,
    RULE_SHIFT_TASKS,
    CONTRADICTION_TESTS,
)
from .pccf_attention import patch_model_attention, update_pi_all, restore_model_attention


def _validate_experiment_data():
    for test in COUNTERFACTUAL_TESTS:
        if test["answer_correct"] == test["answer_prior"]:
            raise ValueError(f"counterfactual test {test['id']} has identical answers")
    for test in CONTRADICTION_TESTS:
        if test["answer_early"] == test["answer_late"]:
            raise ValueError(f"contradiction test {test['id']} has identical answers")


# ── Model Loading ─────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str, device: str = "cuda", dtype: str = "float16"):
    """Load model and tokenizer. Uses default (SDPA) attention for compatibility."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    print(f"[Load] Loading {model_name} ({dtype}) to {device}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[Load] Loaded ({n_params:.2f}B params)")
    return model, tokenizer


def format_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_text


# ── Metrics ───────────────────────────────────────────────────────────

def compute_normalized_logprob(model, tokenizer, prompt: str, answer: str) -> float:
    """
    Compute length-normalized log probability of `answer` given `prompt`.
    Returns a higher (less negative) score for answers the model prefers.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_text = prompt + answer
    full_ids = tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt").to(model.device)
    if full_ids.shape[1] <= len(prompt_ids):
        return float("-inf")
    answer_ids = full_ids[0, len(prompt_ids):].tolist()
    ans_len = len(answer_ids)
    seq_len = full_ids.shape[1]

    with torch.no_grad():
        outputs = model(full_ids)
    logits = outputs.logits.float()  # (1, seq_len, vocab)

    total_logprob = 0.0
    for i, tid in enumerate(answer_ids):
        pos = seq_len - ans_len + i - 1
        if 0 <= pos < logits.shape[1]:
            lp = F.log_softmax(logits[0, pos], dim=-1)
            total_logprob += lp[tid].item()

    return total_logprob / ans_len


def compute_answer_probs(model, tokenizer, prompt: str, options: list[str]) -> dict:
    """
    Compute normalized probability for each option.
    Returns dict mapping option -> probability in [0, 1].
    """
    scores = {}
    for opt in options:
        scores[opt] = compute_normalized_logprob(model, tokenizer, prompt, opt)

    vals = np.array(list(scores.values()))
    vals = vals - np.max(vals)
    probs = np.exp(vals) / np.sum(np.exp(vals))
    return {opt: float(p) for opt, p in zip(scores.keys(), probs)}


def compute_labeled_answer_probs(model, tokenizer, prompt: str, options: dict[str, str]) -> dict:
    """
    Compute normalized probabilities while separating reported labels
    from the exact continuation text scored by the tokenizer.
    """
    scores = {}
    for label, text in options.items():
        scores[label] = compute_normalized_logprob(model, tokenizer, prompt, text)

    vals = np.array(list(scores.values()))
    vals = vals - np.max(vals)
    probs = np.exp(vals) / np.sum(np.exp(vals))
    return {label: float(p) for label, p in zip(scores.keys(), probs)}


# ── Experiment I: Counterfactual Knowledge ────────────────────────────

def run_experiment_counterfactual(model, tokenizer, pi_values=None):
    """
    Test whether PCCF precision modulation helps the model follow
    counterfactual premises over pre-training priors.
    """
    if pi_values is None:
        pi_values = {
            "standard": 1.0,
            "pccf_07": 0.7,
            "pccf_05": 0.5,
            "pccf_03": 0.3,
            "pccf_01": 0.1,
        }

    results = []
    for test in COUNTERFACTUAL_TESTS:
        prompt = format_prompt(tokenizer, f"{test['prefix']}\n\nQ: {test['question']}")
        correct = test["answer_correct"]
        prior = test["answer_prior"]

        for label, pi in pi_values.items():
            restore_model_attention(model)
            patch_model_attention(model, precision_pi=pi)

            probs = compute_labeled_answer_probs(
                model,
                tokenizer,
                prompt,
                {correct: f" {correct}", prior: f" {prior}"},
            )
            p_correct = probs.get(correct, 0.0)
            p_prior = probs.get(prior, 0.0)

            results.append({
                "test_id": test["id"],
                "category": test["category"],
                "pi": pi,
                "label": label,
                "p_correct": round(p_correct, 4),
                "p_prior": round(p_prior, 4),
                "prefers_context": p_correct > p_prior,
                "margin": round(p_correct - p_prior, 4),
            })

            restore_model_attention(model)
        print(f"  [{test['id']:20s}] done")

    return pd.DataFrame(results)


def analyze_counterfactual(df):
    """Analyze and print counterfactual experiment results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT I: Counterfactual Knowledge Conflict")
    print("=" * 70)

    summary = df.groupby("label").agg(
        mean_p_context=("p_correct", "mean"),
        mean_p_prior=("p_prior", "mean"),
        mean_margin=("margin", "mean"),
        win_rate=("prefers_context", "mean"),
    ).sort_values("mean_margin", ascending=False)
    print(summary.round(4).to_string())

    # Compare best PCCF vs standard
    std_row = summary.loc["standard"]
    best_label = summary.index[0] if summary.index[0] != "standard" else summary.index[1]
    best_row = summary.loc[best_label]
    improvement = best_row["mean_margin"] - std_row["mean_margin"]
    print(f"\nBest PCCF ({best_label}) vs Standard improvement in margin: {improvement:+.4f}")

    # Category breakdown
    print("\nCategory breakdown (standard -> best PCCF):")
    best_pi = df[df["label"] == best_label]["pi"].iloc[0]
    for cat in sorted(df["category"].unique()):
        std_m = df[(df["category"] == cat) & (df["label"] == "standard")]["margin"].mean()
        pccf_m = df[(df["category"] == cat) & (df["label"] == best_label)]["margin"].mean()
        print(f"  {cat:15s}: {std_m:+.4f} -> {pccf_m:+.4f} (delta={pccf_m - std_m:+.4f})")

    # Control items
    ctrl = df[df["category"] == "control"]
    ctrl_std_win = ctrl[ctrl["label"] == "standard"]["prefers_context"].mean()
    ctrl_pccf_win = ctrl[ctrl["label"] == best_label]["prefers_context"].mean()
    print(f"\nControl items win_rate: standard={ctrl_std_win:.2%}, PCCF={ctrl_pccf_win:.2%}")

    return summary


# ── Experiment II: Rule Shift ─────────────────────────────────────────

def run_experiment_rule_shift(model, tokenizer, pi_values=None):
    """Test adaptation to label reversal in few-shot classification."""
    if pi_values is None:
        pi_values = {"standard": 1.0, "pccf_05": 0.5, "pccf_01": 0.1}

    results = []
    for task in RULE_SHIFT_TASKS:
        for label, pi in pi_values.items():
            restore_model_attention(model)
            patch_model_attention(model, precision_pi=pi)

            context = "Classify each review as 'positive' (good) or 'negative' (bad).\n\n"
            for text, lbl in task["before_shift"]:
                context += f"Review: {text}\nLabel: {lbl}\n\n"
            context += task["shift_notice"] + "\n\n"

            correct = 0
            total = 0
            for text, expected in task["after_shift"]:
                q = format_prompt(tokenizer, f"{context}Review: {text}\nLabel:")
                probs = compute_labeled_answer_probs(
                    model,
                    tokenizer,
                    q,
                    {c: f" {c}" for c in task["classes"]},
                )
                predicted = max(probs, key=probs.get)
                if predicted == expected:
                    correct += 1
                total += 1
                context += f"Review: {text}\nLabel: {expected}\n\n"

            acc = correct / total if total > 0 else 0
            results.append({
                "task_id": task["id"], "pi": pi, "label": label,
                "accuracy": round(acc, 4), "correct": correct, "total": total,
            })

            restore_model_attention(model)
        print(f"  [{task['id']}] done")

    return pd.DataFrame(results)


def analyze_rule_shift(df):
    """Analyze rule-shift results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT II: Few-shot Rule Shift (Label Reversal)")
    print("=" * 70)
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        print(f"  {label}: accuracy = {sub['accuracy'].mean():.2%}")
    try:
        pccf = df[df["label"] == "pccf_01"]["accuracy"].mean()
        std = df[df["label"] == "standard"]["accuracy"].mean()
        print(f"  PCCF improvement: {pccf - std:+.2%}")
    except:
        pass


# ── Experiment III: Long Context Contradiction ────────────────────────

def run_experiment_contradiction(model, tokenizer, pi_values=None):
    """Test whether PCCF helps prioritize recent over outdated info in long contexts."""
    if pi_values is None:
        pi_values = {"standard": 1.0, "pccf_05": 0.5, "pccf_01": 0.1}

    results = []
    for test in CONTRADICTION_TESTS:
        context = " ".join(t for t, _ in test["contexts"])
        prompt = format_prompt(tokenizer, f"{context}\n\nQ: {test['question']}")
        options = {
            test["answer_early"]: f" {test['answer_early']}",
            test["answer_late"]: f" {test['answer_late']}",
        }

        for label, pi in pi_values.items():
            restore_model_attention(model)
            patch_model_attention(model, precision_pi=pi)
            probs = compute_labeled_answer_probs(model, tokenizer, prompt, options)
            p_early = probs.get(test["answer_early"], 0.0)
            p_late = probs.get(test["answer_late"], 0.0)

            results.append({
                "test_id": test["id"], "pi": pi, "label": label,
                "p_early": round(p_early, 4), "p_late": round(p_late, 4),
                "prefers_late": p_late > p_early,
                "margin": round(p_late - p_early, 4),
            })

            restore_model_attention(model)
        print(f"  [{test['id']:15s}] done")

    return pd.DataFrame(results)


def analyze_contradiction(df):
    """Analyze contradiction results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT III: Long Context Contradiction")
    print("=" * 70)
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        print(f"  {label}: prefers_late={sub['prefers_late'].mean():.2%}, margin={sub['margin'].mean():+.4f}")
    try:
        pccf = df[df["label"] == "pccf_01"]["prefers_late"].mean()
        std = df[df["label"] == "standard"]["prefers_late"].mean()
        print(f"  PCCF improvement in preferring latest info: {pccf - std:+.2%}")
    except:
        pass


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PCCF-LLM Experiments")
    parser.add_argument("--exp", type=str, default="all",
                        choices=["counterfactual", "rule_shift", "contradiction", "all"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--output-dir", type=str, default="pccf_llm_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    _validate_experiment_data()

    print("=" * 70)
    print("PCCF-LLM Experiments")
    print(f"Model: {args.model}")
    print("=" * 70)

    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.dtype)

    all_summaries = {}

    if args.exp in ("counterfactual", "all"):
        print("\n--- Running Experiment I: Counterfactual ---")
        df1 = run_experiment_counterfactual(model, tokenizer)
        df1.to_csv(f"{args.output_dir}/counterfactual_results.csv", index=False)
        s1 = analyze_counterfactual(df1)
        s1.to_csv(f"{args.output_dir}/counterfactual_summary.csv")

    if args.exp in ("rule_shift", "all"):
        print("\n--- Running Experiment II: Rule Shift ---")
        df2 = run_experiment_rule_shift(model, tokenizer)
        df2.to_csv(f"{args.output_dir}/rule_shift_results.csv", index=False)
        analyze_rule_shift(df2)

    if args.exp in ("contradiction", "all"):
        print("\n--- Running Experiment III: Contradiction ---")
        df3 = run_experiment_contradiction(model, tokenizer)
        df3.to_csv(f"{args.output_dir}/contradiction_results.csv", index=False)
        analyze_contradiction(df3)

    print("\n" + "=" * 70)
    print(f"All experiments complete. Results saved to: {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
