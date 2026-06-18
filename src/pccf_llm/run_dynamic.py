"""
Dynamic PCCF experiment runner with event-triggered precision control.

Uses SpikeDetector + EventTriggeredPCCF from pccf_controller.py:
- Normal operation: pi = 1.0 (standard attention)
- Entropy spike detected -> pi drops to 0.1 for hold_steps tokens
- Recovery to pi = 1.0 after hold expires

This is a cleaner test than the continuous controller:
intervention is binary and event-driven.

Usage:
    python -m pccf_llm.run_dynamic --exp counterfactual
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .experiment_data import COUNTERFACTUAL_TESTS, CONTRADICTION_TESTS
from .pccf_attention import patch_model_attention, update_pi_all, restore_model_attention
from .pccf_controller import EventTriggeredPCCF, compute_entropy_from_logits


def _validate_experiment_data():
    for test in COUNTERFACTUAL_TESTS:
        if test["answer_correct"] == test["answer_prior"]:
            raise ValueError(f"counterfactual test {test['id']} has identical answers")
    for test in CONTRADICTION_TESTS:
        if test["answer_early"] == test["answer_late"]:
            raise ValueError(f"contradiction test {test['id']} has identical answers")


# ── Model ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name, device="cuda", dtype="float16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch_dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return model, tokenizer


# ── Answer scoring ────────────────────────────────────────────────────

def format_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_text


def normalized_logprob(model, tokenizer, prompt, answer):
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
    logits = outputs.logits.float()
    total = 0.0
    for i, tid in enumerate(answer_ids):
        pos = seq_len - ans_len + i - 1
        if 0 <= pos < logits.shape[1]:
            total += F.log_softmax(logits[0, pos], dim=-1)[tid].item()
    return total / ans_len


def answer_probs(model, tokenizer, prompt, options):
    scores = {opt: normalized_logprob(model, tokenizer, prompt, opt) for opt in options}
    vals = np.array(list(scores.values()))
    vals -= np.max(vals)
    probs = np.exp(vals) / np.sum(np.exp(vals))
    return {opt: float(p) for opt, p in zip(scores.keys(), probs)}


def labeled_answer_probs(model, tokenizer, prompt, options):
    scores = {label: normalized_logprob(model, tokenizer, prompt, text) for label, text in options.items()}
    vals = np.array(list(scores.values()))
    vals -= np.max(vals)
    probs = np.exp(vals) / np.sum(np.exp(vals))
    return {label: float(p) for label, p in zip(scores.keys(), probs)}


# ── Dynamic experiment with event-triggered PCCF ─────────────────────

def run_event_counterfactual(
    model,
    tokenizer,
    output_dir="pccf_llm_results_dynamic",
    pi_min=0.1,
    hold_steps=20,
    recovery_steps=20,
    cooldown_steps=40,
    threshold_sigma=3.0,
):
    """Run counterfactual test with event-triggered precision control."""
    os.makedirs(output_dir, exist_ok=True)
    controller = EventTriggeredPCCF(
        pi_min=pi_min,
        hold_steps=hold_steps,
        recovery_steps=recovery_steps,
        cooldown_steps=cooldown_steps,
    )
    controller.spike_detector.threshold_sigma = threshold_sigma

    results = []
    traces = {}

    for test in COUNTERFACTUAL_TESTS:
        prompt = format_prompt(tokenizer, f"{test['prefix']}\n\nQ: {test['question']}")
        correct = test["answer_correct"]
        prior = test["answer_prior"]
        options = {correct: f" {correct}", prior: f" {prior}"}

        # ── Standard (pi=1.0) ──
        restore_model_attention(model)
        probs_std = labeled_answer_probs(model, tokenizer, prompt, options)

        # ── Event-triggered PCCF ──
        controller.reset()
        restore_model_attention(model)
        patch_model_attention(model, precision_pi=1.0)

        # Feed prompt token by token, tracking entropy
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        trace = []

        for i in range(1, prompt_ids.shape[1] + 1):
            chunk = prompt_ids[:, :i]
            with torch.no_grad():
                out = model(chunk)
            entropy = compute_entropy_from_logits(out.logits)
            state = controller.step(entropy)
            update_pi_all(model, state["pi"])
            trace.append({
                "token": i, "pi": state["pi"],
                "state": state["state"], "entropy": round(entropy, 4),
            })

        # Evaluate with dynamic pi
        probs_dyn = labeled_answer_probs(model, tokenizer, prompt, options)

        results.append({
            "test_id": test["id"],
            "category": test["category"],
            "margin_std": round(probs_std.get(correct, 0) - probs_std.get(prior, 0), 4),
            "margin_dyn": round(probs_dyn.get(correct, 0) - probs_dyn.get(prior, 0), 4),
            "p_correct_std": round(probs_std.get(correct, 0), 4),
            "p_correct_dyn": round(probs_dyn.get(correct, 0), 4),
            "prefers_correct_std": probs_std.get(correct, 0) > probs_std.get(prior, 0),
            "prefers_correct_dyn": probs_dyn.get(correct, 0) > probs_dyn.get(prior, 0),
            "triggers": controller.trigger_count,
            "trigger_info": controller.trigger_log[:3],  # first 3 triggers
        })

        traces[test["id"]] = {
            "trace": trace,
            "triggers": controller.trigger_log,
        }
        print(f"  [{test['id']:20s}] std={results[-1]['margin_std']:+.4f} "
              f"dyn={results[-1]['margin_dyn']:+.4f} "
              f"triggers={controller.trigger_count}")

    restore_model_attention(model)

    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/event_counterfactual.csv", index=False)

    # Save full traces
    with open(f"{output_dir}/event_counterfactual_traces.json", "w") as f:
        json.dump(traces, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 60)
    print("EVENT-TRIGGERED PCCF: Counterfactual Knowledge")
    print("=" * 60)

    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat]
        print(f"  {cat:15s}: std={sub['margin_std'].mean():+.4f} "
              f"dyn={sub['margin_dyn'].mean():+.4f} "
              f"triggers={sub['triggers'].mean():.1f}")

    std_win = df["prefers_correct_std"].mean()
    dyn_win = df["prefers_correct_dyn"].mean()
    std_m = df["margin_std"].mean()
    dyn_m = df["margin_dyn"].mean()
    print(f"\nOverall: std_win={std_win:.1%} dyn_win={dyn_win:.1%}")
    print(f"         std_margin={std_m:+.4f} dyn_margin={dyn_m:+.4f}")
    print(f"         delta={dyn_m - std_m:+.4f}")
    print(f"         avg triggers={df['triggers'].mean():.1f}")

    return df


def run_event_contradiction(
    model,
    tokenizer,
    output_dir="pccf_llm_results_dynamic",
    pi_min=0.1,
    hold_steps=20,
    recovery_steps=20,
    cooldown_steps=40,
    threshold_sigma=3.0,
):
    """Event-triggered PCCF on long context contradiction."""
    os.makedirs(output_dir, exist_ok=True)
    controller = EventTriggeredPCCF(
        pi_min=pi_min,
        hold_steps=hold_steps,
        recovery_steps=recovery_steps,
        cooldown_steps=cooldown_steps,
    )
    controller.spike_detector.threshold_sigma = threshold_sigma

    results = []
    for test in CONTRADICTION_TESTS:
        context = " ".join(t for t, _ in test["contexts"])
        prompt = format_prompt(tokenizer, f"{context}\n\nQ: {test['question']}")
        options = {
            test["answer_early"]: f" {test['answer_early']}",
            test["answer_late"]: f" {test['answer_late']}",
        }

        # Standard
        restore_model_attention(model)
        probs_std = labeled_answer_probs(model, tokenizer, prompt, options)

        # Event-triggered
        controller.reset()
        restore_model_attention(model)
        patch_model_attention(model, precision_pi=1.0)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        for i in range(1, prompt_ids.shape[1] + 1):
            chunk = prompt_ids[:, :i]
            with torch.no_grad():
                out = model(chunk)
            entropy = compute_entropy_from_logits(out.logits)
            state = controller.step(entropy)
            update_pi_all(model, state["pi"])

        probs_dyn = labeled_answer_probs(model, tokenizer, prompt, options)
        results.append({
            "test_id": test["id"],
            "margin_std": round(probs_std.get(test["answer_late"], 0) - probs_std.get(test["answer_early"], 0), 4),
            "margin_dyn": round(probs_dyn.get(test["answer_late"], 0) - probs_dyn.get(test["answer_early"], 0), 4),
            "prefers_late_std": probs_std.get(test["answer_late"], 0) > probs_std.get(test["answer_early"], 0),
            "prefers_late_dyn": probs_dyn.get(test["answer_late"], 0) > probs_dyn.get(test["answer_early"], 0),
            "triggers": controller.trigger_count,
        })
        print(f"  [{test['id']:15s}] std={results[-1]['margin_std']:+.4f} "
              f"dyn={results[-1]['margin_dyn']:+.4f} triggers={controller.trigger_count}")

    restore_model_attention(model)
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/event_contradiction.csv", index=False)

    std_late = df["prefers_late_std"].mean()
    dyn_late = df["prefers_late_dyn"].mean()
    print(f"\nContradiction: prefers_late std={std_late:.1%} dyn={dyn_late:.1%}")
    return df


# ── Token-by-token trace visualization ────────────────────────────────

def print_trace_example(test_id, traces, tokenizer, limit=80):
    """Print a human-readable token-by-token trace with pi state."""
    if test_id not in traces:
        return
    trace = traces[test_id]["trace"]
    triggers = traces[test_id]["triggers"]
    print(f"\n--- Trace: {test_id} ({len(triggers)} triggers) ---")
    for t in trace[:limit]:
        marker = " ⚡" if t["state"] in ("triggered", "holding") else ""
        bar = "█" * int(t["pi"] * 20) if t["pi"] > 0 else "_"
        print(f"  token{t['token']:3d} pi={t['pi']:.2f} {bar:20s} ent={t['entropy']:.3f} {t['state']}{marker}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all", choices=["counterfactual", "contradiction", "all"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="pccf_llm_results_dynamic")
    parser.add_argument("--trace", action="store_true", help="Print token-by-token traces")
    parser.add_argument("--trace-id", type=str, default="capital_1",
                        help="Which test to trace (only with --trace)")
    parser.add_argument("--pi-min", type=float, default=0.1)
    parser.add_argument("--hold-steps", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="Sigma threshold for spike detection")
    args = parser.parse_args()

    _validate_experiment_data()
    model, tokenizer = load_model_and_tokenizer(args.model)

    if args.trace:
        from .pccf_controller import EventTriggeredPCCF
        # Trace a single test case token by token
        test = next(t for t in COUNTERFACTUAL_TESTS if t["id"] == args.trace_id)
        prompt = format_prompt(tokenizer, f"{test['prefix']}\n\nQ: {test['question']}")
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        tokens = [tokenizer.decode([tid]) for tid in prompt_ids[0]]

        controller = EventTriggeredPCCF(
            pi_min=args.pi_min, hold_steps=args.hold_steps,
        )
        controller.spike_detector.threshold_sigma = args.threshold

        patch_model_attention(model, precision_pi=1.0)
        print(f"\n{'='*60}")
        print(f"Trace: {args.trace_id} (sigma_threshold={args.threshold})")
        print(f"Prompt: {prompt[:120]}...")
        print(f"{'='*60}")
        print(f"{'tok':>4s} {'pi':>5s} {'entropy':>8s} {'sigma':>6s} {'state':>12s} token")
        print("-" * 60)

        for i in range(1, prompt_ids.shape[1] + 1):
            chunk = prompt_ids[:, :i]
            with torch.no_grad():
                out = model(chunk)
            entropy = compute_entropy_from_logits(out.logits)
            state = controller.step(entropy)
            update_pi_all(model, state["pi"])
            detection = state.get("detection", {})
            marker = " <<< TRIGGER" if state["triggered"] else ""
            print(f"{i:4d} {state['pi']:5.2f} {entropy:8.4f} "
                  f"{detection.get('sigma', 0):6.2f} {state['state']:>12s} "
                  f"{tokens[i-1]}{marker}")

        restore_model_attention(model)
        return

    if args.exp in ("counterfactual", "all"):
        print("\n=== Event-Triggered PCCF: Counterfactual ===")
        df1 = run_event_counterfactual(
            model,
            tokenizer,
            args.output_dir,
            pi_min=args.pi_min,
            hold_steps=args.hold_steps,
            threshold_sigma=args.threshold,
        )

    if args.exp in ("contradiction", "all"):
        print("\n=== Event-Triggered PCCF: Contradiction ===")
        df2 = run_event_contradiction(
            model,
            tokenizer,
            args.output_dir,
            pi_min=args.pi_min,
            hold_steps=args.hold_steps,
            threshold_sigma=args.threshold,
        )


if __name__ == "__main__":
    main()
