"""
PCCF-PID-lite closed-loop precision control.

This is a mechanism-validation runner. It treats PCCF precision as a control
variable and uses answer-margin feedback to adjust pi over a small number of
iterations. It is not a deployable black-box decoding method because the target
and distractor answers are known by the benchmark evaluator.

Control interpretation:
  - counterfactual tasks: reduce prior dominance by lowering pi on selected layers
  - contradiction tasks: preserve late evidence by raising/sharpening pi
"""

import argparse
import os
import re
from dataclasses import dataclass, field

import pandas as pd

from .pccf_attention import patch_model_attention, restore_model_attention
from .run_hallucination_baselines import (
    ScoredItem,
    build_items,
    DATASET_CHOICES,
    format_prompt,
    load_model_and_tokenizer,
    print_item_preview,
    score_binary_options,
    summarize,
)


@dataclass
class PIDController:
    kp: float = 0.25
    ki: float = 0.05
    kd: float = 0.10
    target_margin: float = 0.20
    pi_min: float = 0.4
    pi_max: float = 1.4
    pi: float = 1.0
    integral: float = 0.0
    prev_error: float = 0.0
    trace: list[dict] = field(default_factory=list)

    def step(self, observed_margin: float, direction: float) -> float:
        """
        Update pi from observed margin.

        error > 0 means target evidence is not yet dominant enough.
        direction = -1 lowers pi under positive error.
        direction = +1 raises pi under positive error.
        """
        error = max(0.0, self.target_margin - observed_margin)
        self.integral += error
        derivative = error - self.prev_error
        raw = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.pi = min(self.pi_max, max(self.pi_min, self.pi + direction * raw))
        self.prev_error = error
        self.trace.append({
            "observed_margin": round(observed_margin, 6),
            "error": round(error, 6),
            "control": round(direction * raw, 6),
            "pi": round(self.pi, 6),
        })
        return self.pi

    def step_proxy(self, proxy_error: float, direction: float) -> float:
        """Update pi from an unlabeled proxy error in [0, +inf)."""
        error = max(0.0, proxy_error)
        self.integral += error
        derivative = error - self.prev_error
        raw = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.pi = min(self.pi_max, max(self.pi_min, self.pi + direction * raw))
        self.prev_error = error
        self.trace.append({
            "observed_margin": None,
            "error": round(error, 6),
            "control": round(direction * raw, 6),
            "pi": round(self.pi, 6),
        })
        return self.pi


def control_direction(item: ScoredItem, mode: str) -> float:
    if mode == "auto":
        if item.task == "contradiction":
            return +1.0
        return -1.0
    if mode == "lower":
        return -1.0
    if mode == "raise":
        return +1.0
    raise ValueError(f"Unsupported direction mode: {mode}")


def proxy_error(item: ScoredItem, mode: str) -> float:
    """
    Unlabeled intervention pressure.

    This deliberately avoids target/distractor probabilities. It only uses the
    prompt/task form as a proxy for whether old priors or early context may be
    over-dominant.
    """
    if mode == "oracle_margin":
        raise ValueError("oracle_margin is not a proxy mode")
    text = item.prompt.lower()
    if mode == "task":
        return 1.0 if item.task in {"counterfactual", "contradiction"} else 0.0
    if mode == "keyword":
        cues = [
            "alternate universe", "alternative timeline", "simulation", "where",
            "not", "rather than", "instead", "update", "correction", "important",
            "revised", "moved", "changed", "replaced", "from now on",
        ]
        hits = sum(1 for cue in cues if cue in text)
        return min(1.0, hits / 2.0)
    if mode == "hybrid":
        return max(proxy_error(item, "task"), proxy_error(item, "keyword"))
    if mode == "evidence":
        return proxy_error(item, "hybrid")
    raise ValueError(f"Unsupported feedback mode: {mode}")


def confidence_gate(scores: dict, threshold: float) -> bool:
    """Return True when the model is uncertain enough to allow intervention."""
    return abs(scores["margin"]) < threshold


def _normalize_terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def evidence_support(prompt: str, answer: str) -> float:
    prompt_terms = set(_normalize_terms(prompt))
    answer_terms = _normalize_terms(answer)
    if not answer_terms:
        return 0.0
    hits = sum(1 for term in answer_terms if term in prompt_terms)
    return hits / len(answer_terms)


def evidence_gate(item: ScoredItem, scores: dict, min_gap: float) -> bool:
    """
    Gate intervention using context evidence, not correctness labels.

    If the model already prefers the option with stronger direct prompt support,
    do not intervene. If it prefers the less-supported option and the support gap
    is large enough, intervention is allowed.
    """
    target_support = evidence_support(item.prompt, item.target)
    distractor_support = evidence_support(item.prompt, item.distractor)
    support_gap = target_support - distractor_support
    if abs(support_gap) < min_gap:
        return False
    model_prefers_target = scores["margin"] > 0
    evidence_prefers_target = support_gap > 0
    return model_prefers_target != evidence_prefers_target


def evidence_direction(item: ScoredItem) -> float | None:
    target_support = evidence_support(item.prompt, item.target)
    distractor_support = evidence_support(item.prompt, item.distractor)
    support_gap = target_support - distractor_support
    if support_gap == 0:
        return None
    return +1.0 if support_gap > 0 else -1.0


def evidence_margin(scores: dict, direction: float) -> float:
    """Margin toward the option with stronger prompt evidence."""
    return direction * scores["margin"]


def evaluate_with_pi(model, tokenizer, item: ScoredItem, pi: float, layer_mode: str, conflict_aware: bool) -> dict:
    restore_model_attention(model)
    if pi != 1.0:
        patch_model_attention(model, precision_pi=pi, layer_mode=layer_mode)
    prompt = format_prompt(tokenizer, item.prompt, conflict_aware=conflict_aware)
    return score_binary_options(model, tokenizer, prompt, item.target, item.distractor)


def probe_control_direction(
    model,
    tokenizer,
    item: ScoredItem,
    std_scores: dict,
    layer_mode: str,
    conflict_aware: bool,
    evidence_dir: float,
    low_pi: float,
    high_pi: float,
    min_gain: float,
) -> tuple[float | None, dict]:
    """
    Choose a control direction using unlabeled evidence preference.

    The target/distractor labels are only candidate options. The gate asks:
    which pi moves probability toward the option that is more directly supported
    by the prompt text?
    """
    std_ev_margin = evidence_margin(std_scores, evidence_dir)
    low_scores = evaluate_with_pi(model, tokenizer, item, low_pi, layer_mode, conflict_aware)
    high_scores = evaluate_with_pi(model, tokenizer, item, high_pi, layer_mode, conflict_aware)
    low_gain = evidence_margin(low_scores, evidence_dir) - std_ev_margin
    high_gain = evidence_margin(high_scores, evidence_dir) - std_ev_margin

    probe = {
        "std_evidence_margin": round(std_ev_margin, 6),
        "low_gain": round(low_gain, 6),
        "high_gain": round(high_gain, 6),
        "low_pi": low_pi,
        "high_pi": high_pi,
    }
    if low_gain <= min_gain and high_gain <= min_gain:
        return None, probe
    return (-1.0 if low_gain >= high_gain else +1.0), probe


def run_pid_lite(
    model,
    tokenizer,
    items: list[ScoredItem],
    layer_mode: str,
    direction_mode: str,
    conflict_aware: bool,
    steps: int,
    kp: float,
    ki: float,
    kd: float,
    target_margin: float,
    pi_min: float,
    pi_max: float,
    feedback_mode: str,
    selection: str,
    confidence_threshold: float,
    evidence_gap: float,
    probe_low_pi: float,
    probe_high_pi: float,
    probe_min_gain: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    traces = []

    for item in items:
        std = evaluate_with_pi(model, tokenizer, item, 1.0, layer_mode, False)
        controller = PIDController(
            kp=kp,
            ki=ki,
            kd=kd,
            target_margin=target_margin,
            pi_min=pi_min,
            pi_max=pi_max,
            pi=1.0,
        )
        direction = control_direction(item, direction_mode)
        probe_info = {}

        current = std
        best_pi = 1.0
        best_scores = std
        best_margin = std["margin"]
        if feedback_mode == "oracle_margin":
            allow_intervention = True
        elif feedback_mode == "probe":
            ev_dir = evidence_direction(item)
            if ev_dir is None:
                allow_intervention = False
            else:
                probed_direction, probe_info = probe_control_direction(
                    model,
                    tokenizer,
                    item,
                    std,
                    layer_mode,
                    conflict_aware,
                    ev_dir,
                    probe_low_pi,
                    probe_high_pi,
                    probe_min_gain,
                )
                allow_intervention = probed_direction is not None
                if allow_intervention:
                    direction = probed_direction
        elif feedback_mode == "evidence":
            allow_intervention = evidence_gate(item, std, evidence_gap)
        else:
            allow_intervention = confidence_gate(std, confidence_threshold)

        for step_idx in range(steps):
            if not allow_intervention:
                feedback_value = 0.0
                next_pi = 1.0
                controller.trace.append({
                    "observed_margin": None,
                    "error": 0.0,
                    "control": 0.0,
                    "pi": 1.0,
                })
            elif feedback_mode == "oracle_margin":
                next_pi = controller.step(current["margin"], direction)
                feedback_value = current["margin"]
            elif feedback_mode == "probe":
                feedback_value = 1.0
                next_pi = controller.step_proxy(feedback_value, direction)
            else:
                feedback_value = proxy_error(item, feedback_mode)
                next_pi = controller.step_proxy(feedback_value, direction)
            current = evaluate_with_pi(
                model,
                tokenizer,
                item,
                next_pi,
                layer_mode,
                conflict_aware=conflict_aware,
            )
            trace_row = controller.trace[-1].copy()
            trace_row.update({
                "item_id": item.id,
                "task": item.task,
                "step": step_idx + 1,
                "feedback_mode": feedback_mode,
                "feedback_value": feedback_value,
                **probe_info,
                "eval_margin": round(current["margin"], 6),
                "eval_p_target": round(current["p_target"], 6),
            })
            traces.append(trace_row)
            if current["margin"] > best_margin:
                best_margin = current["margin"]
                best_pi = next_pi
                best_scores = current

        if selection == "final":
            chosen_pi = controller.pi
            chosen_scores = current
        elif selection == "best":
            chosen_pi = best_pi
            chosen_scores = best_scores
        else:
            raise ValueError(f"Unsupported selection mode: {selection}")

        rows.append({
            "item_id": item.id,
            "task": item.task,
            "category": item.category,
            "baseline": "standard",
            "pi": 1.0,
            "conflict_aware": False,
            "target": item.target,
            "distractor": item.distractor,
            **std,
        })
        rows.append({
            "item_id": item.id,
            "task": item.task,
            "category": item.category,
            "baseline": "pccf_pid_lite",
            "pi": round(chosen_pi, 6),
            "conflict_aware": conflict_aware,
            "target": item.target,
            "distractor": item.distractor,
            **chosen_scores,
        })
        print(
            f"  [{item.task}:{item.id}] std={std['margin']:+.4f} "
            f"pid={chosen_scores['margin']:+.4f} pi={chosen_pi:.3f}"
        )

    restore_model_attention(model)
    return pd.DataFrame(rows), pd.DataFrame(traces)


def plot_pid_traces(traces: pd.DataFrame, output_dir: str, dataset: str):
    if traces.empty:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping PID trace plots.")
        return []
    paths = []
    summary = traces.groupby(["task", "step"]).agg(
        mean_pi=("pi", "mean"),
        mean_margin=("eval_margin", "mean"),
    ).reset_index()

    for metric, ylabel in [("mean_pi", "Precision pi"), ("mean_margin", "Evaluation margin")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for task, sub in summary.groupby("task"):
            ax.plot(sub["step"], sub[metric], marker="o", label=task)
        ax.set_xlabel("PID step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"PCCF-PID-lite {ylabel} trajectory")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(output_dir, f"{dataset}_pid_{metric}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Run PCCF-PID-lite closed-loop precision control.")
    parser.add_argument("--dataset", default="local", choices=DATASET_CHOICES)
    parser.add_argument("--dataset-file", default=None,
                        help="JSONL file for --dataset jsonl. Fields: prompt,target,distractor,id,category,task.")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run-items", action="store_true",
                        help="Load and print benchmark items without loading a model.")
    parser.add_argument("--output-dir", default="pccf_pid_lite_results")
    parser.add_argument("--layer-mode", default="late")
    parser.add_argument("--direction", default="auto", choices=["auto", "lower", "raise"])
    parser.add_argument("--conflict-aware", action="store_true")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--feedback-mode", default="oracle_margin",
                        choices=["oracle_margin", "task", "keyword", "hybrid", "evidence", "probe"],
                        help="oracle_margin uses labels for feasibility; other modes are unlabeled proxies.")
    parser.add_argument("--selection", default=None, choices=["best", "final"],
                        help="best uses labeled evaluation to choose pi; final uses the controller's final pi.")
    parser.add_argument("--confidence-threshold", type=float, default=0.35,
                        help="Unlabeled proxy mode intervenes only when abs(standard margin) is below this value.")
    parser.add_argument("--evidence-gap", type=float, default=0.25,
                        help="Evidence proxy intervenes when prompt-support gap exceeds this value and model preference disagrees.")
    parser.add_argument("--probe-low-pi", type=float, default=0.7)
    parser.add_argument("--probe-high-pi", type=float, default=1.3)
    parser.add_argument("--probe-min-gain", type=float, default=0.02)
    parser.add_argument("--kp", type=float, default=0.25)
    parser.add_argument("--ki", type=float, default=0.05)
    parser.add_argument("--kd", type=float, default=0.10)
    parser.add_argument("--target-margin", type=float, default=0.20)
    parser.add_argument("--pi-min", type=float, default=0.4)
    parser.add_argument("--pi-max", type=float, default=1.4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    items = build_items(args.dataset, args.limit, args.dataset_file)
    if not items:
        raise RuntimeError(f"No items loaded for dataset {args.dataset}")
    if args.dry_run_items:
        print_item_preview(items, args.dataset)
        return
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.device,
        args.dtype,
        local_files_only=args.local_files_only,
    )
    selection = args.selection
    if selection is None:
        selection = "best" if args.feedback_mode == "oracle_margin" else "final"

    df, traces = run_pid_lite(
        model,
        tokenizer,
        items,
        layer_mode=args.layer_mode,
        direction_mode=args.direction,
        conflict_aware=args.conflict_aware,
        steps=args.steps,
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        target_margin=args.target_margin,
        pi_min=args.pi_min,
        pi_max=args.pi_max,
        feedback_mode=args.feedback_mode,
        selection=selection,
        confidence_threshold=args.confidence_threshold,
        evidence_gap=args.evidence_gap,
        probe_low_pi=args.probe_low_pi,
        probe_high_pi=args.probe_high_pi,
        probe_min_gain=args.probe_min_gain,
    )
    summary = summarize(df)
    plot_paths = plot_pid_traces(traces, args.output_dir, args.dataset)

    results_path = os.path.join(args.output_dir, f"{args.dataset}_pid_results.csv")
    trace_path = os.path.join(args.output_dir, f"{args.dataset}_pid_trace.csv")
    summary_path = os.path.join(args.output_dir, f"{args.dataset}_pid_summary.csv")
    df.to_csv(results_path, index=False)
    traces.to_csv(trace_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("PCCF-PID-lite Summary")
    print("=" * 80)
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved: {results_path}")
    print(f"Saved: {trace_path}")
    print(f"Saved: {summary_path}")
    for path in plot_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
