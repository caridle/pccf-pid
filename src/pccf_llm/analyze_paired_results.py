"""
Paired statistical summaries for PCCF binary-option experiments.

The script compares two baselines on matched item_id rows and reports:
  - target-preference rates with Wilson confidence intervals
  - paired improvement/regression counts
  - exact two-sided McNemar/binomial sign-test p-value
  - mean margin delta with bootstrap confidence interval
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def exact_mcnemar_p(improvements: int, regressions: int) -> float:
    discordant = improvements + regressions
    if discordant == 0:
        return 1.0
    k = min(improvements, regressions)
    cdf = sum(math.comb(discordant, i) for i in range(k + 1)) / (2 ** discordant)
    return min(1.0, 2 * cdf)


def bootstrap_ci(values: np.ndarray, reps: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[idx].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def compare(
    path: str,
    control: str,
    reference: str,
    output_dir: str,
    reps: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(path)
    required = {"item_id", "baseline", "prefers_target", "margin", "p_target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    ref = df[df["baseline"] == reference].set_index("item_id")
    ctl = df[df["baseline"] == control].set_index("item_id")
    common = ref.index.intersection(ctl.index)
    if len(common) == 0:
        raise ValueError(f"No overlapping item_id rows for {reference!r} and {control!r}")
    ref = ref.loc[common].sort_index()
    ctl = ctl.loc[common].sort_index()

    paired = pd.DataFrame({
        "item_id": ref.index,
        "reference_prefers_target": ref["prefers_target"].astype(bool).to_numpy(),
        "control_prefers_target": ctl["prefers_target"].astype(bool).to_numpy(),
        "reference_margin": ref["margin"].astype(float).to_numpy(),
        "control_margin": ctl["margin"].astype(float).to_numpy(),
        "reference_p_target": ref["p_target"].astype(float).to_numpy(),
        "control_p_target": ctl["p_target"].astype(float).to_numpy(),
    })
    paired["margin_delta"] = paired["control_margin"] - paired["reference_margin"]
    paired["p_target_delta"] = paired["control_p_target"] - paired["reference_p_target"]
    paired["paired_outcome"] = np.select(
        [
            (~paired["reference_prefers_target"]) & paired["control_prefers_target"],
            paired["reference_prefers_target"] & (~paired["control_prefers_target"]),
            paired["reference_prefers_target"] & paired["control_prefers_target"],
        ],
        ["improvement", "regression", "both_correct"],
        default="both_wrong",
    )

    n = len(paired)
    ref_success = int(paired["reference_prefers_target"].sum())
    ctl_success = int(paired["control_prefers_target"].sum())
    improvements = int((paired["paired_outcome"] == "improvement").sum())
    regressions = int((paired["paired_outcome"] == "regression").sum())
    ref_ci = wilson_interval(ref_success, n)
    ctl_ci = wilson_interval(ctl_success, n)
    margin_delta = paired["margin_delta"].to_numpy()
    p_delta = paired["p_target_delta"].to_numpy()
    margin_ci = bootstrap_ci(margin_delta, reps, seed)
    p_ci = bootstrap_ci(p_delta, reps, seed + 1)

    summary = {
        "input": path,
        "reference": reference,
        "control": control,
        "n": n,
        "reference_target_wins": ref_success,
        "control_target_wins": ctl_success,
        "reference_target_win_rate": ref_success / n,
        "control_target_win_rate": ctl_success / n,
        "reference_wilson_95": ref_ci,
        "control_wilson_95": ctl_ci,
        "paired_improvements": improvements,
        "paired_regressions": regressions,
        "both_correct": int((paired["paired_outcome"] == "both_correct").sum()),
        "both_wrong": int((paired["paired_outcome"] == "both_wrong").sum()),
        "mcnemar_exact_p": exact_mcnemar_p(improvements, regressions),
        "mean_reference_margin": float(paired["reference_margin"].mean()),
        "mean_control_margin": float(paired["control_margin"].mean()),
        "mean_margin_delta": float(margin_delta.mean()),
        "margin_delta_bootstrap_95": margin_ci,
        "mean_p_target_delta": float(p_delta.mean()),
        "p_target_delta_bootstrap_95": p_ci,
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
    }

    os.makedirs(output_dir, exist_ok=True)
    label = f"{reference}_vs_{control}".replace("/", "_")
    paired_path = os.path.join(output_dir, f"{label}_paired_items.csv")
    summary_path = os.path.join(output_dir, f"{label}_paired_summary.json")
    paired.to_csv(paired_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["paired_items_path"] = paired_path
    summary["summary_path"] = summary_path
    return summary, paired


def main():
    parser = argparse.ArgumentParser(description="Analyze paired PCCF binary-option results.")
    parser.add_argument("--results", required=True, help="CSV with item_id, baseline, prefers_target, margin.")
    parser.add_argument("--reference", default="standard")
    parser.add_argument("--control", required=True)
    parser.add_argument("--output-dir", default="pccf_paired_stats")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    summary, _ = compare(
        args.results,
        args.control,
        args.reference,
        args.output_dir,
        args.bootstrap_reps,
        args.seed,
    )
    print("=" * 80)
    print(f"Paired comparison: {args.reference} vs {args.control}")
    print("=" * 80)
    print(f"n = {summary['n']}")
    print(
        f"{args.reference}: {summary['reference_target_win_rate']:.4f} "
        f"Wilson95=({summary['reference_wilson_95'][0]:.4f}, {summary['reference_wilson_95'][1]:.4f})"
    )
    print(
        f"{args.control}: {summary['control_target_win_rate']:.4f} "
        f"Wilson95=({summary['control_wilson_95'][0]:.4f}, {summary['control_wilson_95'][1]:.4f})"
    )
    print(
        "paired improvements/regressions = "
        f"{summary['paired_improvements']}/{summary['paired_regressions']}, "
        f"McNemar exact p = {summary['mcnemar_exact_p']:.6f}"
    )
    print(
        f"mean margin delta = {summary['mean_margin_delta']:.6f} "
        f"bootstrap95=({summary['margin_delta_bootstrap_95'][0]:.6f}, "
        f"{summary['margin_delta_bootstrap_95'][1]:.6f})"
    )
    print(f"Saved: {summary['paired_items_path']}")
    print(f"Saved: {summary['summary_path']}")


if __name__ == "__main__":
    main()
