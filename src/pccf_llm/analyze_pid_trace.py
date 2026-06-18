"""Summarize PCCF-PID trace behavior.

This diagnostic is intentionally descriptive. It reports whether the controller
intervened, whether the probe selected lower or higher precision, and how often
the final selected pi differs from standard inference.
"""

import argparse
import json
import os

import pandas as pd


def summarize_trace(results_path: str, trace_path: str, output_dir: str) -> dict:
    results = pd.read_csv(results_path)
    traces = pd.read_csv(trace_path)

    pid = results[results["baseline"] == "pccf_pid_lite"].copy()
    if pid.empty:
        raise ValueError(f"No pccf_pid_lite rows found in {results_path}")

    final_pi_counts = pid["pi"].round(6).value_counts().sort_index()
    changed = pid[pid["pi"].round(6) != 1.0]
    lower = pid[pid["pi"] < 1.0]
    higher = pid[pid["pi"] > 1.0]

    item_probe = traces.groupby("item_id", as_index=False).agg(
        low_gain=("low_gain", "first"),
        high_gain=("high_gain", "first"),
        first_pi=("pi", "first"),
        final_trace_pi=("pi", "last"),
        final_trace_margin=("eval_margin", "last"),
    )
    item_probe["probe_direction"] = "none"
    item_probe.loc[item_probe["first_pi"] < 1.0, "probe_direction"] = "lower"
    item_probe.loc[item_probe["first_pi"] > 1.0, "probe_direction"] = "higher"
    probe_counts = item_probe["probe_direction"].value_counts().to_dict()

    merged = pid.merge(item_probe, on="item_id", how="left")
    rows = []
    for task, sub in merged.groupby("task"):
        rows.append({
            "task": task,
            "n": int(len(sub)),
            "final_pi_changed": int((sub["pi"].round(6) != 1.0).sum()),
            "final_pi_lower": int((sub["pi"] < 1.0).sum()),
            "final_pi_higher": int((sub["pi"] > 1.0).sum()),
            "mean_pid_margin": float(sub["margin"].mean()),
            "mean_pid_p_target": float(sub["p_target"].mean()),
        })

    summary = {
        "results": results_path,
        "trace": trace_path,
        "n_items": int(len(pid)),
        "final_pi_changed": int(len(changed)),
        "final_pi_lower": int(len(lower)),
        "final_pi_higher": int(len(higher)),
        "final_pi_counts": {str(k): int(v) for k, v in final_pi_counts.items()},
        "probe_direction_counts": {str(k): int(v) for k, v in probe_counts.items()},
        "by_task": rows,
    }

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "pid_trace_summary.json")
    item_path = os.path.join(output_dir, "pid_trace_item_summary.csv")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    item_probe.to_csv(item_path, index=False)
    summary["summary_path"] = summary_path
    summary["item_summary_path"] = item_path
    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize PCCF-PID trace behavior.")
    parser.add_argument("--results", required=True, help="PID results CSV.")
    parser.add_argument("--trace", required=True, help="PID trace CSV.")
    parser.add_argument("--output-dir", default="pccf_pid_trace_stats")
    args = parser.parse_args()

    summary = summarize_trace(args.results, args.trace, args.output_dir)
    print("=" * 80)
    print("PCCF-PID trace summary")
    print("=" * 80)
    print(f"n_items = {summary['n_items']}")
    print(
        "final pi changed/lower/higher = "
        f"{summary['final_pi_changed']}/{summary['final_pi_lower']}/{summary['final_pi_higher']}"
    )
    print(f"final pi counts = {summary['final_pi_counts']}")
    print(f"probe direction counts = {summary['probe_direction_counts']}")
    print(f"Saved: {summary['summary_path']}")
    print(f"Saved: {summary['item_summary_path']}")


if __name__ == "__main__":
    main()
