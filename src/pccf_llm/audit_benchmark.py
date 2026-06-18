"""
Audit whether benchmark items satisfy PCCF-PID evidence-gate preconditions.

PCCF-PID-lite is only meaningful when the benchmark exposes a candidate answer
that is more directly supported by the current prompt/evidence than its
distractor. This script checks that condition before any model/PID run.
"""

import argparse
import json
import os
from dataclasses import asdict

import pandas as pd

from .run_hallucination_baselines import DATASET_CHOICES, build_items
from .run_pid_lite import evidence_direction, evidence_support


DEFAULT_THRESHOLDS = [0.05, 0.10, 0.25]


def audit_items(items, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for item in items:
        target_support = evidence_support(item.prompt, item.target)
        distractor_support = evidence_support(item.prompt, item.distractor)
        support_gap = target_support - distractor_support
        direction = evidence_direction(item)
        if direction is None:
            direction_label = "ambiguous"
        elif direction > 0:
            direction_label = "target"
        else:
            direction_label = "distractor"

        row = {
            "item_id": item.id,
            "task": item.task,
            "category": item.category,
            "target": item.target,
            "distractor": item.distractor,
            "target_support": target_support,
            "distractor_support": distractor_support,
            "support_gap": support_gap,
            "abs_support_gap": abs(support_gap),
            "evidence_direction": direction_label,
            "target_aligned": support_gap > 0,
            "ambiguous_direction": direction is None,
        }
        for threshold in thresholds:
            suffix = _threshold_suffix(threshold)
            row[f"abs_gap_ge_{suffix}"] = abs(support_gap) >= threshold
            row[f"target_gap_ge_{suffix}"] = support_gap >= threshold
            row[f"distractor_gap_ge_{suffix}"] = support_gap <= -threshold
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_audit(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for (task, category), sub in df.groupby(["task", "category"], dropna=False):
        row = {
            "task": task,
            "category": category,
            "n": int(len(sub)),
            "zero_direction": int((sub["evidence_direction"] == "ambiguous").sum()),
            "target_direction": int((sub["evidence_direction"] == "target").sum()),
            "distractor_direction": int((sub["evidence_direction"] == "distractor").sum()),
            "mean_target_support": float(sub["target_support"].mean()),
            "mean_distractor_support": float(sub["distractor_support"].mean()),
            "mean_support_gap": float(sub["support_gap"].mean()),
            "median_abs_support_gap": float(sub["abs_support_gap"].median()),
        }
        for threshold in thresholds:
            suffix = _threshold_suffix(threshold)
            row[f"abs_gap_ge_{suffix}"] = int(sub[f"abs_gap_ge_{suffix}"].sum())
            row[f"target_gap_ge_{suffix}"] = int(sub[f"target_gap_ge_{suffix}"].sum())
            row[f"distractor_gap_ge_{suffix}"] = int(sub[f"distractor_gap_ge_{suffix}"].sum())
            row[f"target_eligible_rate_{suffix}"] = float(sub[f"target_gap_ge_{suffix}"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task", "category"]).reset_index(drop=True)


def summarize_overall(df: pd.DataFrame, thresholds: list[float]) -> dict:
    result = {
        "n": int(len(df)),
        "zero_direction": int((df["evidence_direction"] == "ambiguous").sum()),
        "target_direction": int((df["evidence_direction"] == "target").sum()),
        "distractor_direction": int((df["evidence_direction"] == "distractor").sum()),
        "mean_target_support": float(df["target_support"].mean()),
        "mean_distractor_support": float(df["distractor_support"].mean()),
        "mean_support_gap": float(df["support_gap"].mean()),
        "median_abs_support_gap": float(df["abs_support_gap"].median()),
    }
    for threshold in thresholds:
        suffix = _threshold_suffix(threshold)
        result[f"abs_gap_ge_{suffix}"] = int(df[f"abs_gap_ge_{suffix}"].sum())
        result[f"target_gap_ge_{suffix}"] = int(df[f"target_gap_ge_{suffix}"].sum())
        result[f"distractor_gap_ge_{suffix}"] = int(df[f"distractor_gap_ge_{suffix}"].sum())
        result[f"target_eligible_rate_{suffix}"] = float(df[f"target_gap_ge_{suffix}"].mean())
    return result


def collect_examples(df: pd.DataFrame, limit: int, min_gap: float) -> dict:
    suffix = _threshold_suffix(min_gap)
    columns = [
        "item_id",
        "task",
        "category",
        "target",
        "distractor",
        "target_support",
        "distractor_support",
        "support_gap",
        "evidence_direction",
    ]
    ambiguous = df[df["evidence_direction"] == "ambiguous"].head(limit)
    wrong_way = df[df["distractor_gap_ge_" + suffix]].head(limit)
    eligible = df[df["target_gap_ge_" + suffix]].head(limit)
    return {
        "ambiguous": ambiguous[columns].to_dict(orient="records"),
        "distractor_supported": wrong_way[columns].to_dict(orient="records"),
        "target_eligible": eligible[columns].to_dict(orient="records"),
    }


def export_eligible_jsonl(items, df: pd.DataFrame, path: str, min_gap: float) -> int:
    by_id = {item.id: item for item in items}
    eligible = df[df["support_gap"] >= min_gap]
    with open(path, "w", encoding="utf-8") as handle:
        for _, row in eligible.iterrows():
            item = by_id[row["item_id"]]
            payload = asdict(item)
            payload["audit_target_support"] = float(row["target_support"])
            payload["audit_distractor_support"] = float(row["distractor_support"])
            payload["audit_support_gap"] = float(row["support_gap"])
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return int(len(eligible))


def _threshold_suffix(value: float) -> str:
    return str(value).replace(".", "p")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description="Audit benchmark evidence-gate compatibility for PCCF-PID.")
    parser.add_argument("--dataset", default="local", choices=DATASET_CHOICES)
    parser.add_argument("--dataset-file", default=None,
                        help="JSONL file for --dataset jsonl. Fields: prompt,target,distractor,id,category,task.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="pccf_benchmark_audit")
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--min-gap", type=float, default=0.25,
                        help="Primary gap threshold used for examples and target-eligibility headline.")
    parser.add_argument("--example-limit", type=int, default=5)
    parser.add_argument("--export-eligible-jsonl", action="store_true",
                        help="Write items with support_gap >= --min-gap as JSONL for downstream benchmark runs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    thresholds = sorted(set(args.thresholds + [args.min_gap]))
    items = build_items(args.dataset, args.limit, args.dataset_file)
    if not items:
        raise RuntimeError(f"No items loaded for dataset {args.dataset}")

    df = audit_items(items, thresholds)
    by_task = summarize_audit(df, thresholds)
    overall = summarize_overall(df, thresholds)
    examples = collect_examples(df, args.example_limit, args.min_gap)

    prefix = args.dataset
    if args.dataset == "jsonl" and args.dataset_file:
        prefix = os.path.splitext(os.path.basename(args.dataset_file))[0]
    rows_path = os.path.join(args.output_dir, f"{prefix}_audit_items.csv")
    summary_path = os.path.join(args.output_dir, f"{prefix}_audit_summary.csv")
    json_path = os.path.join(args.output_dir, f"{prefix}_audit_summary.json")
    eligible_path = os.path.join(args.output_dir, f"{prefix}_eligible_gap{_threshold_suffix(args.min_gap)}.jsonl")

    df.to_csv(rows_path, index=False)
    by_task.to_csv(summary_path, index=False)
    eligible_count = None
    if args.export_eligible_jsonl:
        eligible_count = export_eligible_jsonl(items, df, eligible_path, args.min_gap)
    payload = {
        "dataset": args.dataset,
        "dataset_file": args.dataset_file,
        "limit": args.limit,
        "thresholds": thresholds,
        "primary_min_gap": args.min_gap,
        "overall": overall,
        "by_task": by_task.to_dict(orient="records"),
        "examples": examples,
        "eligible_jsonl": eligible_path if args.export_eligible_jsonl else None,
        "eligible_jsonl_count": eligible_count,
        "schema_note": (
            "target_gap_ge_* counts items where direct prompt support favors the labeled target. "
            "PCCF-PID evidence/probe modes require at least a non-ambiguous evidence direction; "
            "strict context-faithfulness tests should have many target_gap_ge_0p25 items."
        ),
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False)

    headline_suffix = _threshold_suffix(args.min_gap)
    print("=" * 80)
    print(f"Benchmark audit: dataset={args.dataset}, n={len(df)}")
    print("=" * 80)
    print(by_task.round(4).to_string(index=False))
    print()
    print(
        "Primary target-eligible items "
        f"(support_gap >= {args.min_gap:g}): {overall[f'target_gap_ge_{headline_suffix}']}/{overall['n']}"
    )
    print(f"Saved: {rows_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {json_path}")
    if args.export_eligible_jsonl:
        print(f"Saved: {eligible_path} ({eligible_count} items)")


if __name__ == "__main__":
    main()
