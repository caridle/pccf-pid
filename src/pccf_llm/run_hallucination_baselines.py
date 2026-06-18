"""
Hallucination and context-faithfulness baselines for PCCF.

This runner compares:
  - standard decoding/scoring
  - global PCCF attention scaling
  - conflict-aware prompting
  - conflict-aware prompting plus PCCF scaling

The default dataset is the local counterfactual/context-conflict suite so the
script works without downloading external data. If `datasets` is installed,
TruthfulQA multiple-choice can be loaded with --dataset truthfulqa.
"""

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .experiment_data import COUNTERFACTUAL_TESTS, CONTRADICTION_TESTS
from .pccf_attention import patch_model_attention, restore_model_attention


DATASET_CHOICES = [
    "local",
    "counterfactual",
    "contradiction",
    "truthfulqa",
    "truthfulqa_mc",
    "fever_label",
    "memotrap",
    "jsonl",
]

CONFLICT_AWARE_INSTRUCTION = (
    "Answer using the information in the prompt. If the prompt conflicts with "
    "common knowledge or your prior knowledge, treat the prompt as authoritative."
)


@dataclass
class ScoredItem:
    id: str
    category: str
    prompt: str
    target: str
    distractor: str
    task: str


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    dtype: str = "float16",
    local_files_only: bool = False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if device != "cuda":
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return model, tokenizer


def format_prompt(tokenizer, user_text: str, conflict_aware: bool = False) -> str:
    system = "You are a helpful assistant."
    if conflict_aware:
        system = f"{system} {CONFLICT_AWARE_INSTRUCTION}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user_text}"


def normalized_logprob(model, tokenizer, prompt: str, answer: str) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_text = prompt + answer
    full_ids = tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt").to(model.device)
    if full_ids.shape[1] <= len(prompt_ids):
        return float("-inf")

    answer_ids = full_ids[0, len(prompt_ids):].tolist()
    ans_len = len(answer_ids)
    seq_len = full_ids.shape[1]

    with torch.no_grad():
        logits = model(full_ids).logits.float()

    total_logprob = 0.0
    for i, token_id in enumerate(answer_ids):
        pos = seq_len - ans_len + i - 1
        if 0 <= pos < logits.shape[1]:
            total_logprob += F.log_softmax(logits[0, pos], dim=-1)[token_id].item()
    return total_logprob / max(ans_len, 1)


def score_binary_options(model, tokenizer, prompt: str, target: str, distractor: str) -> dict:
    scores = {
        "target": normalized_logprob(model, tokenizer, prompt, f" {target}"),
        "distractor": normalized_logprob(model, tokenizer, prompt, f" {distractor}"),
    }
    vals = np.array([scores["target"], scores["distractor"]])
    vals -= np.max(vals)
    probs = np.exp(vals) / np.sum(np.exp(vals))
    return {
        "p_target": float(probs[0]),
        "p_distractor": float(probs[1]),
        "margin": float(probs[0] - probs[1]),
        "prefers_target": bool(probs[0] > probs[1]),
    }


def local_counterfactual_items(limit: int | None = None) -> list[ScoredItem]:
    items = []
    for test in COUNTERFACTUAL_TESTS:
        items.append(ScoredItem(
            id=test["id"],
            category=test["category"],
            prompt=f"{test['prefix']}\n\nQ: {test['question']}",
            target=test["answer_correct"],
            distractor=test["answer_prior"],
            task="counterfactual",
        ))
    return items[:limit] if limit else items


def local_contradiction_items(limit: int | None = None) -> list[ScoredItem]:
    items = []
    for test in CONTRADICTION_TESTS:
        context = " ".join(text for text, _ in test["contexts"])
        items.append(ScoredItem(
            id=test["id"],
            category="long_context",
            prompt=f"{context}\n\nQ: {test['question']}",
            target=test["answer_late"],
            distractor=test["answer_early"],
            task="contradiction",
        ))
    return items[:limit] if limit else items


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install `datasets` to use public benchmarks: pip install datasets"
        ) from exc
    return load_dataset


def truthfulqa_items(limit: int | None = None) -> list[ScoredItem]:
    load_dataset = _require_datasets()
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    items = []
    for idx, row in enumerate(ds):
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        if 1 not in labels or 0 not in labels:
            continue
        target = choices[labels.index(1)]
        distractor = choices[labels.index(0)]
        items.append(ScoredItem(
            id=f"truthfulqa_{idx}",
            category=row.get("category", "truthfulqa"),
            prompt=f"Q: {row['question']}\nA:",
            target=target,
            distractor=distractor,
            task="truthfulqa",
        ))
        if limit and len(items) >= limit:
            break
    return items


def truthfulqa_mc_items(limit: int | None = None) -> list[ScoredItem]:
    """
    EleutherAI's simplified TruthfulQA-MC format.

    This is kept separate from the original `truthful_qa` loader because the
    schemas differ: this variant stores a flat `choices` list and one label.
    """
    load_dataset = _require_datasets()
    ds = load_dataset("EleutherAI/truthful_qa_mc", split="validation")
    items = []
    for idx, row in enumerate(ds):
        choices = list(row["choices"])
        label = row["label"]
        if not choices:
            continue
        if not isinstance(label, int):
            label = choices.index(str(label)) if str(label) in choices else int(label)
        if label < 0 or label >= len(choices):
            continue
        target = choices[label]
        distractor = next((choice for i, choice in enumerate(choices) if i != label), None)
        if distractor is None:
            continue
        items.append(ScoredItem(
            id=f"truthfulqa_mc_{idx}",
            category=row.get("category", "truthfulqa_mc"),
            prompt=f"Q: {row['question']}\nA:",
            target=target,
            distractor=distractor,
            task="truthfulqa_mc",
        ))
        if limit and len(items) >= limit:
            break
    return items


def fever_label_items(limit: int | None = None, split: str = "labelled_dev") -> list[ScoredItem]:
    """
    FEVER claim-label scoring.

    Prefer the official FEVER loader when available. Newer `datasets` releases
    reject script-based datasets, so this falls back to `maxzoech/fever`, a
    script-free mirror that includes evidence text.
    """
    load_dataset = _require_datasets()
    load_attempts = [
        ("fever", ("v1.0",), [split, "validation", "dev", "train"]),
        ("maxzoech/fever", tuple(), ["test", "train"]),
    ]
    last_error = None
    ds = None
    for dataset_name, dataset_args, split_candidates in load_attempts:
        for split_name in split_candidates:
            try:
                ds = load_dataset(dataset_name, *dataset_args, split=split_name)
                break
            except Exception as exc:
                last_error = exc
        if ds is not None:
            break
    if ds is None:
        raise RuntimeError("Could not load FEVER from official loader or maxzoech/fever mirror.") from last_error

    label_map = {
        "SUPPORTS": ("supported", "refuted"),
        "REFUTES": ("refuted", "supported"),
    }
    items = []
    for idx, row in enumerate(ds):
        label = row.get("label")
        if label not in label_map:
            continue
        target, distractor = label_map[label]
        evidence = row.get("evidence")
        evidence_text = f"Evidence: {evidence}\n" if evidence else ""
        items.append(ScoredItem(
            id=f"fever_{row.get('id', idx)}",
            category="fever_label",
            prompt=(
                "Determine whether the claim is supported or refuted.\n"
                f"{evidence_text}"
                f"Claim: {row['claim']}\nAnswer:"
            ),
            target=target,
            distractor=distractor,
            task="fever_label",
        ))
        if limit and len(items) >= limit:
            break
    return items


def memotrap_items(limit: int | None = None) -> list[ScoredItem]:
    """
    MemoTrap memorization-trap scoring.

    Each item asks for a quote ending in a specified word while presenting a
    familiar phrase prefix. The target option follows the explicit instruction;
    the distractor is the memorized phrase completion.
    """
    load_dataset = _require_datasets()
    ds = load_dataset("Albertmade/memo-trap", split="train")
    items = []
    for idx, row in enumerate(ds):
        choices = list(row["classes"])
        answer_index = int(row["answer_index"])
        if answer_index < 0 or answer_index >= len(choices):
            continue
        target = str(choices[answer_index]).strip()
        distractor = next((str(choice).strip() for i, choice in enumerate(choices) if i != answer_index), None)
        if not target or not distractor:
            continue
        items.append(ScoredItem(
            id=f"memotrap_{idx}",
            category=f"round_{row.get('round', 'unknown')}",
            prompt=str(row["prompt"]),
            target=target,
            distractor=distractor,
            task="memotrap",
        ))
        if limit and len(items) >= limit:
            break
    return items


def jsonl_items(path: str, limit: int | None = None) -> list[ScoredItem]:
    """
    Load benchmark items from JSONL.

    Required fields: prompt, target, distractor.
    Optional fields: id, category, task.
    """
    items = []
    with open(path, encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"prompt", "target", "distractor"} - set(row)
            if missing:
                raise ValueError(f"JSONL row {idx + 1} missing fields: {sorted(missing)}")
            items.append(ScoredItem(
                id=str(row.get("id", f"jsonl_{idx}")),
                category=str(row.get("category", "jsonl")),
                prompt=str(row["prompt"]),
                target=str(row["target"]),
                distractor=str(row["distractor"]),
                task=str(row.get("task", "jsonl")),
            ))
            if limit and len(items) >= limit:
                break
    return items


def print_item_preview(items: list[ScoredItem], dataset: str, max_rows: int = 5):
    preview = pd.DataFrame([item.__dict__ for item in items[:max_rows]])
    print(f"Loaded {len(items)} items for dataset={dataset}")
    text = preview.to_string(index=False)
    print(text.encode("ascii", errors="backslashreplace").decode("ascii"))


def build_items(dataset: str, limit: int | None, dataset_file: str | None = None) -> list[ScoredItem]:
    if dataset == "local":
        items = local_counterfactual_items(None) + local_contradiction_items(None)
        return items[:limit] if limit else items
    if dataset == "counterfactual":
        return local_counterfactual_items(limit)
    if dataset == "contradiction":
        return local_contradiction_items(limit)
    if dataset == "truthfulqa":
        return truthfulqa_items(limit)
    if dataset == "truthfulqa_mc":
        return truthfulqa_mc_items(limit)
    if dataset == "fever_label":
        return fever_label_items(limit)
    if dataset == "memotrap":
        return memotrap_items(limit)
    if dataset == "jsonl":
        if not dataset_file:
            raise ValueError("--dataset-file is required when --dataset jsonl")
        return jsonl_items(dataset_file, limit)
    raise ValueError(f"unknown dataset: {dataset}")


def run_baselines(model, tokenizer, items: list[ScoredItem], pi: float, layer_mode: str) -> pd.DataFrame:
    baselines = [
        {"label": "standard", "pi": 1.0, "conflict_aware": False},
        {"label": f"global_pccf_pi{pi:g}", "pi": pi, "conflict_aware": False},
        {"label": "conflict_prompt", "pi": 1.0, "conflict_aware": True},
        {"label": f"conflict_prompt_pccf_pi{pi:g}", "pi": pi, "conflict_aware": True},
    ]

    rows = []
    for item in items:
        for baseline in baselines:
            restore_model_attention(model)
            if baseline["pi"] != 1.0:
                patch_model_attention(model, precision_pi=baseline["pi"], layer_mode=layer_mode)
            prompt = format_prompt(tokenizer, item.prompt, conflict_aware=baseline["conflict_aware"])
            scores = score_binary_options(model, tokenizer, prompt, item.target, item.distractor)
            rows.append({
                "item_id": item.id,
                "task": item.task,
                "category": item.category,
                "baseline": baseline["label"],
                "pi": baseline["pi"],
                "conflict_aware": baseline["conflict_aware"],
                "target": item.target,
                "distractor": item.distractor,
                **scores,
            })
        print(f"  [{item.task}:{item.id}] done")

    restore_model_attention(model)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["task", "baseline"])
        .agg(
            n=("item_id", "count"),
            target_win_rate=("prefers_target", "mean"),
            mean_p_target=("p_target", "mean"),
            mean_margin=("margin", "mean"),
        )
        .reset_index()
        .sort_values(["task", "mean_margin"], ascending=[True, False])
    )


def main():
    parser = argparse.ArgumentParser(description="Run PCCF hallucination/context-faithfulness baselines.")
    parser.add_argument("--dataset", default="local", choices=DATASET_CHOICES)
    parser.add_argument("--dataset-file", default=None,
                        help="JSONL file for --dataset jsonl. Fields: prompt,target,distractor,id,category,task.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--pi", type=float, default=0.7)
    parser.add_argument("--layer-mode", default="all",
                        help="PCCF layer selection: all, early, middle, late, lastN, or layers:i,j,k.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="pccf_hallucination_baselines")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Use only locally cached HuggingFace model files.")
    parser.add_argument("--dry-run-items", action="store_true",
                        help="Load and print benchmark items without loading a model.")
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
    df = run_baselines(model, tokenizer, items, args.pi, args.layer_mode)
    summary = summarize(df)

    results_path = os.path.join(args.output_dir, f"{args.dataset}_results.csv")
    summary_path = os.path.join(args.output_dir, f"{args.dataset}_summary.csv")
    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved: {results_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
