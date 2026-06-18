"""Free-generation smoke test for PCCF benchmark items.

This script is a qualitative boundary check, not the main evaluation protocol.
It samples a small number of items and records short greedy generations under
standard prompting and conflict-aware prompting. Optional PCCF attention scaling
can be enabled for a rough actuator check, but the output should not be treated
as a scored hallucination benchmark without a separate generation evaluator.
"""

import argparse
import json
import os

import torch

from .pccf_attention import patch_model_attention, restore_model_attention
from .run_hallucination_baselines import build_items, DATASET_CHOICES, format_prompt, load_model_and_tokenizer


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_generation_smoke(
    model,
    tokenizer,
    items,
    max_new_tokens: int,
    pccf_pi: float | None,
    layer_mode: str,
) -> list[dict]:
    rows = []
    for item in items:
        for label, conflict_aware, pi in [
            ("standard_generation", False, None),
            ("conflict_prompt_generation", True, None),
            ("pccf_generation", False, pccf_pi),
        ]:
            if pi is None and label == "pccf_generation":
                continue
            restore_model_attention(model)
            if pi is not None:
                patch_model_attention(model, precision_pi=pi, layer_mode=layer_mode)
            prompt = format_prompt(tokenizer, item.prompt, conflict_aware=conflict_aware)
            completion = generate_text(model, tokenizer, prompt, max_new_tokens)
            rows.append({
                "item_id": item.id,
                "task": item.task,
                "category": item.category,
                "condition": label,
                "pi": pi if pi is not None else 1.0,
                "target": item.target,
                "distractor": item.distractor,
                "completion": completion,
            })
            print(f"[{item.id}:{label}] {completion}")
    restore_model_attention(model)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run qualitative free-generation smoke tests.")
    parser.add_argument("--dataset", default="jsonl", choices=DATASET_CHOICES)
    parser.add_argument("--dataset-file", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--pccf-pi", type=float, default=None)
    parser.add_argument("--layer-mode", default="late")
    parser.add_argument("--output-dir", default="pccf_generation_smoke")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    items = build_items(args.dataset, args.limit, args.dataset_file)
    if not items:
        raise RuntimeError(f"No items loaded for dataset {args.dataset}")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.device,
        args.dtype,
        local_files_only=args.local_files_only,
    )
    rows = run_generation_smoke(
        model,
        tokenizer,
        items,
        args.max_new_tokens,
        args.pccf_pi,
        args.layer_mode,
    )
    path = os.path.join(args.output_dir, f"{args.dataset}_generation_smoke.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
