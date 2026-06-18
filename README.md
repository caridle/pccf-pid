# PCCF-PID Evidence-Gated Precision Control

This repository contains the code, audited data subset, and result artifacts supporting the associated manuscript:

**Evidence-Gated Precision Control for Context-Faithful Language Model Inference: A PCCF-PID Framework**

The repository is organized for review and reproducibility. It does not include model weights.

## Contents

- `src/pccf_llm/`: LLM attention-scaling, PCCF-PID, benchmark-audit, trace-analysis, and paired-statistics scripts.
- `src/verify_pccf.py`: controlled scalar concept-drift experiment.
- `src/verify_pccf_attention.py`: controlled symbolic rule-shift experiment.
- `data/`: benchmark audit files and the audited MemoTrap eligible JSONL subset.
- `results/`: result CSV/JSON artifacts reported in the manuscript.
- `EVIDENCE_MAP.md`: table-by-table map from manuscript claims to committed artifacts.

## Environment

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

For GPU runs, install a PyTorch build compatible with your CUDA version before running the LLM scripts.

## Reproduce MemoTrap Runs

The associated manuscript reports Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct runs on the audited MemoTrap subset in `data/memotrap_eligible_gap0p25.jsonl`.

Set model paths as needed. The 1.5B model can be loaded by Hugging Face model id if available locally or online. The 7B command below assumes a local model directory.

```bash
cd src
python -m pccf_llm.run_hallucination_baselines --dataset jsonl --dataset-file ../data/memotrap_eligible_gap0p25.jsonl --limit 88 --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype float16 --output-dir ../results/qwen15_memotrap/reproduced_baseline
python -m pccf_llm.run_pid_lite --dataset jsonl --dataset-file ../data/memotrap_eligible_gap0p25.jsonl --limit 88 --feedback-mode probe --selection final --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype float16 --output-dir ../results/qwen15_memotrap/reproduced_pid
```

For 7B:

```bash
cd src
python -m pccf_llm.run_hallucination_baselines --dataset memotrap --limit 88 --model /path/to/Qwen2.5-7B-Instruct --device cuda --dtype float16 --output-dir ../results/qwen7b_memotrap/reproduced_baseline --local-files-only
python -m pccf_llm.run_pid_lite --dataset memotrap --limit 88 --feedback-mode probe --selection final --model /path/to/Qwen2.5-7B-Instruct --device cuda --dtype float16 --output-dir ../results/qwen7b_memotrap/reproduced_pid --local-files-only
```

## Paired Statistics

```bash
cd src
python -m pccf_llm.analyze_paired_results --results ../results/qwen15_memotrap/pccf_qwen_memotrap_pid_88/jsonl_pid_results.csv --reference standard --control pccf_pid_lite --output-dir ../results/qwen15_memotrap/reproduced_stats
python -m pccf_llm.analyze_paired_results --results ../results/qwen7b_memotrap/pccf_qwen7b_memotrap_pid_88/memotrap_pid_results.csv --reference standard --control pccf_pid_lite --output-dir ../results/qwen7b_memotrap/reproduced_stats
```

The committed `results/` directory already contains the paired summaries used by the associated manuscript.

## Benchmark Audit

```bash
cd src
python -m pccf_llm.audit_benchmark --dataset memotrap --limit 100 --output-dir ../data/reproduced_audit --export-eligible-jsonl
```

The audit checks whether candidate answers expose a usable evidence direction before PCCF-PID is applied.

## Main Reported Files

- Qwen2.5-1.5B MemoTrap:
  - `results/qwen15_memotrap/pccf_qwen_memotrap_baseline_88/`
  - `results/qwen15_memotrap/pccf_qwen_memotrap_pid_88/`
  - `results/qwen15_memotrap/pccf_qwen_memotrap_stats/`
- Qwen2.5-7B MemoTrap:
  - `results/qwen7b_memotrap/pccf_qwen7b_memotrap_baseline_88/`
  - `results/qwen7b_memotrap/pccf_qwen7b_memotrap_pid_88/`
  - `results/qwen7b_memotrap/pccf_qwen7b_memotrap_stats/`

## Notes

The LLM experiments use binary candidate scoring to isolate the precision-control mechanism. They are not open-ended generation benchmarks, and the associated manuscript does not claim broad hallucination elimination.
