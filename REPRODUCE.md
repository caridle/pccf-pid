# Reproduction Checklist

Run commands from the repository root unless noted.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Verify Existing Paired Summaries

```bash
cd src
python -m pccf_llm.analyze_paired_results --results ../results/qwen15_memotrap/pccf_qwen_memotrap_pid_88/jsonl_pid_results.csv --reference standard --control pccf_pid_lite --output-dir ../results/qwen15_memotrap/reproduced_stats
python -m pccf_llm.analyze_paired_results --results ../results/qwen7b_memotrap/pccf_qwen7b_memotrap_pid_88/memotrap_pid_results.csv --reference standard --control pccf_pid_lite --output-dir ../results/qwen7b_memotrap/reproduced_stats
```

Expected headline results:

- Qwen2.5-1.5B: standard 0.6818, PCCF-PID 0.7614, paired improvements/regressions 8/1.
- Qwen2.5-7B: standard 0.8068, PCCF-PID 0.8750, paired improvements/regressions 6/0.

## 3. Re-run From Models

Use `README.md` commands and replace model identifiers or local paths as needed. Model weights are not included in this repository.
