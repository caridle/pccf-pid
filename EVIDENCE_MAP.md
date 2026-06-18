# Evidence Map

This file maps the manuscript tables and claims to committed artifacts.

## Benchmark Audit

- Table 2, local PCCF suite: `data/local_audit_summary.csv`, `data/local_audit_items.csv`
- Table 2, MemoTrap eligible subset: `data/memotrap_eligible_gap0p25_audit_summary.csv`, `data/memotrap_eligible_gap0p25_audit_items.csv`, `data/memotrap_eligible_gap0p25.jsonl`
- Table 2, TruthfulQA adapter: `data/truthfulqa_audit_summary.csv`, `data/truthfulqa_audit_items.csv`
- Table 2, FEVER adapter: `data/fever_label_audit_summary.csv`, `data/fever_label_audit_items.csv`

## Controlled PCCF Experiments

- Scalar concept-drift summary: `results/pccf_scalar_majorrev_summary.csv`
- Scalar p-values and sensitivity: `results/pccf_scalar_majorrev_pvalues_mse.csv`, `results/pccf_scalar_majorrev_pvalues_recovery.csv`, `results/pccf_scalar_majorrev_recovery_sensitivity.csv`
- Symbolic rule-shift summary: `results/pccf_toylm_majorrev_summary.csv`
- Source scripts: `src/verify_pccf.py`, `src/verify_pccf_attention.py`

## Local LLM Static and PID-lite Results

- Table 7 and Table 8 local/static evidence is reproduced by the LLM scripts in `src/pccf_llm/` and the reported local result artifacts in the main project workspace. The public GitHub package centers the auditable MemoTrap subset and paired statistics below.

## Qwen2.5-1.5B MemoTrap

- Baselines summary: `results/qwen15_memotrap/pccf_qwen_memotrap_baseline_88/jsonl_summary.csv`
- PCCF-PID summary: `results/qwen15_memotrap/pccf_qwen_memotrap_pid_88/jsonl_pid_summary.csv`
- PCCF-PID item results: `results/qwen15_memotrap/pccf_qwen_memotrap_pid_88/jsonl_pid_results.csv`
- PCCF-PID trace: `results/qwen15_memotrap/pccf_qwen_memotrap_pid_88/jsonl_pid_trace.csv`
- Standard vs PCCF-PID paired comparison: `results/qwen15_memotrap/pccf_qwen_memotrap_stats/standard_vs_pccf_pid_lite_paired_summary.json`
- Standard vs conflict-prompt + PCCF paired comparison: `results/qwen15_memotrap/pccf_qwen_memotrap_stats/standard_vs_conflict_prompt_pccf_pi0.7_paired_summary.json`
- Conflict-prompt + PCCF vs PCCF-PID paired comparison: `results/qwen15_memotrap/pccf_qwen_memotrap_stats/conflict_prompt_pccf_pi0.7_vs_pccf_pid_lite_paired_summary.json`

## Qwen2.5-7B MemoTrap

- Baselines summary: `results/qwen7b_memotrap/pccf_qwen7b_memotrap_baseline_88/memotrap_summary.csv`
- PCCF-PID summary: `results/qwen7b_memotrap/pccf_qwen7b_memotrap_pid_88/memotrap_pid_summary.csv`
- PCCF-PID item results: `results/qwen7b_memotrap/pccf_qwen7b_memotrap_pid_88/memotrap_pid_results.csv`
- PCCF-PID trace: `results/qwen7b_memotrap/pccf_qwen7b_memotrap_pid_88/memotrap_pid_trace.csv`
- Standard vs PCCF-PID paired comparison: `results/qwen7b_memotrap/pccf_qwen7b_memotrap_stats/standard_vs_pccf_pid_lite_paired_summary.json`
- Standard vs global PCCF paired comparison: `results/qwen7b_memotrap/pccf_qwen7b_memotrap_stats/global_pccf/standard_vs_global_pccf_pi0.7_paired_summary.json`
- Global PCCF vs PCCF-PID paired comparison: `results/qwen7b_memotrap/pccf_qwen7b_memotrap_stats/pid_vs_global/global_pccf_pi0.7_vs_pccf_pid_lite_paired_summary.json`

## PID-term Ablation

- P-only paired summary: `results/ablation/pccf_qwen_memotrap_pid_ablation_stats/p_only/standard_vs_pccf_pid_lite_paired_summary.json`
- I-only paired summary: `results/ablation/pccf_qwen_memotrap_pid_ablation_stats/i_only/standard_vs_pccf_pid_lite_paired_summary.json`
- D-only paired summary: `results/ablation/pccf_qwen_memotrap_pid_ablation_stats/d_only/standard_vs_pccf_pid_lite_paired_summary.json`

## Trace Evidence

- Trace aggregate: `results/trace/pccf_qwen_memotrap_trace_stats/pid_trace_summary.json`
- Item-level trace summary: `results/trace/pccf_qwen_memotrap_trace_stats/pid_trace_item_summary.csv`

## Main Analysis Scripts

- Benchmark audit: `src/pccf_llm/audit_benchmark.py`
- PCCF attention scaling: `src/pccf_llm/pccf_attention.py`
- PCCF-PID controller: `src/pccf_llm/pccf_controller.py`
- Baseline runs: `src/pccf_llm/run_hallucination_baselines.py`
- PID-lite runs: `src/pccf_llm/run_pid_lite.py`
- Paired statistics: `src/pccf_llm/analyze_paired_results.py`
- Trace analysis: `src/pccf_llm/analyze_pid_trace.py`
