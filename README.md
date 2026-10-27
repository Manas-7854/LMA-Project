# Verify What Must Be True: Obligation Ledgers for Scientific Claim Verification

LMA Group Project 2026 — Team Comfortably Numpy.

## Setup
```
pip install -r requirements.txt
python -m pytest -q tests
```

## Reproduce the current results
```
python scripts/audit_dataset.py          # reports/dataset_audit.{md,json}, reports/figures/
python scripts/shortcut_baselines.py     # results/metrics/shortcut_baselines.json
python scripts/eval_retrieval.py         # results/metrics/retrieval_bm25_dev.json
python scripts/run_baselines.py --system b1 --split dev --n 50 --profile mock --exp-id smoke_mock_b1
```
Real runs use `--profile qwen3_8b_vllm` against a vLLM OpenAI-compatible server (`configs/model.yaml`).

## Docs
`docs/PROJECT_AUDIT.md`, `docs/PROJECT_STATUS.md`, `docs/EXPERIMENT_PROTOCOL.md` (draft), `reports/dataset_audit.md`.
