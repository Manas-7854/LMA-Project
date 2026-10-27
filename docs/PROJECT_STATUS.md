# Project status

## 2026-10-27, 16:39 UTC / 22:09 IST — Protocol amendment 1: M2 consolidation audit

* Team chose option 2 from protocol §11. Recorded as dated Amendment 1 (protocol §12).
* Ledger: `Audit.kind` ("audit" | "consolidation"); consolidations only extend the evidence of SATISFIED core
  obligations (`Ledger.evidence`), never change status. `scifact_v1` uses consolidated evidence.
* M2 and M2-random run the consolidation step after each round (`consolidate` in `src/oblig/systems/scifact.py`);
  `Context.m2_consolidate=False` gives the pre-amendment ablation.
* 41 tests passing (4 new for consolidation, 1 updated). Mock re-runs of m2/m2_random on 30 train_dev claims:
  9 / 6 consolidation calls, ≤ 9 abstracts, no errors (mock outputs only).

## 2026-10-27, 16:36 UTC / 22:06 IST — M2 and all added controls implemented (SciFact)

**Done:**
* `src/oblig/systems/scifact.py` now has 12 systems: b0, b1, b1_9, b1_sc, b2, m1, m1_shared, m1_untyped, m1_llmagg,
  m2, m2_random, m2_claimiter (design: protocol §11).
* LLM client takes a per-call temperature and sample index (B1-SC seeds 13..17; default cache keys unchanged).
  Ledger has an `atomic` type for M1-untyped and a `retrieval_rounds` log (targets, queries, new abstracts, stop reason).
* Runner reports iteration stats: claims entering iteration, verdict flips that fix or introduce errors, abstracts
  seen, stop reasons.
* Mock smoke runs of all 12 systems on 30 train_dev claims: no call errors, no invalid citations, M2 ≤ 9 abstracts
  with no duplicates. Mock scores mean nothing.
* 37 unit tests passing (9 new for iteration/controls).

**Open decision:** under the single-abstract SUPPORTS rule, M2 can rarely turn NEI into SUPPORTS, because obligations
resolved in round 0 cite old abstracts and are terminal (protocol §11, finding). The team must choose to keep the
rule or add a consolidation audit before Gate 3.

**Still blocked:** GPU for Qwen3-8B (Gate 7 hand-check, throughput pilot, B1-SC token matching); BGE-base download.

## 2026-10-27, 16:25 UTC / 21:55 IST — RQs frozen (Gate 2); B2 and M1 implemented (SciFact)

**Decision (team):** the refined RQ1/RQ2 and six added controls (protocol §9) are approved and frozen.

**Done:**
* Ledger: `src/oblig/ledger/schema.py` (typed obligations, append-only audits, enforced transitions, JSON-schema
  serialisation) and `aggregation.py` (`scifact_v1` proposal rule; `pubhealth_v1` candidate, untested on data).
* Planner/Auditor/B1/B2 prompts and schemas (`src/oblig/llm/prompts_scifact.py`); citation resolution against shown
  evidence (`src/oblig/agents/citations.py`).
* Systems B0, B1, B2, M1, M1-shared under one 5-abstract budget (`src/oblig/systems/scifact.py`); runner
  `scripts/run_scifact.py` (resumable, guards on held-out set and stale configs); SciFact claim-level and
  abstract-level metrics (`src/oblig/evaluation/scifact.py`).
* Mock smoke runs of all five systems on 30 train_dev claims (`results/predictions/smoke_sf_*.jsonl`): every
  ledger validates, budgets hold, no invalid citations. Mock scores are meaningless.
* 28 unit tests passing (verdict-rule edge cases, transitions, citations, budget, metrics).

**Not done / blockers:** Gate 7 (hand-check ledgers produced by Qwen3-8B) needs a GPU; M2, B1-9, and the other four
added controls are not implemented; BGE-base download still pending.

**Commands:** `python scripts/run_scifact.py --system {b0,b1,b2,m1,m1_shared} --eval-set train_dev --n 30 --profile mock_scifact --exp-id smoke_sf_<system>`; `python -m pytest -q tests`

## 2026-10-26, 19:58 UTC / 2026-10-27, 01:28 IST — SciFact retrieval pipeline (Phase 5, SciFact)

**Decision (team):** SciFact is the primary testbed for RQ1/RQ2; PUBHEALTH is the secondary testbed.

**Done (measured):**
* `src/oblig/data/scifact.py`: claims, corpus, claim-level labels, gold rationale sentence ids, `train_dev` split.
* Found that the shipped SciFact CV folds mix claims_train and claims_dev (55–68 dev claims in every fold's dev
  part). Development therefore uses `train_dev` (200 claims from claims_train only, seed 13).
* `src/oblig/retrieval/{dense,hybrid}.py`: sentence-transformers retriever with on-disk embedding cache; RRF fusion.
* Retrieval on train_dev (125 claims with evidence): Recall@5 BM25 0.887, MiniLM 0.865, hybrid 0.882;
  Recall@10 0.913 / 0.905 / 0.956. claims_dev (held-out) not yet scored.
* 16 unit tests passing.

**Commands run:** `python scripts/eval_retrieval_scifact.py --eval-set train_dev`; `python -m pytest -q tests`

* Literature review written (`docs/LITERATURE_REVIEW.md`, 27 works, 54 references checked against OpenAlex/arXiv; uncertain entries marked). Novelty: partial overlap; reframed as a budget-controlled study. Proposed RQ refinements and six added controls in EXPERIMENT_PROTOCOL §9, pending team approval (Gate 2 not yet passed).

**Blockers:** BGE-base (proposal's dense model) not cached — needs a ~440 MB download; MiniLM used meanwhile.
GPU for Qwen3-8B still unresolved.

## 2026-10-26, 10:42 UTC / 16:12 IST — Phase 0 and Phase 1 complete; Gate 4 partially

**Done (measured):**
* Repository audit (`docs/PROJECT_AUDIT.md`): repo was data-only; no proposal file present; CPU-only machine.
* Dataset audit (`reports/dataset_audit.md/.json`, figures): 9,804 / 1,214 / 1,233 valid rows; 37 malformed rows excluded.
* Leakage: claim_id encodes label (shortcut macro-F1 0.594 on test); genre/style shortcuts 0.48–0.58; verdict
  sentences masked (1,811 sentences, 11.3% of articles).
* BM25 retrieval pilot on all 1,214 dev claims over the pooled masked corpus (`results/metrics/retrieval_bm25_dev.json`).
* B0/B1 pipeline smoke test end to end with the mock backend on 50 dev claims (`results/predictions/smoke_mock_b{0,1}.jsonl`).
  The mock only checks plumbing; its scores mean nothing.
* 10 unit tests passing.

**Commands run:**
```
python scripts/audit_dataset.py
python scripts/shortcut_baselines.py
python scripts/eval_retrieval.py
python scripts/run_baselines.py --system b0 --split dev --n 50 --profile mock --exp-id smoke_mock_b0
python scripts/run_baselines.py --system b1 --split dev --n 50 --profile mock --exp-id smoke_mock_b1
python -m pytest -q tests
```

**Blockers:**
1. No CUDA GPU → no Qwen3-8B/vLLM. Need cluster access (or authorised alternative).
2. Real-model smoke test needs a model download (locally cached Qwen2.5-0.5B lacks weights).
3. Team decision: SciFact primary vs PUBHEALTH primary (protocol §8).

**Next:** literature review (Phase 2), SciFact loader/retrieval with gold-rationale recall, dense + hybrid retrieval.
