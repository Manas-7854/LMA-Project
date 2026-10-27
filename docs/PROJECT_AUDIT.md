# Project audit (Phase 0)

Audited 2026-10-26, 10:42 UTC / 16:12 IST, on a clone of https://github.com/Nishanth-nishu/LMA-Project
(branch `main`, commit `3e0ce53`, identical to the upstream `Manas-7854/LMA-Project` HEAD).

## 1. What existed before this work

| item | state |
|---|---|
| git history | 1 commit ("added the dataset", 2026-10-21) |
| code, notebooks, configs, README, dependency files | **none** |
| evaluation code, checkpoints, experiment outputs, logs | none |
| `data/PUBHEALTH/{train,dev,test}.tsv` | 77 MB total; 9,832 / 1,221 / 1,235 parsed rows |
| `data/raw/scifact/` + `scifact.tar.gz` | official SciFact release: claims train/dev/test, 5,183-abstract corpus, 5 CV folds; macOS `._*` resource-fork files included |
| project proposal | **not in the repository**; the protocol here is built from the proposal as summarised in the project brief |

## 2. Environment

* Windows 11, Intel CPU (20 logical cores), 15.7 GB RAM, Intel UHD 770 iGPU, **no CUDA GPU**, 228 GB free disk.
* Python 3.11.9; torch 2.14.0+cpu, transformers 5.19.0, sentence-transformers 6.1.0, scikit-learn 1.9.1,
  pandas 3.0.5. vLLM is not installed and cannot run here (needs a CUDA GPU).
* HF cache holds `Qwen2.5-0.5B-Instruct` (config/tokenizer only — weights missing), `all-MiniLM-L6-v2`,
  `ms-marco-MiniLM-L-6-v2`. Not BGE-base, not Qwen3-8B.
* **Consequence:** Qwen3-8B via vLLM (proposal) is infeasible on this machine. Pipelines are built
  backend-agnostic (OpenAI-compatible endpoint) so they run unchanged against a GPU vLLM server.

## 3. Added in this phase

| path | purpose |
|---|---|
| `src/oblig/data/pubhealth.py` | TSV loader, validity filter, canonical labels, opaque example ids, allowed-input whitelist |
| `src/oblig/data/leakage.py` | cross-split overlap, near-duplicates, verdict-phrase detection |
| `src/oblig/data/masking.py` | sentence splitting, verdict-sentence masking |
| `src/oblig/retrieval/{corpus,bm25}.py` | deterministic passage corpus with provenance; sparse BM25 |
| `src/oblig/llm/{client,prompts}.py` | hf / openai(vLLM) / mock backends, SQLite cache, JSON-schema validation, failure recording; B0/B1 prompts |
| `src/oblig/evaluation/classification.py` | macro-F1, per-class, confusion matrix, paired bootstrap |
| `src/oblig/utils/cache.py` | cache keyed on model, revision, decoding, prompt version, messages |
| `scripts/audit_dataset.py`, `scripts/shortcut_baselines.py`, `scripts/eval_retrieval.py`, `scripts/run_baselines.py` | audit, leakage baselines, retrieval pilot, B0/B1 runner with run manifest |
| `configs/{dataset,model,retrieval,experiment_matrix}.yaml` | configuration |
| `tests/test_foundation.py` | 10 tests (all passing) |
| `reports/dataset_audit.{md,json}`, `reports/figures/*.png`, `reports/near_duplicate_claims.csv` | dataset audit |

Raw data files were not modified.

## 4. Conflicts between the proposal and the PUBHEALTH direction

1. **Evidence.** SciFact has sentence-level gold rationales; PUBHEALTH has none. RQ2's "evidence recall" is only
   measurable on SciFact (PUBHEALTH gives silver document-level signals).
2. **Labels.** 3-way SUPPORTS/REFUTES/NEI vs 4-way true/false/mixture/unproven. `mixture` is not NEI (it asserts
   conflicting evidence); `unproven` is closest to NEI. The proposal's verdict rule has no `mixture` outcome and
   needs a separate PUBHEALTH rule.
3. **Corpus.** SciFact supplies an abstract corpus. PUBHEALTH's main_text is the fact-checker's own write-up
   (written knowing the verdict), or for "true" wire news, the news article itself. Using it as evidence requires
   masking and makes the task partly reading comprehension.
4. **Shortcuts.** PUBHEALTH labels leak through claim_id, genre and metadata (macro-F1 0.48–0.59 without any LLM);
   SciFact has no comparable artefact found so far.
5. **SciFact test labels are unpublished**: the proposal's SciFact evaluation must run on dev.

## 5. Technical debt / risks

* No GPU: every reported LLM result depends on securing cluster access.
* `fact_checkers` field is not cleanly parseable (comma-split credentials).
* Source URLs: 2,318 malformed; external pages are not downloaded and are not part of the corpus.
* `._*` macOS files in `data/raw/scifact/` are junk (left in place; not deleted without team agreement).

## 6. Recommended sequence

1. Team decisions (EXPERIMENT_PROTOCOL §8): primary testbed, GPU access.
2. Phase 2 literature review → freeze RQs.
3. SciFact loader + retrieval (abstract corpus, gold rationale recall) mirroring the PUBHEALTH modules.
4. Dense (BGE-base) + RRF hybrid; measure against BM25 on both datasets' dev.
5. On GPU: Qwen3-8B pilot (50 dev claims) for B0/B1 → measured throughput → compute budget.
6. B2, then M1 (planner/auditor/ledger/aggregation) with hand-checked ledgers (Gate 7), then M2 (Gate 8).
7. Freeze protocol (Gate 3) → test runs → analysis, annotation, report.
