# Detailed explanation of the work so far

**Project:** Verify What Must Be True: Obligation Ledgers for Scientific Claim Verification
(LMA Group Project 2026, Team Comfortably Numpy)
**Covers:** 2026-10-26 to 2026-10-28; branch `obligation-ledger-foundation`, on top of the dataset-only commit `3e0ce53`.

> **Read this first.** No real language model has been run yet. Every LLM system has run only on a scripted
> stand-in ("mock") that exercises the code paths. Mock scores mean nothing about whether the method works.
> The numbers below that *are* real are the dataset statistics, the non-LLM shortcut baselines, and the retrieval
> measurements.

---

## 1. Where we started

The repository held two datasets and nothing else: no code, configs, README or proposal file.

| dataset | location | contents |
|---|---|---|
| SciFact | `data/raw/scifact/` | claims (train 809, dev 300, test 300 **unlabelled**), a corpus of 5,183 abstracts, 5 cross-validation folds |
| PUBHEALTH | `data/PUBHEALTH/` | train/dev/test TSVs of public-health claims, with fact-check articles, explanations and four labels |

The machine used is a Windows laptop with an Intel CPU, 16 GB RAM and **no CUDA GPU**. The proposal's model
(Qwen3-8B on vLLM) therefore cannot run here. All code talks to an OpenAI-compatible endpoint, so it will run
unchanged against a GPU vLLM server.

**Both datasets in the repo are used; nothing was downloaded from outside.** The team decided that SciFact is the
primary testbed for the research questions and PUBHEALTH the secondary one (§5).

---

## 2. Phase 0–1: auditing the data (PUBHEALTH)

Script: `scripts/audit_dataset.py` → `reports/dataset_audit.md`, `reports/dataset_audit.json`, `reports/figures/`.
All numbers are measured from the files.

### 2.1 Size and labels

| split | rows parsed | valid | malformed (excluded, not repaired) |
|---|---|---|---|
| train | 9,832 | 9,804 | 28 |
| dev | 1,221 | 1,214 | 7 |
| test | 1,235 | 1,233 | 2 |

- Labels are `true / false / mixture / unproven`. The raw files say **"mixture"**, not "mixed".
- Pooled counts are 6,306 / 3,769 / 1,799 / 377. The brief's estimates (~6,000 / 3,000 / 1,800 / 300) match these
  pooled totals, not the train split.
- `unproven` is only 3–4% of each split (45 test claims), so its F1 will be noisy.
- The 37 malformed rows come from records broken by embedded line breaks, mostly TruthOrFiction items.
- Records without sources: 5,706, confirming the teammate's "~5,700".
- Article length: mean 714 words, median 610. Records without sources: mean 612, median 490.

### 2.2 Leakage: the most important finding

1. **`claim_id` gives away the label.** Every record with an id below 8936 is "true" (4,151 records, mostly
   wire-news headlines). A rule using nothing but id ranges reaches **macro-F1 0.594 on test**. The id is therefore
   never shown to a model. Internal ids are opaque hashes, and passages appear in prompts as `E1…Ek`.
   My own first smoke test leaked this through passage ids; the leak was caught and fixed.
2. **Writing style predicts the label.** "True" records are mostly news articles; false and mixture records are
   mostly fact-check write-ups. Simple text classifiers with no LLM reach on test:

   | shortcut baseline (`scripts/shortcut_baselines.py`) | test macro-F1 |
   |---|---|
   | always "true" | 0.164 |
   | claim text only (TF-IDF + logistic regression) | 0.482 |
   | metadata only (fact-checker, subjects, has-sources) | 0.495 |
   | article text with verdict sentences removed | 0.576 |
   | `claim_id` ranges | 0.594 |

   Any LLM result on PUBHEALTH has to be compared against these, and reported separately for news and fact-check
   records.
3. **Articles state their own verdict.** PolitiFact's rating word is already stripped ("We rate it ."), but the
   sentence and the rating definitions remain. These sentences are masked: 1,811 sentences in 11% of articles
   (`src/oblig/data/masking.py`).
4. **Explanation overlap.** The gold explanation appears word-for-word in the article for 4.7% of records. The
   explanation is never a model input.
5. **Split overlap is small.** 4 test claims have a near-duplicate in another split, and about 94 articles are
   shared across splits.

### 2.3 Consequences for the design

- PUBHEALTH has **no gold evidence sentences**, so it cannot measure evidence recall (needed for RQ2). It can only
  give "silver" signals, such as whether the claim's own article was retrieved.
- Its four labels don't map onto SciFact's SUPPORTS / REFUTES / NEI. "Mixture" is not "not enough info", so it needs
  its own verdict rule, which is not yet chosen.

---

## 3. SciFact data handling

`src/oblig/data/scifact.py`

- Claim-level labels come from the evidence: SUPPORTS 332 / REFUTES 173 / NEI 304 (train), 124 / 64 / 112 (dev).
  No claim has both supporting and refuting evidence.
- **`claims_test.jsonl` has no labels**, so the held-out evaluation set is `claims_dev.jsonl` (300 claims).
- **The shipped cross-validation folds are unsafe.** Every fold's dev part contains 55–68 claims from
  `claims_dev`, so using them for development would leak the held-out set. Development instead uses `train_dev`:
  a fixed, label-balanced 200-claim split drawn only from `claims_train` (seed 13; 82 / 43 / 75).
- The runner refuses to score `claims_dev` without an explicit `--heldout` flag.

---

## 4. Retrieval

Code: `src/oblig/retrieval/` (BM25 written from scratch; dense retrieval with sentence-transformers and cached
embeddings; reciprocal-rank-fusion hybrid).

### 4.1 SciFact (`scripts/eval_retrieval_scifact.py`, 125 train_dev claims with evidence)

| retriever | Recall@5 | Recall@10 | Recall@20 |
|---|---|---|---|
| BM25 | 0.887 | 0.913 | 0.940 |
| MiniLM dense | 0.865 | 0.905 | 0.960 |
| BM25 + MiniLM hybrid (proposal design) | 0.882 | 0.956 | 0.976 |

- At the proposal's budget of 5 abstracts, the three are tied at this sample size.
- **This bounds RQ2.** At 5 abstracts, only about 12% of gold abstracts are missed, so targeted re-retrieval has
  little left to find. Simply taking 10 abstracts recovers about 7 points of it, so the extra-evidence control
  (B1-9) is a strong baseline.
- The proposal's dense model, BGE-base, isn't on this machine (about a 440 MB download), so MiniLM, already cached,
  stands in. The held-out set has not been scored.

### 4.2 PUBHEALTH (`scripts/eval_retrieval.py`, all 1,214 dev claims)

The corpus pools all articles, with verdict sentences masked. The claim's own article is retrieved in the top 5 for
79% of claims with 4-sentence passages, and for 85% with whole articles. This is a silver signal only.

---

## 5. Research questions (literature review → frozen)

`docs/LITERATURE_REVIEW.md` covers 27 works, with 54 references checked against OpenAlex/arXiv.

**Novelty verdict:** the mechanism partially overlaps prior work. The closest papers are:
- FactAgent (Trinh et al., 2025): decompose → retrieve per part → label each part → all-must-hold, on SciFact-Open.
  Its decomposition *hurt* with a 3B model.
- FIRE (2025): decides per claim whether to search again.
- Vladika et al. (2025): iterative questions on SciFact.
- Evidence-Ledger Adjudication (2026): a claim-level "ledger" under a similar name.

The project is therefore framed as **a budget-controlled study** rather than a new method.

**Frozen at Gate 2** (`docs/EXPERIMENT_PROTOCOL.md` §9, approved 2026-10-27):
- **RQ1:** with model, retriever and evidence budget fixed, does checking each obligation separately beat direct
  RAG and sub-question RAG on SciFact? Which part explains any gain: evidence fetched per obligation, the
  per-obligation statuses, or the deterministic rule?
- **RQ2:** does re-retrieval aimed only at unresolved obligations beat a single pass, and beat simply adding more
  abstracts, at equal extra evidence? At what cost?
- **RQ3 (optional):** robustness on PUBHEALTH.

---

## 6. The obligation ledger

`src/oblig/ledger/`

- **Obligations:** typed. Core types are subject, outcome and relation; qualifiers are population, comparator,
  timepoint, scope and other. At most 6 per claim. "Core" vs "qualifier" is set by the type in code, never by the
  model. Duplicates are dropped and logged.
- **Statuses:** SATISFIED / CONTRADICTED / PARTIAL / UNSUPPORTED.
- **History is append-only.** Every check is stored with its round number. SATISFIED and CONTRADICTED are final,
  and round numbers must increase; the code enforces this.
- **Citations are checked.** The model cites abstracts as `A1…Ak` plus sentence numbers. These are checked against
  what it was actually shown, and the quoted text comes from the corpus. SATISFIED or CONTRADICTED without a valid
  citation is treated as UNSUPPORTED.
- **Verdict rule (`scifact_v1`, the proposal's):**
  - REFUTES if any obligation is contradicted;
  - SUPPORTS if every core obligation is satisfied by one abstract;
  - otherwise NEI.

  The reason is logged for every claim. A four-class PUBHEALTH rule (`pubhealth_v1`) exists as a candidate only.

---

## 7. The 12 SciFact systems

`src/oblig/systems/scifact.py`. Every retrieval system sees at most 5 distinct abstracts per claim, except where
noted.

| id | what it does | why it's there |
|---|---|---|
| B0 | verdict from the claim alone | how much the model knows without evidence |
| B1 | direct RAG: top-5 for the claim, one verdict | standard baseline |
| B1-9 | B1 with 9 abstracts | does M2 help only because it sees more text? |
| B1-SC | B1 with 5 sampled verdicts and a majority vote | spend similar compute without decomposing |
| B2 | sub-questions → retrieval → one verdict, no statuses | decomposition without a ledger |
| M1 | obligations → per-obligation retrieval → per-obligation audit → rule | the proposed method |
| M1-shared | M1, but every obligation is checked on B1's top-5 | does per-obligation evidence matter? |
| M1-untyped | untyped atomic facts instead of typed obligations | does typing matter? |
| M1-LLMagg | M1's ledger, verdict from the model instead of the rule | does the deterministic rule matter? |
| M2 | M1 + up to 2 rounds of targeted re-retrieval (≤2 new abstracts each, ≤9 total) + consolidation | the proposed iterative method |
| M2-random | M2, but follow-up queries go to random obligations | does targeting by status matter? |
| M2-claimiter | claim-level iteration while the verdict is NEI, same caps | FIRE/QACheck-style competitor |

### Protocol amendment 1 (2026-10-27): consolidation audit

Implementing M2 exposed a problem. Obligations satisfied in round 0 are final and point at old abstracts. Evidence
M2 finds in a new abstract therefore almost never satisfied "all core obligations in one abstract", so M2 could
rarely turn NEI into SUPPORTS.

The fix the team approved: after each round, every other satisfied core obligation is re-checked against the newly
cited abstract alone. This **adds evidence only**; it never changes a status, and there is no new retrieval. It is
recorded in `docs/EXPERIMENT_PROTOCOL.md` §12, and `m2_consolidate=False` gives the original design for comparison.

---

## 8. Running and checking

- **Runner:** `scripts/run_scifact.py`. Writes one prediction file per run plus metrics and a run record (git
  commit, model, settings, time). It resumes after interruption, refuses to reuse outputs made with different
  settings, and guards the held-out set.
- **Metrics:** SciFact claim-level macro-F1, and abstract-level label and label+rationale F1. The sentence-level
  number is a simplified version and labelled as such. Paired bootstrap confidence intervals are implemented.
- **Cost logging:** model calls, tokens, latency, failures, consolidation calls, verdict flips (fixing or
  introducing errors), and stop reasons.
- **Tests:** 41 unit tests, all passing (`python -m pytest -q tests`). They cover parsing, leakage guards, retrieval
  determinism, ledger rules, every verdict-rule edge case, citations, budgets, iteration caps and stop reasons,
  sampling seeds, consolidation, and metrics.
- **Smoke runs:** all 12 systems ran on 30 train_dev claims with the mock (`results/predictions/smoke_sf_*.jsonl`).
  There were no failed calls and no invalid citations, budgets held, and M2 never exceeded 9 abstracts.

---

## 9. What is not done

1. **No real-model results.** These need a GPU running Qwen3-8B. They include:
   - the throughput pilot and compute estimate;
   - Gate 7, hand-checking real ledgers;
   - matching B1-SC's sample count to M1's token cost;
   - every reported number.
2. **BGE-base** (the proposal's dense model) is not downloaded; MiniLM is used meanwhile.
3. **Gate 3 (final protocol freeze)** still needs the PUBHEALTH four-class rule chosen on dev and the dense-model
   choice fixed.
4. **Not started:** PUBHEALTH runs of the LLM systems, held-out evaluation, human annotation of 50 claims,
   robustness splits, the failure analysis, and the final report.

---

## 10. Where to look

| topic | file |
|---|---|
| running log of every step | `docs/PROJECT_STATUS.md` |
| repository and environment audit | `docs/PROJECT_AUDIT.md` |
| dataset audit and leakage | `reports/dataset_audit.md` |
| literature review | `docs/LITERATURE_REVIEW.md` |
| protocol, frozen RQs, amendments | `docs/EXPERIMENT_PROTOCOL.md` |
| systems and controls list | `configs/experiment_matrix.yaml` |
| model / retrieval / dataset settings | `configs/*.yaml` |
| how to reproduce | `README.md` |
