# Experiment protocol — DRAFT (not frozen)

Status: research questions and controls frozen (§9, Gate 2). The rest is frozen at Gate 3. Originally drafted after the Phase 0/1 audit. It is frozen at Gate 3, after the literature review (Phase 2) and
after the team decides the open questions in §8. Until then no test-set LLM run is made.

## 1. Tasks (kept separate; never pooled into one headline number)

| | SciFact (original proposal) | PUBHEALTH (team's new direction) |
|---|---|---|
| labels | SUPPORTS / REFUTES / NEI | true / false / mixture / unproven |
| evidence | 5,183-abstract corpus, sentence-level gold rationales | pooled, verdict-masked fact-check / news articles; no gold sentence evidence |
| held-out eval | dev (300; test labels unpublished) | official test (1,233 valid) |
| prompt development | `train_dev`: seeded, label-stratified 200-claim split of claims_train (82 S / 43 R / 75 NEI). The shipped CV folds are **not** used: every fold's dev part contains 55–68 claims from claims_dev | dev (1,214) |
| evidence metrics | gold rationale recall/precision, abstract-level label+rationale F1 | silver only: own-article recall@k, explanation-sentence recall |

**Consequence for the research questions.** RQ2 asks whether targeted re-retrieval improves *evidence recall*. Only
SciFact has gold evidence to measure that. PUBHEALTH can answer the *verdict* part of RQ1/RQ2 and adds a
generalisation test, but its evidence metrics are silver. Recommendation: SciFact remains the primary testbed for
RQ1/RQ2 as proposed; PUBHEALTH is the second, larger testbed with its own four-class aggregation rule (§4).

## 2. PUBHEALTH inputs

* Model input: claim text only.
* Evidence corpus: main_text of all 12,251 valid records (train+dev+test), split into 4-sentence passages
  (stride 4), verdict sentences masked. The claim's own article is in the pool and must be *retrieved*.
* Forbidden as input: label, explanation, claim_id (encodes label), fact_checkers, subjects, sources,
  date_published, split name. Passage ids are opaque hashes; prompts show passages as `E1..Ek`.
* Few-shot exemplars (if any) come from train only and never from articles shared with test.

## 3. Systems (one model, one retriever, matched budgets)

| id | system | evidence per claim (PUBHEALTH draft) |
|---|---|---|
| B0 | zero-shot LLM | 0 |
| B1 | direct RAG, BM25(claim) | 5 passages |
| B1-9 | direct RAG, matched to M2's max budget | 5 + max extra of M2 (9 for SciFact abstracts) |
| B2 | sub-question RAG (no ledger, no statuses) | 5 passages total, split across sub-questions |
| M1 | obligation RAG, single pass | 5 passages total, split across obligations |
| M2 | iterative obligation RAG | M1 + ≤2 rounds × ≤2 new passages/abstracts per round |
| M3 | debate (optional) | only if compute remains |

SciFact keeps the proposal's settings: top-5 abstracts, ≤6 obligations, ≤2 extra rounds of ≤2 new abstracts.

**SciFact retrieval (measured on train_dev, 125 claims with evidence; `results/metrics/retrieval_scifact_train_dev.json`).**
Recall@5 / @10: BM25 0.887 / 0.913; MiniLM dense 0.865 / 0.905; BM25+MiniLM RRF 0.882 / 0.956. At the k=5 budget the
three are indistinguishable at this sample size; the hybrid's advantage appears at k=10. Abstract-level evidence
recall at k=5 already misses only ~12% of gold abstracts, which is the most any re-retrieval (M2 or B1-9) could recover;
simply extending the claim query to k=10 recovers ~7 points of it. RQ2's evidence-recall effect is therefore bounded
and small on SciFact, and B1-9 is a strong control. The 5,183-abstract corpus is small, so retrieval is
easier than in an open setting (cf. SciFact-Open). Selected: hybrid RRF (proposal design); the dense model is
BGE-base once downloaded, else MiniLM, and the choice is recorded before any held-out run.

## 4. Aggregation

* SciFact: proposal rule — REFUTES if any obligation CONTRADICTED; SUPPORTS if every core obligation SATISFIED from
  one abstract and none CONTRADICTED; else NEI.
* PUBHEALTH (to be fixed on dev before Gate 3, candidates): false if a *core* obligation is CONTRADICTED and none
  SATISFIED; mixture if both SATISFIED and CONTRADICTED core obligations occur; true if all core SATISFIED;
  unproven otherwise. Compared against LLM aggregation over the same ledger. Chosen on dev only.

## 5. Metrics and statistics

Primary: macro-F1 (four-class PUBHEALTH; three-class SciFact claim-level). Secondary: per-class P/R/F1, accuracy,
confusion matrix, invalid-output rate, invalid-citation rate, evidence metrics (§1), calls/tokens/latency per claim.
Paired bootstrap (10,000 resamples, seed 13) 95% CIs for every system difference. Failed or unparseable outputs count
as wrong (`__invalid__`), never dropped.

Mandatory reference rows for PUBHEALTH: majority, claim_id-bin shortcut, claim-only TF-IDF LR, masked-article
TF-IDF LR, metadata LR (`results/metrics/shortcut_baselines.json`). PUBHEALTH results are also reported separately for
the wire-news block (claim_id < 8936, all `true`) and the fact-check remainder — the split is used for analysis only.

## 6. Generalisation splits (PUBHEALTH)

Held-out source site (wire-news vs PolitiFact vs Snopes/TruthOrFiction vs Health News Review), inferred from
id ranges and metadata and documented. Held-out fact-checker requires a cleaned checker mapping first (credentials
are comma-split in the raw field). Since the LLM systems are not trained, these splits only restrict which data may be
used for prompt development.

## 7. Model and decoding

Qwen3-8B, non-thinking mode, served by vLLM, temperature 0, seed 13, revision pinned on first download. One profile
(`configs/model.yaml: qwen3_8b_vllm`) for all controlled systems.

## 8. Open decisions for the team (block Gate 3)

1. Primary testbed: SciFact (as proposed) + PUBHEALTH (secondary), or PUBHEALTH primary?
2. GPU: the audit machine is CPU-only; which cluster/GPU will serve Qwen3-8B?
3. Whether the PUBHEALTH corpus should exclude each claim's own article (harder, "open" setting) as a second condition.

## 9. Research questions — FROZEN (Gate 2 passed)

Approved by the team on 2026-10-27; frozen from this point. Changes require a dated protocol amendment recorded here.
Gate 3 (full protocol freeze) still needs the PUBHEALTH aggregation rule and the dense-model choice fixed on development data.

Source: `docs/LITERATURE_REVIEW.md` §4–5. Verdict: the mechanism partially overlaps prior work (closest: FactAgent,
Trinh et al. 2025, decompose → per-subclaim retrieval → per-subclaim labels → AND rule, on SciFact-Open; FIRE,
NAACL Findings 2025, per-claim confidence-driven re-search; Vladika et al. NAACL 2025, iterative question-guided
verification on SciFact; Evidence-Ledger Adjudication, arXiv 2607.26512, a claim-level 4-relation "ledger").
No "first" claims. The contribution is framed as a **budget-controlled study** of typed obligation ledgers and
status-targeted re-retrieval with an open 8B model.

* **RQ1 (refined).** With model, retriever and evidence-token budget fixed, does per-obligation verification beat
  direct RAG and sub-question RAG on SciFact, and which component explains any difference: obligation-aligned
  evidence vs shared claim-level evidence, per-obligation statuses, or the deterministic rule vs LLM aggregation?
  H1a gain concentrated in REFUTES/NEI recall; H1b gain shrinks with shared evidence (Akhter et al. 2026);
  H1c risk: decomposition/auditing noise at 8B cancels the gain (FactAgent with a 3B model).
* **RQ2 (refined).** Does status-targeted re-retrieval (PARTIAL/UNSUPPORTED only) beat single-pass and beat
  untargeted extra evidence at equal added evidence, and at what cost? Report gain per added abstract and per LLM
  call, status-transition matrix, share of claims entering M2, harmful vs helpful verdict flips, false-SUPPORTS on NEI.
  Bound: on train_dev, recall@5 already misses only ~12% of gold abstracts (§3), so the evidence-recall effect is small.
* **RQ3 (optional, secondary).** Robustness on PUBHEALTH with its own frozen four-class rule, stratified by genre.

Added controls (all same model, caps and evidence budget):
B1-SC direct RAG with self-consistency (token-matched); M1-shared obligations audited on shared claim evidence;
M1-untyped FActScore-style untyped decomposition; M1-LLMagg LLM aggregation over the same ledger;
M2-random re-retrieval for random obligations; M2-claimiter FIRE/QACheck-style claim-level iteration under M2's caps.

## 10. B2 / M1 implementation decisions (SciFact; `src/oblig/systems/scifact.py`)

Design choices, not measured results. Gate 7 (hand-checked ledgers from the real model) is still open.

* **Evidence budget.** B1, B2, M1 and M1-shared each show the model at most 5 distinct abstracts per claim. B2 and M1
  build that pool by round-robin over their per-query rankings (rank 1 of every query, then rank 2, ...). So
  decomposition changes *which* abstracts are seen, not how many. Token cost differs (M1 makes 1 + n_obligations
  calls) and is reported per claim.
* **Queries.** B2 sub-question / M1 obligation query = `"{query} {claim}"` (proposal: combine with the claim).
* **M1 per-obligation evidence.** Each obligation is audited on the pool abstracts within its own top-5. If none are,
  it gets its highest-ranked pool abstract. **M1-shared** audits every obligation on B1's claim-level top-5.
* **B2 has no statuses.** It makes one decomposition call and one verdict call that sees the sub-questions and the
  pooled abstracts.
* **Planner.** At most 6 obligations. Criticality comes from the type (subject/outcome/relation = core), never from
  the model. Duplicate (type, normalised description) obligations are dropped and recorded. If the Planner fails,
  the ledger is empty and the verdict is NEI, counted as a planner failure.
* **Auditor citations.** These are abstract aliases A1..Ak plus sentence numbers, resolved against the abstracts
  actually shown. Quoted text is taken from the corpus. A SATISFIED/CONTRADICTED answer without a valid citation
  keeps its raw status but has effective status UNSUPPORTED. Aggregation uses effective statuses.
* **Ledger.** Audits are append-only with round numbers. SATISFIED/CONTRADICTED are terminal, and rounds must
  increase. These invariants are enforced in code (`src/oblig/ledger/schema.py`).
* **Verdict.** This is the proposal rule `scifact_v1`, with the reason logged per claim. On SUPPORTS, the rationale is
  the core obligations' cited sentences in the shared abstract. Abstract-level scoring keeps the first 3 sentences,
  as in the SciFact metric.
* **Held-out guard.** `run_scifact.py` refuses claims_dev without `--heldout`, and refuses to resume a prediction
  file produced with a different configuration.

## 11. M2 and added controls: implementation (SciFact; `src/oblig/systems/scifact.py`)

These are design choices, not results. All controls use the same model, retriever, caps and prompts as the system
they control.

* **M2.** It starts with M1 (round 0), then runs at most 2 rounds. Each round:
  1. One follow-up-query call per PARTIAL/UNSUPPORTED obligation (the Auditor's rationale is included).
  2. Up to 2 abstracts the claim hasn't seen yet are retrieved, round-robin across the follow-up queries
     (`"{query} {claim}"`).
  3. Every unresolved obligation is re-audited on its previous evidence plus the new abstracts.

  At most 5 + 2×2 = 9 distinct abstracts per claim. Resolved obligations are never re-audited, and history is
  append-only.

  Stop reasons are recorded per claim: all obligations resolved / no new evidence / round cap reached. If a
  follow-up-query call fails, the obligation's own query is used and the error is logged. The record keeps the M1
  verdict, so helpful and harmful flips can be counted. `needs_more_retrieval` is recorded but does not trigger
  anything; the trigger is the status.
* **M2-random.** Identical to M2, except each round's follow-up queries are written for the same *number* of
  obligations drawn uniformly from all obligations (seeded per claim). Re-audit still covers the unresolved
  obligations. This isolates whether targeting by status matters.
* **M2-claimiter.** Claim-level iteration in the style of FIRE/QACheck: a B1 verdict; while it is NEI, one claim-level
  follow-up query and up to 2 new abstracts, at most 2 rounds (9 abstracts). There is no calibrated confidence, so
  NEI is the trigger.
* **B1-9.** B1 with top-9 abstracts (M2's maximum).
* **B1-SC.** B1 evidence, 5 verdicts sampled at temperature 0.7 (seeds 13..17), majority vote. A tie goes to NEI if
  NEI is among the tied labels, otherwise to the label sampled first. The sample count is to be token-matched to M1
  from the GPU pilot and recorded before Gate 3.
* **M1-untyped.** FActScore-style atomic facts (≤6), each a core obligation of type `atomic`; otherwise M1.
* **M1-LLMagg.** The same ledger as M1. The verdict comes from one LLM call over the obligation list (criticality,
  status, rationale, quoted evidence). The rule's verdict is also stored (`rule_label`) for a paired comparison.

### Finding from implementation (needs a team decision before Gate 3)

Under `scifact_v1`, SUPPORTS needs every core obligation satisfied **in one abstract**. Resolved obligations are
terminal in M2. So when M2 finds the missing evidence in a *new* abstract, the obligations satisfied earlier still
cite an *old* abstract. SUPPORTS can then only fire if the newly satisfied obligation cites an old abstract. In
practice, M2 can change NEI → REFUTES far more easily than NEI → SUPPORTS. A unit test reproduces this
(`tests/test_iterative.py::test_m2_targets_unresolved_and_preserves_history`). In the mock run, 11/30 claims entered
iteration and 0 changed verdict (mock outputs only).

Options:
1. Keep the rule as specified and report the asymmetry.
2. Amend M2 with a *consolidation audit*: when a new abstract satisfies a core obligation, re-check the other
   satisfied core obligations against that abstract only. This costs extra calls, stays within the 9-abstract cap,
   and is recorded as a new round.

Either choice has to be fixed before any held-out run. **Resolved: option 2 adopted, see §12.**

## 12. Protocol amendments

### Amendment 1 — 2026-10-27, 16:39 UTC / 22:09 IST: M2 consolidation audit

**Decided by:** the team (approval in the project session), before any held-out or real-model run.
**Reason:** §11 finding. Under `scifact_v1`, resolved obligations are terminal and cite their round-0 abstracts. So
evidence that M2 finds in a new abstract could almost never produce SUPPORTS.

**Change (M2 and M2-random only; M1 and every other system unchanged):**
1. After the re-audits of a round, take every abstract cited by a core obligation that became SATISFIED in that round.
2. For each such abstract, re-check every other SATISFIED core obligation that does not already cite it, using the
   standard Auditor prompt with **that abstract alone** (call role `consolidation`).
3. A consolidation is stored as an audit of `kind: "consolidation"` in the same round. It **never changes the
   obligation's status**. Only a SATISFIED consolidation with a valid citation adds evidence (`Ledger.evidence`).
   Any other outcome, including CONTRADICTED, is recorded and has no effect on the verdict.
4. No new retrieval: the 9-abstract cap is unchanged. Each (obligation, abstract) pair is checked at most once, and
   qualifiers are never consolidated.

**Effects to report:** consolidation calls per claim (counted in cost) and verdict flips with and without the step.
`Context.m2_consolidate = False` reproduces the original design for the ablation. RQ2's question is unchanged; M2
now has one more call type, and the comparisons with B1-9, M2-random and M2-claimiter still hold the evidence caps
fixed. M2-random gets the same consolidation, so it differs from M2 only in targeting.

**Tests:** `tests/test_ledger.py::test_consolidation_invariants`, `::test_scifact_rule_uses_consolidated_evidence`;
`tests/test_iterative.py::test_m2_targets_unresolved_and_preserves_history`,
`::test_m2_without_consolidation_keeps_the_asymmetry`,
`::test_failed_consolidation_adds_no_evidence_and_never_changes_status`.

