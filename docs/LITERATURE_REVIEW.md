# Literature Review: Obligation Ledgers for Scientific Claim Verification

Project: *Verify What Must Be True: Obligation Ledgers for Scientific Claim Verification*
Scope: positioning the obligation-ledger pipeline (typed decomposition, per-obligation hybrid retrieval, per-obligation Auditor statuses, a deterministic aggregation rule, and optional status-driven re-retrieval, M2) against prior work.

Unless a number is attributed to a specific source that was read (abstract, HTML or PDF), it is not reported. "Not reported" means that I did not find the detail in the parts of the source I read. It does not mean the paper lacks it.

---

## 1. Search log

**Date:** 2026-10-27. Only the date was recorded, not clock times; the searches fall on that calendar date in both UTC and IST.

**Tools and sources**
- Web search (standard and extended modes), with paper pages read through WebFetch: arXiv abstract and HTML pages, ACL Anthology pages, IJCAI proceedings.
- arXiv export API, used in bulk to read the abstracts and comments fields (venue notes) for 30 papers.
- Full text of 3 PDFs extracted with `pdftotext`: Alvarez et al. 2024 (SCitance), Tan et al. 2026 (SciTrue), Akhter et al. 2026 (IJCAI-26).
- Reference verification: the reference-lookup MCP tools (`verify_bibtex`, backed by OpenAlex and arXiv). All 56 entries were checked; status is given per entry in Section 6.

**Queries (abridged)**
1. `SciTrue scientific claim verification`; `SciTrue Tan Li Gahegan ...`
2. `SCitance claim verification citation sentences`
3. `atomic claim decomposition verification status tracking re-retrieval unverified subclaims fact-checking LLM agent`
4. `SciFact LLM zero-shot retrieval-augmented claim verification decomposition 2024 2025`
5. `PICO decomposition scientific claim verification LLM per-element evidence`
6. `fact-checking agent re-retrieve evidence only for unverified sub-claims iterative verification "sub-claim" status`
7. `claim decomposition SciFact subclaims verification LLM decompose-then-verify results`
8. `scientific claim verification agent iterative retrieval SciFact LLM multi-agent 2025`
9. `claim verification "missing information" targeted follow-up retrieval sub-question insufficient evidence`
10. `debate multi-agent scientific claim verification SciFact`
11. `"evidence ledger" OR "verification ledger" LLM claim verification retrieval`
12. `Decomposition Dilemmas ...`; `Minimal Evidence Group Identification ...`
13. `LLM fact-checking sub-claim "partially supported" verdict per subclaim evidence retrieval loop ...`
14. `LLM claim verification checklist of required conditions ... overgeneralization population scope`

**Result of the search for the novelty threat.** None of the searches found a system that combines all of the following: (a) a typed decomposition into core and qualifier obligations, (b) a persistent per-unit status from a closed set such as SATISFIED/CONTRADICTED/PARTIAL/UNSUPPORTED with cited sentences, (c) a deterministic aggregation rule over those statuses, and (d) re-retrieval triggered only for units in particular statuses. Every one of these ingredients does appear somewhere in prior work, and several systems combine two or three of them (Section 4).

---

## 2. Closely related methods

The entries are grouped by theme. Each entry gives: RQ, architecture, data, baselines, metrics, findings, limitations, similarity to the ledger, and the exact difference.

### 2.1 Scientific claim verification: datasets and pre-LLM systems

**(1) SciFact, Wadden et al., EMNLP 2020.**
- *RQ / task:* Defines scientific claim verification: select the abstracts that SUPPORT or REFUTE a claim, and identify rationale sentences for each.
- *Data:* 1.4K expert-written claims, paired with abstracts annotated with labels and rationales.
- *Architecture:* A pipeline baseline (VeriSci) of abstract retrieval, rationale selection and label prediction.
- *Findings:* Simple domain adaptation substantially improves over models trained on Wikipedia or political news.
- *Limitations:* A small closed corpus. Claims were written by annotators from citances, and the negations were also written by hand. Later work cites lexical biases in those negations; Alvarez et al. 2024 attribute this point to Wadden et al. 2020 and Wright et al. 2022.
- *Similarity:* This is our primary testbed. SciFact's rationale annotation is a form of minimal evidence set: each rationale is meant to be sufficient on its own. That fits our rule of "SUPPORTS only if every core obligation is satisfied from one abstract."
- *Difference:* SciFact labels claim–abstract pairs. Its official metrics are abstract-level and sentence-level, not claim-level. We must state which of our metrics maps onto which official one.

**(2) SciFact-Open, Wadden et al., Findings of EMNLP 2022.**
- *RQ:* Do SciFact-trained systems hold up against a realistic open corpus?
- *Data:* A test collection over 500K abstracts, with evidence collected by pooling the predictions of four systems.
- *Findings:* Systems developed on small corpora drop by at least 15 F1. The paper also notes cases where the evidence supports only a *special case* of the claim.
- *Limitations:* Pooled annotation is incomplete by construction.
- *Similarity:* The "special case" phenomenon is exactly what our qualifier obligations (population, scope) are meant to catch, and what our PARTIAL status is meant to mark.
- *Difference:* SciFact-Open is a benchmark, not a method. It is also the natural corpus-shift test, which we currently do not use.

**(3) MultiVerS, Wadden et al., Findings of NAACL 2022.**
- *Architecture:* A Longformer encoder over the claim and the full abstract, with multitask heads for the label and for rationales. It is weakly supervised for domain adaptation.
- *Findings:* Outperforms two competitive baselines on three scientific claim verification datasets, and is particularly strong in zero- and few-shot domain adaptation.
- *Similarity:* It is the strongest supervised reference point on SciFact.
- *Difference:* It makes one holistic decision per claim–abstract pair. It does not decompose the claim, and it records no per-condition reasoning.

**(4) PUBHEALTH, Kotonya & Toni, EMNLP 2020.**
- *Data:* 11.8K public-health claims, with journalist-written explanations and four labels (true, false, mixture, unproven).
- *Tasks:* Veracity prediction and explanation generation, plus three coherence properties of explanations.
- *Findings:* Training on in-domain data helps for claims that need expertise.
- *Limitations:* The evidence comes from fact-check articles. Glockner et al. (2022) argue that this kind of evidence "leakage" makes fact-checking unrealistic. I did not check whether they analyse PUBHEALTH by name, so treat that link as unverified.
- *Similarity:* Our secondary testbed. "Mixture" relates to PARTIAL, and "unproven" relates to NEI/UNSUPPORTED.
- *Difference:* PUBHEALTH has no closed abstract corpus and no rationale annotations, so our "SUPPORTS from one abstract" rule and our sentence-citation metrics do not carry over directly.

**(5) Generating scientific claims for zero-shot fact checking (CLAIMGEN), Wright et al., ACL 2022.**
- *Method:* Generates atomic, verifiable claims from scientific sentences (CLAIMGEN-BART, CLAIMGEN-ENTITY), plus KBIN for negations.
- *Findings:* Zero-shot systems trained on the generated claims reach "up to 90%" of the performance of fully supervised models.
- *Relevance:* Early use of *atomic* scientific claims.
- *Difference:* Atomicity is used to generate training data, not to verify claims at inference time.

### 2.2 LLM-era results on SciFact (the numbers we must situate against)

**(6) SCitance: "Zero-shot Scientific Claim Verification Using LLMs and Citation Text", Alvarez, Bennett & Wang, SDP Workshop 2024.** This paper exists. "SCitance" is the name of its in-context-learning dataset (repository `larchlab/scitance`); it is not a method.
- *RQ:* Can citances plus LLM-generated negations replace curated claim–evidence pairs?
- *Data:* SCitance is derived from SciFact citances, with 251 SUPPORT, 225 REFUTE and 180 NEI instances. The paper also evaluates on the SciFact test set through the public leaderboard.
- *Retrieval for SciFact:* Dense retrieval with the S-PubMedBert-MS-MARCO-SCIFACT checkpoint, top-3 abstracts per claim.
- *Results, SciFact test, abstract-level F1 (paper Table 3):* MultiVerS 72.5; GPT-4 zero-shot 68.6; GPT-4 few-shot with citance examples 71.7; GPT-4 few-shot with claim examples 72.2; GPT-3.5 zero-shot 35.0.
- *Limitations (stated by the authors):* Results are sensitive to the prompt; explainability is not addressed; one dataset.
- *Relevance:* This is the closest published "zero-shot LLM + RAG on SciFact" baseline, and its numbers are on *test*. Our numbers are on dev, so they are not directly comparable.
- *Difference:* Holistic prompting, with no decomposition.

**(7) Step-by-Step Fact Verification for Medical Claims, Vladika, Hacajová & Matthes, NAACL 2025 (short).**
- *Architecture:* Inspired by QACheck. It generates up to 5 follow-up questions, retrieves evidence for each (web search or the LLM's internal knowledge), and lets a reasoner decide whether to ask more questions or output a verdict. A variant uses logic predicates of the form verb(subject, object).
- *Data:* SciFact, HealthFC, CoVERt.
- *Baseline:* A three-part pipeline (retrieval, evidence selection, DeBERTa NLI).
- *Metric:* Binary F1, with NEI removed.
- *Results, SciFact F1:* Pipeline with whole-web evidence 82.7; GPT-4o-mini step-by-step with whole web 87.6; LLaMA-3.1-70B 84.3; Mixtral 84.1.
- *Limitations:* No NEI class; web evidence is not filtered; follow-up questions can drift onto one aspect of the claim; knowledge conflicts.
- *Similarity:* This is **the most direct prior work on iterative, decomposition-driven LLM verification for SciFact.**
- *Difference:* Its questions are generated on the fly and are not a typed checklist. It keeps no status per question. Stopping is decided holistically by the LLM. There is no deterministic aggregation, no NEI, and no closed SciFact corpus (it uses web evidence).

**(8) Towards Robust Fact-Checking: A Multi-Agent System with Advanced Evidence Retrieval (FactAgent), Trinh, Nguyen & Hy, arXiv 2025.**
- *Architecture:* Four agents: decomposition into subclaims; generation of sub-queries per subclaim; web evidence retrieval; and a verdict agent that labels *each subclaim*. A claim is SUPPORTED only if every predicate is True.
- *Data:* FEVEROUS, HOVER, and SciFact-Open (100 validation claims, binary labels, web search restricted to before Oct 2020).
- *Results, SciFact-Open macro-F1:* With GPT-4o-mini, the multi-agent system scores 0.770, against FOLK 0.737 and Direct 0.497. With Qwen-2.5-3B, the multi-agent system scores 0.422, *below* Direct at 0.625.
- *Limitations:* 100 claims; binary labels; web search instead of the corpus; no iterative retrieval.
- *Similarity:* Decomposition, per-subclaim retrieval, per-subclaim labels and a deterministic AND aggregation. That is RQ1's mechanism minus typing and minus PARTIAL.
- *Difference:* No qualifier/core typing, no 4-way status, no single-abstract constraint, no status-driven re-retrieval. **Warning sign:** with small models the decomposition *hurt*. This is directly relevant to Qwen3-8B.

**(9) Evidence-Ledger Adjudication for Claim–Evidence Traceability, Chen, Yu & Wang, arXiv 2026.** This is a naming collision with our "ledger".
- *Method:* Pairs each *whole* claim with a fixed evidence packet and assigns one of four relations (supports, contradicts, missing evidence, mixed evidence). The predicted relation then sets a review route through a fixed mapping.
- *Data:* 2,335 rows from AVeriTeC, CLIMATE-FEVER and SciFact (300 SciFact dev rows, using cited abstracts).
- *Results:* Overall relation accuracy 0.676 and macro-F1 0.601 for the agent condition. On SciFact, accuracy is 0.853 and macro-F1 0.643.
- *Limitations:* No retrieval at all (the evidence packets are given). The LLM is not named in the parts I read. There are internal inconsistencies about which baseline is "best".
- *Similarity:* The ledger metaphor; a 4-way status set that includes "missing" and "mixed"; deterministic routing on status.
- *Difference:* The unit is the claim, not obligations within it. There is no retrieval or re-retrieval. We must cite it and avoid suggesting that "ledger" is our coinage.

**(10) DeepSciVerify, Sadeghi et al., arXiv 2026.**
- *Method:* Stage 1 verifies a claim against its cited abstract and defers uncertain cases. Stage 2 retrieves full-text passages only for the deferred cases.
- *Data:* SCitance.
- *Results:* 86.7 micro-F1, which is 4.5 above abstract-only baselines. 67% of instances are resolved without full-text retrieval.
- *Similarity:* *Uncertainty-triggered, selective* evidence escalation with a cost accounting. This is RQ2's logic applied at claim level.
- *Difference:* Escalation is triggered by per-claim uncertainty, not by per-obligation status. It escalates to more of the same document (full text) rather than to new abstracts. No decomposition is reported.

**(11) When Retrieval Helps and Distracts, Deka & Singh, arXiv 2026.**
- *Method:* Compares LLMs with and without PubMed RAG, fine-tuned LLMs and biomedical encoders on CARE-XAI (five sources, including SciFact and PubMedQA). Introduces the Bio-GRACE diagnostic.
- *Findings (qualitative, from the abstract):* Retrieval helps on PubMed-aligned sources such as SciFact but can distract on broader public-health claims. Encoders are best at predicting verdicts alone.
- *Relevance:* This directly predicts that our gains on SciFact may not transfer to PUBHEALTH. The paper supports a selective-retrieval framing.

### 2.3 Claim decomposition and decompose-then-verify

**(12) WiCE, Kamoi et al., EMNLP 2023.**
- *Data:* Entailment over real Wikipedia claim–evidence pairs, *with sub-sentence (subclaim) labels and a minimal subset of evidence sentences per subclaim*. It proposes GPT-3.5 claim decomposition that improves entailment models at test time.
- *Similarity:* Per-subclaim labels with per-subclaim minimal evidence. This is very close to our "status plus cited sentence ids".
- *Difference:* WiCE is a dataset and an entailment study with given evidence. It has no retrieval loop and no typed obligations, and it is not scientific.

**(13) FActScore, Min et al., EMNLP 2023.**
- *Method:* Splits a generation into atomic facts and scores the fraction supported by a knowledge source. The automated estimator has less than 2% error. In the human study ChatGPT reaches 58%.
- *Similarity:* The canonical atomic-unit verification with a binary status per unit.
- *Difference:* It evaluates long-form generations rather than verifying claims. The aggregate is a precision ratio, not a verdict. It is not targeted at science and does not re-retrieve.

**(14) SAFE, Wei et al., NeurIPS 2024.**
- *Method:* An LLM splits a response into facts and verifies each fact with a *multi-step* search loop.
- *Results:* 72% agreement with crowd annotators over about 16k facts; SAFE "wins" 76% of 100 disagreements; it is more than 20x cheaper than humans.
- *Similarity:* Per-unit iterative search.
- *Difference:* The search loop runs for every fact rather than being triggered by status. It uses Google Search, not a corpus. The aggregation is F1@K.

**(15) VeriScore, Song, Kim & Iyyer, Findings of EMNLP 2024.**
- *Method:* Extracts only *verifiable* claims before verification.
- *Relevance:* Supports our choice to drop or mark obligations that cannot be checked (for example, hedges).
- *Difference:* Evaluates long-form generations; no status-driven retrieval.

**(16) ClaimDecomp, Chen et al., EMNLP 2022.**
- *Data:* Over 1,000 political claims decomposed into yes/no subquestions, both literal and *implied*.
- *Findings:* Predicting the complete set of subquestions without evidence is hard. Subquestions help retrieve evidence and derive veracity.
- *Similarity:* Subquestions as "what must be true".
- *Difference:* Not typed, not scientific, no ledger or loop.

**(17) ProgramFC, Pan et al., ACL 2023.**
- *Method:* An LLM writes a reasoning program; each sub-task is executed by a handler (QA, fact verification, logical operators).
- *Findings:* Outperforms seven baselines on HOVER and FEVEROUS.
- *Similarity:* Symbolic, deterministic composition of sub-results.
- *Difference:* Program steps are not persistent statuses with evidence, and no step is re-executed with new retrieval when it is unresolved.

**(18) HiSS, Zhang & Gao, AACL 2023.**
- *Method:* Splits news claims into subclaims and verifies each through step-by-step QA prompting.
- *Findings:* Outperforms the fully supervised state of the art on two misinformation datasets.
- *Difference:* The aggregation is LLM-internal; no status ledger; news domain.

**(19) Factcheck-Bench, Wang et al., Findings of EMNLP 2024.**
- *Data:* An annotation pipeline and benchmark with labels at claim, sentence and document level for LLM outputs.
- *Findings:* FacTool, FactScore and Perplexity struggle to identify false claims; the best F1 is 0.63.
- *Relevance:* The canonical decompose → retrieve → verify → aggregate pipeline with labels per claim.
- *Difference:* General domain; no targeted re-retrieval.

**(20) Decomposition analyses: Hu, Long & Wang (NAACL 2025); Wanner et al. (*SEM 2024); Akhter et al. (IJCAI-26); Lu et al. 2025.** These four papers are the main *risk* literature.
- *Hu et al.:* Effects of decomposition are inconsistent, with a trade-off between accuracy gains and decomposition noise. They provide a taxonomy of decomposition errors.
- *Wanner et al.:* FActScore-style results are sensitive to the decomposition method. They propose DecompScore.
- *Akhter et al.:* Decomposition helps significantly only when evidence is *granular and aligned to each sub-claim*. Reusing claim-level evidence for every sub-claim fails to help and often hurts (PHEMEPlus, MMM-Fact, COVID-Fact). Under noisy sub-claim labels, conservative abstention limits error propagation.
- *Lu et al.:* Hand-crafted decomposition policies are misaligned with the verifier's preferred atomicity. RL-learned decomposition improves verification accuracy by 0.12 on a 0–1 scale.
- *Relevance:* Akhter et al.'s distinction between aligned and repeated claim-level evidence is the single most useful design axis for our RQ1 ablation. Our per-obligation retrieval corresponds to the aligned setup, and verifying obligations against shared claim-level evidence corresponds to the repeated setup. Their abstention result supports our conservative "else NEI" rule.

**(21) Minimal Evidence Groups (MEG), Li et al., TrustNLP 2025.**
- *Method:* Identifying a minimal evidence group is reduced to Set Cover over entailment judgements of full or partial support.
- *Results:* +18.4 (WiCE) and +34.8 (SciFact) absolute over LLM prompting.
- *Similarity:* "Full vs partial support" judged per evidence group; minimal sufficient sets.
- *Difference:* The decomposition is over *evidence* rather than over claim obligations, and there is no iterative retrieval. Our rule "all core obligations satisfied from one abstract" is a restricted MEG: one abstract must cover all core obligations.

**(22) Dense X Retrieval, Chen et al., EMNLP 2024.** Introduces propositions (atomic factoids) as the *retrieval unit*. Indexing the corpus by propositions beats passage-level indexing for retrieval.
- *Relevance:* This is atomicity on the corpus side. Our obligations are atomic on the query side, while our corpus stays at abstract and sentence level. It is a possible orthogonal ablation.

### 2.4 Question-guided and iterative retrieval for verification

**(23) QACheck, Pan et al., EMNLP 2023 demo.**
- *Architecture:* Five modules: claim verifier ("is there enough information?"), question generator, QA module, QA validator, reasoner. It asks questions iteratively until the verifier judges the context sufficient, and gives the evidence source per question.
- *Similarity:* Closest in spirit to M2: a sufficiency check that drives more retrieval.
- *Difference:* Sufficiency is judged once over the whole claim. Questions are not tied to persistent typed obligations and carry no 4-way status.

**(24) FIRE, Xie et al., Findings of NAACL 2025.**
- *Method:* For each *atomic* claim, the agent decides by its confidence whether to answer or to issue another search.
- *Results:* Slightly better than strong frameworks while cutting LLM cost by 7.6x and search cost by 16.5x on average.
- *Similarity:* Adaptive, per-unit retrieval with explicit cost accounting. **This is the key comparison for RQ2.**
- *Difference:* FIRE assumes that decomposition was done upstream. It is driven by confidence rather than by status. There is no claim-level deterministic aggregation, and it is not scientific.

**(25) AFEV ("Fact in Fragments"), Zheng et al., arXiv 2025.** A later version appears to be in *Expert Systems with Applications* (ScienceDirect listing, not verified).
- *Method:* *Iteratively* extracts atomic facts, with each step's label and rationale conditioning the next extraction. It adds evidence reranking and context-specific demonstrations.
- *Aggregation:* A secondary summary (not verified in the paper) describes the claim-level rule as: any atomic fact False gives False; all True gives True; otherwise Unverifiable.
- *Results:* Reports state of the art on five benchmarks; I did not identify which.
- *Similarity:* Iteration driven by feedback, together with a rule-based aggregation over fact labels. If the secondary summary is accurate, this is close to our REFUTES/SUPPORTS/NEI rule.
- *Difference:* Iteration refines the *decomposition* rather than re-retrieving for unresolved units. There is no typing and no scientific corpus.

**(26) Recon, Answer, Verify (RAV), Shukla, Dutta & Bhattacharyya, arXiv 2025.**
- *Method:* A question generator, answer generator and label generator iterate over sub-questions about different aspects of the claim.
- *Results:* RAWFC +25.28% over the prior state of the art; HOVER +1.54/+4.94/+1.78 points (2/3/4-hop).
- *Difference:* As with QACheck, coverage of open aspects is implicit in the loop rather than recorded as a status.

**(27) EMULATE, Hong, Luo & Wan, FEVER Workshop 2025.** A multi-agent system in which each agent handles a small subtask, such as ranking search results or judging page content. It retrieves evidence iteratively for *atomic* claims. I did not read its datasets or numbers.

**Generic RAG control methods (context only).**
- Self-Ask (Press et al., Findings EMNLP 2023): explicit follow-up questions, optionally answered by search.
- IRCoT (Trivedi et al., ACL 2023): interleaves retrieval with chain-of-thought; up to +21 retrieval and +15 QA points.
- Least-to-Most prompting (Zhou et al., ICLR 2023).
- FLARE (Jiang et al., EMNLP 2023): retrieves when upcoming tokens have low confidence.
- Self-RAG (Asai et al., ICLR 2024): trained reflection tokens decide when to retrieve and critique.
- Adaptive-RAG (Jeong et al., NAACL 2024): a classifier routes queries by complexity.
- CRAG (Yan et al., 2024): a retrieval evaluator triggers corrective actions, plus decompose-then-recompose filtering of documents.

These methods decide *when* to retrieve from model confidence or a learned critic. None of them keeps a typed per-obligation ledger. Our sub-question RAG baseline is the Self-Ask/least-to-most family with the statuses removed.

### 2.5 Attribution, grounding verifiers, debate, calibration

- **ALCE** (Gao et al., EMNLP 2023): a benchmark for LLM answers with citations, with automatic citation recall and precision. On ELI5, even the best models lack complete citation support 50% of the time.
- **AIS** (Rashkin et al., Computational Linguistics 2023) and **AttrScore** (Yue et al., Findings EMNLP 2023): human and automatic attribution evaluation.
- *Relevance of these three:* They give us metrics for whether the Auditor's cited sentence ids actually entail the status assigned.

**MiniCheck** (Tang, Laban & Durrett, EMNLP 2024): a 770M checker trained on synthetic data reaches GPT-4 accuracy on LLM-AggreFact at about 400x lower cost. It is a natural cheap alternative Auditor, or an independent judge of the Auditor's citations.

**SciTrue** (Tan, Li & Gahegan, EACL 2026 demo). This paper exists, as an EACL 2026 system demonstration. A separate NTCIR-19 SciClaimEval entry also uses the name SciTrue (arXiv 2609.00654); I did not read or verify it.
- *Architecture:* GPT-4o with Semantic Scholar retrieval (at most 15 articles). It labels evidence as fully or conditionally supporting or refuting, and records the *assumptions or conditions* behind each judgement (for example, a mouse study or a specific demographic). It then synthesises a summary and an overall verdict (fully/mostly/partially/not supported), and *afterwards* decomposes the summary into subclaims, each with a sentence, paragraph, contribution label and assumptions.
- *Evaluation:* Human evaluation of about 300 attributions over 60 claims against GPT-4o-search-preview and Perplexity Sonar Pro. No SciFact evaluation.
- *Similarity:* Per-subclaim evidence linking, explicit conditions or qualifiers, and graded partial support in science.
- *Difference:* The decomposition comes after the verdict and serves explanation. It is not a checklist that drives verification. The verdict is produced by an LLM rather than by a rule, there is no status-driven re-retrieval, and there is no benchmark accuracy.

**Debate.**
- Du et al. (ICML 2024) report gains in factuality from multi-agent debate.
- DebateCV (He et al., WWW 2026): two debaters and a moderator. Zero-shot moderators lean towards neutral verdicts; the authors propose Debate-SFT.
- *Relevance:* Debate is a competing way to surface the "subtle errors" that our CONTRADICTED qualifier statuses target. It is worth one sentence in the paper and is not a required baseline.

**Calibration, abstention and shift.**
- Kamath, Jia & Liang (ACL 2020): selective prediction under domain shift.
- Wen et al. (2024): a survey of abstention in LLMs.
- Akhter et al. (above): abstention reduces error propagation in decomposition.
- Glockner et al. (EMNLP 2022): leaked and insufficient counter-evidence.
- SciFact-Open: a drop of at least 15 F1 under corpus shift.
- Deka & Singh 2026: retrieval utility depends on the source.
- Peters & Chin-Yee 2025: LLM summaries of science overgeneralise by dropping scope qualifiers. This motivates qualifier obligations, but I read the result only through search snippets, so no numbers are given here.

---

## 3. Comparison table

Key: **Y** = yes; **P** = partial or implicit; **N** = no; **?** = not reported in what I read.

Columns:
- **Dec** = decomposes the claim
- **PerUnit** = verification per unit
- **Status** = explicit status per unit
- **EvLink** = statuses linked to evidence
- **StatRR** = re-retrieval driven by status
- **DetAgg** = deterministic aggregation
- **Sci** = scientific or biomedical domain

| Method | Dec | PerUnit | Status | EvLink | StatRR | DetAgg | Sci |
|---|---|---|---|---|---|---|---|
| SciFact / VeriSci (2020) | N | N | N | Y (claim-level rationales) | N | N | Y |
| MultiVerS (2022) | N | N | N | Y (claim-level) | N | N | Y |
| SCitance / Alvarez et al. (2024) | N | N | N | N | N | N | Y |
| WiCE (2023) | Y | Y | Y | Y (minimal evidence/subclaim) | N | ? | N |
| FActScore (2023) | Y | Y | Y (binary) | P | N | Y (ratio) | N |
| SAFE (2024) | Y | Y | Y (binary) | Y | P (per-fact loop, not status-gated) | Y (F1@K) | N |
| VeriScore (2024) | Y | Y | Y | ? | N | Y | N |
| ClaimDecomp (2022) | Y (subquestions) | Y | Y (yes/no) | P | N | N | N |
| ProgramFC (2023) | Y | Y | P | P | N | Y (program logic) | N |
| HiSS (2023) | Y | Y | N | N | P | N | N |
| Factcheck-Bench (2024) | Y | Y | Y | Y | N | P | N |
| QACheck (2023) | P (iterative questions) | Y | N | Y | P (sufficiency-driven) | N | N |
| FIRE (2025) | N (assumes atomic input) | Y | N (confidence) | Y | P (confidence-driven) | N | N |
| AFEV (2025) | Y (iterative) | Y | Y | ? | P (re-extraction, not re-retrieval) | Y (secondary source; unverified) | N |
| RAV (2025) | Y (sub-questions) | Y | N | Y | P | N | N |
| Vladika et al. (2025) | Y (questions/predicates) | Y | N | Y | P (LLM decides more questions) | N | Y |
| FactAgent / Trinh et al. (2025) | Y | Y | Y (binary) | Y | N | Y (all-True) | P (SciFact-Open subset) |
| MEG / Li et al. (2025) | N (decomposes evidence) | P | P (full/partial) | Y | N | Y (set cover) | P |
| SciTrue (2026) | P (post hoc subclaims) | P | Y (contribution label) | Y (sentence + paragraph) | N | N | Y |
| Evidence-Ledger Adjudication (2026) | N | N | Y (4 relations, claim-level) | Y (packet) | N | Y (routing) | P |
| DeepSciVerify (2026) | N | N | N (defer flag) | Y | P (uncertainty-gated escalation) | ? | Y |
| Akhter et al. (2026) (analysis) | Y | Y | Y | Y (aligned evidence) | N | ? | N |
| **Ours: obligation ledger** | **Y (typed core/qualifier)** | **Y** | **Y (4-way)** | **Y (sentence ids)** | **Y (PARTIAL/UNSUPPORTED only)** | **Y** | **Y** |

---

## 4. Gap assessment

**Verdict: partially overlapping. The contribution is not novel as a mechanism, but it is defensible as a controlled study of a specific combination.**

What is already done:
- *Decompose → retrieve per unit → verify per unit → deterministic AND aggregation* has been done, including on SciFact-Open, by FactAgent (Trinh et al. 2025). Rule-based aggregation over atomic labels is reported for AFEV, and SAFE, FActScore and Factcheck-Bench all label atomic units.
- *Per-unit labels with minimal per-unit evidence* is WiCE's annotation scheme. "Full vs partial support" judgements underpin MEG.
- *Iterative, gap-driven retrieval for scientific and medical claims, on SciFact specifically,* is Vladika et al. (2025), building on QACheck. Confidence-driven retrieval per atomic claim with cost accounting is FIRE. Uncertainty-gated escalation with a "67% resolved without escalation" cost analysis is DeepSciVerify, on SCitance and therefore SciFact-derived data.
- *A 4-way relation ledger with "missing" and "mixed" labels and deterministic routing* is Evidence-Ledger Adjudication (2026), at claim level. That paper also uses SciFact dev.
- *Qualifier/condition awareness in science* (assumptions such as a mouse study or a demographic) appears in SciTrue, but post hoc.

What I could not find:
1. A typed obligation schema (core vs qualifier) whose types enter the aggregation rule. In our rule, core obligations gate SUPPORTS, while any obligation, qualifiers included, can trigger REFUTES.
2. Re-retrieval triggered by a *closed status set per unit* (PARTIAL/UNSUPPORTED) rather than by model confidence or a holistic sufficiency judgement, with hard caps.
3. A single-abstract constraint for SUPPORTS, which aligns with SciFact's abstract-level evaluation and is a restricted minimal evidence group.
4. A controlled comparison with matched budgets (same model, retriever and evidence budget) between holistic RAG, sub-question RAG without statuses, and the ledger, on SciFact's closed corpus with an open 8B model.

Items 1–3 are design refinements. Item 4 is the defensible scientific contribution. Two findings in prior work make item 4 *non-trivial* rather than a foregone conclusion:
- FactAgent's decomposition hurt with a 3B model (0.422 vs 0.625 Direct on SciFact-Open).
- Akhter et al. and Hu et al. show that decomposition gains depend on aligned evidence and on decomposition noise.

**Proposed framing.**
> *"When does verifying what must be true help? A budget-controlled study of typed obligation ledgers and status-targeted re-retrieval for scientific claim verification with an open 8B model."*

Claim the contribution as:
- (i) isolating *which component* of decompose-then-verify helps: aligned per-obligation retrieval, per-obligation auditing, or rule-based aggregation;
- (ii) testing whether re-retrieval driven by status beats re-retrieval driven by confidence, untargeted, or random, at equal added evidence and at measured cost;
- (iii) error analysis by obligation type (qualifier failures versus core failures).

Do not claim to be "the first" to decompose, to use a ledger, or to retrieve iteratively for SciFact. Cite FactAgent, Vladika et al., FIRE, QACheck, WiCE, Evidence-Ledger Adjudication and SciTrue explicitly as the closest work.

---

## 5. Recommended revisions

### 5.1 RQ1 (revised)
> **RQ1.** With the model (Qwen3-8B, non-thinking), retriever (BM25+BGE, RRF) and *evidence-token budget* held fixed, does obligation-ledger verification improve SciFact-dev label macro-F1 over direct RAG and over sub-question RAG? Which component accounts for any gain: (a) evidence retrieved per obligation (aligned) versus shared claim-level evidence, (b) auditing per obligation with statuses, or (c) deterministic versus LLM aggregation?

Hypotheses:
- **H1a:** The ledger outperforms direct RAG on REFUTES recall. The mechanism is that qualifier obligations surface contradictions. Report the result separately for SUPPORT and CONTRADICT.
- **H1b:** The gain shrinks or disappears when obligations are audited against shared claim-level evidence. This replicates Akhter et al.'s aligned-versus-repeated finding in science.
- **H1c (risk):** With an 8B model, errors from decomposition and auditing can cancel the gains, as with FactAgent on small models. Measure decomposition quality on a sample (for example, a DecompScore-style or manual check).

### 5.2 RQ2 (revised)
> **RQ2.** Starting from the M1 ledger, does re-retrieval targeted at PARTIAL/UNSUPPORTED obligations improve over (i) adding the same number of abstracts untargeted at claim level, (ii) re-retrieving for randomly chosen obligations, and (iii) confidence-driven claim-level iteration in the style of FIRE? Report gains per added abstract and per LLM call or token.

Hypotheses:
- **H2a:** Targeted re-retrieval beats (i) and (ii) at equal added evidence.
- **H2b:** Most gains come from UNSUPPORTED→SATISFIED transitions on core obligations. Report a status-transition matrix and the fraction of claims that trigger M2. FIRE and DeepSciVerify both report the share of cases that need further retrieval, so this number is expected.
- **H2c:** M2 increases false SUPPORTS on NEI claims. This is the cost of seeing more text, so report the NEI false-positive rate.

### 5.3 Optional RQ3 (shift and abstention)
> Does the ledger's conservative NEI default give better selective-prediction behaviour (risk–coverage) and smaller degradation on PUBHEALTH than direct RAG?

Deka & Singh 2026 predict that retrieval may *distract* on public-health claims, so frame PUBHEALTH results as a robustness check. Pre-register the label mapping (mixture ↔ PARTIAL-dominated ledger, unproven ↔ NEI) before seeing results.

### 5.4 Baselines and ablations to add
1. **Aligned versus shared evidence ablation (required):** audit the same obligations against the claim-level top-k pool instead of per-obligation retrieval.
2. **LLM-aggregation ablation:** give the ledger to the LLM for the final verdict instead of the rule. This isolates the deterministic rule.
3. **Untyped decomposition, FActScore/WiCE style (required):** atomic facts without core/qualifier types, the same statuses and an all-must-hold rule. This isolates *typing*.
4. **Re-retrieval for random obligations** at an equal number of rounds and abstracts (required for RQ2).
5. **Confidence-driven iteration at claim level, FIRE/QACheck style,** with the same caps (≤2 rounds, ≤2 abstracts per round). This is required, because it is the strongest prior form of adaptive retrieval.
6. **Compute-matched direct RAG:** self-consistency over N samples, so that the token count matches the ledger's multiple Auditor calls.
7. **Reference numbers:**
   - Cite the SciFact *test* numbers (MultiVerS 72.5; GPT-4 few-shot 72.2) as context only, since our evaluation is on dev.
   - If feasible, run the public MultiVerS checkpoint on dev, using the same retrieval or the oracle-abstract setting, for a comparison on the same split.
   - Note that Evidence-Ledger Adjudication reports 0.853 accuracy on 300 SciFact dev rows, but with *given* evidence, so it is not comparable to an open-retrieval setting.
8. **Optional:** use MiniCheck as an alternative Auditor or as an independent judge of whether the cited sentences entail the status (ALCE/AIS-style citation precision and recall).

### 5.5 Metrics and reporting
- Report official SciFact abstract-level metrics (label only, and label plus rationale) alongside claim-level macro-F1, since the Auditor's sentence ids allow rationale scoring.
- Report bootstrap confidence intervals. SciFact dev is small (300 claims).
- Report cost: LLM calls, tokens, abstracts read.
- Report a breakdown by obligation type.
- Acknowledge possible pretraining contamination of SciFact abstracts for Qwen3-8B, and include a claim-only, no-retrieval baseline as a contamination probe. The planned zero-shot baseline covers this if it is labelled as such.

---

## 6. References and verification status

Every entry was checked with the reference-lookup tool `verify_bibtex` (OpenAlex/arXiv) on 2026-10-27. "Verified" means that the title, first author and year matched. Venue notes come from the arXiv comments field or the ACL Anthology, IJCAI and arXiv pages that I read, and are marked where unconfirmed.

| # | Reference | Status |
|---|---|---|
| 1 | Wadden, Lin, Lo, Wang, van Zuylen, Cohan, Hajishirzi. *Fact or Fiction: Verifying Scientific Claims.* EMNLP 2020. arXiv:2004.14974 | Verified (arXiv) |
| 2 | Wadden, Lo, Kuehl, Cohan, Beltagy, Wang, Hajishirzi. *SciFact-Open: Towards open-domain scientific claim verification.* Findings EMNLP 2022. arXiv:2210.13777 | Verified (arXiv) |
| 3 | Wadden, Lo, Wang, Cohan, Beltagy, Hajishirzi. *MultiVerS: Improving scientific claim verification with weak supervision and full-document context.* Findings NAACL 2022. arXiv:2112.01640 | Verified (arXiv; arXiv year 2021) |
| 4 | Kotonya, Toni. *Explainable Automated Fact-Checking for Public Health Claims.* EMNLP 2020. arXiv:2010.09926 | Verified (arXiv) |
| 5 | Wright, Wadden, Lo, Kuehl, Cohan, Augenstein, Wang. *Generating Scientific Claims for Zero-Shot Scientific Fact Checking.* ACL 2022. arXiv:2203.12990 | Verified (arXiv) |
| 6 | Sarrouti et al. *Evidence-based Fact-Checking of Health-related Claims.* Findings EMNLP 2021. doi:10.18653/v1/2021.findings-emnlp.297 | Verified (OpenAlex title match); not discussed in the body |
| 7 | Alvarez, Bennett, Wang. *Zero-shot Scientific Claim Verification Using LLMs and Citation Text.* SDP Workshop 2024, pp. 269–276. doi:10.18653/v1/2024.sdp-1.25 | Verified (OpenAlex); full text read |
| 8 | Vladika, Hacajová, Matthes. *Step-by-Step Fact Verification System for Medical Claims with Explainable Reasoning.* NAACL 2025 (short), pp. 805–816. doi:10.18653/v1/2025.naacl-short.68; arXiv:2502.14765 | Verified (DOI + arXiv); HTML read |
| 9 | Trinh, Nguyen, Hy. *Towards Robust Fact-Checking: A Multi-Agent System with Advanced Evidence Retrieval.* arXiv:2506.17878 (2025) | Verified (arXiv); preprint, no venue found; HTML read |
| 10 | Chen, Yu, Wang. *Evidence-Ledger Adjudication for Claim-Evidence Traceability.* arXiv:2607.26512 (2026) | Verified (arXiv); preprint |
| 11 | Sadeghi, Khajavi, Adhikari, Tessier. *DeepSciVerify: Verifying Scientific Claim–Citation Alignment via LLM-Driven Evidence Escalation.* arXiv:2605.27710 (2026) | Verified (arXiv); abstract only |
| 12 | Deka, Singh. *When Retrieval Helps and Distracts: Evaluating Evidence-Generating LLMs for Biomedical Claim Verification.* arXiv:2608.01409 (2026) | Verified (arXiv); abstract only |
| 13 | Tan, Li, Gahegan. *SciTrue: Evidence-Grounded Claim Verification in Science.* EACL 2026 System Demonstrations, pp. 397–406. doi:10.18653/v1/2026.eacl-demo.27 | Verified (DOI/OpenAlex); full text read |
| 14 | Kamoi, Goyal, Rodriguez, Durrett. *WiCE: Real-World Entailment for Claims in Wikipedia.* EMNLP 2023. arXiv:2303.01432 | Verified (arXiv) |
| 15 | Min et al. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP 2023. arXiv:2305.14251 | Verified (arXiv) |
| 16 | Wei et al. *Long-form factuality in large language models* (SAFE). NeurIPS 2024. arXiv:2403.18802 | Verified (arXiv) |
| 17 | Song, Kim, Iyyer. *VeriScore: Evaluating the factuality of verifiable claims in long-form text generation.* arXiv:2406.19276 (2024) | Verified (arXiv); venue (Findings EMNLP 2024) not confirmed |
| 18 | Chen, Sriram, Choi, Durrett. *Generating Literal and Implied Subquestions to Fact-check Complex Claims* (ClaimDecomp). arXiv:2205.06938; EMNLP 2022 | Verified (arXiv); venue not confirmed from the page I read |
| 19 | Pan, Wu, Lu, Luu, Wang, Kan, Nakov. *Fact-Checking Complex Claims with Program-Guided Reasoning* (ProgramFC). ACL 2023. arXiv:2305.12744 | Verified (arXiv) |
| 20 | Pan, Lu, Kan, Nakov. *QACheck: A Demonstration System for Question-Guided Multi-Hop Fact-Checking.* EMNLP 2023 Demos. arXiv:2310.07609 | Verified (arXiv) |
| 21 | Zhang, Gao. *Towards LLM-based Fact Verification on News Claims with a Hierarchical Step-by-Step Prompting Method* (HiSS). AACL 2023. arXiv:2310.00305 | Verified (arXiv) |
| 22 | Wang et al. *Factcheck-Bench: Fine-Grained Evaluation Benchmark for Automatic Fact-checkers.* arXiv:2311.09000 | Verified (arXiv); venue (Findings EMNLP 2024) not confirmed |
| 23 | Chen et al. *Dense X Retrieval: What Retrieval Granularity Should We Use?* arXiv:2312.06648 | Verified (arXiv); venue (EMNLP 2024) not confirmed |
| 24 | Hu, Long, Wang. *Decomposition Dilemmas: Does Claim Decomposition Boost or Burden Fact-Checking Performance?* NAACL 2025. arXiv:2411.02400 | Verified (arXiv) |
| 25 | Wanner, Ebner, Jiang, Dredze, Van Durme. *A Closer Look at Claim Decomposition.* *SEM 2024. arXiv:2403.11903 | Verified (arXiv); venue taken from the ACL Anthology URL in search results |
| 26 | Akhter, Ruggeri, Bilal, Procter, Liakata. *The Alignment Bottleneck in Decomposition-Based Claim Verification.* IJCAI-26. arXiv:2602.10380 | Verified (arXiv); IJCAI proceedings PDF read |
| 27 | Lu, Ziems, Dang, Jiang. *Optimizing Decomposition for Optimal Claim Verification.* arXiv:2503.15354 (2025) | Verified (arXiv); venue not confirmed |
| 28 | Li, Chen, Kapadia, Ouyang, Zhang. *Minimal Evidence Group Identification for Claim Verification.* arXiv:2404.15588; TrustNLP 2025 | Verified (arXiv); venue from search results only |
| 29 | Xie et al. *FIRE: Fact-checking with Iterative Retrieval and Verification.* Findings NAACL 2025. arXiv:2411.00784 | Verified (arXiv) |
| 30 | Zheng et al. *Fact in Fragments: Deconstructing Complex Claims via LLM-based Atomic Fact Extraction and Verification* (AFEV). arXiv:2506.07446 (2025) | Verified (arXiv); journal version unverified; aggregation rule from a secondary source, unverified |
| 31 | Shukla, Dutta, Bhattacharyya. *Recon, Answer, Verify: Agents in Search of Truth.* arXiv:2507.03671 (2025) | Verified (arXiv); an EMNLP 2025 Industry version appears in search results, unverified |
| 32 | Hong, Luo, Wan. *EMULATE: A Multi-Agent Framework for Determining the Veracity of Atomic Claims by Emulating Human Actions.* FEVER Workshop 2025, pp. 179–183. doi:10.18653/v1/2025.fever-1.13 | Verified (DOI) |
| 33 | Press et al. *Measuring and Narrowing the Compositionality Gap in Language Models* (Self-Ask). Findings EMNLP 2023. arXiv:2210.03350 | Verified (arXiv) |
| 34 | Trivedi, Balasubramanian, Khot, Sabharwal. *Interleaving Retrieval with Chain-of-Thought Reasoning ...* (IRCoT). ACL 2023. arXiv:2212.10509 | Verified (arXiv) |
| 35 | Zhou et al. *Least-to-Most Prompting Enables Complex Reasoning in Large Language Models.* arXiv:2205.10625 | Verified (arXiv); venue (ICLR 2023) not confirmed |
| 36 | Jiang et al. *Active Retrieval Augmented Generation* (FLARE). EMNLP 2023. arXiv:2305.06983 | Verified (arXiv) |
| 37 | Asai, Wu, Wang, Sil, Hajishirzi. *Self-RAG ...* arXiv:2310.11511 | Verified (arXiv); venue (ICLR 2024) not confirmed |
| 38 | Jeong, Baek, Cho, Hwang, Park. *Adaptive-RAG ...* NAACL 2024. arXiv:2403.14403 | Verified (arXiv) |
| 39 | Yan, Gu, Zhu, Ling. *Corrective Retrieval Augmented Generation* (CRAG). arXiv:2401.15884 | Verified (arXiv); preprint |
| 40 | Gao, Yen, Yu, Chen. *Enabling Large Language Models to Generate Text with Citations* (ALCE). EMNLP 2023. arXiv:2305.14627 | Verified (arXiv) |
| 41 | Rashkin et al. *Measuring Attribution in Natural Language Generation Models* (AIS). arXiv:2112.12870 | Verified (arXiv); journal (Computational Linguistics) not confirmed |
| 42 | Yue et al. *Automatic Evaluation of Attribution by Large Language Models.* arXiv:2305.06311 | Verified (arXiv); venue not confirmed |
| 43 | Tang, Laban, Durrett. *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents.* EMNLP 2024. arXiv:2404.10774 | Verified (arXiv) |
| 44 | Du et al. *Improving Factuality and Reasoning in Language Models through Multiagent Debate.* arXiv:2305.14325 | Verified (arXiv); venue (ICML 2024) not confirmed |
| 45 | He et al. *Debating Truth: Debate-driven Claim Verification with Multiple LLM Agents* (DebateCV). WWW 2026. arXiv:2507.19090; doi:10.1145/3774904.3792993 | Verified (arXiv + DOI) |
| 46 | Dhuliawala et al. *Chain-of-Verification Reduces Hallucination in Large Language Models.* arXiv:2309.11495 | Verified (arXiv); not discussed in the body (optional citation) |
| 47 | Kamath, Jia, Liang. *Selective Question Answering under Domain Shift.* ACL 2020. arXiv:2006.09462 | Verified (arXiv); venue not confirmed |
| 48 | Wen et al. *Know Your Limits: A Survey of Abstention in Large Language Models.* arXiv:2407.18418 | Verified (arXiv) |
| 49 | Glockner, Hou, Gurevych. *Missing Counter-Evidence Renders NLP Fact-Checking Unrealistic for Misinformation.* EMNLP 2022. arXiv:2210.13865 | Verified (arXiv) |
| 50 | Peters, Chin-Yee. *Generalization Bias in Large Language Model Summarization of Scientific Research.* arXiv:2504.00025 (2025) | Verified (arXiv); content read only through search snippets |
| 51 | Robertson, Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond.* 2009. doi:10.1561/1500000019 | Verified (DOI) |
| 52 | Cormack, Clarke, Büttcher. *Reciprocal rank fusion outperforms Condorcet and individual rank learning methods.* SIGIR 2009. doi:10.1145/1571941.1572114 | Verified (DOI) |
| 53 | Xiao et al. *C-Pack: Packed Resources For General Chinese Embeddings* (BGE). arXiv:2309.07597 | Verified (arXiv), after correcting the title |
| 54 | Yang et al. *Qwen3 Technical Report.* arXiv:2505.09388 | Verified (arXiv) |

**Not verified or uncertain:**
- The SciTrue NTCIR-19 SciClaimEval paper (arXiv 2609.00654): seen in search results only; not read.
- The AFEV journal version and its aggregation rule.
- The NAACL 2025 status of FIRE: the arXiv comment says "Findings of NAACL"; the ACL Anthology PDF URL `2025.findings-naacl.158` was seen in search results.
- Several venue notes, as marked in the table.

I did not cite FOLK, CliVER/CoVERt, HealthFC, MedFact, CREDENCE or "Distill and Align Decomposition", because I did not read them closely enough.
