"""Versioned SciFact prompts and output schemas (B0, B1, B2, Planner, Auditor).

Each system prompt starts with a role tag ("ROLE: ...") used only by the mock
backend to route test outputs. Bump a version when its wording changes.
"""
from __future__ import annotations

from ..data.scifact import LABELS
from ..ledger.schema import CORE_TYPES, MAX_OBLIGATIONS, QUALIFIER_TYPES, STATUSES

LABEL_DEFINITIONS = (
    "Labels:\n"
    "- SUPPORTS: the evidence establishes the claim.\n"
    "- REFUTES: the evidence contradicts the claim.\n"
    "- NEI: the evidence neither establishes nor contradicts the claim.")

_CITATIONS = {"type": "array", "items": {
    "type": "object", "required": ["doc", "sentences"],
    "properties": {"doc": {"type": "string"},
                   "sentences": {"type": "array", "items": {"type": "integer"}}}}}


def _system(role: str, body: str) -> dict:
    return {"role": "system", "content": f"ROLE: {role}\n{body} Answer with a single JSON object only."}


def format_abstracts(abstracts: list[tuple[str, str, list[str]]]) -> str:
    """abstracts: (alias, title, sentences). Sentences are numbered from 0."""
    blocks = []
    for alias, title, sents in abstracts:
        lines = "\n".join(f"  ({i}) {s}" for i, s in enumerate(sents))
        blocks.append(f"[{alias}] {title}\n{lines}")
    return "\n\n".join(blocks)


# -- B0 ----------------------------------------------------------------------------------
B0_VERSION = "sf-b0-v1"
B0_SCHEMA = {"type": "object", "required": ["verdict", "justification"],
             "properties": {"verdict": {"enum": list(LABELS)}, "justification": {"type": "string"}}}


def b0_messages(claim: str) -> list[dict]:
    return [_system("verdict", "You verify scientific claims."),
            {"role": "user", "content": (
                f"Judge the claim from your own knowledge.\n{LABEL_DEFINITIONS}\n\nClaim: {claim}\n\n"
                'Return JSON: {"verdict": "SUPPORTS|REFUTES|NEI", "justification": "<one or two sentences>"}')}]


# -- B1 ----------------------------------------------------------------------------------
B1_VERSION = "sf-b1-v1"
VERDICT_SCHEMA = {"type": "object", "required": ["verdict", "citations", "justification"],
                  "properties": {"verdict": {"enum": list(LABELS)}, "citations": _CITATIONS,
                                 "justification": {"type": "string"}}}


def b1_messages(claim: str, abstracts) -> list[dict]:
    return [_system("verdict", "You verify scientific claims against research abstracts."),
            {"role": "user", "content": (
                f"Judge the claim using only the abstracts below.\n{LABEL_DEFINITIONS}\n\n"
                f"Abstracts:\n{format_abstracts(abstracts)}\n\nClaim: {claim}\n\n"
                "Cite the abstract ids and sentence numbers your verdict relies on (none for NEI).\n"
                'Return JSON: {"verdict": "SUPPORTS|REFUTES|NEI", "citations": [{"doc": "A1", "sentences": [0]}], '
                '"justification": "<one or two sentences>"}')}]


# -- B2: ordinary sub-questions, one verdict, no statuses ----------------------------------
B2_DECOMP_VERSION = "sf-b2-decomp-v1"
B2_DECOMP_SCHEMA = {"type": "object", "required": ["questions"],
                    "properties": {"questions": {"type": "array", "minItems": 1, "maxItems": MAX_OBLIGATIONS,
                                                 "items": {"type": "string"}}}}


def b2_decomp_messages(claim: str) -> list[dict]:
    return [_system("subquestions", "You help search the scientific literature."),
            {"role": "user", "content": (
                f"Write up to {MAX_OBLIGATIONS} short search questions whose answers would help verify the claim. "
                f"Use as few as the claim needs.\n\nClaim: {claim}\n\n"
                'Return JSON: {"questions": ["<question>", ...]}')}]


B2_VERDICT_VERSION = "sf-b2-verdict-v1"


def b2_verdict_messages(claim: str, questions: list[str], abstracts) -> list[dict]:
    qs = "\n".join(f"- {q}" for q in questions)
    return [_system("verdict", "You verify scientific claims against research abstracts."),
            {"role": "user", "content": (
                f"Judge the claim using only the abstracts below, which were retrieved for these questions:\n{qs}\n\n"
                f"{LABEL_DEFINITIONS}\n\nAbstracts:\n{format_abstracts(abstracts)}\n\nClaim: {claim}\n\n"
                "Cite the abstract ids and sentence numbers your verdict relies on (none for NEI).\n"
                'Return JSON: {"verdict": "SUPPORTS|REFUTES|NEI", "citations": [{"doc": "A1", "sentences": [0]}], '
                '"justification": "<one or two sentences>"}')}]


# -- M1 Planner ----------------------------------------------------------------------------
PLANNER_VERSION = "sf-planner-v1"
PLANNER_SCHEMA = {"type": "object", "required": ["obligations"], "properties": {"obligations": {
    "type": "array", "minItems": 1, "maxItems": MAX_OBLIGATIONS, "items": {
        "type": "object", "required": ["description", "type", "expected_condition", "query"],
        "properties": {"description": {"type": "string", "minLength": 1},
                       "type": {"enum": list(CORE_TYPES + QUALIFIER_TYPES)},
                       "expected_condition": {"type": "string"}, "query": {"type": "string", "minLength": 1}}}}}}


def planner_messages(claim: str) -> list[dict]:
    return [_system("planner", "You decompose scientific claims into what evidence must establish."),
            {"role": "user", "content": (
                "List the atomic conditions that evidence must establish for the claim to be true.\n"
                f"Core types (what the claim is about): {', '.join(CORE_TYPES)} "
                "(subject = intervention/exposure/entity; outcome = measured property or effect; "
                "relation = direction or kind of relation between them).\n"
                f"Qualifier types (restrictions): {', '.join(QUALIFIER_TYPES)}.\n"
                f"Use at most {MAX_OBLIGATIONS}. Include a qualifier only if the claim states it. "
                "Do not repeat a condition. Each needs a short search query.\n\n"
                f"Claim: {claim}\n\n"
                'Return JSON: {"obligations": [{"description": "<condition>", "type": "<type>", '
                '"expected_condition": "<what the evidence must show>", "query": "<search query>"}]}')}]


# -- M1 Auditor ----------------------------------------------------------------------------
AUDITOR_VERSION = "sf-auditor-v1"
AUDITOR_SCHEMA = {"type": "object",
                  "required": ["status", "citations", "rationale", "needs_more_retrieval"],
                  "properties": {"status": {"enum": list(STATUSES)}, "citations": _CITATIONS,
                                 "rationale": {"type": "string"}, "needs_more_retrieval": {"type": "boolean"}}}

STATUS_DEFINITIONS = (
    "Statuses:\n"
    "- SATISFIED: the evidence establishes the condition.\n"
    "- CONTRADICTED: the evidence directly conflicts with the condition.\n"
    "- PARTIAL: the evidence addresses the condition but is incomplete or not specific enough.\n"
    "- UNSUPPORTED: no evidence establishes or contradicts the condition.")


def auditor_messages(claim: str, description: str, expected: str, abstracts) -> list[dict]:
    return [_system("auditor", "You check one condition of a scientific claim against research abstracts."),
            {"role": "user", "content": (
                f"Claim: {claim}\nCondition to check: {description}\nThe evidence must show: {expected}\n\n"
                f"{STATUS_DEFINITIONS}\n\nAbstracts:\n{format_abstracts(abstracts)}\n\n"
                "Judge only this condition, using only these abstracts. SATISFIED and CONTRADICTED must cite "
                "the sentences that show it.\n"
                'Return JSON: {"status": "SATISFIED|CONTRADICTED|PARTIAL|UNSUPPORTED", '
                '"citations": [{"doc": "A1", "sentences": [0]}], "rationale": "<one sentence>", '
                '"needs_more_retrieval": true|false}')}]


# -- M1-untyped: FActScore-style atomic facts ----------------------------------------------
FACTS_VERSION = "sf-facts-v1"
FACTS_SCHEMA = {"type": "object", "required": ["facts"], "properties": {"facts": {
    "type": "array", "minItems": 1, "maxItems": MAX_OBLIGATIONS, "items": {
        "type": "object", "required": ["fact", "query"],
        "properties": {"fact": {"type": "string", "minLength": 1}, "query": {"type": "string", "minLength": 1}}}}}}


def facts_messages(claim: str) -> list[dict]:
    return [_system("facts", "You break scientific claims into atomic facts."),
            {"role": "user", "content": (
                f"Break the claim into at most {MAX_OBLIGATIONS} atomic facts, each a short standalone statement, "
                "with a short search query for each. Use as few as the claim needs.\n\n"
                f"Claim: {claim}\n\n"
                'Return JSON: {"facts": [{"fact": "<statement>", "query": "<search query>"}]}')}]


# -- M2: targeted follow-up query for one unresolved obligation ------------------------------
REQUERY_VERSION = "sf-requery-v1"
REQUERY_SCHEMA = {"type": "object", "required": ["missing", "query"],
                  "properties": {"missing": {"type": "string"}, "query": {"type": "string", "minLength": 1}}}


def requery_messages(claim: str, description: str, expected: str, status: str, rationale: str) -> list[dict]:
    return [_system("requery", "You plan literature searches for scientific fact-checking."),
            {"role": "user", "content": (
                f"Claim: {claim}\nCondition: {description}\nThe evidence must show: {expected}\n"
                f"Current status: {status}\nAuditor's note: {rationale or '(none)'}\n\n"
                "State what information is still missing, and write one new search query for it.\n"
                'Return JSON: {"missing": "<what is missing>", "query": "<search query>"}')}]


# -- M2-claimiter: claim-level follow-up query (FIRE/QACheck-style) --------------------------
CLAIM_REQUERY_VERSION = "sf-claim-requery-v1"


def claim_requery_messages(claim: str, justification: str) -> list[dict]:
    return [_system("claim_requery", "You plan literature searches for scientific fact-checking."),
            {"role": "user", "content": (
                f"Claim: {claim}\nThe abstracts read so far were not enough to decide. "
                f"Note: {justification or '(none)'}\n\n"
                "State what information is still missing, and write one new search query for it.\n"
                'Return JSON: {"missing": "<what is missing>", "query": "<search query>"}')}]


B1_ITER_VERSION = "sf-b1-iter-v1"   # same wording as B1; separate version keeps caches apart


# -- M1-LLMagg: LLM verdict over the ledger --------------------------------------------------
AGGREGATE_VERSION = "sf-aggregate-v1"


def aggregate_messages(claim: str, rows: list[dict]) -> list[dict]:
    lines = []
    for r in rows:
        ev = "; ".join(f'"{e}"' for e in r["evidence"]) or "none"
        lines.append(f"- [{r['criticality']}] {r['description']}: {r['status']}. {r['rationale']} Evidence: {ev}")
    return [_system("aggregate", "You verify scientific claims from an evidence audit."),
            {"role": "user", "content": (
                "Each condition of the claim was checked separately against research abstracts:\n"
                + "\n".join(lines) + f"\n\n{LABEL_DEFINITIONS}\n\nClaim: {claim}\n\n"
                'Decide the verdict. Return JSON: {"verdict": "SUPPORTS|REFUTES|NEI", '
                '"justification": "<one or two sentences>"}')}]
