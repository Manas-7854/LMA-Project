"""Deterministic verdict policies over a ledger's current (effective) statuses.

scifact_v1 is the proposal's rule and is fixed. pubhealth_v1 is a *candidate*
four-class rule; it is selected or revised on PUBHEALTH dev before Gate 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import CONTRADICTED, PARTIAL, SATISFIED, Ledger


@dataclass
class Verdict:
    label: str
    reason: str
    rationale_docs: dict[str, list[int]] = field(default_factory=dict)   # doc_id -> cited sentence ids


def cited_evidence(ledger: Ledger, status: str, core_only: bool = False) -> dict[str, list[int]]:
    out: dict[str, set[int]] = {}
    for o in ledger.obligations:
        a = ledger.latest(o.obligation_id)
        if a and a.effective_status == status and (not core_only or o.criticality == "core"):
            for e in ledger.evidence(o.obligation_id):
                out.setdefault(e.doc_id, set()).update(e.sentence_ids)
    return {d: sorted(s) for d, s in out.items()}


def scifact_v1(ledger: Ledger) -> Verdict:
    """REFUTES if any obligation CONTRADICTED; SUPPORTS if every core obligation is
    SATISFIED by sentences of one abstract and none is CONTRADICTED; else NEI.
    Qualifiers never block SUPPORTS."""
    core = [o for o in ledger.obligations if o.criticality == "core"]
    if not core:
        return Verdict("NEI", "no core obligations")
    if any(ledger.status(o.obligation_id) == CONTRADICTED for o in ledger.obligations):
        return Verdict("REFUTES", "an obligation is contradicted", cited_evidence(ledger, CONTRADICTED))
    if not all(ledger.status(o.obligation_id) == SATISFIED for o in core):
        return Verdict("NEI", "a core obligation is not satisfied")
    docs_per_ob = [{e.doc_id for e in ledger.evidence(o.obligation_id)} for o in core]
    common = set.intersection(*docs_per_ob)
    if not common:
        return Verdict("NEI", "core obligations satisfied only across different abstracts")
    doc = sorted(common)[0]
    sents = sorted({i for o in core for e in ledger.evidence(o.obligation_id) if e.doc_id == doc
                    for i in e.sentence_ids})
    return Verdict("SUPPORTS", "all core obligations satisfied in one abstract", {doc: sents})


def pubhealth_v1(ledger: Ledger) -> Verdict:
    """Candidate: mixture if core obligations are both satisfied and contradicted
    (or a core one is only PARTIAL while another is contradicted); false if a core
    obligation is contradicted and none satisfied; true if all core satisfied;
    unproven otherwise."""
    core = [o for o in ledger.obligations if o.criticality == "core"]
    if not core:
        return Verdict("unproven", "no core obligations")
    st = [ledger.status(o.obligation_id) for o in core]
    if CONTRADICTED in st and (SATISFIED in st or PARTIAL in st):
        return Verdict("mixture", "core obligations both established and contradicted",
                       {**cited_evidence(ledger, SATISFIED, True), **cited_evidence(ledger, CONTRADICTED, True)})
    if CONTRADICTED in st:
        return Verdict("false", "a core obligation is contradicted", cited_evidence(ledger, CONTRADICTED, True))
    if all(s == SATISFIED for s in st):
        return Verdict("true", "all core obligations satisfied", cited_evidence(ledger, SATISFIED, True))
    return Verdict("unproven", "core obligations unresolved")


POLICIES = {"scifact_v1": scifact_v1, "pubhealth_v1": pubhealth_v1}
