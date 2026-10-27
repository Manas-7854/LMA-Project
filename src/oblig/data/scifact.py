"""SciFact loading, claim-level labels and development splits.

Label space is SUPPORTS / REFUTES / NEI and is never merged with PUBHEALTH's.
claims_test.jsonl is unlabelled, so claims_dev.jsonl is the held-out evaluation
set. The shipped cross_validation folds mix train and dev claims and are not
used: prompt/retrieval development uses a seeded split of claims_train only.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

LABELS = ("SUPPORTS", "REFUTES", "NEI")
_EVIDENCE_LABEL = {"SUPPORT": "SUPPORTS", "CONTRADICT": "REFUTES"}


def default_root() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "raw" / "scifact" / "data"


@dataclass(frozen=True)
class Abstract:
    doc_id: str
    title: str
    sentences: list[str]
    structured: bool


@dataclass
class Claim:
    id: int
    claim: str
    label: str | None                                    # None for unlabelled test claims
    evidence: dict[str, list[int]] = field(default_factory=dict)          # doc_id -> rationale sentence ids
    evidence_sets: dict[str, list[list[int]]] = field(default_factory=dict)  # doc_id -> rationale sets
    doc_labels: dict[str, str] = field(default_factory=dict)              # doc_id -> SUPPORTS/REFUTES
    cited_doc_ids: list[str] = field(default_factory=list)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_corpus(root: Path | None = None) -> dict[str, Abstract]:
    rows = _read_jsonl((root or default_root()) / "corpus.jsonl")
    return {str(r["doc_id"]): Abstract(str(r["doc_id"]), r["title"].strip(),
                                       [s.strip() for s in r["abstract"]], bool(r["structured"])) for r in rows}


def claim_label(evidence: dict) -> str | None:
    """Claim-level label: NEI without evidence, else the (single) evidence label."""
    labels = {_EVIDENCE_LABEL[e["label"]] for ev in evidence.values() for e in ev}
    if not labels:
        return "NEI"
    if len(labels) > 1:
        raise ValueError("claim has both supporting and refuting evidence")
    return labels.pop()


def load_claims(split: str, root: Path | None = None) -> list[Claim]:
    out = []
    for r in _read_jsonl((root or default_root()) / f"claims_{split}.jsonl"):
        ev = r.get("evidence")
        c = Claim(r["id"], r["claim"].strip(), None if ev is None else claim_label(ev),
                  cited_doc_ids=[str(d) for d in r.get("cited_doc_ids", [])])
        for doc, sets in (ev or {}).items():
            c.evidence_sets[doc] = [sorted(s["sentences"]) for s in sets]
            c.evidence[doc] = sorted({i for s in sets for i in s["sentences"]})
            c.doc_labels[doc] = _EVIDENCE_LABEL[sets[0]["label"]]
        out.append(c)
    return out


def development_split(claims: list[Claim], n_dev: int = 200, seed: int = 13) -> tuple[list[Claim], list[Claim]]:
    """Label-stratified (train_part, dev_part) split of claims_train for development."""
    rng = random.Random(seed)
    by_label: dict[str, list[Claim]] = {}
    for c in sorted(claims, key=lambda c: c.id):
        by_label.setdefault(c.label, []).append(c)
    dev = []
    for label in LABELS:
        group = by_label.get(label, [])
        rng.shuffle(group)
        dev.extend(group[: round(n_dev * len(group) / len(claims))])
    dev_ids = {c.id for c in dev}
    return [c for c in claims if c.id not in dev_ids], sorted(dev, key=lambda c: c.id)
