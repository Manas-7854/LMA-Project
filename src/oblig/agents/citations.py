"""Resolve model citations against the evidence actually shown to the model."""
from __future__ import annotations

from ..data.scifact import Abstract
from ..ledger.schema import EvidenceRef


def alias_abstracts(doc_ids: list[str], corpus: dict[str, Abstract]):
    """Return prompt tuples (alias, title, sentences) and alias -> doc_id."""
    shown = [(f"A{i + 1}", corpus[d].title, corpus[d].sentences) for i, d in enumerate(doc_ids)]
    return shown, {f"A{i + 1}": d for i, d in enumerate(doc_ids)}


def resolve(citations: list[dict], alias: dict[str, str],
            corpus: dict[str, Abstract]) -> tuple[list[EvidenceRef], list[dict]]:
    """Keep citations whose alias was shown and whose sentence numbers exist; text comes
    from the corpus. Everything else is returned as invalid, never silently dropped."""
    valid: dict[str, set[int]] = {}
    invalid = []
    for c in citations:
        doc = alias.get(c.get("doc", ""))
        if doc is None:
            invalid.append({**c, "reason": "unknown document"})
            continue
        n = len(corpus[doc].sentences)
        good = [i for i in c.get("sentences", []) if 0 <= i < n]
        bad = [i for i in c.get("sentences", []) if not 0 <= i < n]
        if bad:
            invalid.append({"doc": c["doc"], "sentences": bad, "reason": "sentence out of range"})
        if good:
            valid.setdefault(doc, set()).update(good)
        elif not bad:
            invalid.append({**c, "reason": "no sentences"})
    refs = [EvidenceRef(d, sorted(s), " ".join(corpus[d].sentences[i] for i in sorted(s)))
            for d, s in valid.items()]
    return refs, invalid
