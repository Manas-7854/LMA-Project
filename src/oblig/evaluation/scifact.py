"""SciFact metrics: claim-level macro-F1 and abstract-level label / label+rationale F1.

Abstract level follows the SciFact definitions: a predicted (abstract, label) is
correct if the abstract is a gold evidence abstract with that label; it is also
correctly *rationalised* if some gold rationale set is contained in the first
three predicted sentences. NEI predictions contribute no abstracts.
Sentence-level numbers here are a simplified rationale P/R/F1 (a sentence counts
if it is in a gold rationale of that abstract), not the official sentence metric.
"""
from __future__ import annotations

from ..data.scifact import LABELS, Claim
from .classification import classification_report

MAX_RATIONALE = 3


def _f1(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "tp": tp, "n_pred": n_pred, "n_gold": n_gold}


def abstract_level(claims: list[Claim], preds: dict[int, dict]) -> dict:
    tp_label = tp_rat = n_pred = n_gold = 0
    s_tp = s_pred = s_gold = 0
    for c in claims:
        n_gold += len(c.evidence)
        s_gold += sum(len(v) for v in c.evidence.values())
        p = preds[c.id]
        if p["pred_label"] not in ("SUPPORTS", "REFUTES"):
            continue
        for doc, sents in p.get("rationale", {}).items():
            n_pred += 1
            sents = list(sents)[:MAX_RATIONALE]
            s_pred += len(sents)
            if doc in c.evidence:
                s_tp += len(set(sents) & set(c.evidence[doc]))
                if c.doc_labels[doc] == p["pred_label"]:
                    tp_label += 1
                    if any(set(g) <= set(sents) for g in c.evidence_sets[doc]):
                        tp_rat += 1
    return {"label_only": _f1(tp_label, n_pred, n_gold), "label_rationale": _f1(tp_rat, n_pred, n_gold),
            "rationale_sentences_simplified": _f1(s_tp, s_pred, s_gold)}


def evaluate(claims: list[Claim], preds: dict[int, dict]) -> dict:
    gold = [c.label for c in claims]
    pred = [preds[c.id]["pred_label"] for c in claims]
    return {"claim_level": classification_report(gold, pred, labels=LABELS),
            "abstract_level": abstract_level(claims, preds)}
