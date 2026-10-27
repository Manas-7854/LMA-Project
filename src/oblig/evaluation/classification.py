"""Classification metrics and paired bootstrap for prediction files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from ..data.pubhealth import LABELS

INVALID = "__invalid__"  # prediction for a failed/unparseable call; always counted wrong


def read_predictions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def macro_f1(gold, pred, labels=LABELS) -> float:
    return float(f1_score(gold, pred, labels=list(labels), average="macro", zero_division=0))


def classification_report(gold: list[str], pred: list[str], labels=LABELS) -> dict:
    pred = [p if p in labels else INVALID for p in pred]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=list(labels), zero_division=0)
    return {"n": len(gold), "macro_f1": round(macro_f1(gold, pred, labels), 4),
            "accuracy": round(float(accuracy_score(gold, pred)), 4),
            "invalid_predictions": int(sum(x == INVALID for x in pred)),
            "per_class": {l: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                              "f1": round(float(f[i]), 4), "support": int(s[i])} for i, l in enumerate(labels)},
            "confusion_matrix": {"labels": list(labels) + [INVALID],
                                 "matrix": confusion_matrix(gold, pred, labels=list(labels) + [INVALID]).tolist()}}


def paired_bootstrap(gold, pred_a, pred_b, metric=macro_f1, n_boot: int = 10_000, seed: int = 13) -> dict:
    """95% CI of metric(b) - metric(a) by resampling the same examples for both systems."""
    gold, pred_a, pred_b = map(np.asarray, (gold, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(gold)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = metric(gold[idx], pred_b[idx]) - metric(gold[idx], pred_a[idx])
    return {"delta": round(metric(gold, pred_b) - metric(gold, pred_a), 4),
            "ci95": [round(float(np.percentile(deltas, 2.5)), 4), round(float(np.percentile(deltas, 97.5)), 4)],
            "p_delta_le_0": round(float((deltas <= 0).mean()), 4), "n": n, "n_boot": n_boot}
