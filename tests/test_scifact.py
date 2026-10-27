import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.data.scifact import claim_label, development_split, load_claims, load_corpus  # noqa: E402
from oblig.retrieval.bm25 import BM25  # noqa: E402
from oblig.retrieval.hybrid import HybridRetriever, rrf  # noqa: E402


def test_split_labels():
    assert Counter(c.label for c in load_claims("train")) == {"SUPPORTS": 332, "NEI": 304, "REFUTES": 173}
    assert Counter(c.label for c in load_claims("dev")) == {"SUPPORTS": 124, "NEI": 112, "REFUTES": 64}
    assert all(c.label is None for c in load_claims("test"))


def test_claim_label_rule():
    assert claim_label({}) == "NEI"
    assert claim_label({"1": [{"sentences": [0], "label": "CONTRADICT"}]}) == "REFUTES"


def test_development_split_is_train_only_and_deterministic():
    train = load_claims("train")
    a, b = development_split(train)
    a2, b2 = development_split(train)
    dev_ids = {c.id for c in load_claims("dev")}
    assert [c.id for c in b] == [c.id for c in b2] and len(b) == 200
    assert not ({c.id for c in b} & {c.id for c in a}) and not ({c.id for c in b} & dev_ids)


def test_gold_rationales_resolve_to_corpus_sentences():
    corpus = load_corpus()
    assert len(corpus) == 5183
    for c in load_claims("dev"):
        for doc, sents in c.evidence.items():
            assert doc in corpus
            assert all(0 <= i < len(corpus[doc].sentences) for i in sents)


def test_rrf_fuses_and_is_deterministic():
    fused = rrf([[("a", 9), ("b", 5)], [("b", 0.9), ("c", 0.8)]], k=60)
    assert fused[0][0] == "b" and {d for d, _ in fused} == {"a", "b", "c"}
    assert rrf([[("x", 1)], [("y", 1)]]) == [("x", 1 / 61), ("y", 1 / 61)]


def test_hybrid_with_bm25_only_matches_bm25_order():
    texts, ids = ["aspirin lowers stroke risk", "statins lower cholesterol", "aspirin and bleeding"], ["1", "2", "3"]
    bm = BM25().fit(texts, ids)
    h = HybridRetriever([bm], depth=3)
    assert [d for d, _ in h.search("aspirin stroke", 2)] == [d for d, _ in bm.search("aspirin stroke", 2)]
