"""Run a SciFact system and write one prediction file, metrics and a run manifest.

Resumable: claims already in the prediction file are skipped. The held-out set
(claims_dev) requires --heldout, which is only to be used after Gate 3.

Usage:
  python scripts/run_scifact.py --system m1 --eval-set train_dev --n 20 --profile mock_scifact --exp-id smoke_sf_m1
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.data.scifact import development_split, load_claims, load_corpus  # noqa: E402
from oblig.evaluation.scifact import evaluate  # noqa: E402
from oblig.llm import prompts_scifact as P  # noqa: E402
from oblig.llm.client import LLMClient, LLMConfig  # noqa: E402
from oblig.llm.mock_scifact import mock_scifact  # noqa: E402
from oblig.retrieval.factory import build_retriever  # noqa: E402
from oblig.systems.scifact import SYSTEMS, Context  # noqa: E402
from oblig.utils.cache import Cache  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    ap.add_argument("--eval-set", choices=["train_dev", "dev"], default="train_dev")
    ap.add_argument("--heldout", action="store_true", help="required for --eval-set dev (after Gate 3 only)")
    ap.add_argument("--n", type=int, default=None, help="first n claims by id (development pilots)")
    ap.add_argument("--retriever", default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--exp-id", required=True)
    args = ap.parse_args()
    if args.eval_set == "dev" and not args.heldout:
        sys.exit("claims_dev is the held-out set: pass --heldout only after the protocol is frozen (Gate 3)")

    rcfg = yaml.safe_load(open(ROOT / "configs" / "retrieval.yaml", encoding="utf-8"))["scifact"]
    prof = yaml.safe_load(open(ROOT / "configs" / "model.yaml", encoding="utf-8"))[args.profile]
    cfg = LLMConfig(**prof)
    if args.profile == "mock_scifact":
        cfg.extra["mock_fn"] = mock_scifact
    retriever_name = args.retriever or rcfg.get("retriever", "hybrid_bm25_minilm")
    budget = args.budget or rcfg["evidence_budget_abstracts"]
    corpus = load_corpus()
    ctx = Context(LLMClient(cfg, Cache(ROOT / "results" / "cache" / "llm.sqlite")),
                  build_retriever(retriever_name, corpus), corpus, budget=budget)
    claims = development_split(load_claims("train"))[1] if args.eval_set == "train_dev" else load_claims("dev")
    if args.n:
        claims = claims[: args.n]

    out = ROOT / "results" / "predictions" / f"{args.exp_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Resuming is only allowed with an identical configuration (no stale outputs).
    run_cfg = {"system": args.system, "eval_set": args.eval_set, "profile": prof, "retriever": retriever_name,
               "budget": budget, "policy": ctx.policy, "m2_rounds": ctx.m2_rounds,
               "m2_new_per_round": ctx.m2_new_per_round, "sc_samples": ctx.sc_samples,
               "sc_temperature": ctx.sc_temperature, "seed": ctx.seed, "m2_consolidate": ctx.m2_consolidate,
               "prompt_versions": {k: v for k, v in vars(P).items()
                                                                           if k.endswith("_VERSION")}}
    cfg_path = out.with_suffix(".config.json")
    if out.exists() and cfg_path.exists() and json.loads(cfg_path.read_text()) != run_cfg:
        sys.exit(f"{out.name} was produced with a different configuration; use a new --exp-id")
    cfg_path.write_text(json.dumps(run_cfg, indent=2))
    done = {}
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            done[r["claim_id"]] = r
    t0 = time.perf_counter()
    with open(out, "a", encoding="utf-8") as f:
        for c in claims:
            if c.id in done:
                continue
            t = time.perf_counter()
            rec = SYSTEMS[args.system](ctx, c)
            rec = {"claim_id": c.id, "gold_label": c.label, "system": args.system, "model": cfg.model,
                   **rec, "n_calls": len(rec["calls"]),
                   "prompt_tokens": sum(x["prompt_tokens"] for x in rec["calls"]),
                   "completion_tokens": sum(x["completion_tokens"] for x in rec["calls"]),
                   "n_call_errors": sum(x["error"] is not None for x in rec["calls"]),
                   "wall_s": round(time.perf_counter() - t, 3)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[c.id] = rec
    preds = {c.id: done[c.id] for c in claims}
    metrics = evaluate(claims, preds)
    recs = list(preds.values())
    iterative = [r for r in recs if "verdict_changed" in r]
    m2_stats = {} if not iterative else {
        "claims_entering_iteration": sum(bool(r.get("entered_m2", r.get("rounds"))) for r in iterative),
        "verdicts_changed": sum(r["verdict_changed"] for r in iterative),
        "changes_fixing_error": sum(r["verdict_changed"] and r["pred_label"] == r["gold_label"] for r in iterative),
        "changes_introducing_error": sum(r["verdict_changed"] and r.get("m1_label", r.get("first_label"))
                                         == r["gold_label"] for r in iterative),
        "consolidation_calls": sum(c["role"] == "consolidation" for r in iterative for c in r["calls"]),
        "abstracts_seen_mean": round(sum(len(r["retrieved"]) for r in iterative) / len(iterative), 2),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in iterative))}
    cost = {"calls_per_claim": round(sum(r["n_calls"] for r in recs) / len(recs), 3),
            "prompt_tokens_per_claim": round(sum(r["prompt_tokens"] for r in recs) / len(recs), 1),
            "completion_tokens_per_claim": round(sum(r["completion_tokens"] for r in recs) / len(recs), 1),
            "call_errors": sum(r["n_call_errors"] for r in recs),
            "invalid_citations": sum(len(r.get("invalid_citations", [])) for r in recs)}
    manifest = {"exp_id": args.exp_id, "dataset": "scifact", "system": args.system, "eval_set": args.eval_set,
                "n": len(claims), "profile": args.profile, "llm": {k: v for k, v in prof.items()},
                "retriever": retriever_name, "evidence_budget_abstracts": budget, "policy": ctx.policy,
                "system_params": {k: run_cfg[k] for k in ("m2_rounds", "m2_new_per_round", "sc_samples",
                                                          "sc_temperature", "seed", "m2_consolidate")},
                "git_commit": git_commit(), "python": platform.python_version(), "platform": platform.platform(),
                "run_utc": datetime.now(timezone.utc).isoformat(), "wall_clock_s": round(time.perf_counter() - t0, 1),
                "predictions": str(out.relative_to(ROOT)).replace("\\", "/")}
    (ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "metrics" / f"{args.exp_id}.json").write_text(
        json.dumps({"manifest": manifest, "metrics": metrics, "cost": cost, "iteration": m2_stats}, indent=2))
    cl = metrics["claim_level"]
    print(json.dumps({"exp_id": args.exp_id, "macro_f1": cl["macro_f1"], "invalid": cl["invalid_predictions"],
                      "abstract_label_rationale_f1": metrics["abstract_level"]["label_rationale"]["f1"], **cost}))


if __name__ == "__main__":
    main()
