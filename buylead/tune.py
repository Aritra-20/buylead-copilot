"""Calibrate the neural-embedding refusal threshold on the DEV set only.

    python -m buylead.tune            # needs GEMINI_API_KEY; ~40 LLM calls + ~60 embedding calls

For each dev inquiry it runs the real LLM extractor, then neural search, and records the top
similarity. It picks the threshold that best separates "should match the gold product" from
"out-of-catalog, should refuse", preferring the widest safety margin among equally good cuts.
The result goes to data/embedding_threshold.json and is then used by the agent and the evals.
The held-out set is never used here, so held-out results stay an honest test.
"""
from __future__ import annotations

import json
from pathlib import Path

from .agent import BuyLeadAgent, neural_index
from .extract import ground_requirement
from .retrieve import THRESHOLD_FILE

ROOT = Path(__file__).resolve().parents[1]


def collect(dataset: str = "eval_set.jsonl") -> list[dict]:
    agent = BuyLeadAgent("llm", embeddings="gemini")
    index = neural_index()
    rows = []
    for case in (json.loads(l) for l in (ROOT / "data" / dataset).read_text(encoding="utf-8").splitlines() if l.strip()):
        req, step = agent.extractor.extract(case["inquiry"])
        req = ground_requirement(req, case["inquiry"], step)
        query = " ".join([req.product_query or ""] + req.specs).strip()
        if not req.in_scope or not query:
            continue  # service requests / empty queries are refused before retrieval
        (top, sim), *_ = index.search(query, k=1)
        g = case["gold"]
        rows.append({"id": case["id"], "query": query, "top": top, "sim": sim,
                     "should_refuse": g["action"] == "refuse", "gold": g.get("product")})
        print(f"case {case['id']:>3}: {sim:.3f}  {top:<26} <- {query!r}", flush=True)
    return rows


def best_threshold(rows: list[dict]) -> tuple[float, float, float]:
    """Return (threshold, accuracy, margin). A case is right if: refuse-gold -> sim < t;
    match-gold -> sim >= t and top product == gold product."""
    sims = sorted({r["sim"] for r in rows})
    cuts = [s - 1e-4 for s in sims] + [sims[-1] + 1e-4]
    best = None
    for t in cuts:
        ok = sum((r["sim"] < t) if r["should_refuse"] else (r["sim"] >= t and r["top"] == r["gold"]) for r in rows)
        below = [r["sim"] for r in rows if r["sim"] < t]
        above = [r["sim"] for r in rows if r["sim"] >= t]
        margin = (min(above) if above else 1.0) - (max(below) if below else 0.0)
        mid = ((max(below) if below else t) + (min(above) if above else t)) / 2
        cand = (ok, margin, mid)
        if best is None or cand[:2] > best[:2]:
            best = cand
    ok, margin, mid = best
    return round(mid, 4), ok / len(rows), round(margin, 4)


def main() -> None:
    rows = collect()
    t, acc, margin = best_threshold(rows)
    model = neural_index().embedder.model
    THRESHOLD_FILE.write_text(json.dumps({"threshold": t, "embedding_model": model, "calibrated_on": "eval_set.jsonl (dev)",
                                          "dev_retrieval_accuracy": round(acc, 4), "margin": margin}, indent=2) + "\n",
                              encoding="utf-8")
    print(f"\nthreshold {t}  (dev retrieval accuracy {acc:.1%}, margin {margin})  -> saved to {THRESHOLD_FILE.name}")


if __name__ == "__main__":
    main()
