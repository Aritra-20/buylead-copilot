"""Offline evaluation harness.

    python -m buylead.eval --mode rules          # free baseline, no API key
    python -m buylead.eval --mode llm            # needs ANTHROPIC_API_KEY
    python -m buylead.eval --mode llm --report docs/EVAL_REPORT_LLM.md

Metrics
  action accuracy     match / clarify / refuse decided correctly
  product accuracy    resolved catalog product == gold (in-catalog cases)
  field accuracy      quantity, unit, city, urgency vs gold
  top-3 precision     share of shortlisted suppliers that sell the right product
  local-in-top-3      when a local (same city/state/NCR) supplier exists, is one shortlisted?
  false-match rate    out-of-catalog inquiries that were wrongly matched
  unsafe-output rate  final drafts containing a price/supplier not in the catalog
                      (re-checked independently of the in-pipeline guard)
  guard catches       how often the guards intervened
  latency p50 / p95, tokens/query, cost/query (set BUYLEAD_PRICE_IN / BUYLEAD_PRICE_OUT
                      to your model's USD price per million tokens)
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .agent import BuyLeadAgent
from .catalog import CITY_STATE, NCR, load_suppliers
from .draft import check_draft
from .llm import model_price, provider_name

ROOT = Path(__file__).resolve().parents[1]


def _local_exists(product: str, city: str | None) -> bool:
    if not city:
        return False
    st = CITY_STATE.get(city)
    return any(s["product"] == product and (s["city"] == city or s["state"] == st or (city in NCR and s["city"] in NCR))
               for s in load_suppliers())


def _is_local(s: dict, city: str) -> bool:
    return s["city"] == city or s["state"] == CITY_STATE.get(city) or (city in NCR and s["city"] in NCR)


def _threshold(agent) -> float:
    from .agent import neural_index
    from .retrieve import get_index
    return neural_index().threshold if agent.embeddings == "gemini" else get_index().threshold


def run(mode: str, report: Path | None, dataset: str = "eval_set.jsonl") -> dict:
    agent = BuyLeadAgent(mode)
    cases = [json.loads(l) for l in (ROOT / "data" / dataset).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows, lat, tok_in, tok_out = [], [], [], []
    c = {k: [0, 0] for k in ["action", "product", "quantity", "unit", "city", "urgent", "top3", "local", "false_match", "unsafe"]}

    def score(key, ok):
        c[key][0] += int(bool(ok)); c[key][1] += 1

    guard_catches, llm_errors = 0, 0
    outcome = {"match": 0, "clarify": 0, "refuse": 0, "wrong_match": 0}
    seg_n, seg_refuse, seg_wrong = {"in": 0, "ooc": 0}, {"in": 0, "ooc": 0}, {"in": 0, "ooc": 0}
    for i, case in enumerate(cases, 1):
        g, r = case["gold"], agent.run(case["inquiry"])
        err = f"  LLM ERROR: {r.llm_errors[0]}" if r.llm_errors else ""
        print(f"[{i}/{len(cases)}] case {case['id']}: {r.action} ({r.total_ms / 1000:.1f}s){err}", flush=True)
        lat.append(r.total_ms); tok_in.append(r.tokens_in); tok_out.append(r.tokens_out)
        guard_catches += len(r.guard_events)
        llm_errors += len(r.llm_errors)
        score("action", r.action == g["action"])
        outcome[r.action] += 1
        if r.action == "match" and (g["action"] == "refuse" or r.product != g.get("product")):
            outcome["wrong_match"] += 1   # a bad lead would reach suppliers
        seg = "ooc" if g["action"] == "refuse" else "in"
        seg_n[seg] += 1
        if r.action == "refuse":
            seg_refuse[seg] += 1          # routed to a human
        elif r.action == "match" and (seg == "ooc" or r.product != g.get("product")):
            seg_wrong[seg] += 1           # wrong product -> bad lead to suppliers
        if g["action"] == "refuse":
            score("false_match", r.action != "refuse")
        else:
            score("product", r.product == g["product"])
            score("quantity", r.requirement.quantity == g["quantity"] if g["quantity"] is None
                  else r.requirement.quantity is not None and abs(r.requirement.quantity - g["quantity"]) < 1e-6)
            if g["unit"]:
                score("unit", r.requirement.unit == g["unit"])
            score("city", r.requirement.city == g["city"])
            score("urgent", r.requirement.urgent == g["urgent"])
            if r.suppliers:
                for s in r.suppliers:
                    score("top3", s["product"] == g["product"])
                if g["city"] and _local_exists(g["product"], g["city"]):
                    score("local", any(_is_local(s, g["city"]) for s in r.suppliers))
        for d in r.drafts:
            s = next(x for x in r.suppliers if x["supplier_id"] == d["supplier_id"])
            bad = check_draft(d["text"], s, r.requirement, r.suppliers) or ("S9999" in d["text"])
            score("unsafe", bad)
        rows.append((case, r))

    pct = lambda k: 100 * c[k][0] / c[k][1] if c[k][1] else float("nan")  # noqa: E731
    model = agent.extractor.model if mode == "llm" else "none"
    price = model_price(model) if mode == "llm" else None
    avg_in, avg_out = statistics.mean(tok_in), statistics.mean(tok_out)
    cost = (avg_in * price[0] + avg_out * price[1]) / 1e6 if price else None
    lat_sorted = sorted(lat)
    summary = {
        "mode": mode, "cases": len(cases),
        "provider": (provider_name() or "anthropic") if mode == "llm" else "none",
        "model": model,
        "embeddings": agent.embeddings,
        "match_threshold": round(_threshold(agent), 4),
        "llm_errors": llm_errors,
        "action_accuracy": pct("action"), "product_accuracy": pct("product"),
        "quantity_accuracy": pct("quantity"), "unit_accuracy": pct("unit"), "city_accuracy": pct("city"),
        "urgency_accuracy": pct("urgent"), "top3_precision": pct("top3"), "local_in_top3": pct("local"),
        "false_match_rate": pct("false_match"), "unsafe_output_rate": pct("unsafe"),
        "drafts_checked": c["unsafe"][1], "guard_catches": guard_catches,
        "latency_p50_ms": statistics.median(lat), "latency_p95_ms": lat_sorted[int(0.95 * (len(lat) - 1))],
        "avg_tokens_in": avg_in, "avg_tokens_out": avg_out, "cost_per_query_usd": cost,
        # Outcome mix over ALL cases - feeds the business-impact model (buylead/impact.py)
        "share_match": outcome["match"] / len(cases), "share_clarify": outcome["clarify"] / len(cases),
        "share_refuse": outcome["refuse"] / len(cases), "wrong_match_rate": outcome["wrong_match"] / len(cases),
        # Per segment: in-catalog (gold match/clarify) vs out-of-catalog / not-a-product (gold refuse)
        "in_catalog_cases": seg_n["in"], "out_of_catalog_cases": seg_n["ooc"],
        "in_catalog_refuse_rate": seg_refuse["in"] / max(seg_n["in"], 1),
        "in_catalog_wrong_match_rate": seg_wrong["in"] / max(seg_n["in"], 1),
        "out_of_catalog_refuse_rate": seg_refuse["ooc"] / max(seg_n["ooc"], 1),
        "out_of_catalog_false_match_rate": seg_wrong["ooc"] / max(seg_n["ooc"], 1),
    }
    if report:
        report.write_text(_markdown(summary, rows, dataset), encoding="utf-8")
        report.with_suffix(".json").write_text(json.dumps({**summary, "dataset": dataset}, indent=2) + "\n",
                                               encoding="utf-8")
    return summary


def _markdown(s: dict, rows, dataset: str) -> str:
    f = lambda v: "n/a" if v is None or v != v else f"{v:.1f}%"  # noqa: E731
    cost = ("n/a (no LLM calls)" if s["mode"] != "llm" else "set BUYLEAD_PRICE_IN/OUT") if s["cost_per_query_usd"] is None \
        else f"${s['cost_per_query_usd']:.5f} (≈ ${1000 * s['cost_per_query_usd']:.2f} per 1,000 inquiries, paid-tier list price)"
    out = [f"# Evaluation report - `{s['mode']}` mode", "",
           f"Generated by `python -m buylead.eval --mode {s['mode']} --data {dataset}` on {s['cases']} hand-labelled inquiries "
           "(clean English, Hinglish, typos, missing quantity, missing location, out-of-catalog, prompt injection).", "",
           (f"Provider / model: **{s['provider']} / {s['model']}** · retrieval: **{s['embeddings']}** embeddings, "
            f"match threshold {s['match_threshold']}") if s["mode"] == "llm"
           else f"No LLM calls (rules baseline) · retrieval: **{s['embeddings']}**, match threshold {s['match_threshold']}.", "",
           "| Metric | Value |", "|---|---|",
           f"| Action accuracy (match / clarify / refuse) | {f(s['action_accuracy'])} |",
           f"| Product resolution accuracy | {f(s['product_accuracy'])} |",
           f"| Quantity accuracy | {f(s['quantity_accuracy'])} |",
           f"| Unit accuracy | {f(s['unit_accuracy'])} |",
           f"| City accuracy | {f(s['city_accuracy'])} |",
           f"| Urgency accuracy | {f(s['urgency_accuracy'])} |",
           f"| Top-3 supplier precision | {f(s['top3_precision'])} |",
           f"| Local supplier in top-3 (when one exists) | {f(s['local_in_top3'])} |",
           f"| False-match rate on out-of-catalog inquiries (lower is better) | {f(s['false_match_rate'])} |",
           f"| Unsafe-output rate in final drafts ({s['drafts_checked']} drafts, lower is better) | {f(s['unsafe_output_rate'])} |",
           f"| Guard interventions | {s['guard_catches']} |",
           f"| LLM calls that failed and fell back to rules (lower is better) | {s['llm_errors']} |",
           f"| Latency p50 / p95 | {s['latency_p50_ms']:.1f} ms / {s['latency_p95_ms']:.1f} ms |",
           f"| Avg tokens per query (in / out) | {s['avg_tokens_in']:.0f} / {s['avg_tokens_out']:.0f} |",
           f"| Cost per query | {cost} |", "",
           "## Per-case results", "",
           "| # | Type | Inquiry | Gold | Got | Product | Qty | City | Notes |", "|---|---|---|---|---|---|---|---|---|"]
    for case, r in rows:
        g = case["gold"]
        mark = "OK" if r.action == g["action"] and (g["action"] == "refuse" or r.product == g["product"]) else "**MISS**"
        notes = "; ".join(r.guard_events) or (r.refusal_reason or "")[:60]
        q = "-" if r.requirement.quantity is None else f"{r.requirement.quantity:g} {r.requirement.unit or ''}"
        out.append(f"| {case['id']} | {case['type']} | {case['inquiry'].replace('|', '/')} | {g['action']} | "
                   f"{r.action} {mark} | {r.product or '-'} | {q} | {r.requirement.city or '-'} | {notes.replace('|', '/')} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rules", "llm"], default="rules")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--data", default="eval_set.jsonl", help="eval_set.jsonl (dev) or heldout_set.jsonl (held-out)")
    a = ap.parse_args()
    s = run(a.mode, a.report, a.data)
    for k, v in s.items():
        print(f"{k:22s} {v:.2f}" if isinstance(v, float) else f"{k:22s} {v}")


if __name__ == "__main__":
    main()
