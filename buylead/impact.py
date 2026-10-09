"""Business-impact model: measured eval rates x editable business assumptions.

Nothing here is observed in production - there is no production. The *rates* (how often each engine
routes an inquiry to a human, or sends a wrong-product lead to suppliers, and the LLM cost per
inquiry) are MEASURED on the held-out eval set. The *business inputs* (volume, cost of a human
follow-up, cost of a bad lead, ...) are ASSUMPTIONS, labelled as such and editable in the demo.

Three ways to handle the inquiries that need triage today (messy / incomplete / unclear):
  manual  - a person reads and follows up on every one (assumed error-free, which flatters manual)
  rules   - the regex + TF-IDF engine decides; refusals go to a person
  copilot - the LLM agent decides; refusals go to a person

    python -m buylead.impact          # prints the default scenario
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


@dataclass
class Assumptions:
    monthly_inquiries: int = 100_000      # buyer inquiries per month in the slice being automated
    triage_share: float = 0.40            # share that are messy / incomplete and need triage today
    out_of_catalog_share: float = 0.10    # share of those that are not a catalog product (or not a purchase)
    human_minutes: float = 2.0            # minutes of a lead-ops person per manual follow-up
    human_cost_per_hour_inr: float = 600  # loaded hourly cost of that person
    suppliers_per_lead: int = 3           # suppliers each matched inquiry is sent to
    bad_lead_cost_inr: float = 40         # cost per supplier who receives an irrelevant lead (credit/refund + churn risk)
    usd_inr: float = 88                   # exchange rate for LLM cost


ASSUMPTION_NOTES = {
    "monthly_inquiries": "Buyer inquiries per month in scope",
    "triage_share": "Share that are messy / incomplete and need triage today",
    "out_of_catalog_share": "Share of those that match no catalog product",
    "human_minutes": "Minutes of lead-ops time per manual follow-up",
    "human_cost_per_hour_inr": "Loaded hourly cost of lead-ops (₹)",
    "suppliers_per_lead": "Suppliers each matched inquiry is sent to",
    "bad_lead_cost_inr": "Cost per supplier receiving an irrelevant lead (₹)",
    "usd_inr": "USD → INR rate",
}


def _load(name: str) -> dict | None:
    p = DOCS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def measured_rates() -> dict[str, dict]:
    """Held-out rates per engine, from the eval JSON written next to each report."""
    out = {}
    for engine, name in [("rules", "EVAL_REPORT_RULES_HELDOUT"), ("copilot", "EVAL_REPORT_LLM_HELDOUT")]:
        j = _load(name)
        if j:
            out[engine] = {k: j[k] for k in ["in_catalog_refuse_rate", "in_catalog_wrong_match_rate",
                                             "out_of_catalog_refuse_rate", "out_of_catalog_false_match_rate"]}
            out[engine]["cost_per_query_usd"] = j.get("cost_per_query_usd") or 0.0
            out[engine]["source"] = f"docs/{name}.md · {j.get('cases')} held-out cases" + (
                f" · {j.get('model')} · {j.get('embeddings')} retrieval" if engine == "copilot" else "")
    return out


def scenario(rates: dict | None, a: Assumptions) -> dict:
    """Monthly outcome for one way of handling the triage pool."""
    pool = a.monthly_inquiries * a.triage_share
    in_cat, ooc = pool * (1 - a.out_of_catalog_share), pool * a.out_of_catalog_share
    if rates is None:   # manual: a person touches every inquiry
        human, bad, ai_usd = pool, 0.0, 0.0
    else:
        human = in_cat * rates["in_catalog_refuse_rate"] + ooc * rates["out_of_catalog_refuse_rate"]
        bad = in_cat * rates["in_catalog_wrong_match_rate"] + ooc * rates["out_of_catalog_false_match_rate"]
        ai_usd = pool * rates["cost_per_query_usd"]
    human_hours = human * a.human_minutes / 60
    human_inr = human_hours * a.human_cost_per_hour_inr
    bad_leads = bad * a.suppliers_per_lead
    bad_inr = bad_leads * a.bad_lead_cost_inr
    ai_inr = ai_usd * a.usd_inr
    return {"inquiries_in_pool": pool, "auto_handled": pool - human, "human_followups": human,
            "human_hours": human_hours, "bad_leads_to_suppliers": bad_leads,
            "human_cost_inr": human_inr, "bad_lead_cost_inr": bad_inr, "llm_cost_inr": ai_inr,
            "total_cost_inr": human_inr + bad_inr + ai_inr}


def compare(a: Assumptions | None = None) -> dict:
    a = a or Assumptions()
    rates = measured_rates()
    res = {"manual": scenario(None, a)}
    for engine in ["rules", "copilot"]:
        if engine in rates:
            res[engine] = scenario(rates[engine], a)
    base = res["manual"]["total_cost_inr"]
    for v in res.values():
        v["saving_vs_manual_inr"] = base - v["total_cost_inr"]
    return {"assumptions": asdict(a), "rates": rates, "scenarios": res}


def _fmt_inr(x: float) -> str:
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:,.2f} lakh"
    return f"₹{x:,.0f}"


def markdown_table(result: dict) -> str:
    s = result["scenarios"]
    cols = [c for c in ["manual", "rules", "copilot"] if c in s]
    label = {"manual": "Manual triage", "rules": "Rules engine", "copilot": "LLM copilot"}
    rows = [("Handled without a person", lambda v: f"{v['auto_handled']:,.0f}"),
            ("Manual follow-ups", lambda v: f"{v['human_followups']:,.0f}"),
            ("Lead-ops hours", lambda v: f"{v['human_hours']:,.0f}"),
            ("Wrong-product leads sent to suppliers", lambda v: f"{v['bad_leads_to_suppliers']:,.0f}"),
            ("LLM cost", lambda v: _fmt_inr(v["llm_cost_inr"])),
            ("Total monthly cost", lambda v: _fmt_inr(v["total_cost_inr"])),
            ("Saving vs manual", lambda v: _fmt_inr(v["saving_vs_manual_inr"]))]
    out = ["| Per month | " + " | ".join(label[c] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    out += [f"| {name} | " + " | ".join(f(s[c]) for c in cols) + " |" for name, f in rows]
    return "\n".join(out)


if __name__ == "__main__":
    r = compare()
    print(markdown_table(r))
    print("\nAssumptions:", r["assumptions"])
    for k, v in r["rates"].items():
        print(f"{k}: {v['source']}")
