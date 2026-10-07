# 📦 BuyLead Copilot

**An AI agent that turns messy B2B buyer inquiries into structured RFQs, matches them to the right suppliers with RAG, and drafts quote requests - with guardrails against hallucination and prompt injection.**

> "helmat 200 pcs urjent delhi" → *Industrial Safety Helmet · 200 pieces · Delhi · urgent* → 3 ranked NCR suppliers with reasons → 3 ready-to-send quote requests.

Built as a product + AI prototype for B2B marketplaces such as IndiaMART. All supplier data is **synthetic** (generated with a fixed seed in `scripts/generate_catalog.py`).

![screenshot](docs/screenshot.png)

---

## Why this problem
Buyers on B2B marketplaces type short, messy, often Hinglish inquiries with missing quantity or location.
Matching them badly costs **two** customers at once: the buyer gets irrelevant quotes, and the supplier pays for a lead they can't serve. See the one-page **[PRD](docs/PRD.md)** for users, trade-offs, metrics and what I deliberately did *not* build.

## How it works

```
buyer text ──► 1. EXTRACT ──► 2. GROUND ──► 3. RETRIEVE ──► 4. VERIFY ──► 5. DECIDE ──► 6. DRAFT + GUARD
               Claude tool    drop values    vector search   LLM yes/no    match /       RFQ per supplier,
               call (JSON     not in the     over catalog,   "is this      clarify /     blocked if it invents
               schema) or     buyer's own    rank suppliers  really what   refuse        a price or supplier
               rules          words          with reasons    they want?"
```

| Step | What | AI concepts used |
|---|---|---|
| Extract | System prompt + 3 few-shot examples + **forced tool call** with a JSON schema → always-valid structured output | Structured prompting, few-shot, tool use |
| Ground | Every quantity / city / spec must appear in the buyer's text, otherwise it's dropped and logged | Hallucination detection |
| Retrieve | Product resolution by cosine similarity over word + character n-gram vectors (typo-tolerant); **refuses below a confidence threshold** | Embeddings, vector search, RAG |
| Verify | A cheap second LLM call rejects false matches vector search can't (e.g. "turmeric *powder*" ≠ "*powder*-free gloves") | LLM-as-judge, re-ranking |
| Decide | `match` / `clarify` (asks one question when quantity is missing) / `refuse` (routes to a human) | Agent decision policy |
| Draft + guard | Drafts use only catalog facts; any rupee value or supplier not in the record → fallback template | Output guardrails, prompt-injection defence |

Every LLM step falls back to the rules engine if the API fails, and every run produces a **trace** with latency, tokens and guard events.

## Evaluation
Two hand-labelled sets: a **40-case dev set** and a **20-case held-out set** written after the baseline was built (not tuned on). Both cover clean English, Hinglish, typos, missing quantity, missing location, out-of-catalog products and prompt-injection attempts.

**Rules baseline (no LLM)**: these are measured results, reproducible with the commands below:

| Metric | Dev (40) | Held-out (20) |
|---|---|---|
| Action accuracy (match / clarify / refuse) | 95.0% | 85.0% |
| Product resolution accuracy | 100.0% | 93.8% |
| Quantity accuracy | 100.0% | 93.8% |
| Top-3 supplier precision | 100.0% | 100.0% |
| False-match rate on out-of-catalog (↓) | 33.3% | 50.0% |
| Unsafe-output rate in drafts (↓) | 0.0% | 0.0% |
| Latency p50 | ~2 ms | ~2 ms |

**What this shows:** the baseline looks great on the set it was built against and degrades on unseen phrasing - especially **false matches** ("laptops" → office chairs, "copper scrap" → copper wire, "turmeric powder" → powder-free gloves). That gap is exactly what the LLM extractor + verifier are for. Full per-case tables: [`docs/EVAL_REPORT_RULES.md`](docs/EVAL_REPORT_RULES.md), [`docs/EVAL_REPORT_RULES_HELDOUT.md`](docs/EVAL_REPORT_RULES_HELDOUT.md).

**LLM mode** - run it yourself with an API key; the report adds tokens/query, cost/query and p95 latency:
```bash
export ANTHROPIC_API_KEY=...            # BUYLEAD_PRICE_IN / BUYLEAD_PRICE_OUT = USD per 1M tokens, for cost
python -m buylead.eval --mode llm --report docs/EVAL_REPORT_LLM.md
python -m buylead.eval --mode llm --data heldout_set.jsonl --report docs/EVAL_REPORT_LLM_HELDOUT.md
```

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py                    # UI; works without an API key in "Rules baseline" mode
pytest -q                               # 11 offline tests, incl. a fake "hallucinating" LLM
```
```python
from buylead import BuyLeadAgent
r = BuyLeadAgent("llm").run("bhai 2000 corrugated box 5 ply chahiye Surat me")
r.action, r.requirement, r.suppliers, r.drafts, r.trace
```

## Repo map
```
buylead/extract.py    LLM + rules extractors, prompt, JSON schema, grounding guard
buylead/retrieve.py   vector index, confidence threshold, supplier ranking with reasons
buylead/draft.py      LLM verifier, RFQ drafters, output guard
buylead/agent.py      orchestration + decision policy (match / clarify / refuse)
buylead/eval.py       evaluation harness + markdown reports
data/                 synthetic catalog, dev + held-out eval sets
docs/PRD.md           one-page PRD: problem, trade-offs, metrics, risks
tests/                offline tests (fake Anthropic client)
```

## Limitations & next steps
- Catalog is synthetic and small (231 suppliers / 20 products); real catalogs need neural embeddings + a vector DB (the `EmbeddingIndex` is built to be swapped).
- The match threshold was tuned on the dev set - hence the held-out set; re-tune whenever the catalog changes.
- Next: LLM-vs-baseline comparison on cost and p95 latency, then a shadow launch measuring "% inquiries with ≥1 relevant quote in 24h".

---
Built by [Aritra Pal](https://www.aritrapal.me) · Python, Claude API, scikit-learn, Streamlit · built with Claude Code
