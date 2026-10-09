# 📦 BuyLead Copilot

**An AI agent that turns messy B2B buyer inquiries into structured RFQs, matches them to the right suppliers with RAG, and drafts quote requests - with guardrails against hallucination and prompt injection.**

> "helmat 200 pcs urjent delhi" → *Industrial Safety Helmet · 200 pieces · Delhi · urgent* → 3 ranked NCR suppliers with reasons → 3 ready-to-send quote requests.

**▶ Live demo: [buylead-copilot.streamlit.app](https://buylead-copilot.streamlit.app)** (Gemini engine)

Built as a product + AI prototype for B2B marketplaces, where buyer inquiries are high-volume, short and often Hinglish. All supplier data is **synthetic** (generated with a fixed seed in `scripts/generate_catalog.py`).

![screenshot](docs/screenshot.png)

---

## 2-minute demo
1. **Try one inquiry** → pick `helmat 200 pcs urjent delhi`: typos and Hinglish become a clean RFQ, 3 ranked suppliers with reasons, and drafted quote requests.
2. Pick `1000 kg organic turmeric powder Erode`: nothing in the catalog fits, so it **routes to a person instead of guessing** (the rules baseline matches it to *powder-free gloves*).
3. Pick the `Ignore previous instructions...` example: the injected supplier and price never reach a draft.
4. **Batch triage** → run the queue: how many inquiries are handled without a person, and how many leads go out.
5. **Business impact** → change the assumptions and watch cost, lead-ops hours and bad leads move.

## Why this problem
Buyers on B2B marketplaces type short, messy, often Hinglish inquiries with missing quantity or location.
Matching them badly costs **two** customers at once: the buyer gets irrelevant quotes, and the supplier pays for a lead they can't serve. See the one-page **[PRD](docs/PRD.md)** for users, trade-offs, metrics and what I deliberately did *not* build.

## How it works

```
buyer text ──► 1. EXTRACT ──► 2. GROUND ──► 3. RETRIEVE ──► 4. VERIFY ──► 5. DECIDE ──► 6. DRAFT + GUARD
               LLM tool call  drop values    vector search   LLM yes/no    match /       RFQ per supplier,
               call (JSON     not in the     over catalog,   "is this      clarify /     blocked if it invents
               schema) or     buyer's own    rank suppliers  really what   refuse        a price or supplier
               rules          words          with reasons    they want?"
```

| Step | What | AI concepts used |
|---|---|---|
| Extract | System prompt + 3 few-shot examples + **forced tool call** with a JSON schema (Gemini: JSON mode with the same schema) → always-valid structured output | Structured prompting, few-shot, tool use |
| Ground | Every quantity / city / spec must appear in the buyer's text, otherwise it's dropped and logged | Hallucination detection |
| Retrieve | Product resolution by cosine similarity over **neural embeddings** (Gemini `gemini-embedding-2`) in LLM mode, or word + character n-gram TF-IDF vectors (typo-tolerant, offline) in the rules baseline; **refuses below a confidence threshold calibrated on the dev set** | Embeddings, vector search, RAG |
| Verify | A cheap second LLM call rejects false matches vector search can't (e.g. "turmeric *powder*" ≠ "*powder*-free gloves") | LLM-as-judge, re-ranking |
| Decide | `match` / `clarify` (asks one question when quantity is missing) / `refuse` (routes to a human) | Agent decision policy |
| Draft + guard | Drafts use only catalog facts; any rupee value or supplier not in the record → fallback template | Output guardrails, prompt-injection defence |

Every LLM step falls back to the rules engine if the API fails (and neural retrieval falls back to TF-IDF), and every run produces a **trace** with latency, tokens and guard events.

**Model-agnostic:** the agent runs on **Claude** or **Gemini** behind one interface (`buylead/llm.py`), so the same prompts, guardrails and eval harness compare providers like-for-like. Pick one with `BUYLEAD_PROVIDER=anthropic|gemini`, or just set one API key.

## Evaluation
Two hand-labelled sets: a **40-case dev set** and a **20-case held-out set** written after the baseline was built (not tuned on). Both cover clean English, Hinglish, typos, missing quantity, missing location, out-of-catalog products and prompt-injection attempts.

**Rules baseline vs. Gemini (`gemini-3.5-flash-lite`, TF-IDF retrieval; the neural-embedding rerun is next)**: measured results, reproducible with the commands below:

| Metric | Rules · Dev (40) | Gemini · Dev (40) | Rules · Held-out (20) | Gemini · Held-out (20) |
|---|---|---|---|---|
| Action accuracy (match / clarify / refuse) | 95.0% | **100.0%** | 85.0% | **100.0%** |
| Product resolution accuracy | 100.0% | 100.0% | 93.8% | **100.0%** |
| Quantity accuracy | 100.0% | 100.0% | 93.8% | **100.0%** |
| City accuracy | 100.0% | 100.0% | 93.8% | 93.8% |
| Top-3 supplier precision | 100.0% | 100.0% | 100.0% | 100.0% |
| False matches on out-of-catalog inquiries (↓) | 33.3% (2/6) | **0% (0/6)** | 50.0% (2/4) | **0% (0/4)** |
| Unsafe-output rate in drafts (↓) | 0% | 0% (90 drafts) | 0% | 0% (42 drafts) |
| LLM calls that failed and fell back to rules | - | 0 | - | 0 |
| Avg tokens per query (in / out) | 0 / 0 | 1,303 / 276 | 0 / 0 | 1,263 / 265 |
| Est. LLM cost per 1,000 inquiries (paid list price) | $0 | ≈ $1.08 | $0 | ≈ $1.04 |
| Latency p50 | ~2 ms | ~30 s\* | ~2 ms | ~30 s\* |

**What this shows:** the rules baseline looks great on the set it was built against and degrades on unseen phrasing - especially **false matches** ("laptops" → office chairs, "copper scrap" → copper wire, "turmeric powder" → powder-free gloves). With the LLM extractor + verifier, held-out action accuracy rises from **85% to 100%** and false matches fall from **50% to 0%**, while the output guards keep unsafe drafts at 0%. The trade-off is cost and speed: ~1,550 tokens and up to 5 LLM calls per inquiry instead of a 2 ms rules lookup - which is why the PRD keeps rules as the fallback.

\* Latency is dominated by the free-tier throttle (10 requests/min ≈ 6 s between calls, up to 5 calls per inquiry), not the model; unthrottled, the live demo answers a full inquiry (extract, verify, 3 drafts) in ~4 s (single run; not yet benchmarked). Caveat: the sets are small (60 cases, 10 out-of-catalog), so treat 100% / 0% as "no errors observed", not a guarantee. Full per-case tables: [`EVAL_REPORT_RULES.md`](docs/EVAL_REPORT_RULES.md), [`EVAL_REPORT_RULES_HELDOUT.md`](docs/EVAL_REPORT_RULES_HELDOUT.md), [`EVAL_REPORT_LLM.md`](docs/EVAL_REPORT_LLM.md), [`EVAL_REPORT_LLM_HELDOUT.md`](docs/EVAL_REPORT_LLM_HELDOUT.md).

**LLM mode** - run it yourself with an API key; the report adds the provider/model, tokens/query, cost/query, p95 latency and how many LLM calls failed and fell back to rules:
```bash
# Gemini (free tier key: https://aistudio.google.com/apikey)
export GEMINI_API_KEY=...               # PowerShell: $env:GEMINI_API_KEY = "..."
# or Claude:  export ANTHROPIC_API_KEY=...
# optional: BUYLEAD_PRICE_IN / BUYLEAD_PRICE_OUT = USD per 1M tokens, for cost/query
python -m buylead.eval --mode llm --report docs/EVAL_REPORT_LLM.md
python -m buylead.eval --mode llm --data heldout_set.jsonl --report docs/EVAL_REPORT_LLM_HELDOUT.md
```
With neural embeddings, calibrate the refusal threshold on the **dev set only** first (the held-out set is never used for tuning):
```bash
python -m buylead.tune                  # writes data/embedding_threshold.json
```
Requests are retried with backoff on rate limits. On a free-tier key, also set `BUYLEAD_RPM=10` to stay under the per-minute limit. Cost per query is computed from paid-tier list prices built into `buylead/llm.py` (override with `BUYLEAD_PRICE_IN` / `BUYLEAD_PRICE_OUT`).

**Demo cost guardrails** (the public demo runs on a paid key): inquiries are capped at 300 characters, each visitor gets 20 LLM runs, and an in-app ledger stops LLM calls after **$0.25/day** of estimated spend (≈ under $8/month), falling back to the free rules engine. The ledger resets on app restart, so the hard stop is a budget alert on the billing account.

## Business impact (illustrative)
There is no production traffic, so this is a **model, not an observed result**: engine rates (how often an inquiry goes to a person, or a wrong-product lead goes to suppliers, and LLM cost) are **measured on the held-out set**; the business inputs are **assumptions** you can change in the demo's *Business impact* tab or in `buylead/impact.py`.

Default assumptions: 100,000 inquiries/month, 40% need triage today, 10% of those match no catalog product, 2 min of lead-ops time per manual follow-up at ₹600/hour, each lead goes to 3 suppliers, ₹40 cost per irrelevant lead a supplier receives, ₹88/USD.

| Per month | Manual triage | Rules engine | LLM copilot |
|---|---|---|---|
| Handled without a person | 0 | 35,750 | 36,000 |
| Manual follow-ups | 40,000 | 4,250 | 4,000 |
| Lead-ops hours | 1,333 | 142 | 133 |
| Wrong-product leads sent to suppliers | 0 | 6,000 | 0 |
| LLM cost | ₹0 | ₹0 | ₹3,666 |
| Total monthly cost | ₹8.00 lakh | ₹3.25 lakh | ₹83,666 |
| Saving vs manual | ₹0 | ₹4.75 lakh | ₹7.16 lakh |

**Reading it:** the rules engine already removes most manual work but sends ~6,000 wrong-product leads to suppliers each month - the cost a marketplace pays in supplier trust. The LLM copilot automates about the same share with **no wrong leads observed**, for ~₹3,700/month of LLM spend. Manual triage is assumed error-free (flattering it), and "0 wrong leads" means none in 4 out-of-catalog held-out cases, not a guarantee. The **Batch triage** tab shows the same split on a queue you paste in.

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py                    # UI; works without an API key in "Rules baseline" mode
pytest -q                               # 15 offline tests, incl. fake "hallucinating" Claude and Gemini clients
```
```python
from buylead import BuyLeadAgent
r = BuyLeadAgent("llm").run("bhai 2000 corrugated box 5 ply chahiye Surat me")
r.action, r.requirement, r.suppliers, r.drafts, r.trace
```

## Repo map
```
buylead/llm.py        provider switch: Claude or Gemini behind one interface (throttling, retries)
buylead/extract.py    LLM + rules extractors, prompt, JSON schema, grounding guard
buylead/retrieve.py   vector index, confidence threshold, supplier ranking with reasons
buylead/draft.py      LLM verifier, RFQ drafters, output guard
buylead/agent.py      orchestration + decision policy (match / clarify / refuse)
buylead/eval.py       evaluation harness + markdown reports
data/                 synthetic catalog, dev + held-out eval sets
docs/PRD.md           one-page PRD: problem, trade-offs, metrics, risks
tests/                offline tests (fake Claude and Gemini clients)
```

## Limitations & next steps
- Catalog is synthetic and small (231 suppliers / 20 products); a real catalog would put the same `search()` interface on a vector DB.
- The match threshold was tuned on the dev set - hence the held-out set; re-tune whenever the catalog changes.
- Next: measure unthrottled latency and paid-tier cost per inquiry, run the same evals on Claude for a provider comparison, grow the eval set (especially out-of-catalog cases), then a shadow launch measuring "% inquiries with ≥1 relevant quote in 24h".

---
Built by [Aritra Pal](https://www.aritrapal.me) · Python, Claude API / Gemini API, scikit-learn, Streamlit · built with Claude Code
