# PRD - BuyLead Copilot (one-pager)

**Author:** Aritra Pal · **Status:** v1 prototype shipped · **Data:** synthetic supplier catalog (231 suppliers, 20 products)

## 1. Problem
On a B2B marketplace, a large share of buyer inquiries arrive as short, messy text - Hinglish, typos, no
quantity, no location ("helmat 200 pcs urjent delhi", "office chair ka best price batao").
Today these either need a human to call the buyer back, or get blasted to loosely-matched suppliers.
Both are expensive:
- **Buyer:** slow or irrelevant quotes → drops off and goes to WhatsApp / a local trader.
- **Supplier:** paid-for leads that don't fit their product or MOQ → churn from paid plans.
- **Marketplace:** tele-calling cost per lead, and lower lead-to-quote conversion.

## 2. Users & jobs to be done
| User | Job |
|---|---|
| Buyer (SME owner, procurement exec) | "Get me 3 credible quotes for exactly what I need, fast, without filling a long form." |
| Supplier | "Only send me leads I can actually serve (my product, my MOQ, my region)." |
| Internal lead-ops team | "Tell me which leads need a human, and why." |

## 3. What v1 does
1. **Extract** - turn the inquiry into a structured requirement (product, specs, quantity, unit, city, urgency, language).
2. **Ground** - drop any extracted value that isn't in the buyer's own words (anti-hallucination).
3. **Retrieve** - vector search over the catalog to resolve the product; rank suppliers by spec fit, location, rating, GST verification, response time and MOQ fit, with reasons.
4. **Verify** - a second LLM check that the matched product is really what the buyer wants.
5. **Decide** - `match`, `clarify` (ask one question) or `refuse` (route to a human).
6. **Draft** - RFQ message per supplier; an output guard blocks invented prices/suppliers.

## 4. Key product decisions & trade-offs
| Decision | Why | What we gave up |
|---|---|---|
| **Refuse below a confidence threshold** instead of always returning results | A wrong match costs two users (buyer + supplier) and erodes trust in leads | Some recall: genuine but oddly-phrased inquiries go to a human |
| **Ask for quantity, don't ask for city** | Quantity changes price and MOQ fit; city only changes ranking, so we fall back to nationwide | Slightly less local results when city is missing |
| **One clarifying question max** | Every extra question is a drop-off point | Some leads still incomplete |
| **Drafts use only catalog facts; guard falls back to a template** | Safety > fluency - a fake price in a supplier message is a trust incident | LLM drafts are sometimes replaced by plainer templates |
| **Rules engine kept as fallback** | The product must work if the LLM API is down or slow | Rules are weaker on unseen phrasing (see held-out eval) |
| **Small, cheap model by default** | This is a high-volume, per-lead workflow; unit cost matters | Some accuracy vs. a frontier model - measure before upgrading |

## 5. What we chose NOT to build (v1)
- Price negotiation / auto-accepting quotes - high risk, needs supplier consent.
- Voice / image inquiries - real demand, but text covers the core flow first.
- Supplier-side auto-replies - separate product with separate trust questions.
- Real-time inventory - catalog prices are "listed ranges", clearly labelled as such.

## 6. Success metrics
- **North Star:** % of inquiries that receive ≥ 1 relevant quote within 24 h.
- **Input metrics:** action accuracy, false-match rate (out-of-catalog), top-3 supplier precision,
  local-supplier-in-top-3, clarification rate, unsafe-output rate (target 0), cost and p95 latency per lead.
- **Guardrail:** supplier lead-rejection rate must not rise.

## 7. Evaluation (see `docs/EVAL_REPORT_*.md`)
40-case dev set + 20-case held-out set covering clean English, Hinglish, typos, missing fields,
out-of-catalog products and prompt-injection attempts. The rules baseline does well on the dev set it was
built against but drops on the held-out set (false matches on "laptops" → chairs, "copper scrap" → copper
wire). That gap is the case for the LLM extractor + verifier, which must beat the baseline on the
held-out set before rollout.

## 8. Risks
| Risk | Mitigation |
|---|---|
| Hallucinated quantity/city/spec | `ground_requirement` guard, unit-tested with a fake "hallucinating" model |
| Prompt injection in buyer text | Untrusted-input instructions in every prompt + output guard on drafts |
| False matches (vector search matches on shared words like "powder", "steel") | Confidence threshold + LLM verifier |
| LLM outage / latency spike | Automatic rules fallback per step |
| Threshold over-fit to eval set | Separate held-out set; re-tune when catalog changes |

## 9. Next steps
1. Run LLM mode on both eval sets; compare to baseline on accuracy, cost and p95 latency.
2. Replace TF-IDF vectors with neural embeddings + a vector DB; re-tune the threshold.
3. Shadow-launch on a slice of real inquiries; measure quote-within-24h vs. control.
