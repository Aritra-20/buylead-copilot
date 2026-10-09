"""The BuyLead Copilot agent - orchestrates extract -> ground -> retrieve -> verify -> decide -> draft.

Decision policy (the product trade-offs, see docs/PRD.md):
  * REFUSE  when the inquiry is out of scope or nothing in the catalog is a confident match.
            A wrong supplier wastes the buyer's time AND sends the supplier a junk lead.
  * CLARIFY when the product is clear but the quantity is missing. Quantity drives price and
            MOQ fit, so we ask one question instead of guessing - and still preview suppliers.
  * MATCH   otherwise. A missing city is not blocking: we fall back to nationwide suppliers.
"""
from __future__ import annotations

import time

from .catalog import products
from .draft import LLMDrafter, LLMVerifier, check_draft, template_draft
from .extract import LLMExtractor, RuleExtractor, ground_requirement
from .llm import GeminiEmbedder, describe_error, embeddings_backend
from .retrieve import NeuralIndex, retrieve
from .schema import Result, TraceStep


_NEURAL: dict[str, NeuralIndex] = {}   # product vectors are embedded once per process


def neural_index(embedder=None) -> NeuralIndex:
    embedder = embedder or GeminiEmbedder()
    key = getattr(embedder, "model", "custom")
    if key not in _NEURAL:
        _NEURAL[key] = NeuralIndex(embedder)
    return _NEURAL[key]


class BuyLeadAgent:
    def __init__(self, mode: str = "rules", client=None, model: str | None = None, embedder=None,
                 embeddings: str | None = None):
        if mode not in {"rules", "llm"}:
            raise ValueError("mode must be 'rules' or 'llm'")
        self.mode = mode
        self.rules = RuleExtractor()
        if mode == "llm":
            self.extractor = LLMExtractor(client, model)
            self.verifier = LLMVerifier(self.extractor.client, self.extractor.model)
            self.drafter = LLMDrafter(self.extractor.client, self.extractor.model)
        else:
            self.extractor, self.verifier, self.drafter = self.rules, None, None
        # Retrieval backend: neural (Gemini embeddings) in LLM mode, TF-IDF otherwise / as fallback.
        self.embeddings = embeddings or ("gemini" if embedder is not None else embeddings_backend(mode))
        self._embedder = embedder

    def run(self, inquiry: str) -> Result:
        trace: list[TraceStep] = []

        # 1. Extract (LLM falls back to rules on any API error, so the product never hard-fails)
        try:
            req, step = self.extractor.extract(inquiry)
        except Exception as e:  # noqa: BLE001
            req, step = self.rules.extract(inquiry)
            step.notes.append(f"LLM extraction failed ({describe_error(e)}); used rules fallback")
        req = ground_requirement(req, inquiry, step)
        trace.append(step)
        res = Result(inquiry=inquiry, mode=self.mode, requirement=req, action="match", trace=trace)

        if not req.in_scope:
            res.action, res.refusal_reason = "refuse", "This looks like a service/job request, not a product purchase."
            return res

        # 2. Retrieve (neural embeddings fall back to TF-IDF on any API error, and the failure is logged)
        fallback_note = None
        if self.embeddings == "gemini":
            try:
                product, sim, _, suppliers, rstep = retrieve(req, neural_index(self._embedder))
            except Exception as e:  # noqa: BLE001
                fallback_note = f"neural embedding failed ({describe_error(e)}); used TF-IDF fallback"
                product, sim, _, suppliers, rstep = retrieve(req)
        else:
            product, sim, _, suppliers, rstep = retrieve(req)
        if fallback_note:
            rstep.notes.append(fallback_note)
        trace.append(rstep)
        res.retrieval_score = round(sim, 3)

        # 3. Verify the match (LLM mode only)
        if product and self.verifier is not None:
            try:
                ok, vstep = self.verifier.verify(inquiry, product, products()[product]["category"])
                trace.append(vstep)
                if not ok:
                    vstep.notes.append(f"GUARD: verifier rejected match '{product}'")
                    product, suppliers = None, []
            except Exception as e:  # noqa: BLE001
                trace.append(TraceStep("verify:llm", 0, notes=[f"verifier failed ({describe_error(e)}); kept match"]))

        if product is None:
            res.action = "refuse"
            res.refusal_reason = ("We don't have a confident match for this product in the supplier catalog. "
                                  "Routing to a human category specialist instead of guessing.")
            return res

        res.product, res.suppliers = product, suppliers
        if req.quantity is not None and req.unit is None:
            req.unit = products()[product]["unit"]

        # 4. Decide
        if req.quantity is None:
            res.action = "clarify"
            unit = products()[product]["unit"]
            loc = "" if req.city else " and which city should it be delivered to"
            res.clarifying_question = f"How many {unit}s of {product.lower()} do you need{loc}?"
            return res

        # 5. Draft + output guard
        t0 = time.perf_counter()
        dstep = TraceStep(f"draft:{self.mode}", 0)
        for s in suppliers:
            text, source = None, "template"
            if self.drafter is not None:
                try:
                    text, ti, to = self.drafter.draft(s, req, product)
                    dstep.tokens_in += ti
                    dstep.tokens_out += to
                    source = "llm"
                except Exception as e:  # noqa: BLE001
                    dstep.notes.append(f"LLM draft failed ({describe_error(e)}); used template")
            if text is None:
                text = template_draft(s, req, product)
            problems = check_draft(text, s, req, suppliers)
            if problems:
                dstep.notes.append(f"GUARD: draft to {s['supplier_id']} blocked ({'; '.join(problems)}) -> template")
                text, source = template_draft(s, req, product), "template"
            res.drafts.append({"supplier_id": s["supplier_id"], "supplier_name": s["supplier_name"],
                               "text": text, "source": source})
        dstep.ms = (time.perf_counter() - t0) * 1000
        trace.append(dstep)
        return res
