"""Step 2 - RAG over the supplier catalog.

Retrieval works in two stages:
  1. Product resolution - vector search (TF-IDF word + character n-gram embeddings,
     cosine similarity) over one document per catalog product. Character n-grams make
     it robust to typos ("helmat", "wier") without any external embedding API.
     If the best similarity is below a threshold the agent REFUSES instead of forcing
     a wrong match - the most common way marketplace search loses buyer trust.
  2. Supplier ranking - suppliers of the resolved product are scored on spec match,
     location, rating, GST verification, response time and MOQ feasibility, and every
     score comes with human-readable reasons.

Two interchangeable indexes: `EmbeddingIndex` (TF-IDF, offline baseline) and `NeuralIndex`
(Gemini neural embeddings, used in LLM mode). A real catalog would add a vector DB behind the same
`search()` interface.
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .catalog import CITY_STATE, DATA_DIR, NCR, load_suppliers, products
from .schema import Requirement, TraceStep

# Below this cosine similarity we refuse rather than guess. Tuned on the eval set
# (see docs/EVAL_REPORT.md) - re-tune when the catalog or embedding model changes.
MATCH_THRESHOLD = 0.20
THRESHOLD_FILE = DATA_DIR / "embedding_threshold.json"
DEFAULT_NEURAL_THRESHOLD = 0.50   # placeholder until calibrated with `python -m buylead.tune`

EXTRA_SYNONYMS = {
    "Corrugated Box": "carton cartons boxes",
    "Basmati Rice": "chawal sella golden steam 1121 1509",
    "Cotton Fabric": "kapda poplin cambric",
    "Mono PERC Solar Panel": "solar plate panels watt",
    "Industrial Safety Helmet": "helmets isi construction",
    "Copper House Wire": "wires frls fr coil",
    "Ergonomic Office Chair": "chairs headrest mesh",
    "LED Flood Light": "lights flood",
    "Stainless Steel Hex Bolt": "bolts ss",
    "MS Hex Nut": "nuts",
    "Self Drilling Screw": "screws",
    "BOPP Packing Tape": "tapes",
    "Nitrile Gloves": "glove powder free",
    "Safety Shoes": "shoe",
}


def _product_texts() -> tuple[list[str], list[str], list[str]]:
    """Per catalog product: name, a keyword document (for TF-IDF) and a readable description (for neural)."""
    prods = products()
    specs_by_product: dict[str, set] = {}
    for s in load_suppliers():
        specs_by_product.setdefault(s["product"], set()).add(s["spec"])
    names = list(prods)
    kw_docs, prose_docs = [], []
    for n in names:
        specs = " ".join(sorted(specs_by_product[n]))
        kw_docs.append(" ".join([n, prods[n]["category"], prods[n]["keywords"], EXTRA_SYNONYMS.get(n, ""), specs]).lower())
        prose_docs.append(f"{n}. Category: {prods[n]['category']}. Also called: {prods[n]['keywords']} "
                          f"{EXTRA_SYNONYMS.get(n, '')}. Typical specs: {specs}.")
    return names, kw_docs, prose_docs


class EmbeddingIndex:
    """Product-level vector index over TF-IDF word + character n-gram vectors (offline, no API).

    Used by the rules baseline, and as the fallback if neural embeddings are unavailable.
    `NeuralIndex` below is the drop-in neural replacement - the rest of the pipeline does not change.
    """

    kind = "tfidf"

    def __init__(self):
        self.names, docs, _ = _product_texts()
        self.threshold = MATCH_THRESHOLD
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(docs)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(docs)
        self.matrix = self._embed(docs)

    def _embed(self, texts: list[str]):
        return normalize(hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr())

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        sims = (self._embed([query.lower()]) @ self.matrix.T).toarray()[0]
        order = np.argsort(-sims)[:k]
        return [(self.names[i], float(sims[i])) for i in order]


def neural_threshold() -> float:
    """Refusal threshold for neural similarity: BUYLEAD_EMBED_THRESHOLD, else the value calibrated on the
    dev set by `python -m buylead.tune` (data/embedding_threshold.json), else a conservative default."""
    if os.getenv("BUYLEAD_EMBED_THRESHOLD"):
        return float(os.environ["BUYLEAD_EMBED_THRESHOLD"])
    if THRESHOLD_FILE.exists():
        return float(json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))["threshold"])
    return DEFAULT_NEURAL_THRESHOLD


class NeuralIndex:
    """Product-level index over neural embeddings (Gemini `gemini-embedding-2`).

    Catches meaning that shared characters miss ("hard hats" -> safety helmet, "solar module" -> panel)
    and avoids shared-word false matches ("turmeric powder" vs "powder-free gloves").
    Product vectors are computed once per process; each inquiry costs one query embedding.
    """

    kind = "gemini"

    def __init__(self, embedder, threshold: float | None = None):
        self.embedder = embedder
        self.names, _, prose = _product_texts()
        self.matrix = embedder.embed_documents(self.names, prose)
        self.threshold = neural_threshold() if threshold is None else threshold

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        sims = self.matrix @ self.embedder.embed_query(query)
        order = np.argsort(-sims)[:k]
        return [(self.names[i], float(sims[i])) for i in order]


_INDEX: EmbeddingIndex | None = None


def get_index() -> EmbeddingIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = EmbeddingIndex()
    return _INDEX


def _clean(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def rank_suppliers(product: str, req: Requirement, k: int = 3) -> list[dict]:
    buyer_state = CITY_STATE.get(req.city or "")
    ranked = []
    for s in load_suppliers():
        if s["product"] != product:
            continue
        score, reasons = 0.0, []
        spec_clean = _clean(s["spec"])
        hits = [sp for sp in req.specs if _clean(sp) and _clean(sp) in spec_clean]
        if req.specs:
            score += 0.35 * len(hits) / len(req.specs)
            if hits:
                reasons.append("Spec match: " + ", ".join(hits))
        if req.city:
            if s["city"] == req.city:
                score += 0.30; reasons.append(f"Same city ({s['city']})")
            elif req.city in NCR and s["city"] in NCR:
                score += 0.24; reasons.append("Delhi NCR supplier")
            elif buyer_state and s["state"] == buyer_state:
                score += 0.18; reasons.append(f"Same state ({s['state']})")
        score += 0.15 * (s["rating"] - 3.4) / 1.5
        if s["gst_verified"]:
            score += 0.10; reasons.append("GST verified")
        resp_w = 0.12 if req.urgent else 0.05
        score += resp_w * (1 / s["avg_response_hrs"])
        if req.urgent and s["avg_response_hrs"] <= 2:
            reasons.append(f"Replies in ~{s['avg_response_hrs']}h")
        if req.quantity is not None and s["moq"] > req.quantity:
            score *= 0.5; reasons.append(f"MOQ {s['moq']} is above your quantity")
        ranked.append({**s, "score": round(score, 3), "reasons": reasons})
    ranked.sort(key=lambda r: -r["score"])
    return ranked[:k]


def retrieve(req: Requirement, index=None) -> tuple[str | None, float, list[tuple[str, float]], list[dict], TraceStep]:
    index = index or get_index()
    t0 = time.perf_counter()
    step = TraceStep(f"retrieve:{index.kind}", 0)
    query = " ".join([req.product_query or ""] + req.specs).strip()
    if not query:
        step.ms = (time.perf_counter() - t0) * 1000
        return None, 0.0, [], [], step
    candidates = index.search(query, k=3)
    best, sim = candidates[0]
    step.notes.append(f"top products: " + ", ".join(f"{n} ({s:.2f})" for n, s in candidates))
    if sim < index.threshold:
        step.notes.append(f"best similarity {sim:.2f} < threshold {index.threshold:.2f} -> no confident match")
        step.ms = (time.perf_counter() - t0) * 1000
        return None, sim, candidates, [], step
    suppliers = rank_suppliers(best, req)
    step.ms = (time.perf_counter() - t0) * 1000
    return best, sim, candidates, suppliers, step
