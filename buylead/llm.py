"""LLM provider switch - Claude (Anthropic) or Gemini (Google).

The agent only ever calls `client.messages.create(...)` in the Anthropic shape. For Gemini,
`GeminiClient` exposes that same method and translates it, so extract / verify / draft and
all the guardrails run unchanged on either provider.

Pick the provider with BUYLEAD_PROVIDER=anthropic|gemini. If it is not set, the provider is
chosen from whichever API key is present (ANTHROPIC_API_KEY first, then GEMINI_API_KEY).
"""
from __future__ import annotations

import json
import os
import re
import time
from types import SimpleNamespace

from .schema import UNITS

DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "gemini": "gemini-3.5-flash-lite"}
DEFAULT_EMBED_MODEL = "gemini-embedding-2"

# Paid-tier list prices, USD per 1M tokens (input, output), from ai.google.dev/gemini-api/docs/pricing
# (checked Oct 2026). Used for cost-per-query when BUYLEAD_PRICE_IN/OUT are not set.
PRICES = {"gemini-3.5-flash-lite": (0.30, 2.50)}


def model_price(model: str) -> tuple[float, float] | None:
    p_in, p_out = os.getenv("BUYLEAD_PRICE_IN"), os.getenv("BUYLEAD_PRICE_OUT")
    if p_in and p_out:
        return float(p_in), float(p_out)
    return PRICES.get(model)


def provider_name() -> str | None:
    """Return 'anthropic', 'gemini', or None when no API key is configured."""
    p = os.getenv("BUYLEAD_PROVIDER", "").strip().lower()
    if p in DEFAULT_MODELS:
        return p
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return None


def default_model(provider: str) -> str:
    return os.getenv("BUYLEAD_MODEL") or DEFAULT_MODELS[provider]


def make_client():
    """Build the client for the configured provider (lazy imports: rules mode needs neither SDK)."""
    p = provider_name() or "anthropic"
    if p == "gemini":
        return GeminiClient()
    import anthropic
    return anthropic.Anthropic()


def resolve_model(client, model: str | None) -> str:
    if model:
        return model
    return getattr(client, "model", None) or default_model("anthropic")


class _Throttle:
    """Keeps requests under the free tier's requests-per-minute limit."""

    def __init__(self, rpm: float):
        self.gap = 60.0 / rpm if rpm > 0 else 0.0
        self.last = 0.0

    def wait(self):
        delay = self.last + self.gap - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.last = time.monotonic()


def _retryable(e: Exception) -> bool:
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    return code in (429, 500, 503) or "RESOURCE_EXHAUSTED" in str(e) or "UNAVAILABLE" in str(e)


def _parse_json(text: str) -> dict:
    """Parse the model's JSON, tolerating ```json fences or stray prose around the object."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


class GeminiClient:
    """Gemini behind the Anthropic `messages.create` interface used by the agent.

    * A forced tool call becomes Gemini JSON mode, with the tool's JSON schema spelled out in
      the system instruction. The parsed JSON comes back as a `tool_use` block, exactly like Claude.
    * Plain text calls map system prompt -> system_instruction.
    * Requests are throttled only if BUYLEAD_RPM is set (e.g. 10 on the free tier; paid-tier
      limits are far higher), and always retried with backoff on 429 / 5xx.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None, genai_client=None):
        if genai_client is None:
            from google import genai  # pip install google-genai
            genai_client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        self._genai = genai_client
        self.model = model or default_model("gemini")
        self._throttle = _Throttle(float(os.getenv("BUYLEAD_RPM") or 0))
        self.messages = SimpleNamespace(create=self._create)

    def _generate(self, system: str, user: str, max_tokens: int, json_mode: bool):
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            # Generous ceiling: some Gemini models spend output tokens on internal reasoning,
            # and a tight cap (e.g. the verifier's 5 tokens) would return an empty answer.
            max_output_tokens=max(max_tokens, 1024),
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        for attempt in range(5):
            self._throttle.wait()
            try:
                return self._genai.models.generate_content(model=self.model, contents=user, config=config)
            except Exception as e:  # noqa: BLE001
                if attempt == 4 or not _retryable(e):
                    raise
                time.sleep(15 * (attempt + 1))

    def _create(self, *, model=None, max_tokens=400, system="", messages=(), tools=None, tool_choice=None,
                temperature=0, **_):
        user = "\n\n".join(m["content"] for m in messages if m.get("role") == "user")
        if tools:
            tool = tools[0]
            system = (f"{system}\n\nReturn ONLY one JSON object (no prose, no code fences) that matches this "
                      f"JSON schema for `{tool['name']}`:\n{json.dumps(tool['input_schema'])}")
            resp = self._generate(system, user, max_tokens, json_mode=True)
            data = _parse_json(resp.text or "")
            if data.get("unit") not in UNITS:
                data["unit"] = None
            block = SimpleNamespace(type="tool_use", name=tool["name"], input=data)
        else:
            resp = self._generate(system, user, max_tokens, json_mode=False)
            block = SimpleNamespace(type="text", text=(resp.text or "").strip())
        um = getattr(resp, "usage_metadata", None)
        tokens_in = getattr(um, "prompt_token_count", 0) or 0
        tokens_out = (getattr(um, "candidates_token_count", 0) or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
        return SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out))


class GeminiEmbedder:
    """Neural text embeddings via the Gemini API (`gemini-embedding-2`, 100+ languages).

    Vectors are L2-normalised so a dot product is cosine similarity. Uses the task prefixes the
    model expects for asymmetric search (short query vs. longer product document).
    """

    def __init__(self, api_key: str | None = None, model: str | None = None, genai_client=None, dim: int = 768):
        if genai_client is None:
            from google import genai
            genai_client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        self._genai = genai_client
        self.model = model or os.getenv("BUYLEAD_EMBED_MODEL") or DEFAULT_EMBED_MODEL
        self.dim = dim

    def _embed(self, texts: list[str]):
        import numpy as np
        from google.genai import types

        out = []
        for i in range(0, len(texts), 50):
            for attempt in range(5):
                try:
                    res = self._genai.models.embed_content(
                        model=self.model, contents=texts[i:i + 50],
                        config=types.EmbedContentConfig(output_dimensionality=self.dim))
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 4 or not _retryable(e):
                        raise
                    time.sleep(15 * (attempt + 1))
            out.extend(e.values for e in res.embeddings)
        v = np.asarray(out, dtype="float32")
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    def embed_documents(self, titles: list[str], texts: list[str]):
        return self._embed([f"title: {t} | text: {x}" for t, x in zip(titles, texts)])

    def embed_query(self, text: str):
        return self._embed([f"task: search result | query: {text}"])[0]


def embeddings_backend(mode: str) -> str:
    """'gemini' (neural) in LLM mode with a Gemini key AND a calibrated threshold; else 'tfidf'.

    Neural similarities live on a different scale from TF-IDF, so an uncalibrated threshold could
    silently refuse good inquiries. Until `python -m buylead.tune` has written the threshold (or
    BUYLEAD_EMBED_THRESHOLD is set), the agent keeps TF-IDF. BUYLEAD_EMBEDDINGS=gemini|tfidf forces a choice.
    """
    from .catalog import DATA_DIR

    choice = os.getenv("BUYLEAD_EMBEDDINGS", "").strip().lower()
    if mode != "llm" or choice == "tfidf":
        return "tfidf"
    if choice == "gemini":
        return "gemini"
    has_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    calibrated = bool(os.getenv("BUYLEAD_EMBED_THRESHOLD")) or (DATA_DIR / "embedding_threshold.json").exists()
    return "gemini" if has_key and calibrated else "tfidf"
