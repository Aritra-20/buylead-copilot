"""Keep tests offline: strip API keys and BuyLead settings from the environment for every test."""
import pytest


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for k in ["ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "BUYLEAD_PROVIDER", "BUYLEAD_MODEL",
              "BUYLEAD_EMBEDDINGS", "BUYLEAD_EMBED_THRESHOLD", "BUYLEAD_RPM", "BUYLEAD_PRICE_IN", "BUYLEAD_PRICE_OUT"]:
        monkeypatch.delenv(k, raising=False)
