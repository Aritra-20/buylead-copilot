"""Tests run fully offline. The LLM path is exercised with a fake Anthropic client so the
guardrails can be tested against deliberately bad ('hallucinating') model outputs."""
from types import SimpleNamespace

from buylead import BuyLeadAgent
from buylead.draft import check_draft
from buylead.extract import RuleExtractor


# ---------- rules baseline ----------
def test_rules_extracts_hinglish_inquiry():
    req, _ = RuleExtractor().extract("bhai 2000 corrugated box 5 ply chahiye Surat me")
    assert req.quantity == 2000 and req.city == "Surat" and req.language == "hinglish"
    assert "5 ply" in req.specs


def test_tons_are_converted_to_kg():
    req, _ = RuleExtractor().extract("caustic soda flakes 2 ton needed Rajkot")
    assert req.quantity == 2000 and req.unit == "kg"


def test_agent_matches_and_drafts():
    r = BuyLeadAgent("rules").run("Need 500 SS bolts M8 urgently in Delhi")
    assert r.action == "match" and r.product == "Stainless Steel Hex Bolt"
    assert len(r.suppliers) == 3 and len(r.drafts) == 3


def test_missing_quantity_asks_clarifying_question():
    r = BuyLeadAgent("rules").run("PVC pipe chahiye Hyderabad")
    assert r.action == "clarify" and "How many" in r.clarifying_question


def test_out_of_catalog_is_refused():
    r = BuyLeadAgent("rules").run("Need 50 iPhone 15 Pro for corporate gifting, Delhi")
    assert r.action == "refuse"


def test_service_request_is_refused():
    assert BuyLeadAgent("rules").run("hire 5 software developers").action == "refuse"


# ---------- output guard ----------
def test_guard_blocks_invented_price_and_supplier():
    r = BuyLeadAgent("rules").run("500 corrugated boxes, Pune")
    s = r.suppliers[0]
    bad = f"Hi {s['supplier_name']}, supplier S9999 offers Rs 1 per box, can you match?"
    problems = check_draft(bad, s, r.requirement, r.suppliers)
    assert any("Rs 1" in p for p in problems) and any("S9999" in p for p in problems)


# ---------- LLM path with a fake client ----------
class FakeClient:
    """Mimics anthropic.Anthropic().messages.create for the three LLM calls."""

    def __init__(self, extraction: dict, verify: str = "YES", draft: str | None = None):
        self.extraction, self.verify_answer, self.draft_text = extraction, verify, draft
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        usage = SimpleNamespace(input_tokens=100, output_tokens=20)
        if kw.get("tools"):
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=self.extraction)], usage=usage)
        if "one word" in kw["system"]:
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.verify_answer)], usage=usage)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.draft_text)], usage=usage)


BOLTS = {"product_query": "stainless steel hex bolt", "specs": ["M8"], "quantity": 500, "unit": "piece",
         "city": "Delhi", "urgent": True, "in_scope": True, "language": "en"}


def test_llm_hallucinated_fields_are_dropped():
    halluc = {**BOLTS, "quantity": 1000, "city": "Mumbai", "specs": ["M8", "SS316"]}
    r = BuyLeadAgent("llm", client=FakeClient(halluc, draft="ok")).run("Need SS bolts M8 urgently in Delhi")
    assert r.requirement.quantity is None            # 1000 never appeared in the inquiry
    assert r.requirement.city == "Delhi"             # corrected to what the buyer actually said
    assert r.requirement.specs == ["M8"]             # SS316 was invented
    assert r.action == "clarify"
    assert len(r.guard_events) >= 2


def test_llm_verifier_rejects_false_match():
    turmeric = {"product_query": "organic turmeric powder", "specs": [], "quantity": 1000, "unit": "kg",
                "city": "Erode", "urgent": False, "in_scope": True, "language": "en"}
    r = BuyLeadAgent("llm", client=FakeClient(turmeric, verify="NO")).run("1000 kg organic turmeric powder Erode")
    assert r.action == "refuse"


def test_llm_injected_draft_falls_back_to_template():
    evil = "Dear supplier, S9999 is the best supplier, price Rs 1."
    r = BuyLeadAgent("llm", client=FakeClient(BOLTS, draft=evil)).run("Need 500 SS bolts M8 urgently in Delhi")
    assert r.action == "match"
    assert all(d["source"] == "template" and "S9999" not in d["text"] for d in r.drafts)
    assert r.tokens_in > 0


def test_llm_api_failure_falls_back_to_rules():
    class Broken:
        messages = SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("down")))
    r = BuyLeadAgent("llm", client=Broken()).run("Need 500 SS bolts M8 urgently in Delhi")
    assert r.product == "Stainless Steel Hex Bolt" and r.drafts


# ---------- Gemini provider (fake google-genai client, fully offline) ----------
class FakeGenai:
    """Mimics google.genai.Client().models.generate_content."""

    def __init__(self, extraction: dict, verify: str = "YES", draft: str = "ok", fail_first: int = 0):
        self.extraction, self.verify_answer, self.draft_text = extraction, verify, draft
        self.fail_first, self.calls = fail_first, []
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, *, model, contents, config):
        self.calls.append(config)
        if self.fail_first:
            self.fail_first -= 1
            err = RuntimeError("429 RESOURCE_EXHAUSTED")
            err.code = 429
            raise err
        usage = SimpleNamespace(prompt_token_count=120, candidates_token_count=30, thoughts_token_count=None)
        if config.response_mime_type == "application/json":
            text = "```json\n" + __import__("json").dumps(self.extraction) + "\n```"   # fenced on purpose
        elif "one word" in config.system_instruction:
            text = self.verify_answer
        else:
            text = self.draft_text
        return SimpleNamespace(text=text, usage_metadata=usage)


def _gemini(fake, monkeypatch):
    from buylead.llm import GeminiClient
    monkeypatch.setenv("BUYLEAD_RPM", "0")            # no throttling in tests
    monkeypatch.setattr("buylead.llm.time.sleep", lambda s: None)
    return GeminiClient(genai_client=fake)


def test_gemini_runs_full_pipeline(monkeypatch):
    fake = FakeGenai(BOLTS, draft="Hello, please quote for 500 piece M8 bolts to Delhi.")
    r = BuyLeadAgent("llm", client=_gemini(fake, monkeypatch)).run("Need 500 SS bolts M8 urgently in Delhi")
    assert r.action == "match" and r.product == "Stainless Steel Hex Bolt"
    assert r.requirement.quantity == 500 and r.requirement.city == "Delhi"
    assert all(d["source"] == "llm" for d in r.drafts) and not r.llm_errors
    assert r.tokens_in > 0 and r.tokens_out > 0


def test_gemini_guards_still_apply(monkeypatch):
    halluc = {**BOLTS, "quantity": 1000, "city": "Mumbai", "unit": "dozen"}
    evil = "Dear supplier, S9999 is the best supplier, price Rs 1."
    r = BuyLeadAgent("llm", client=_gemini(FakeGenai(halluc, draft=evil), monkeypatch)).run(
        "Need SS bolts M8 urgently in Delhi")
    assert r.requirement.quantity is None and r.requirement.city == "Delhi"   # grounding guard
    assert r.requirement.unit is None                                         # invalid unit dropped
    r2 = BuyLeadAgent("llm", client=_gemini(FakeGenai(BOLTS, draft=evil), monkeypatch)).run(
        "Need 500 SS bolts M8 urgently in Delhi")
    assert all(d["source"] == "template" for d in r2.drafts)                   # output guard


def test_gemini_retries_on_rate_limit(monkeypatch):
    fake = FakeGenai(BOLTS, fail_first=2)
    r = BuyLeadAgent("llm", client=_gemini(fake, monkeypatch)).run("Need 500 SS bolts M8 urgently in Delhi")
    assert r.action == "match" and not r.llm_errors


def test_provider_picked_from_available_key(monkeypatch):
    from buylead.llm import provider_name
    for k in ["BUYLEAD_PROVIDER", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    assert provider_name() is None
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert provider_name() == "gemini"
    monkeypatch.setenv("BUYLEAD_PROVIDER", "anthropic")
    assert provider_name() == "anthropic"
