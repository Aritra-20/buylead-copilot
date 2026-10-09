"""Data structures shared across the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

UNITS = ["piece", "pair", "box", "roll", "coil", "kg", "metre", "unit"]


@dataclass
class Requirement:
    """Structured version of a buyer's free-text inquiry."""

    product_query: Optional[str] = None      # normalised product name, e.g. "stainless steel hex bolt"
    specs: list[str] = field(default_factory=list)  # e.g. ["M8", "SS304"]
    quantity: Optional[float] = None         # always in base unit (tons are converted to kg)
    unit: Optional[str] = None               # one of UNITS
    city: Optional[str] = None
    urgent: bool = False
    in_scope: bool = True                    # False if the inquiry is not a request to buy goods
    language: str = "en"                     # "en" | "hinglish"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TraceStep:
    step: str
    ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class Result:
    inquiry: str
    mode: str
    requirement: Requirement
    action: str                               # "match" | "clarify" | "refuse"
    product: Optional[str] = None             # catalog product resolved by retrieval
    retrieval_score: float = 0.0
    suppliers: list[dict] = field(default_factory=list)
    drafts: list[dict] = field(default_factory=list)
    clarifying_question: Optional[str] = None
    refusal_reason: Optional[str] = None
    trace: list[TraceStep] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return sum(s.ms for s in self.trace)

    @property
    def tokens_in(self) -> int:
        return sum(s.tokens_in for s in self.trace)

    @property
    def tokens_out(self) -> int:
        return sum(s.tokens_out for s in self.trace)

    @property
    def guard_events(self) -> list[str]:
        return [n for s in self.trace for n in s.notes if n.startswith("GUARD")]

    @property
    def llm_errors(self) -> list[str]:
        """LLM calls that failed and fell back to rules/templates (so an 'LLM' eval can't silently be rules)."""
        return [n for s in self.trace for n in s.notes if "failed (" in n]
