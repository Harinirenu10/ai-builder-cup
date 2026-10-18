"""Turn explanation_facts into 1-3 plain sentences with Gemini flash (garment language only)."""

import json
import logging
import re
from typing import Any

from pydantic import BaseModel

from fitcore import llm
from fitcore.schemas import FitResult

logger = logging.getLogger("fitcore.fit.explain")

SYSTEM = (
    "You write short fit explanations for a fashion shopper. Use only the facts provided. "
    "Do not add any fact, number, size, material or advice that is not in the facts. "
    "Describe the garment (its ease, length, cut, fabric and alterations) and never judge the body: "
    "no words about weight, body shape, flaws or what is flattering. "
    "Write 1 to 3 sentences in the requested language. Keep numbers and units exactly as given."
)

# Words that must never reach a shopper (CLAUDE.md language rule).
_BANNED = re.compile(
    r"\b(weight|overweight|underweight|fat|fatty|chubby|plump|skinny|curvy|bulky|flaw|flaws|flawed|"
    r"flatter|flatters|flattering|unflattering|figure|problem areas?|body type|body shape|your body|"
    r"slimming|hide|hides|disguise)\b",
    re.IGNORECASE,
)
_NUM = re.compile(r"\d+(?:\.\d+)?")


class ExplainOut(BaseModel):
    text: str


def _numbers(text: str) -> set[float]:
    return {float(n) for n in _NUM.findall(text)}


def _acceptable(text: str, facts: list[str]) -> bool:
    if not text.strip() or _BANNED.search(text):
        return False
    return _numbers(text) <= _numbers(" ".join(facts))


def explain(fit_result: FitResult, language: str = "en", client: Any | None = None) -> str:
    """Explain the fit in 1-3 sentences. Falls back to the plain facts if the output breaks a rule."""
    facts = fit_result.explanation_facts
    if not facts:
        return ""
    prompt = json.dumps({"language": language, "facts": facts}, ensure_ascii=False)
    out = llm.generate_json("flash", prompt, ExplainOut, system=SYSTEM, client=client)
    text = out.text.strip()
    if not _acceptable(text, facts):
        logger.warning("explain(): model output rejected by guardrails; returning facts instead")
        return " ".join(facts[:3])
    return text
