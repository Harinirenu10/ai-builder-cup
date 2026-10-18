"""A neutral, facts-only case summary for the returns desk (Gemini flash through fitcore.llm).

The model sees only the facts built by `case_facts`: no shopper free text, no body measurements. Its output must not
contain accusatory or body-judging words and must not introduce numbers that are not in the facts; otherwise a plain
template summary is used instead.
"""

import json
import logging
import re
from typing import Any

from pydantic import BaseModel

from fitcore import llm

logger = logging.getLogger("fitcore.returns.summary")

SYSTEM = (
    "You write a short case summary for a person deciding on a clothing return. Use only the facts provided. "
    "State facts and numbers neutrally in 2 to 4 sentences: what was ordered, what was reported, what the measurements show, "
    "what the rules decided and why. Never speculate about the shopper's honesty or intentions, never accuse or suggest blame, "
    "and never describe anyone's body. Describe the garment and the process only. Keep numbers and units exactly as given."
)

# Words that must never appear (accusation, blame, body judgement).
BANNED = re.compile(
    r"\b(fraud\w*|lying|liar\w*|lie|lies|lied|cheat\w*|scam\w*|dishonest\w*|deceiv\w*|deceit\w*|suspicious|suspect\w*|fake\w*|abus\w*|"
    r"gaming|misuse\w*|bad faith|blame\w*|accus\w*|guilty|untrustworthy|manipulat\w*|"
    r"weight|overweight|underweight|fat|chubby|skinny|curvy|flaw\w*|flatter\w*|unflattering|figure|body type|body shape)\b",
    re.IGNORECASE,
)
_NUM = re.compile(r"\d+(?:\.\d+)?")


class SummaryOut(BaseModel):
    text: str


def case_facts(*, product_title: str, ordered_size: str, reason: str, guarantee: bool, recommended_size: str | None,
               confidence: float | None, routing_route: str, routing_status: str, decision_log: list[str],
               spec_cm: dict[str, float] | None, measured_cm: dict[str, float] | None, has_note: bool,
               days_since_order: int) -> dict[str, Any]:
    """The only input the model gets. Garment numbers only; the shopper's note is reduced to whether one exists."""
    diffs = {z: round(measured_cm[z] - spec_cm[z], 1) for z in (measured_cm or {}) if spec_cm and z in spec_cm}
    return {
        "product": product_title, "ordered_size": ordered_size, "reported_reason": reason.replace("_", " "),
        "fit_guarantee": guarantee, "size_recommended_at_purchase": recommended_size,
        "confidence_at_purchase": None if confidence is None else round(confidence, 2),
        "days_since_order": days_since_order, "shopper_left_a_note": has_note,
        "ordered_size_spec_cm": spec_cm or {}, "measured_garment_cm": measured_cm or {}, "measured_minus_spec_cm": diffs,
        "routing": routing_route.replace("_", " "), "status": routing_status.replace("_", " "), "rule_log": decision_log,
    }


def _numbers(text: str) -> set[float]:
    return {float(n) for n in _NUM.findall(text)}


def acceptable(text: str, facts: dict[str, Any]) -> bool:
    if not text.strip() or BANNED.search(text):
        return False
    return _numbers(text) <= _numbers(json.dumps(facts))


def template_summary(facts: dict[str, Any]) -> str:
    """Deterministic fallback: the same facts in fixed sentences."""
    parts = [f"{facts['product']}, size {facts['ordered_size']}, ordered {facts['days_since_order']} days ago. Reported reason: {facts['reported_reason']}."]
    if facts["fit_guarantee"]:
        parts.append(f"The fit guarantee applies (recommended size {facts['size_recommended_at_purchase']}, confidence {facts['confidence_at_purchase']}).")
    else:
        parts.append("The fit guarantee does not apply.")
    if facts["measured_minus_spec_cm"]:
        worst = max(facts["measured_minus_spec_cm"].items(), key=lambda kv: abs(kv[1]))
        parts.append(f"The unboxing measurement differs from the ordered size's spec by up to {abs(worst[1])} cm ({worst[0].replace('_', ' ')}).")
    else:
        parts.append("There is no comparable unboxing measurement.")
    parts.append(f"Routing: {facts['routing']}; status: {facts['status']}.")
    return " ".join(parts)


def summarize(facts: dict[str, Any], client: Any | None = None, ollama_transport: Any | None = None) -> tuple[str, str]:
    """(text, source) where source is 'model' or 'template'."""
    try:
        out = llm.generate_json("flash", json.dumps(facts, ensure_ascii=False), SummaryOut, system=SYSTEM, client=client, ollama_transport=ollama_transport)
    except Exception as exc:
        logger.warning("summary model call failed (%s); using the template", type(exc).__name__)
        return template_summary(facts), "template"
    text = out.text.strip()
    if not acceptable(text, facts):
        logger.warning("summary rejected by guardrails; using the template")
        return template_summary(facts), "template"
    return text, "model"
