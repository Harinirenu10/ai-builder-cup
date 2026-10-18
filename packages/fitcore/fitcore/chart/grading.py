"""Grading check: every zone must grow from one size to the next, by an amount inside the expected grade.

Flags a value smaller than the previous size, a step bigger than the usual grade, and a step smaller than the usual grade
(including no growth where growth is expected). Gemini flash words each flag as one sentence; a template does if it cannot.
"""

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from fitcore import llm
from fitcore.fit.config import get_config
from fitcore.read.normalize import size_rank
from fitcore.returns.summary import BANNED
from fitcore.schemas import Category, MeasurementZone as Z

logger = logging.getLogger("fitcore.chart.grading")
EPS = 1e-6


class GradingFlag(BaseModel):
    size: str                       # the larger size of the pair: the step into this size is the problem
    previous_size: str
    zone: Z
    kind: Literal["decrease", "jump_too_large", "jump_too_small"]
    severity: Literal["high", "medium"]
    value_cm: float
    previous_cm: float
    delta_cm: float
    expected_min_cm: float
    expected_max_cm: float
    explanation: str = ""
    explained_by: Literal["template", "model"] = "template"


def step_range(category: Category, zone: Z) -> tuple[float, float]:
    cfg = get_config()
    pair = cfg.overrides.get(category, {}).get(zone) or cfg.default_step_cm.get(zone)
    return (pair[0], pair[1]) if pair else (0.0, 99.0)


def order_labels(labels: list[str]) -> list[str]:
    """Smallest first when every label can be ranked (S, M, L, 32, 34, ...); otherwise as given."""
    ranks = [size_rank(label) for label in labels]
    if labels and all(r is not None for r in ranks):
        return [lab for _, lab in sorted(zip(ranks, labels), key=lambda t: t[0])]  # type: ignore[arg-type]
    return list(labels)


def check_grading(category: Category, sizes: dict[str, dict[Z, float]], tolerance_cm: float | None = None) -> list[GradingFlag]:
    """Compare each size with the next smaller one that has a value, zone by zone.

    If sizes in between have no value for the zone, the expected growth is multiplied by the number of steps skipped. A jump may be
    `tolerance_cm` outside its range before it is flagged (default from config: measured values carry about 1 cm of noise);
    a decrease is flagged whatever its size.
    """
    tol = get_config().jump_tolerance_cm if tolerance_cm is None else tolerance_cm
    flags: list[GradingFlag] = []
    ordered = order_labels(list(sizes))
    zones = sorted({z for row in sizes.values() for z in row}, key=lambda z: z.value)
    for zone in zones:
        lo, hi = step_range(category, zone)
        prev_label: str | None = None
        prev_index = 0
        for index, label in enumerate(ordered):
            if zone not in sizes[label]:
                continue
            if prev_label is not None:
                steps = index - prev_index
                delta = round(sizes[label][zone] - sizes[prev_label][zone], 2)
                kind: str | None = None
                if delta < -EPS:
                    kind = "decrease"
                elif delta > hi * steps + tol + EPS:
                    kind = "jump_too_large"
                elif delta < lo * steps - tol - EPS:
                    kind = "jump_too_small"
                if kind:
                    flags.append(GradingFlag(size=label, previous_size=prev_label, zone=zone, kind=kind,  # type: ignore[arg-type]
                                             severity="high" if kind == "decrease" else "medium", value_cm=sizes[label][zone],
                                             previous_cm=sizes[prev_label][zone], delta_cm=delta,
                                             expected_min_cm=round(lo * steps, 2), expected_max_cm=round(hi * steps, 2)))
            prev_label, prev_index = label, index
    order = {s: i for i, s in enumerate(ordered)}
    return sorted(flags, key=lambda f: (order[f.size], f.zone.value))


def _zone(z: Z) -> str:
    return z.value.replace("_", " ")


def template_explanation(f: GradingFlag) -> str:
    z = _zone(f.zone)
    if f.kind == "decrease":
        return f"{f.size} {z} is {abs(f.delta_cm):.1f} cm smaller than {f.previous_size}: likely a mislabeled sample or a measuring slip."
    if f.kind == "jump_too_large":
        return (f"{f.size} {z} grows {f.delta_cm:.1f} cm from {f.previous_size}, more than the usual {f.expected_min_cm:.1f} to "
                f"{f.expected_max_cm:.1f} cm step: check both photos.")
    if f.delta_cm <= EPS:
        return f"{f.size} {z} is the same as {f.previous_size}, but the usual step is {f.expected_min_cm:.1f} to {f.expected_max_cm:.1f} cm: check the photos."
    return (f"{f.size} {z} grows only {f.delta_cm:.1f} cm from {f.previous_size}, less than the usual {f.expected_min_cm:.1f} to "
            f"{f.expected_max_cm:.1f} cm step: check both photos.")


SYSTEM = (
    "You explain flags from a size-chart grading check to a clothing seller. For each flag write exactly ONE short sentence "
    "that states the size, the zone and the difference in cm, and the most likely cause (a mislabeled sample, a measuring slip, "
    "or a photo to re-check). Use only the numbers given. Describe the garment and the chart; never mention people or bodies. "
    "Return the sentences in the same order, one per flag."
)
_NUM = re.compile(r"\d+(?:\.\d+)?")


class Explanations(BaseModel):
    sentences: list[str] = Field(default_factory=list)


def _one_sentence(text: str) -> bool:
    return len(re.findall(r"[.!?](?:\s|$)", text.strip())) <= 1 and "\n" not in text.strip()


def _numbers(text: str) -> set[float]:
    return {float(n) for n in _NUM.findall(text)}


def apply_templates(flags: list[GradingFlag]) -> list[GradingFlag]:
    """Word every flag with the fixed template (no model). This is what runs while a seller is still editing the table."""
    for f in flags:
        f.explanation, f.explained_by = template_explanation(f), "template"
    return flags


def explain_flags(flags: list[GradingFlag], client: Any | None = None, ollama_transport: Any | None = None) -> list[GradingFlag]:
    """Fill `explanation` on every flag: one flash call for all of them, with a template for any sentence that breaks the rules."""
    apply_templates(flags)
    if not flags:
        return flags
    facts = [{"size": f.size, "previous_size": f.previous_size, "zone": _zone(f.zone), "kind": f.kind.replace("_", " "), "value_cm": f.value_cm,
              "previous_cm": f.previous_cm, "delta_cm": f.delta_cm, "usual_step_cm": [f.expected_min_cm, f.expected_max_cm]} for f in flags]
    try:
        out = llm.generate_json("flash", json.dumps(facts), Explanations, system=SYSTEM, client=client, ollama_transport=ollama_transport)
    except Exception as exc:
        logger.warning("grading explanations failed (%s); using templates", type(exc).__name__)
        return flags
    if len(out.sentences) != len(flags):
        logger.warning("model returned %d sentences for %d flags; using templates", len(out.sentences), len(flags))
        return flags
    for f, fact, text in zip(flags, facts, out.sentences):
        text = text.strip()
        if text and _one_sentence(text) and not BANNED.search(text) and _numbers(text) <= _numbers(json.dumps(fact)) | {float(i) for i in range(10)}:
            f.explanation, f.explained_by = text, "model"
    return flags
