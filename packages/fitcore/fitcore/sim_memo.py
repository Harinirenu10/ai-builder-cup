"""A one-page merchandiser memo from the simulation's statistics (Gemini pro through fitcore.llm).

The model sees only `memo_facts(result)`: every number already written the way it may be quoted (percentages with one decimal, cm).
A memo is used only if every number in it appears in the facts, it is short enough for a page, and it has no accusatory or
body-judging wording; otherwise a template written from the same facts is used. The caveat about the synthetic population is
added by code, always, so it never depends on the model.
"""

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from fitcore import llm
from fitcore.returns.summary import BANNED
from fitcore.sim import SimResult

logger = logging.getLogger("fitcore.sim_memo")
MAX_WORDS = 380
ISSUE_MIN_PCT = 10.0            # a zone is mentioned if at least this share of a size's bodies are below or above its range
MAX_ISSUES = 8


class MemoSection(BaseModel):
    heading: str
    body: str


class Memo(BaseModel):
    headline: str
    sections: list[MemoSection]
    source: Literal["model", "template"] = "template"

    def render(self) -> str:
        return "\n\n".join([self.headline, *(f"{s.heading}\n{s.body}" for s in self.sections)])


class MemoOut(BaseModel):
    """What the model returns (the caveat is not its job)."""
    headline: str = Field(description="One line")
    sections: list[MemoSection]


def pct(x: float) -> float:
    return round(x * 100.0, 1)


def memo_facts(r: SimResult) -> dict[str, Any]:
    """The only input the model gets. Shares are percentages with one decimal; nothing needs computing."""
    issues = []
    for size, zones in r.zone_fit.items():
        for z, sh in zones.items():
            lo, hi = pct(sh.low), pct(sh.high)
            if max(lo, hi) >= ISSUE_MIN_PCT:
                kind, v = (sh.low_word, lo) if lo >= hi else (sh.high_word, hi)
                issues.append({"size": size, "zone": z.value.replace("_", " "), "problem": kind, "percent_of_that_size_buyers": v})
    issues = sorted(issues, key=lambda i: -i["percent_of_that_size_buyers"])[:MAX_ISSUES]
    facts: dict[str, Any] = {
        "product": {"category": r.category.value, "fit": r.fit_type.value, "chart_type": r.chart_type, "sizes": r.sizes},
        "population": {"bodies": r.n_bodies, "source": r.population_source, "synthetic": True,
                       "reweighted_to_a_target_market": r.reweighted},
        "best_size_share_percent": {s: pct(v) for s, v in r.size_mix.items()},
        "no_size_recommended_percent": pct(r.no_recommendation_share),
        "no_good_size_percent": pct(r.no_good_size_share),
        "covered_percent": pct(r.coverage),
        "zones_outside_the_range": issues,
    }
    if r.regrade is not None:
        facts["regrade"] = {
            "covered_percent_before": pct(r.regrade.coverage_before), "covered_percent_after": pct(r.regrade.coverage_after),
            "changes": [{"size": c.size, "zone": c.zone.value.replace("_", " "), "from_cm": c.from_cm, "to_cm": c.to_cm, "change_cm": c.delta_cm,
                         "covered_percent_gained_by_this_change_alone": pct(c.coverage_gain_alone)} for c in r.regrade.changes[:MAX_ISSUES]]}
    return facts


SYSTEM = (
    "You write a one-page memo for a clothing merchandiser about a size chart, using ONLY the statistics provided. "
    "Quote numbers exactly as written in the statistics (a percentage as given, for example 27.1%, a size change in cm as given). "
    "Never calculate a new number: no sums, differences, averages or rounding. If you need a figure that is not given, say it in words. "
    "Write at most 300 words as a headline and two to four short sections (what the population says, where the chart loses buyers, "
    "what to change in the grading). Describe the garment, the sizes and the chart; never describe or judge anyone's body. "
    "The statistics use field names such as covered_percent or no_good_size_percent: never repeat a field name, write natural sentences "
    "(for example: the chart covers 27.1% of the bodies). Lead with the finding that matters most to a merchandiser, then what to do about it. "
    "Do not write a caveat about the population: it is added separately."
)

_NUM = re.compile(r"\d+(?:\.\d+)?")


def numbers_in(text: str) -> set[float]:
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)          # 4,000 is 4000
    return {float(n) for n in _NUM.findall(text)}


def cites_only_the_facts(text: str, facts: dict[str, Any]) -> bool:
    return numbers_in(text) <= numbers_in(json.dumps(facts))


def caveat(r: SimResult) -> MemoSection:
    return MemoSection(heading="Caveat", body=r.banner)


def template_memo(r: SimResult, facts: dict[str, Any]) -> Memo:
    best = ", ".join(f"{s} {v}%" for s, v in facts["best_size_share_percent"].items())
    pop = facts["population"]
    sections = [MemoSection(heading="What the population says", body=(
        f"Of {pop['bodies']} synthetic bodies, the best size is {best}. No size is recommended for {facts['no_size_recommended_percent']}%, and "
        f"{facts['no_good_size_percent']}% have no size where every hard-to-alter zone is ideal, so the chart covers {facts['covered_percent']}%."))]
    if facts["zones_outside_the_range"]:
        lines = "; ".join(f"{i['size']} {i['zone']}: {i['percent_of_that_size_buyers']}% {i['problem']}" for i in facts["zones_outside_the_range"][:5])
        sections.append(MemoSection(heading="Where the chart loses buyers", body=f"Among buyers of each size, the largest shares outside the zone's range are: {lines}."))
    rg = facts.get("regrade")
    if rg and rg["changes"]:
        lines = "; ".join(f"{c['size']} {c['zone']} {c['from_cm']} to {c['to_cm']} cm" for c in rg["changes"][:5])
        sections.append(MemoSection(heading="Suggested regrade", body=(
            f"Moving {lines} raises coverage from {rg['covered_percent_before']}% to {rg['covered_percent_after']}% in this population.")))
    elif rg:
        sections.append(MemoSection(heading="Suggested regrade", body=f"No change of up to 2 cm in a zone raises coverage above {rg['covered_percent_before']}%."))
    return Memo(headline=f"{r.category.value.capitalize()} chart: {facts['covered_percent']}% of the synthetic population has a good size", sections=sections, source="template")


def write_memo(r: SimResult, client: Any | None = None, ollama_transport: Any | None = None) -> Memo:
    """The memo for a simulation result: from the model if it follows the rules, otherwise the template. Never raises."""
    facts = memo_facts(r)
    tail = caveat(r)
    try:
        out = llm.generate_json("pro", json.dumps(facts, ensure_ascii=False), MemoOut, system=SYSTEM, client=client, ollama_transport=ollama_transport)
    except Exception as exc:
        logger.warning("memo model call failed (%s); using the template", type(exc).__name__)
        m = template_memo(r, facts)
        m.sections.append(tail)
        return m
    text = out.headline + " " + " ".join(s.heading + " " + s.body for s in out.sections)
    problems = []
    if not out.headline.strip() or not out.sections:
        problems.append("empty")
    if BANNED.search(text):
        problems.append("banned wording")
    if not cites_only_the_facts(text, facts):
        problems.append("a number that is not in the statistics")
    if len(text.split()) > MAX_WORDS:
        problems.append("too long")
    if problems:
        logger.warning("memo rejected (%s); using the template", ", ".join(problems))
        m = template_memo(r, facts)
    else:
        m = Memo(headline=out.headline.strip(), sections=[MemoSection(heading=s.heading.strip(), body=s.body.strip()) for s in out.sections], source="model")
    m.sections.append(tail)
    return m
