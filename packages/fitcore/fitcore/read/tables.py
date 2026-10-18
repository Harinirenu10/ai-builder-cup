"""Read a size chart straight out of a page's own HTML table, with no model call.

Most shops (and the demo store) publish the chart as a real `<table>`. The extraction model is
still what reads a page's category, fabric and reviews, but for a structured chart it is the
wrong tool twice over: it is the slowest step in a cache-missing read, and a small local model
drops the chart entirely on a large share of attempts even when the numbers are right there in
the text -- measured on the demo shirt, which returned its chart on roughly one attempt in
three. A table that is already machine readable should be read as data.

What comes out is deliberately the same lenient shape the model fills in (RawSize / RawMeasure,
labels exactly as printed), so everything downstream -- label-to-zone mapping, unit inference,
body-vs-garment resolution, range averaging, cm conversion -- stays in
fitcore.read.normalize.build_chart and is not duplicated or second-guessed here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Literal

from fitcore.read.models import RawMeasure, RawSize
from fitcore.read.normalize import normalize_zone
from fitcore.schemas import Category

_SIZE_TOKEN = re.compile(
    r"^(xxs|xs|s|m|l|xl|xxl|2xl|xxxl|3xl|4xl|xxxxl|5xl|6xl|one ?size|free ?size"
    r"|\d{1,3}(\.\d)?|\d{1,2}\s*-\s*\d{1,2})$",
    re.I,
)
# "38-40", "38 - 40", "38–40" (en dash), "38 to 40"
_RANGE = re.compile(r"^(\d{1,3}(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d{1,3}(?:\.\d+)?)$", re.I)
_NUMBER = re.compile(r"(\d{1,3}(?:\.\d+)?)")

_CM_HINT = re.compile(r"\bcm\b|centimet", re.I)
_IN_HINT = re.compile(r"\bin\b|\binch(es)?\b|\"", re.I)
_GARMENT_HINT = re.compile(r"garment measurement|measured flat|laid flat|flat measurement|garment (chest|width)", re.I)
_BODY_HINT = re.compile(r"to fit|body measurement|fits? (chest|waist|hip)|your measurement", re.I)
_SIZE_CHART_HINT = re.compile(r"size (chart|guide)|measurement", re.I)


@dataclass
class _Table:
    rows: list[list[str]] = field(default_factory=list)
    context: str = ""  # nearby text (heading/caption) used only as unit / chart-type evidence


class _TableReader(HTMLParser):
    """Collect every table as a grid of cell strings, plus the text that introduced it."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[_Table] = []
        self._stack: list[_Table] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._recent: list[str] = []  # last few non-table text runs, for context

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "table":
            self._stack.append(_Table(context=" ".join(self._recent[-6:])[-400:]))
        elif tag == "tr" and self._stack:
            self._row = []
        elif tag in ("td", "th") and self._stack:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            if self._row is not None:
                self._row.append(text)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._stack and any(c for c in self._row):
                self._stack[-1].rows.append(self._row)
            self._row = None
        elif tag == "table" and self._stack:
            self.tables.append(self._stack.pop())

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)
        elif not self._stack:
            text = re.sub(r"\s+", " ", data).strip()
            if text:
                self._recent.append(text)


@dataclass
class ParsedChart:
    """A chart read from HTML, in the same lenient shape the model would have returned."""

    sizes: list[RawSize]
    unit: Literal["cm", "in", "unknown"] = "unknown"
    chart_type: Literal["body", "garment", "unknown"] = "unknown"
    chart_type_evidence: str | None = None


def _cell_measure(label: str, cell: str) -> RawMeasure | None:
    cell = cell.strip()
    if not cell:
        return None
    ranged = _RANGE.match(cell)
    if ranged:
        lo, hi = float(ranged.group(1)), float(ranged.group(2))
        return RawMeasure(label=label, value_min=min(lo, hi), value_max=max(lo, hi))
    found = _NUMBER.search(cell)
    if not found:
        return None
    return RawMeasure(label=label, value_min=float(found.group(1)))


def _looks_like_size(cell: str) -> bool:
    return bool(_SIZE_TOKEN.match(cell.strip()))


def _measurement_labels(labels: list[str], category: Category | None) -> int:
    """How many of these labels downstream can actually map to a zone."""
    return sum(1 for lab in labels if lab.strip() and normalize_zone(lab, category) is not None)


def _rows_as_chart(rows: list[list[str]], category: Category | None) -> list[RawSize]:
    """Sizes down the rows: first column is the size label, header row names the measurements."""
    if len(rows) < 2:
        return []
    header = rows[0]
    if _measurement_labels(header[1:], category) < 1:
        return []
    out: list[RawSize] = []
    for row in rows[1:]:
        if not row or not row[0].strip() or not _looks_like_size(row[0]):
            continue
        measures: list[RawMeasure] = []
        for i, cell in enumerate(row[1:], start=1):
            if i >= len(header):
                break
            label = header[i].strip()
            if not label or normalize_zone(label, category) is None:
                continue
            m = _cell_measure(label, cell)
            if m is not None:
                measures.append(m)
        if measures:
            out.append(RawSize(label=row[0].strip(), measurements=measures))
    return out


def _transpose(rows: list[list[str]]) -> list[list[str]]:
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    return [list(col) for col in zip(*padded)]


def _parse_table(table: _Table, category: Category | None) -> ParsedChart | None:
    rows = [r for r in table.rows if any(c.strip() for c in r)]
    if len(rows) < 2:
        return None

    sizes = _rows_as_chart(rows, category)
    if not sizes:
        # Charts are just as often laid out the other way round: measurements down the first
        # column, one column per size.
        sizes = _rows_as_chart(_transpose(rows), category)
    if not sizes:
        return None

    flat = " ".join(c for r in rows for c in r) + " " + table.context
    unit: Literal["cm", "in", "unknown"] = "unknown"
    if _CM_HINT.search(flat):
        unit = "cm"
    elif _IN_HINT.search(flat):
        unit = "in"

    chart_type: Literal["body", "garment", "unknown"] = "unknown"
    evidence: str | None = None
    garment, body = _GARMENT_HINT.search(flat), _BODY_HINT.search(flat)
    # Only one of the two may claim it: a page saying both is ambiguous, so leave it unknown and
    # let normalize.resolve_chart_type apply its own rules rather than guessing here.
    if garment and not body:
        chart_type, evidence = "garment", garment.group(0)
    elif body and not garment:
        chart_type, evidence = "body", body.group(0)

    return ParsedChart(sizes=sizes, unit=unit, chart_type=chart_type, chart_type_evidence=evidence)


def parse_size_chart_html(html: str, category: Category | None = None) -> ParsedChart | None:
    """Best size chart found in this page's HTML tables, or None if no table is one.

    Prefers a table that a size-chart heading or caption introduces, then the one carrying the
    most usable measurements, so a shipping or care table next to the real chart cannot win.
    """
    if not html or "<table" not in html.lower():
        return None
    reader = _TableReader()
    try:
        reader.feed(html)
    except Exception:
        return None

    scored: list[tuple[int, int, ParsedChart]] = []
    for table in reader.tables:
        parsed = _parse_table(table, category)
        if parsed is None:
            continue
        labelled = _SIZE_CHART_HINT.search(table.context) is not None
        cells = sum(len(s.measurements) for s in parsed.sizes)
        scored.append((1 if labelled else 0, cells, parsed))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return scored[0][2]
