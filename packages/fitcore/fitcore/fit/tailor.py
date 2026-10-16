"""Tailor card: the size to buy, the alterations, and only the measurements a tailor needs."""

import json
import logging
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from fitcore import llm
from fitcore.fit.config import get_config
from fitcore.schemas import FitResult, Measurements, MeasurementZone, TailorCard, TailorLines

logger = logging.getLogger("fitcore.fit.tailor")

SYSTEM = (
    "You translate a tailor's work card. Translate only the words. "
    "Keep every number, every unit (cm) and every size label exactly as written. "
    "Return the same number of alteration_lines and measurement_lines in the same order."
)
_NUM = re.compile(r"\d+(?:\.\d+)?")

_FONT_CANDIDATES = {
    "default": ["arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "ta": [
        "Nirmala.ttc", "Latha.ttf", "NotoSansTamil-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf",
        "/usr/share/fonts/truetype/lohit-tamil/Lohit-Tamil.ttf",
    ],
}


def _label(zone: MeasurementZone) -> str:
    return zone.value.replace("_", " ").capitalize()


def _english_lines(card_size: str, fit: FitResult, measurements: dict[MeasurementZone, float]) -> TailorLines:
    return TailorLines(
        title="Tailor card",
        size_line=f"Size to buy: {card_size}",
        alteration_lines=[a.text for a in fit.alterations] or ["No alterations needed"],
        measurement_lines=[f"{_label(z)}: {v:.1f} cm" for z, v in measurements.items()],
    )


def _signature(lines: TailorLines) -> tuple[list[str], int]:
    text = " ".join([lines.size_line, *lines.alteration_lines, *lines.measurement_lines])
    return _NUM.findall(text), len(re.findall(r"cm", text))


def _translation_ok(src: TailorLines, dst: TailorLines, size: str) -> bool:
    if len(src.alteration_lines) != len(dst.alteration_lines) or len(src.measurement_lines) != len(dst.measurement_lines):
        return False
    src_nums, src_cm = _signature(src)
    dst_nums, dst_cm = _signature(dst)
    return sorted(src_nums) == sorted(dst_nums) and src_cm == dst_cm and size in dst.size_line


def tailor_needed_measurements(fit: FitResult, body: Measurements) -> dict[MeasurementZone, float]:
    """Body zones touched by an alteration, plus the category's hard zones so the tailor can check the base size."""
    cfg = get_config()
    zones: list[MeasurementZone] = []
    for a in fit.alterations:
        zones.append(cfg.body_ref.get(a.zone, a.zone))
    zones.extend(cfg.hard_zones[fit.category])
    out: dict[MeasurementZone, float] = {}
    for z in zones:
        v = body.get_cm(z)
        if v is not None and z not in out:
            out[z] = round(v, 1)
    return out


def tailor_card(
    fit_result: FitResult,
    measurements: Measurements,
    languages: list[str] | None = None,
    client: Any | None = None,
) -> TailorCard:
    """Build the card JSON. Non-English text comes from Gemini; numbers and units are verified unchanged."""
    languages = languages or ["en", "ta"]
    if fit_result.recommended_size is None:
        raise ValueError("No recommended size, so there is nothing to put on a tailor card")
    needed = tailor_needed_measurements(fit_result, measurements)
    en = _english_lines(fit_result.recommended_size, fit_result, needed)
    card = TailorCard(size=fit_result.recommended_size, alterations=fit_result.alterations, measurements=needed, lines={"en": en})
    for lang in languages:
        if lang == "en":
            continue
        prompt = json.dumps({"target_language": lang, "card": en.model_dump()}, ensure_ascii=False)
        try:
            out = llm.generate_json("flash", prompt, TailorLines, system=SYSTEM, client=client)
        except Exception as exc:  # network or quota: keep the English card usable
            logger.warning("tailor_card(): translation to %s failed: %s", lang, exc)
            card.notes.append(f"Translation to {lang} unavailable.")
            continue
        if _translation_ok(en, out, fit_result.recommended_size):
            card.lines[lang] = out
        else:
            logger.warning("tailor_card(): %s translation changed numbers or units; dropped", lang)
            card.notes.append(f"Translation to {lang} was dropped because it changed a number or unit.")
    return card


def _font(lang: str, size: int) -> tuple[ImageFont.ImageFont, bool]:
    for name in _FONT_CANDIDATES.get(lang, _FONT_CANDIDATES["default"]):
        try:
            return ImageFont.truetype(name, size), True
        except OSError:
            continue
    return ImageFont.load_default(), False


def render_png(card: TailorCard, path: str | Path) -> Path:
    """Draw every language block on one PNG. Needs a Tamil-capable font for 'ta' (Nirmala on Windows, Noto Sans Tamil elsewhere)."""
    width, margin = 900, 40
    blocks: list[tuple[str, TailorLines]] = list(card.lines.items())
    line_h = 42
    height = margin * 2 + sum((3 + len(b.alteration_lines) + len(b.measurement_lines)) * line_h + 30 for _, b in blocks)
    img = Image.new("RGB", (width, max(height, 200)), "white")
    draw = ImageDraw.Draw(img)
    y = margin
    for lang, b in blocks:
        font, ok = _font(lang, 28)
        title_font, _ = _font(lang, 36)
        if not ok:
            card.notes.append(f"No font found for {lang}; text may not render.")
        draw.text((margin, y), b.title, fill="black", font=title_font)
        y += line_h + 8
        draw.text((margin, y), b.size_line, fill="black", font=font)
        y += line_h
        for t in b.alteration_lines:
            draw.text((margin, y), f"- {t}", fill=(20, 60, 140), font=font)
            y += line_h
        for t in b.measurement_lines:
            draw.text((margin, y), t, fill=(60, 60, 60), font=font)
            y += line_h
        y += 14
        draw.line((margin, y, width - margin, y), fill=(200, 200, 200), width=2)
        y += 16
    path = Path(path)
    img.save(path, "PNG")
    return path
