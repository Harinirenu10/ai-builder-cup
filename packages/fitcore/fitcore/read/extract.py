"""Extract a Product and ReviewSignals from page text and screenshots with one structured LLM call (flash)."""

import json
from typing import Any

from fitcore import llm
from fitcore.read.models import PageExtraction

MAX_TEXT_CHARS = 20000  # keeps a request inside a 16k-token local model; Gemini has room to spare

SYSTEM = """You read an online clothing product page and fill the schema. Rules:
- Use only what the page text, JSON-LD hints and images show. Never invent numbers. Use null / empty lists / "unknown" when something is absent.
- category must be one of the schema values, or null if the item is not one of those garments (shoes, bags, accessories, ...).
- Size chart: copy each size's measurements in the ORIGINAL units, with each measurement label exactly as printed (for example "To fit chest", "Garment chest", "Across shoulder"). For a range cell like 38-40 give value_min 38 and value_max 40.
- chart_type is "body" when the chart lists the body it is made to fit (labels like "To fit chest", "Body measurements", "Fits chest"), and "garment" when it lists the garment's own dimensions ("Garment measurements", "Chest (flat)"). chart_type_evidence is the label or sentence that decided it. Set circumference_basis to "flat" only when the page says chest/waist/hip are measured flat or across.
- unit is the unit the chart prints (cm or in), or "unknown".
- pre_shrunk is "true" only if the page says pre-shrunk / sanforized / pre-washed, "false" only if it says it will shrink, otherwise "unknown"; pre_shrunk_evidence is the short snippet.
- fabric: each fiber with its percent as printed. fit_type and stretch_class only when the page indicates them.
- reviews: look ONLY at customer review text. reviews_read is how many separate reviews you can see. Each zone signal must be supported by reviewers; mention_count is how many of them said it and can never exceed reviews_read. strength is 0 to 1 from how many agree and how strongly. reviewer_context records height_cm, torso long or short, or a short note only when a reviewer states it. If no reviews are visible, reviews_read is 0 and every list is empty and every count is 0.
- shrinkage_mentions, color_mismatch_mentions and sheerness_mentions count reviewers who mention shrinking in the wash, the colour differing from the photos, or the fabric being see-through.
- Describe the garment only. Do not comment on anyone's body."""


def build_parts(text: str, json_ld: list[dict] | None = None, images: list[bytes] | None = None,
                url: str | None = None) -> list[Any]:
    """Prompt parts: one text part, then the images (chart screenshot first)."""
    chunks = []
    if url:
        chunks.append(f"Page URL: {url}")
    if json_ld:
        chunks.append("JSON-LD hints:\n" + json.dumps(json_ld, ensure_ascii=False)[:6000])
    chunks.append("Visible page text:\n" + text[:MAX_TEXT_CHARS] if text.strip() else "Visible page text: (none; read the images)")
    return ["\n\n".join(chunks), *(images or [])]


def extract(text: str, json_ld: list[dict] | None = None, images: list[bytes] | None = None,
            url: str | None = None, client: Any | None = None, ollama_transport: Any | None = None) -> PageExtraction:
    parts = build_parts(text, json_ld, images, url)
    return llm.generate_json("flash", parts, PageExtraction, system=SYSTEM, client=client, ollama_transport=ollama_transport)
