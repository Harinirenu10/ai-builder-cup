"""Business-facing view of a fit: verdicts only, never measurements or ease values (CLAUDE.md privacy rule)."""

from fitcore.fit.config import get_config
from fitcore.schemas import Alteration, BizAlteration, BizVerdict, BizZoneVerdict, FitResult, Verdict

_HINTS = {
    "shorten_sleeves": "sleeves may need {cm} cm shortening",
    "lengthen_sleeves": "sleeves may need {cm} cm lengthening",
    "shorten_hem": "hem may need {cm} cm shortening",
    "lengthen_hem": "hem may need {cm} cm lengthening",
    "shorten_inseam": "inseam may need {cm} cm shortening",
    "lengthen_inseam": "inseam may need {cm} cm lengthening",
    "take_in_waist": "waist may need {cm} cm taking in",
    "let_out_waist": "waist may need {cm} cm letting out",
}


def alteration_hint(fit: FitResult) -> str | None:
    """One short garment-side hint for the largest suggested alteration, or None."""
    if not fit.alterations:
        return None
    a: Alteration = max(fit.alterations, key=lambda x: x.cm)
    template = _HINTS.get(a.action)
    return template.format(cm=f"{a.cm:g}") if template else None


def guarantee_eligible(fit: FitResult) -> bool:
    """A size was recommended with enough confidence and every hard zone fits now, after washing and in poses."""
    if fit.recommended_size is None or fit.confidence < get_config().guarantee.min_confidence:
        return False
    rec = fit.size_fit(fit.recommended_size)
    if rec is None:
        return False
    for z in rec.zones.values():
        if z.hard and (z.verdict != Verdict.ideal or z.verdict_after_wash != Verdict.ideal or any(p.strain for p in z.poses.values())):
            return False
    return True


def zone_verdicts(fit: FitResult) -> list[tuple[str, Verdict]]:
    """(zone, verdict) of the recommended size; the worse of now and after-wash for each zone."""
    rec = fit.size_fit(fit.recommended_size) if fit.recommended_size else None
    if rec is None:
        return []
    return [(z.zone.value, z.verdict if z.verdict != Verdict.ideal else z.verdict_after_wash) for z in rec.zones.values()]


def business_verdict(fit: FitResult) -> BizVerdict:
    rec = fit.size_fit(fit.recommended_size) if fit.recommended_size else None
    zones = [BizZoneVerdict(zone=z.zone, verdict=z.verdict, verdict_after_wash=z.verdict_after_wash) for z in rec.zones.values()] if rec else []
    return BizVerdict(
        category=fit.category,
        recommended_size=fit.recommended_size,
        confidence=fit.confidence,
        zones=zones,
        alterations=[BizAlteration(zone=a.zone, action=a.action) for a in fit.alterations],
        warning_kinds=sorted({w.kind for w in fit.warnings}),
    )
