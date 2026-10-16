"""Business-facing view of a fit: verdicts only, never measurements or ease values (CLAUDE.md privacy rule)."""

from fitcore.schemas import BizVerdict, BizZoneVerdict, BizAlteration, FitResult


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
