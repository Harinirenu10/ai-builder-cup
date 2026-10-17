"""Convert the reader's review signals into per-zone cm adjustments."""

import operator

from fitcore.fit.config import get_config
from fitcore.schemas import Measurements, ReviewAdjustments, ReviewSignal, ReviewSignals

_OPS = {">=": operator.ge, "<=": operator.le}


def apply(review_signals: list[ReviewSignal], twin: Measurements) -> tuple[ReviewAdjustments, list[str]]:
    """Sum signals into cm per zone (positive = runs smaller). Context signals apply only if the twin matches."""
    cfg = get_config()
    cm: dict = {}
    notes: list[str] = []
    for sig in review_signals:
        label = sig.zone.value.replace("_", " ")
        if sig.context is not None:
            ctx = cfg.contexts[sig.context]
            value = twin.get_cm(ctx.body_zone)
            if value is None:
                notes.append(f"Skipped {sig.context.replace('_', ' ')} signal for {label}: no {ctx.body_zone.value.replace('_', ' ')} measurement.")
                continue
            if not _OPS[ctx.op](value, ctx.threshold_cm):
                notes.append(
                    f"Skipped {sig.context.replace('_', ' ')} signal for {label}: "
                    f"{ctx.body_zone.value.replace('_', ' ')} {value:.1f} cm does not match ({ctx.op} {ctx.threshold_cm:.1f} cm)."
                )
                continue
        amount = cfg.strength_cm[sig.strength] * (1 if sig.direction == "smaller" else -1)
        cm[sig.zone] = cm.get(sig.zone, 0.0) + amount
        notes.append(f"Reviews say the {label} runs {sig.direction} ({sig.strength}): adjusted by {amount:+.1f} cm.")
    cap = cfg.max_adjustment_cm
    cm = {z: max(-cap, min(cap, v)) for z, v in cm.items()}
    return ReviewAdjustments(cm=cm), notes


_SMALLER = {"runs_small", "short", "tight"}


def from_review_signals(signals: ReviewSignals) -> list[ReviewSignal]:
    """Reader output to engine signals: direction folded to smaller/larger, strength to a label, torso context applied.

    Signals mentioned by fewer than adapter.min_mentions reviewers are dropped.
    """
    a = get_config().adapter
    out: list[ReviewSignal] = []
    for s in signals.zone_signals:
        if s.mention_count < a.min_mentions:
            continue
        strength = "strong" if s.strength >= a.strong_from else "moderate" if s.strength >= a.moderate_from else "slight"
        torsos = [c.torso for c in s.reviewer_context if c.torso]
        context = None
        if torsos and s.reviewer_context:
            for kind in ("long", "short"):
                if torsos.count(kind) / len(s.reviewer_context) >= a.context_share:
                    context = f"{kind}_torso"
        out.append(ReviewSignal(zone=s.zone, direction="smaller" if s.direction in _SMALLER else "larger",
                                strength=strength, context=context))  # type: ignore[arg-type]
    return out
