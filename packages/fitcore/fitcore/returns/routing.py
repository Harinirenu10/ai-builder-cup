"""Deterministic routing of a return, with a plain-text decision log for every step.

Rules, in order (see also config/returns.yaml):
1. Fit guarantee and the unboxing measurement does not match the label  -> auto-approve an exchange; flag the seller's SKU for a label error.
2. Fit guarantee, too_small or too_big, and the unboxing matches the label -> approve an exchange to the next size; log a model miss.
3. wrong_size_received and the unboxing matches the label                -> human review.
4. No guarantee                                                          -> the merchant's standard policy.
Cases the rules do not cover (no unboxing evidence, a body chart that cannot be compared with a garment, a non-sizing reason on a
guaranteed item) go to a person or to the standard policy. Nothing is ever rejected automatically and nothing penalises the shopper.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from fitcore.fit.config import get_config
from fitcore.schemas import MeasurementZone, SizeChart

Reason = Literal["too_small", "too_big", "wrong_size_received", "quality", "changed_mind", "other"]
Policy = Literal["refund", "exchange", "review"]
SIZE_REASONS = {"too_small", "too_big", "wrong_size_received"}


class OrderItemFacts(BaseModel):
    sku: str
    size: str
    guarantee: bool
    recommended_size_at_purchase: str | None = None
    confidence_at_purchase: float | None = None


class Unboxing(BaseModel):
    """A garment measured when it was unboxed. Garment values only, never body values."""
    id: str
    size_label: str
    measured_cm: dict[MeasurementZone, float]
    overlay_image_url: str | None = None
    source: Literal["simulated", "photo"] = "simulated"


class LogEntry(BaseModel):
    rule: str
    text: str


class LabelCheck(BaseModel):
    comparable: bool
    matches: bool = False
    ordered_size: str
    deviation_cm: float | None = None       # worst zone against the ordered size's chart row
    worst_zone: MeasurementZone | None = None
    closest_size: str | None = None
    closest_deviation_cm: float | None = None
    tolerance_cm: float
    note: str = ""


class Routing(BaseModel):
    route: Literal["auto_exchange", "auto_exchange_next_size", "human_review", "merchant_policy"]
    status: Literal["approved_exchange", "approved_refund", "pending_review"]
    exchange_size: str | None = None
    seller_label_error: bool = False
    model_miss: bool = False
    label_check: LabelCheck | None = None
    decision_log: list[LogEntry] = Field(default_factory=list)


def _fmt(v: float) -> str:
    return f"{v:.1f}"


def label_check(chart: SizeChart, ordered_size: str, unboxing: Unboxing) -> LabelCheck:
    """Compare the measured garment with the chart row of the size that was ordered (and with every other size)."""
    tol = get_config().label_match_tolerance_cm
    if chart.chart_type != "garment":
        return LabelCheck(comparable=False, ordered_size=ordered_size, tolerance_cm=tol,
                          note="The size chart lists body sizes, so a measured garment cannot be compared with the label.")
    if ordered_size not in chart.sizes:
        return LabelCheck(comparable=False, ordered_size=ordered_size, tolerance_cm=tol, note=f"Size {ordered_size} is not in the size chart.")

    def worst(size: str) -> tuple[float, MeasurementZone] | None:
        row = chart.sizes[size]
        diffs = [(abs(unboxing.measured_cm[z] - row[z]), z) for z in unboxing.measured_cm if z in row]
        return max(diffs, key=lambda d: d[0]) if diffs else None

    ordered = worst(ordered_size)
    if ordered is None:
        return LabelCheck(comparable=False, ordered_size=ordered_size, tolerance_cm=tol, note="No measured zone appears in the size chart.")
    others = [(worst(s), s) for s in chart.sizes]
    others = [(w[0], s) for w, s in others if w is not None]
    closest_dev, closest = min(others, key=lambda t: t[0])
    return LabelCheck(comparable=True, matches=ordered[0] <= tol, ordered_size=ordered_size, deviation_cm=round(ordered[0], 2), worst_zone=ordered[1],
                      closest_size=closest, closest_deviation_cm=round(closest_dev, 2), tolerance_cm=tol)


def _neighbour(chart: SizeChart, size: str, step: int) -> str | None:
    """The size one step along the chart (listed smallest to largest), or None at the end."""
    sizes = list(chart.sizes)
    if size not in sizes:
        return None
    i = sizes.index(size) + step
    return sizes[i] if 0 <= i < len(sizes) else None


def _standard_policy(reason: str, item: OrderItemFacts, chart: SizeChart, policy: Policy, window_days: int,
                     ordered_at: datetime, now: datetime, log: list[LogEntry], check: LabelCheck | None) -> Routing:
    days = (now - ordered_at).days
    if check is not None and check.comparable and not check.matches:
        log.append(LogEntry(rule="evidence", text=(
            f"The unboxing measurement differs from the size {item.size} chart by up to {_fmt(check.deviation_cm or 0)} cm "
            f"(tolerance {_fmt(check.tolerance_cm)} cm). This is shown to the person who decides.")))
    if days > window_days:
        log.append(LogEntry(rule="standard_policy", text=f"No fit guarantee. The order is {days} days old, outside the {window_days}-day window, so a person decides. Nothing is rejected automatically."))
        return Routing(route="merchant_policy", status="pending_review", decision_log=log, label_check=check)
    if policy == "refund":
        log.append(LogEntry(rule="standard_policy", text=f"No fit guarantee. The merchant's standard policy approves a refund within {window_days} days (order is {days} days old)."))
        return Routing(route="merchant_policy", status="approved_refund", decision_log=log, label_check=check)
    if policy == "exchange":
        exchange = _neighbour(chart, item.size, 1 if reason == "too_small" else -1 if reason == "too_big" else 0) if reason in ("too_small", "too_big") else None
        extra = f" to size {exchange}" if exchange else ""
        log.append(LogEntry(rule="standard_policy", text=f"No fit guarantee. The merchant's standard policy approves an exchange{extra} within {window_days} days (order is {days} days old)."))
        return Routing(route="merchant_policy", status="approved_exchange", exchange_size=exchange, decision_log=log, label_check=check)
    log.append(LogEntry(rule="standard_policy", text=f"No fit guarantee. The merchant's standard policy is review, so a person decides (order is {days} days old)."))
    return Routing(route="merchant_policy", status="pending_review", decision_log=log, label_check=check)


def route(*, item: OrderItemFacts, reason: Reason, chart: SizeChart, unboxing: Unboxing | None, policy: Policy | None,
          window_days: int | None, ordered_at: datetime, now: datetime) -> Routing:
    cfg = get_config()
    policy = policy or cfg.default_return_policy  # type: ignore[assignment]
    window_days = window_days if window_days is not None else cfg.default_return_window_days
    log: list[LogEntry] = []
    check = label_check(chart, item.size, unboxing) if unboxing is not None else None
    has_evidence = check is not None and check.comparable

    if item.guarantee:
        conf = f"{item.confidence_at_purchase:.2f}" if item.confidence_at_purchase is not None else "n/a"
        log.append(LogEntry(rule="guarantee", text=f"Fit guarantee applies: size {item.size} was bought, the twin recommended {item.recommended_size_at_purchase} with confidence {conf} (needs at least {cfg.guarantee.min_confidence:.2f})."))
    else:
        log.append(LogEntry(rule="guarantee", text="No fit guarantee on this item (the size bought was not the recommended size, or confidence was below the threshold, or there was no twin consent)."))

    if unboxing is None:
        log.append(LogEntry(rule="evidence", text="No unboxing measurement was provided."))
    elif check is not None and not check.comparable:
        log.append(LogEntry(rule="evidence", text=check.note))
    elif check is not None:
        word = "matches" if check.matches else "does not match"
        log.append(LogEntry(rule="evidence", text=(
            f"Unboxing measurement {word} the size {item.size} label: worst zone {check.worst_zone.value.replace('_', ' ')} is off by {_fmt(check.deviation_cm or 0)} cm "
            f"(tolerance {_fmt(check.tolerance_cm)} cm). Closest chart size is {check.closest_size} ({_fmt(check.closest_deviation_cm or 0)} cm off).")))

    # 1. guarantee + label mismatch
    if item.guarantee and has_evidence and not check.matches:  # type: ignore[union-attr]
        closest = f" The garment measures closest to size {check.closest_size}." if check.closest_size != item.size else ""  # type: ignore[union-attr]
        log.append(LogEntry(rule="rule_1", text=f"Rule 1: the garment does not match its label, so the exchange is approved automatically and SKU {item.sku} is flagged for a label error.{closest}"))
        return Routing(route="auto_exchange", status="approved_exchange", exchange_size=item.recommended_size_at_purchase, seller_label_error=True, label_check=check, decision_log=log)

    # 2. guarantee + too_small/too_big + label matches
    if item.guarantee and reason in ("too_small", "too_big") and has_evidence and check.matches:  # type: ignore[union-attr]
        nxt = _neighbour(chart, item.size, 1 if reason == "too_small" else -1)
        if nxt is None:
            log.append(LogEntry(rule="rule_2", text=f"Rule 2 would exchange to the next size, but there is no size {'above' if reason == 'too_small' else 'below'} {item.size} in the chart, so a person decides."))
            return Routing(route="human_review", status="pending_review", model_miss=True, label_check=check, decision_log=log)
        log.append(LogEntry(rule="rule_2", text=f"Rule 2: the garment matches its label but the shopper reports {reason.replace('_', ' ')} under the guarantee, so an exchange to size {nxt} is approved and logged as a model miss for evaluation."))
        return Routing(route="auto_exchange_next_size", status="approved_exchange", exchange_size=nxt, model_miss=True, label_check=check, decision_log=log)

    # 3. wrong size received + label matches
    if reason == "wrong_size_received" and has_evidence and check.matches:  # type: ignore[union-attr]
        log.append(LogEntry(rule="rule_3", text="Rule 3: the shopper says the wrong size arrived but the garment matches its label, so a person reviews it. It is never rejected automatically."))
        return Routing(route="human_review", status="pending_review", label_check=check, decision_log=log)

    # Guaranteed item the rules do not settle: sizing reasons without usable evidence go to a person.
    if item.guarantee and reason in SIZE_REASONS and not has_evidence:
        why = "no unboxing measurement" if unboxing is None else "an unboxing measurement that cannot be compared with this chart"
        log.append(LogEntry(rule="human_review", text=f"Sizing return under the guarantee with {why}, so a person decides and the shopper can add an unboxing measurement."))
        return Routing(route="human_review", status="pending_review", label_check=check, decision_log=log)

    # 4. everything else: the merchant's standard policy (also guaranteed items returned for non-sizing reasons)
    if item.guarantee:
        log.append(LogEntry(rule="standard_policy", text=f"Reason '{reason.replace('_', ' ')}' is not a sizing matter, so the guarantee does not apply and the standard policy does."))
    return _standard_policy(reason, item, chart, policy, window_days, ordered_at, now, log, check)  # type: ignore[arg-type]
