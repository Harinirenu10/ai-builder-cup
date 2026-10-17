from datetime import datetime, timedelta, timezone

import pytest

from fitcore.fit.config import get_config
from fitcore.returns.routing import OrderItemFacts, Unboxing, label_check, route
from fitcore.schemas import MeasurementZone as Z, SizeChart

NOW = datetime(2026, 10, 17, tzinfo=timezone.utc)
CHART = SizeChart(sizes={"S": {"chest": 108, "garment_length": 72}, "M": {"chest": 114, "garment_length": 74}, "L": {"chest": 120, "garment_length": 76}}, chart_type="garment")
BODY_CHART = SizeChart(sizes={"S": {"chest": 90}, "M": {"chest": 96}, "L": {"chest": 102}}, chart_type="body")
TOL = get_config().label_match_tolerance_cm


def item(size="M", guarantee=True, rec="M", conf=0.9):
    return OrderItemFacts(sku="shirt-1", size=size, guarantee=guarantee, recommended_size_at_purchase=rec, confidence_at_purchase=conf)


def measured(row: dict, shift: float = 0.0) -> Unboxing:
    return Unboxing(id="u1", size_label="M", measured_cm={Z(k): v + shift for k, v in row.items()})


M_ROW, S_ROW, L_ROW = CHART.sizes["M"], CHART.sizes["S"], CHART.sizes["L"]


def run(**kw):
    base = dict(item=item(), reason="too_small", chart=CHART, unboxing=measured(M_ROW), policy=None, window_days=None,
                ordered_at=NOW - timedelta(days=3), now=NOW)
    base.update(kw)
    return route(**base)


# (case id, kwargs, route, status, exchange_size, seller_label_error, model_miss, a phrase the decision log must contain)
CASES = [
    ("rule1_label_mismatch_too_small", dict(unboxing=measured(S_ROW)), "auto_exchange", "approved_exchange", "M", True, False, "Rule 1"),
    ("rule1_label_mismatch_any_reason", dict(reason="quality", unboxing=measured(L_ROW)), "auto_exchange", "approved_exchange", "M", True, False, "label error"),
    ("rule1_wrong_size_and_mismatch", dict(reason="wrong_size_received", unboxing=measured(S_ROW)), "auto_exchange", "approved_exchange", "M", True, False, "SKU shirt-1"),
    ("rule2_too_small_matches", dict(reason="too_small"), "auto_exchange_next_size", "approved_exchange", "L", False, True, "model miss"),
    ("rule2_too_big_matches", dict(reason="too_big"), "auto_exchange_next_size", "approved_exchange", "S", False, True, "size S"),
    ("rule2_no_larger_size", dict(item=item("L", rec="L"), unboxing=measured(L_ROW), reason="too_small"), "human_review", "pending_review", None, False, True, "no size above L"),
    ("rule2_no_smaller_size", dict(item=item("S", rec="S"), unboxing=measured(S_ROW), reason="too_big"), "human_review", "pending_review", None, False, True, "no size below S"),
    ("rule3_wrong_size_matches", dict(reason="wrong_size_received"), "human_review", "pending_review", None, False, False, "Rule 3"),
    ("rule3_applies_without_guarantee_too", dict(reason="wrong_size_received", item=item(guarantee=False, rec="L")), "human_review", "pending_review", None, False, False, "never rejected"),
    ("guarantee_size_reason_no_evidence", dict(unboxing=None, reason="too_small"), "human_review", "pending_review", None, False, False, "no unboxing measurement"),
    ("guarantee_wrong_size_no_evidence", dict(unboxing=None, reason="wrong_size_received"), "human_review", "pending_review", None, False, False, "a person decides"),
    ("guarantee_body_chart_cannot_compare", dict(chart=BODY_CHART, unboxing=measured({"chest": 96})), "human_review", "pending_review", None, False, False, "body sizes"),
    ("guarantee_non_sizing_reason_goes_to_policy", dict(reason="changed_mind"), "merchant_policy", "pending_review", None, False, False, "not a sizing matter"),
    ("rule4_default_policy_is_review", dict(item=item(guarantee=False, rec="L", conf=0.5), reason="too_big"), "merchant_policy", "pending_review", None, False, False, "standard policy is review"),
    ("rule4_policy_refund", dict(item=item(guarantee=False, rec="L"), reason="too_big", policy="refund"), "merchant_policy", "approved_refund", None, False, False, "approves a refund"),
    ("rule4_policy_exchange_next_size", dict(item=item(guarantee=False, rec="L"), reason="too_small", policy="exchange"), "merchant_policy", "approved_exchange", "L", False, False, "exchange to size L"),
    ("rule4_policy_exchange_no_direction", dict(item=item(guarantee=False, rec="L"), reason="changed_mind", policy="exchange"), "merchant_policy", "approved_exchange", None, False, False, "approves an exchange"),
    ("rule4_outside_window_goes_to_person", dict(item=item(guarantee=False, rec="L"), policy="refund", ordered_at=NOW - timedelta(days=45)), "merchant_policy", "pending_review", None, False, False, "outside the 30-day window"),
    ("rule4_custom_window", dict(item=item(guarantee=False, rec="L"), policy="refund", window_days=60, ordered_at=NOW - timedelta(days=45)), "merchant_policy", "approved_refund", None, False, False, "within 60 days"),
    ("rule4_mismatch_evidence_is_shown_not_flagged", dict(item=item(guarantee=False, rec="L"), unboxing=measured(S_ROW), policy="refund"), "merchant_policy", "approved_refund", None, False, False, "shown to the person"),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_routing_branches(case):
    _, kw, rte, status, exch, label_err, miss, phrase = case
    r = run(**kw)
    assert (r.route, r.status, r.exchange_size, r.seller_label_error, r.model_miss) == (rte, status, exch, label_err, miss)
    assert r.decision_log and all(e.text and e.rule for e in r.decision_log)
    assert phrase.lower() in " ".join(e.text for e in r.decision_log).lower(), [e.text for e in r.decision_log]


def test_nothing_is_ever_rejected_and_every_status_is_an_approval_or_review():
    reasons = ["too_small", "too_big", "wrong_size_received", "quality", "changed_mind", "other"]
    for guarantee in (True, False):
        for reason in reasons:
            for ub in (None, measured(M_ROW), measured(S_ROW)):
                for policy in (None, "refund", "exchange", "review"):
                    for age in (1, 400):
                        r = run(item=item(guarantee=guarantee), reason=reason, unboxing=ub, policy=policy, ordered_at=NOW - timedelta(days=age))
                        assert r.status in {"approved_exchange", "approved_refund", "pending_review"}
                        assert "reject" not in " ".join(e.text for e in r.decision_log).lower().replace("never rejected", "").replace("rejected automatically", "")


def test_unwanted_flags_only_on_the_documented_branches():
    for case in CASES:
        r = run(**case[1])
        if r.seller_label_error:
            assert r.route == "auto_exchange"
        if r.model_miss:
            assert r.route in ("auto_exchange_next_size", "human_review") and case[1].get("reason") in ("too_small", "too_big")


# ----------------------------------------------------------------------------- label check


@pytest.mark.parametrize("shift,matches", [(0.0, True), (TOL, True), (TOL + 0.01, False), (-TOL, True), (-TOL - 0.01, False)])
def test_tolerance_boundary(shift, matches):
    assert label_check(CHART, "M", measured(M_ROW, shift)).matches is matches


def test_label_check_names_the_closest_size_and_worst_zone():
    c = label_check(CHART, "M", measured({"chest": 108.2, "garment_length": 72.1}))
    assert not c.matches and c.closest_size == "S" and c.worst_zone == Z.chest and c.deviation_cm == pytest.approx(5.8)


def test_only_zones_present_in_both_are_compared():
    c = label_check(CHART, "M", measured({"chest": 114}))
    assert c.matches and c.worst_zone == Z.chest


def test_unknown_size_or_no_shared_zone_is_not_comparable():
    assert not label_check(CHART, "XL", measured(M_ROW)).comparable
    assert not label_check(CHART, "M", measured({"sleeve_length": 60})).comparable


def test_decision_log_states_the_numbers_used():
    text = " ".join(e.text for e in run(unboxing=measured(S_ROW)).decision_log)
    assert "6.0 cm" in text and "tolerance 1.5 cm" in text and "0.90" in text
