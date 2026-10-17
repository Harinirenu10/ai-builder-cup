import pytest

from fitcore.fit import apply_reviews, from_review_signals
from fitcore.schemas import Measurement, Measurements, MeasurementZone as Z, ReviewerContext, ReviewSignals, ZoneSignal


def sig(zone=Z.chest, direction="tight", strength=0.5, n=3, ctx=None):
    return ZoneSignal(zone=zone, direction=direction, strength=strength, mention_count=n, reviewer_context=ctx or [])


@pytest.mark.parametrize("direction,expected", [("runs_small", "smaller"), ("tight", "smaller"), ("short", "smaller"),
                                                 ("runs_large", "larger"), ("loose", "larger"), ("long", "larger")])
def test_direction_folded(direction, expected):
    out = from_review_signals(ReviewSignals(reviews_read=5, zone_signals=[sig(direction=direction)]))
    assert out[0].direction == expected


@pytest.mark.parametrize("strength,label", [(0.0, "slight"), (0.33, "slight"), (0.34, "moderate"), (0.66, "moderate"), (0.67, "strong"), (1.0, "strong")])
def test_strength_labels(strength, label):
    assert from_review_signals(ReviewSignals(reviews_read=5, zone_signals=[sig(strength=strength)]))[0].strength == label


def test_single_mention_is_ignored():
    assert from_review_signals(ReviewSignals(reviews_read=5, zone_signals=[sig(n=1)])) == []


def test_long_torso_context_taken_from_reviewers():
    ctx = [ReviewerContext(torso="long"), ReviewerContext(torso="long"), ReviewerContext(note="5ft 4")]
    out = from_review_signals(ReviewSignals(reviews_read=5, zone_signals=[sig(Z.garment_length, "short", ctx=ctx)]))
    assert out[0].context == "long_torso"


def test_mixed_context_is_not_applied():
    ctx = [ReviewerContext(torso="long"), ReviewerContext(note="a"), ReviewerContext(note="b"), ReviewerContext(note="c")]
    assert from_review_signals(ReviewSignals(reviews_read=5, zone_signals=[sig(Z.garment_length, "short", ctx=ctx)]))[0].context is None


def test_end_to_end_into_apply_respects_twin_context():
    signals = ReviewSignals(reviews_read=6, zone_signals=[sig(Z.garment_length, "short", 0.8, 3, [ReviewerContext(torso="long")])])
    engine_signals = from_review_signals(signals)
    long_twin = Measurements(zones={Z.torso_length: Measurement(value_cm=52, confidence=0.9, source="t")})
    short_twin = Measurements(zones={Z.torso_length: Measurement(value_cm=44, confidence=0.9, source="t")})
    assert apply_reviews(engine_signals, long_twin)[0].cm[Z.garment_length] == pytest.approx(3.0)
    assert apply_reviews(engine_signals, short_twin)[0].cm == {}
