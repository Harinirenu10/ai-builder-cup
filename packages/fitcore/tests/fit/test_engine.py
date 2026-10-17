import pytest

from fitcore.fit.config import EaseRange, get_config
from fitcore.fit.engine import classify, evaluate
from fitcore.schemas import (
    Category, FitPreferences, PoseDelta, PoseDeltas, ReviewAdjustments, Verdict,
)
from tests.fit.helpers import Z, body, product

CFG = get_config()
RNG = EaseRange(min=8, ideal=14, max=20)  # tshirt / regular / chest


@pytest.mark.parametrize("ease,zone,allow,expected", [
    (7.99, Z.chest, 0, Verdict.tight),
    (8.0, Z.chest, 0, Verdict.ideal),
    (14.0, Z.chest, 0, Verdict.ideal),
    (20.0, Z.chest, 0, Verdict.ideal),
    (20.01, Z.chest, 0, Verdict.loose),
    (6.0, Z.chest, 2, Verdict.ideal),      # stretch allowance widens the low side
    (5.99, Z.chest, 2, Verdict.tight),
    (7.99, Z.garment_length, 5, Verdict.short),  # length zones ignore the stretch allowance
    (8.0, Z.garment_length, 5, Verdict.ideal),
    (20.01, Z.garment_length, 0, Verdict.long),
])
def test_verdict_boundaries(ease, zone, allow, expected):
    assert classify(ease, RNG, zone, allow, CFG)[0] == expected


def test_margin_sign_and_size():
    assert classify(5.0, RNG, Z.chest, 0, CFG)[1] == pytest.approx(-3.0)
    assert classify(23.0, RNG, Z.chest, 0, CFG)[1] == pytest.approx(3.0)
    assert classify(14.0, RNG, Z.chest, 0, CFG)[1] == 0.0


def test_inch_chart_matches_cm_chart():
    b = body(chest=96)
    cm = product("tshirt", {"M": {"chest": 111.76}})
    inch = product("tshirt", {"M": {"chest": 44.0}}, unit="in")
    r_cm, r_in = evaluate(b, cm), evaluate(b, inch)
    assert r_in.sizes[0].zones[Z.chest].ease_now_cm == pytest.approx(r_cm.sizes[0].zones[Z.chest].ease_now_cm, abs=0.01)


def test_garment_chart_path():
    r = evaluate(body(chest=96), product("tshirt", {"M": {"chest": 110}}))
    z = r.sizes[0].zones[Z.chest]
    assert z.path == "garment" and z.ease_now_cm == 14.0 and z.verdict == Verdict.ideal


def test_body_chart_path_is_ease_around_ideal():
    sizes = {"S": {"chest": 90}, "M": {"chest": 96}, "L": {"chest": 102}}
    r = evaluate(body(chest=96), product("tshirt", sizes, chart_type="body"))
    by = {s.size: s.zones[Z.chest] for s in r.sizes}
    assert all(z.path == "body" for z in by.values())
    assert by["M"].ease_now_cm == pytest.approx(14.0)      # body on the chart value -> ideal ease
    assert by["S"].ease_now_cm == pytest.approx(8.0) and by["S"].verdict == Verdict.ideal
    assert by["L"].ease_now_cm == pytest.approx(20.0)
    assert r.recommended_size == "M"


def test_personal_offset_and_review_adjustment_follow_formula():
    b, p = body(chest=96), product("tshirt", {"M": {"chest": 110}})
    prefs = FitPreferences(offsets_cm={Category.tshirt: {Z.chest: 2.0}})
    adj = ReviewAdjustments(cm={Z.chest: 1.5})
    z = evaluate(b, p, prefs, adj).sizes[0].zones[Z.chest]
    assert z.ease_now_cm == pytest.approx(14 + 2.0 - 1.5)


def test_wash_rayon_kurta_ideal_length_becomes_short():
    b = body(chest=96, torso_length=46)
    rayon = [{"fiber": "rayon", "percent": 100}]
    sizes = {"M": {"chest": 112, "garment_length": 89}}  # length ease 43 vs kurta/regular 40..60
    r = evaluate(b, product("kurta", sizes, fabric=rayon))
    z = r.sizes[0].zones[Z.garment_length]
    assert z.verdict == Verdict.ideal and z.verdict_after_wash == Verdict.short
    assert any(w.kind == "wash" and w.zone == Z.garment_length for w in r.warnings)
    # pre-shrunk keeps it ideal (shrinkage x 0.3)
    z2 = evaluate(b, product("kurta", sizes, fabric=rayon, pre_shrunk=True)).sizes[0].zones[Z.garment_length]
    assert z2.verdict_after_wash == Verdict.ideal


def test_wash_prefers_size_that_survives_the_wash():
    b = body(chest=96, torso_length=46)
    rayon = [{"fiber": "rayon", "percent": 100}]
    sizes = {"M": {"chest": 112, "garment_length": 89}, "L": {"chest": 114, "garment_length": 94}}
    assert evaluate(b, product("kurta", sizes, fabric=rayon)).recommended_size == "L"


def test_pose_slim_trouser_zero_thigh_ease_strains_when_seated():
    b = body(waist=82, hip=98, thigh=56, inseam=78, back_rise=26)
    p = product("trousers", {"M": {"waist": 86, "hip": 106, "thigh": 56, "inseam": 79, "back_rise": 27}}, fit_type="slim")
    seated = PoseDeltas(poses={"seated": PoseDelta(minutes=120, delta_cm={Z.thigh: 3.0})})
    r = evaluate(b, p, pose_deltas=seated)
    pf = r.sizes[0].zones[Z.thigh].poses["seated"]
    assert pf.strain and pf.ease_cm < 0
    assert any(w.kind == "pose" and w.zone == Z.thigh for w in r.warnings)
    assert all(w.kind != "pose" for w in evaluate(b, p).warnings)


def test_long_pose_strain_costs_more_than_short_pose_strain():
    b = body(hip=98, thigh=56, waist=82)
    p = product("trousers", {"M": {"waist": 86, "hip": 112, "thigh": 64, "inseam": 79}}, stretch="none")

    def pen(minutes):
        d = PoseDeltas(poses={"seated": PoseDelta(minutes=minutes, delta_cm={Z.thigh: 12.0})})
        return evaluate(b, p, pose_deltas=d).sizes[0].zones[Z.thigh].penalty

    assert pen(120) > pen(10) > 0


def test_stretch_allowance_removes_mild_strain():
    b = body(hip=98, thigh=56, waist=82)
    row = {"waist": 86, "hip": 106, "thigh": 63}  # thigh ease 7, jeans/regular min 6
    d = PoseDeltas(poses={"seated": PoseDelta(minutes=60, delta_cm={Z.thigh: 2.0})})
    none = evaluate(b, product("jeans", {"M": row}, stretch="none"), pose_deltas=d).sizes[0].zones[Z.thigh]
    high = evaluate(b, product("jeans", {"M": row}, stretch="high"), pose_deltas=d).sizes[0].zones[Z.thigh]
    assert none.poses["seated"].strain and not high.poses["seated"].strain


@pytest.mark.parametrize("size_chest", [100, 104, 110, 118])
def test_monotonic_larger_chest_never_improves_verdict_once_snug(size_chest):
    """From the first body size that fits, a larger chest only gets tighter (never a better verdict)."""
    p = product("tshirt", {"M": {"chest": size_chest}})
    started, prev_margin = False, 0.0
    for chest in range(60, 140):
        z = evaluate(body(chest=chest), p).sizes[0].zones[Z.chest]
        if not started:
            if z.verdict != Verdict.ideal:
                continue                      # still too loose; not yet at this size's fit
            started = True
        else:
            assert z.verdict in (Verdict.ideal, Verdict.tight)
            assert abs(z.margin_cm) >= abs(prev_margin) - 1e-9
        prev_margin = z.margin_cm
    assert started


def test_alteration_text_and_amount():
    b = body(chest=96, shoulder_width=42, sleeve_length=58, waist=82, torso_length=46)
    sizes = {"M": {"chest": 112, "shoulder_width": 44, "sleeve_length": 61.5, "waist": 98, "garment_length": 72}}
    r = evaluate(b, product("shirt", sizes))
    assert "Shorten sleeves 2.5 cm" in [a.text for a in r.alterations]


def test_waist_taken_in_when_loose():
    b = body(chest=96, shoulder_width=42, sleeve_length=58, waist=82, torso_length=46)
    sizes = {"M": {"chest": 112, "shoulder_width": 44, "sleeve_length": 59, "waist": 120, "garment_length": 72}}
    r = evaluate(b, product("shirt", sizes))
    a = next(a for a in r.alterations if a.zone == Z.waist)
    assert a.action == "take_in_waist" and 0 < a.cm <= 6.0


BODY_ALL = body(chest=96, waist=82, hip=98, shoulder_width=42, sleeve_length=58, torso_length=46,
                thigh=56, inseam=78, back_rise=26)


@pytest.mark.parametrize("category", [c.value for c in Category])
def test_alterations_never_touch_hard_zones(category):
    cat = Category(category)
    hard = set(CFG.hard_zones[cat])
    for fit, ease in CFG.target_ease_cm[cat].items():
        for skew in (-12, -6, -2, 0, 3, 8, 15):
            row = {z.value: BODY_ALL.get_cm(CFG.body_ref.get(z, z)) + e.ideal + skew for z, e in ease.items()}
            sizes = {"A": row, "B": {k: v + 4 for k, v in row.items()}}
            r = evaluate(BODY_ALL, product(category, sizes, fit_type=fit.value))
            for a in r.alterations:
                assert a.zone not in hard
                assert a.zone in CFG.easy_zones
                acts = CFG.actions[a.zone]
                assert a.cm <= max(acts.decrease.max_cm, acts.increase.max_cm)


def test_hard_zone_failing_everywhere_gives_warning_not_size():
    sizes = {"S": {"chest": 90}, "M": {"chest": 94}, "L": {"chest": 98}}  # body 100 -> negative ease everywhere
    r = evaluate(body(chest=100), product("tshirt", sizes))
    assert r.recommended_size is None and r.confidence == 0.0 and r.alterations == []
    assert any(w.kind == "size" and w.zone == Z.chest for w in r.warnings)
    assert r.explanation_facts


def test_confidence_rises_with_score_gap():
    b = body(chest=96)
    near = evaluate(b, product("tshirt", {"A": {"chest": 110}, "B": {"chest": 111}}))
    far = evaluate(b, product("tshirt", {"A": {"chest": 110}, "B": {"chest": 140}}))
    assert far.confidence > near.confidence


def test_missing_hard_body_zone_is_reported():
    r = evaluate(body(chest=96), product("shirt", {"M": {"chest": 112, "shoulder_width": 44}}))
    assert any(w.kind == "data" and w.zone == Z.shoulder_width for w in r.warnings)


def test_no_comparable_zones_raises():
    with pytest.raises(ValueError):
        evaluate(body(neck=38), product("tshirt", {"M": {"chest": 110}}))
