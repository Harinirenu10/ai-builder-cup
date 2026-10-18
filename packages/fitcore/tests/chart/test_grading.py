import json
from unittest.mock import MagicMock

import pytest

from fitcore.chart.grading import GradingFlag, check_grading, explain_flags, order_labels, step_range, template_explanation
from fitcore.chart.review_compare import compare_reviews
from fitcore.schemas import Category, MeasurementZone as Z, ReviewSignals, SizeChart, ZoneSignal

Zc, Zl, Zs = Z.chest, Z.garment_length, Z.shoulder_width


def chart(**cols):
    """chart(chest=[108, 114, 120, 126]) -> sizes S, M, L, XL"""
    labels = ["S", "M", "L", "XL"]
    out: dict[str, dict[Z, float]] = {}
    for zone, vals in cols.items():
        for label, v in zip(labels, vals):
            if v is not None:
                out.setdefault(label, {})[Z(zone)] = v
    return out


GOOD = chart(chest=[108, 114, 120, 126], garment_length=[72, 74, 76, 78], shoulder_width=[44, 46, 48, 50])


def test_a_properly_graded_chart_has_no_flags():
    assert check_grading(Category.shirt, GOOD) == []


def test_value_smaller_than_the_previous_size_is_flagged_high():
    flags = check_grading(Category.shirt, chart(chest=[108, 114, 120, 119]))
    assert len(flags) == 1
    f = flags[0]
    assert (f.size, f.previous_size, f.zone, f.kind, f.severity, f.delta_cm) == ("XL", "L", Zc, "decrease", "high", -1.0)


@pytest.mark.parametrize("step,kind", [(7.0, None), (7.01, "jump_too_large"), (9.0, "jump_too_large"), (3.0, None), (2.99, "jump_too_small"), (0.0, "jump_too_small"), (-0.1, "decrease")])
def test_step_boundaries_for_chest(step, kind):
    flags = check_grading(Category.shirt, {"M": {Zc: 114.0}, "L": {Zc: round(114.0 + step, 2)}}, tolerance_cm=0)
    assert [f.kind for f in flags] == ([kind] if kind else [])


@pytest.mark.parametrize("step,kind", [(8.0, None), (8.01, "jump_too_large"), (2.0, None), (1.99, "jump_too_small"), (-0.1, "decrease")])
def test_measurement_tolerance_widens_the_jump_limits_but_never_the_decrease_rule(step, kind):
    """Default tolerance is 1 cm: a step may be 1 cm outside 3 to 7 before it is flagged; a smaller value is flagged at once."""
    flags = check_grading(Category.shirt, {"M": {Zc: 114.0}, "L": {Zc: round(114.0 + step, 2)}})
    assert [f.kind for f in flags] == ([kind] if kind else [])


def test_inseam_may_stay_the_same_across_sizes_but_not_shrink_or_leap():
    assert check_grading(Category.trousers, chart(inseam=[80, 80, 80, 80])) == []
    assert [(f.size, f.kind) for f in check_grading(Category.trousers, chart(inseam=[80, 80, 79, 83]))] == [("L", "decrease"), ("XL", "jump_too_large")]


def test_category_overrides_change_the_expected_grade():
    assert step_range(Category.shirt, Zl) != step_range(Category.kurta, Zl)
    assert step_range(Category.shirt, Zl) == (0.8, 3.5)
    assert check_grading(Category.kurta, {"M": {Zl: 100.0}, "L": {Zl: 104.0}}) == []             # 4 cm is fine on a kurta
    assert [f.kind for f in check_grading(Category.shirt, {"M": {Zl: 74.0}, "L": {Zl: 80.0}})] == ["jump_too_large"]


def test_each_zone_is_checked_independently_and_flags_are_sorted_by_size():
    flags = check_grading(Category.shirt, chart(chest=[108, 114, 111, 126], garment_length=[72, 74, 76, 70]))
    assert [(f.size, f.zone.value, f.kind) for f in flags] == [("L", "chest", "decrease"), ("XL", "chest", "jump_too_large"), ("XL", "garment_length", "decrease")]


def test_a_missing_size_scales_the_expected_growth_by_the_steps_skipped():
    lengths = [72, 74, 76, 78]                                                              # M exists but has no chest value
    assert check_grading(Category.shirt, chart(chest=[108, None, 120, 126], garment_length=lengths)) == []   # S to L is two steps: 12 cm is inside 6 to 14
    flags = check_grading(Category.shirt, chart(chest=[108, None, 130, 136], garment_length=lengths))        # 22 cm over two steps is too much
    assert [(f.size, f.previous_size, f.kind, f.expected_min_cm, f.expected_max_cm) for f in flags] == [("L", "S", "jump_too_large", 6.0, 14.0)]


def test_zones_present_in_only_some_sizes_are_compared_among_those_sizes():
    sizes = {"S": {Zc: 108.0}, "M": {Zl: 74.0}, "L": {Zc: 112.0}}
    assert [(f.size, f.previous_size, f.kind) for f in check_grading(Category.shirt, sizes)] == [("L", "S", "jump_too_small")]    # 4 cm over two steps needs 6


def test_sizes_given_in_any_order_are_sorted_before_checking():
    shuffled = {"XL": {Zc: 126.0}, "S": {Zc: 108.0}, "L": {Zc: 120.0}, "M": {Zc: 114.0}}
    assert check_grading(Category.shirt, shuffled) == []
    assert order_labels(["XL", "S", "L", "M"]) == ["S", "M", "L", "XL"]
    assert order_labels(["36", "30", "34", "32"]) == ["30", "32", "34", "36"]
    assert order_labels(["Free", "Tall"]) == ["Free", "Tall"]


def test_single_size_has_nothing_to_compare():
    assert check_grading(Category.shirt, {"M": {Zc: 114.0}}) == []


# ----------------------------------------------------------------------------- explanations


def flag(kind="decrease", delta=-1.0, size="XL", prev="L", zone=Zc, lo=3.0, hi=7.0):
    return GradingFlag(size=size, previous_size=prev, zone=zone, kind=kind, severity="high" if kind == "decrease" else "medium",
                       value_cm=119.0, previous_cm=119.0 - delta, delta_cm=delta, expected_min_cm=lo, expected_max_cm=hi)


@pytest.mark.parametrize("f,must", [
    (flag(), "XL chest is 1.0 cm smaller than L: likely a mislabeled sample"),
    (flag("jump_too_large", 9.0), "grows 9.0 cm from L, more than the usual 3.0 to 7.0 cm"),
    (flag("jump_too_small", 1.0), "grows only 1.0 cm from L, less than the usual 3.0 to 7.0 cm"),
    (flag("jump_too_small", 0.0), "is the same as L"),
])
def test_template_explanations_say_the_size_zone_and_numbers(f, must):
    assert must in template_explanation(f)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "t")
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "gemini-exp-custom")


def client_saying(*sentences: str) -> MagicMock:
    c = MagicMock()
    c.models.generate_content.return_value.text = json.dumps({"sentences": list(sentences)})
    return c


def test_model_sentences_are_used_when_they_follow_the_rules():
    fs = [flag(), flag("jump_too_large", 9.0, size="M", prev="S", zone=Zl, lo=0.8, hi=3.5)]
    fs[1].value_cm, fs[1].previous_cm = 83.0, 74.0
    explain_flags(fs, client=client_saying("XL chest is 1.0 cm smaller than L, which points to a mislabeled sample.",
                                            "M garment length grows 9.0 cm from S, far above the usual 0.8 to 3.5 cm, so re-check both photos."))
    assert [f.explained_by for f in fs] == ["model", "model"] and "mislabeled" in fs[0].explanation


@pytest.mark.parametrize("bad", [
    "XL chest is 5.5 cm smaller than L.",                        # a number that is not in the facts
    "XL chest is smaller. It is probably a mislabeled sample.",   # two sentences
    "The seller is lying about the XL chest.",                    # banned wording
    "",
])
def test_sentences_that_break_the_rules_fall_back_to_the_template(bad):
    fs = [flag()]
    explain_flags(fs, client=client_saying(bad))
    assert fs[0].explained_by == "template" and fs[0].explanation == template_explanation(flag())


def test_wrong_number_of_sentences_or_a_model_error_uses_templates():
    fs = [flag(), flag("jump_too_large", 9.0)]
    explain_flags(fs, client=client_saying("Only one."))
    assert all(f.explained_by == "template" for f in fs)
    broken = MagicMock()
    broken.models.generate_content.side_effect = ValueError("boom")
    explain_flags(fs, client=broken)
    assert all(f.explanation for f in fs) and all(f.explained_by == "template" for f in fs)


def test_no_flags_means_no_model_call():
    client = client_saying()
    assert explain_flags([], client=client) == [] and not client.models.generate_content.called


# ----------------------------------------------------------------------------- reviews beside the chart


OLD = SizeChart(sizes={"S": {Zc: 110.0}, "M": {Zc: 116.0}, "L": {Zc: 122.0}, "XL": {Zc: 128.0}}, chart_type="garment")


def reviews(direction="runs_small", zone=Zc, n=3):
    return ReviewSignals(reviews_read=6, zone_signals=[ZoneSignal(zone=zone, direction=direction, strength=0.6, mention_count=n)])


def test_reviewers_say_small_and_the_new_measurement_is_below_the_old_chart():
    new = {"S": {Zc: 108.0}, "M": {Zc: 114.0}, "L": {Zc: 120.0}, "XL": {Zc: 126.0}}
    (n,) = compare_reviews(new, OLD, reviews())
    assert n.delta_cm == -2.0 and n.agreement == "agrees"
    assert n.text == "3 reviewers say it runs small at the chest; your measured chest is 2.0 cm below your old chart."


def test_the_new_measurement_does_not_explain_it_when_it_is_larger():
    new = {s: {Zc: v + 3} for s, v in {"S": 110.0, "M": 116.0, "L": 122.0, "XL": 128.0}.items()}
    (n,) = compare_reviews(new, OLD, reviews())
    assert n.agreement == "differs" and "3.0 cm above your old chart" in n.text


def test_matching_the_old_chart_within_half_a_cm_is_reported_as_the_same():
    new = {s: {Zc: v + 0.3} for s, v in {"S": 110.0, "M": 116.0, "L": 122.0, "XL": 128.0}.items()}
    (n,) = compare_reviews(new, OLD, reviews("tight", n=1))
    assert n.agreement == "same" and n.text.startswith("A reviewer says it feels tight at the chest") and "matches your old chart within 0.5 cm" in n.text


def test_larger_signals_and_other_zones():
    new = {"M": {Z.garment_length: 78.0}}
    old = SizeChart(sizes={"M": {Z.garment_length: 74.0}}, chart_type="garment")
    (n,) = compare_reviews(new, old, reviews("long", Z.garment_length, 2))
    assert n.agreement == "agrees" and "is long at the garment length" in n.text and "4.0 cm above" in n.text


@pytest.mark.parametrize("old,why", [(None, "no earlier garment chart"), (SizeChart(sizes={"M": {Zc: 96.0}}, chart_type="body"), "body sizes"),
                                      (SizeChart(sizes={"M": {Z.waist: 90.0}}, chart_type="garment"), "no earlier garment chart")])
def test_no_comparison_is_said_plainly(old, why):
    (n,) = compare_reviews({"M": {Zc: 114.0}}, old, reviews())
    assert n.agreement == "no_comparison" and n.delta_cm is None and why in n.text


def test_no_reviews_no_notes():
    assert compare_reviews({"M": {Zc: 1.0}}, OLD, None) == [] and compare_reviews({"M": {Zc: 1.0}}, OLD, ReviewSignals()) == []
