import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fitcore.population import generate_standin
from fitcore.returns.summary import BANNED
from fitcore.schemas import Category, Product
from fitcore.sim import simulate
from fitcore.sim_memo import MAX_WORDS, cites_only_the_facts, memo_facts, numbers_in, template_memo, write_memo
from tests.sim.helpers import make_product, population, subset

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "sim_memo_kurta.json"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "t")
    monkeypatch.setenv("GEMINI_PRO_MODEL", "gemini-exp-custom")


@pytest.fixture(scope="module")
def result():
    return simulate(make_product(Category.kurta), population(), regrade=True)


def client_saying(headline: str, *sections: tuple[str, str]) -> MagicMock:
    c = MagicMock()
    c.models.generate_content.return_value.text = json.dumps({"headline": headline, "sections": [{"heading": h, "body": b} for h, b in sections]})
    return c


def test_facts_are_percentages_and_cm_ready_to_quote(result):
    f = memo_facts(result)
    assert f["population"] == {"bodies": 4000, "source": "stand-in", "synthetic": True, "reweighted_to_a_target_market": False}
    assert f["covered_percent"] == round(result.coverage * 100, 1) and f["no_good_size_percent"] == round(result.no_good_size_share * 100, 1)
    assert f["best_size_share_percent"] == {s: round(v * 100, 1) for s, v in result.size_mix.items()}
    shares = [i["percent_of_that_size_buyers"] for i in f["zones_outside_the_range"]]
    assert shares == sorted(shares, reverse=True) and all(v >= 10 for v in shares) and len(shares) <= 8
    assert f["regrade"]["covered_percent_before"] == round(result.regrade.coverage_before * 100, 1)
    assert all(c["change_cm"] == pytest.approx(c["to_cm"] - c["from_cm"], abs=0.011) for c in f["regrade"]["changes"])


def test_facts_without_regrade_have_no_regrade_key():
    assert "regrade" not in memo_facts(simulate(make_product(Category.shirt), subset(population(), 300), regrade=False))


@pytest.mark.parametrize("make", [
    lambda: simulate(make_product(Category.kurta), population(), regrade=True),
    lambda: simulate(make_product(Category.shirt), population(), regrade=False),
    lambda: simulate(make_product(Category.dress, chart_type="body"), population(), regrade=True),
    lambda: simulate(make_product(Category.trousers), population().reweighted(), regrade=True),
])
def test_the_template_memo_cites_only_numbers_in_the_statistics(make):
    r = make()
    facts = memo_facts(r)
    m = template_memo(r, facts)
    assert m.source == "template" and cites_only_the_facts(m.render(), facts) and not BANNED.search(m.render())


def test_numbers_in_reads_thousands_separators_and_decimals():
    assert numbers_in("4,000 bodies, 27.1% and 36.5%, size 32") == {4000.0, 27.1, 36.5, 32.0}


# ----------------------------------------------------------------------------- the model's memo


def valid_body(result):
    f = memo_facts(result)
    return client_saying("Kurta chart coverage", ("Coverage", f"The chart covers {f['covered_percent']}% of the bodies and {f['no_good_size_percent']}% have no good size."))


def test_a_memo_that_quotes_only_the_statistics_is_used_and_the_caveat_is_added_by_code(result):
    m = write_memo(result, client=valid_body(result))
    assert m.source == "model" and m.sections[0].heading == "Coverage"
    assert m.sections[-1].heading == "Caveat" and m.sections[-1].body == result.banner
    assert "not the SMPL shape space" in m.render()


@pytest.mark.parametrize("bad,why", [
    ("The chart covers 99.9% of the bodies.", "invented number"),
    ("The chart covers 27.1% of bodies, a gain of 9.4 points.", "computed number"),
    ("The chart is a fraud on shoppers.", "banned wording"),
    ("The chart does not flatter a heavier figure.", "body wording"),
])
def test_a_memo_that_breaks_the_rules_is_replaced_by_the_template(result, bad, why):
    m = write_memo(result, client=client_saying("Headline", ("Section", bad)))
    assert m.source == "template", why
    assert bad not in m.render() and cites_only_the_facts(m.render(), memo_facts(result)) and m.sections[-1].heading == "Caveat"


def test_an_empty_or_over_long_memo_is_replaced(result):
    assert write_memo(result, client=client_saying("", ("A", "b"))).source == "template"
    c = MagicMock()
    c.models.generate_content.return_value.text = json.dumps({"headline": "H", "sections": []})
    assert write_memo(result, client=c).source == "template"
    long_body = " ".join(["words"] * (MAX_WORDS + 20))
    assert write_memo(result, client=client_saying("Headline", ("S", long_body))).source == "template"


def test_a_model_error_gives_the_template_and_never_raises(result):
    broken = MagicMock()
    broken.models.generate_content.side_effect = ConnectionError("down")
    m = write_memo(result, client=broken)
    assert m.source == "template" and m.sections[-1].heading == "Caveat" and m.headline


def test_the_model_is_given_the_facts_and_nothing_else(result):
    client = valid_body(result)
    write_memo(result, client=client)
    sent = client.models.generate_content.call_args.kwargs["contents"][0]
    assert json.loads(sent) == memo_facts(result)
    assert "weights" not in sent and "height" not in sent                      # no per-body data, only aggregates


# ----------------------------------------------------------------------------- recorded from the live model


def test_the_recorded_gemini_pro_memo_passes_the_rules_and_the_facts_have_not_drifted():
    """tests/fixtures/llm/sim_memo_kurta.json holds the real pro-model memo for the sample kurta on the stand-in population."""
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    product = Product.model_validate(json.loads((ROOT / "samples" / "product_kurta.json").read_text(encoding="utf-8")))
    r = simulate(product, generate_standin(), regrade=True)
    assert memo_facts(r) == fx["facts"]                                          # the statistics are unchanged since it was recorded
    body = [s for s in fx["memo"]["sections"] if s["heading"] != "Caveat"]
    client = client_saying(fx["memo"]["headline"], *[(s["heading"], s["body"]) for s in body])
    m = write_memo(r, client=client)
    assert m.source == "model" and cites_only_the_facts(m.render().replace(r.banner, ""), fx["facts"])
    assert len(m.render().split()) <= MAX_WORDS + 40
