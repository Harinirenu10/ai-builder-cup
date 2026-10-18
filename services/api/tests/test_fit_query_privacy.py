"""The business-facing response must never carry measurements or shape parameters. These tests make that a build failure."""

import json
import re

import pytest

from apilib.models import FitQueryResponse, GiftGiverView
from tests.conftest import KEY, ORIGIN, user

ALLOWED_FIELDS = {"recommended_size", "confidence", "zone_verdicts", "alteration_hint", "guarantee_eligible"}
ZONE_VERDICT_FIELDS = {"zone", "verdict"}
FORBIDDEN_WORDS = re.compile(r"cm|inch|measure|ease|margin|shrink|shape|beta|pose|mesh|height|weight|value|girth|circumference|body|twin|photo|uid|owner", re.I)


def schema_property_names(schema: dict, defs: dict | None = None, seen=None) -> set[str]:
    """Every property name reachable from a JSON schema, following $refs."""
    defs = defs if defs is not None else schema.get("$defs", {})
    seen = seen or set()
    names: set[str] = set()
    for name, sub in schema.get("properties", {}).items():
        names.add(name)
        names |= _walk(sub, defs, seen)
    return names


def _walk(node: dict, defs: dict, seen: set) -> set[str]:
    out: set[str] = set()
    if "$ref" in node:
        ref = node["$ref"].split("/")[-1]
        if ref not in seen:
            seen.add(ref)
            out |= schema_property_names(defs[ref], defs, seen)
    for key in ("items", "additionalProperties"):
        if isinstance(node.get(key), dict):
            out |= _walk(node[key], defs, seen)
    for key in ("anyOf", "allOf", "oneOf"):
        for sub in node.get(key, []):
            out |= _walk(sub, defs, seen)
    if "properties" in node:
        out |= schema_property_names(node, defs, seen)
    return out


def test_response_schema_has_exactly_the_agreed_fields():
    names = schema_property_names(FitQueryResponse.model_json_schema())
    assert names == ALLOWED_FIELDS | ZONE_VERDICT_FIELDS


def test_response_schema_names_hold_no_measurement_words():
    for name in schema_property_names(FitQueryResponse.model_json_schema()):
        assert not FORBIDDEN_WORDS.search(name), name


def test_the_openapi_schema_of_the_endpoint_matches(client):
    spec = client.get("/openapi.json").json()
    op = spec["paths"]["/v1/fit-query"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
    ref = op["$ref"].split("/")[-1]
    names = schema_property_names(spec["components"]["schemas"][ref], spec["components"]["schemas"])
    assert names == ALLOWED_FIELDS | ZONE_VERDICT_FIELDS


def test_gift_giver_view_schema_is_only_size_and_confidence():
    assert schema_property_names(GiftGiverView.model_json_schema()) == {"id", "status", "recommended_size", "confidence"}


def _all_keys(obj) -> set[str]:
    if isinstance(obj, dict):
        return set(obj) | {k for v in obj.values() for k in _all_keys(v)}
    if isinstance(obj, list):
        return {k for v in obj for k in _all_keys(v)}
    return set()


@pytest.mark.parametrize("sleeve_override", [None, 63.5])
def test_real_response_has_no_body_fields_or_numbers(client, product, consent, store, sleeve_override):
    if sleeve_override:
        product["size_chart"]["sizes"]["L"]["sleeve_length"] = sleeve_override
    r = client.post("/v1/fit-query", json={"product": product, "consent_token": consent["token"]}, headers={"X-Merchant-Key": KEY, "Origin": ORIGIN})
    body = r.json()
    assert set(body) == ALLOWED_FIELDS
    assert _all_keys(body) <= ALLOWED_FIELDS | ZONE_VERDICT_FIELDS
    # None of the twin's measurements appear anywhere in the response text.
    twin = store.get("twins", "alice")["measurements"]["zones"]
    text = json.dumps(body)
    for z in twin.values():
        assert not re.search(rf"(?<![\d.]){re.escape(str(z['value_cm']))}(?![\d.])", text), z
    # Only the confidence is numeric (the hint may name the garment change, e.g. 2.5 cm).
    assert isinstance(body["confidence"], float) and all(not isinstance(v["verdict"], (int, float)) for v in body["zone_verdicts"])


def test_error_responses_do_not_leak_body_data(client, product, consent):
    bad = client.post("/v1/fit-query", json={"product": product, "consent_token": consent["token"]}, headers={"X-Merchant-Key": KEY, "Origin": "https://evil.example"})
    assert "chest" not in bad.text and "value_cm" not in bad.text
    client.post(f"/v1/consents/{consent['consent_id']}/revoke", headers=user("alice"))
    revoked = client.post("/v1/fit-query", json={"product": product, "consent_token": consent["token"]}, headers={"X-Merchant-Key": KEY})
    assert revoked.status_code == 403 and "chest" not in revoked.text
