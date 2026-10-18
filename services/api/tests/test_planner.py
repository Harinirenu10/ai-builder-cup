"""Tests for Planner tools, audience-circle overrides, 90-day repeat rules, and DayPlan caching."""

from __future__ import annotations

import datetime
import time
from unittest.mock import MagicMock, patch

from fitcore.schemas import Category, Circle
from apilib.planner import (
    check_repeat,
    default_demo_events,
    get_weather,
    occasion_poses,
    propose_circle,
)
from apilib.stores import MemoryStore
from tests.conftest import user


def test_propose_circle_keywords():
    assert propose_circle("Executive Client Quarterly Review") == "office"
    assert propose_circle("Weekly Team Standup", attendees=["team@company.com"]) == "office"
    assert propose_circle("Family Wedding Reception Dinner") == "family"
    assert propose_circle("Cousin Sangeet Celebration") == "family"
    assert propose_circle("College Alumni Homecoming Reunion") == "college"
    assert propose_circle("Weekend Drinks with Friends") == "friends"
    assert propose_circle("Dentist Appointment") == "other"


def test_audience_90_day_repeat_rules():
    store = MemoryStore()
    uid = "test-shopper"
    item_id = "item-navy-blazer"
    target_date = "2026-10-18"

    # 1. Never worn item -> allowed, no chips
    allowed, rep = check_repeat(store, uid, item_id, "office", target_date)
    assert allowed is True
    assert rep.last_worn_with_circle_on is None
    assert rep.chip_text is None

    # 2. Worn with SAME circle (office) 10 days ago -> DISALLOWED
    ten_days_ago = (datetime.date.fromisoformat(target_date) - datetime.timedelta(days=10)).isoformat()
    store.set("wear_logs", "log1", {
        "user_id": uid,
        "item_id": item_id,
        "circle": "office",
        "date": ten_days_ago,
    })
    allowed_same, rep_same = check_repeat(store, uid, item_id, "office", target_date)
    assert allowed_same is False
    assert rep_same.last_worn_with_circle_on == ten_days_ago

    # 3. Worn with SAME circle 95 days ago -> ALLOWED (outside 90-day window)
    store_old = MemoryStore()
    ninety_five_days_ago = (datetime.date.fromisoformat(target_date) - datetime.timedelta(days=95)).isoformat()
    store_old.set("wear_logs", "log_old", {
        "user_id": uid,
        "item_id": item_id,
        "circle": "office",
        "date": ninety_five_days_ago,
    })
    allowed_old, rep_old = check_repeat(store_old, uid, item_id, "office", target_date)
    assert allowed_old is True
    assert rep_old.last_worn_with_circle_on is None

    # 4. Worn with DIFFERENT circle (office) 20 days ago, now wearing for family -> ALLOWED WITH CHIP
    twenty_days_ago = (datetime.date.fromisoformat(target_date) - datetime.timedelta(days=20)).isoformat()
    store_diff = MemoryStore()
    store_diff.set("wear_logs", "log_diff", {
        "user_id": uid,
        "item_id": item_id,
        "circle": "office",
        "date": twenty_days_ago,
    })
    allowed_diff, rep_diff = check_repeat(store_diff, uid, item_id, "family", target_date)
    assert allowed_diff is True
    assert rep_diff.last_worn_with_circle_on is None
    assert rep_diff.chip_text is not None
    assert "Last worn with Office on" in rep_diff.chip_text


def test_calendar_events_and_circle_override(client):
    uid = "shopper_events"
    # Fetch events (seeds demo events)
    res = client.get("/events?date=2026-10-18", headers=user(uid))
    assert res.status_code == 200
    events = res.json()
    assert len(events) >= 2
    meeting_ev = next(e for e in events if "meeting" in e["title"].lower())
    assert meeting_ev["circle"] == "office"

    # Override circle via PATCH /events/{id}/circle
    override_res = client.patch(
        f"/events/{meeting_ev['id']}/circle",
        json={"circle": "friends"},
        headers=user(uid),
    )
    assert override_res.status_code == 200
    updated_ev = override_res.json()
    assert updated_ev["circle"] == "friends"


def test_wear_logs_endpoints(client):
    uid = "shopper_wear"
    log_payload = {
        "item_id": "demo-shirt-white",
        "circle": "office",
        "date": "2026-10-15",
        "event_id": "event-client-meeting",
    }
    r_post = client.post("/wear-logs", json=log_payload, headers=user(uid))
    assert r_post.status_code == 200
    entry = r_post.json()
    assert entry["item_id"] == "demo-shirt-white"
    assert entry["circle"] == "office"

    # List wear logs
    r_get = client.get("/wear-logs?item_id=demo-shirt-white", headers=user(uid))
    assert r_get.status_code == 200
    assert len(r_get.json()) == 1


def test_day_plan_caching_and_demo_scenario(client):
    uid = "shopper_plan"
    target_date = "2026-10-19"

    # Seed wear log for cross-circle chip demo:
    # item 'demo-kurta-silk-gold' was worn with 'office' on 2026-10-01
    client.post(
        "/wear-logs",
        json={
            "item_id": "demo-kurta-silk-gold",
            "circle": "office",
            "date": "2026-10-01",
        },
        headers=user(uid),
    )

    # 1. First call: generated plan (cached == False)
    res1 = client.get(f"/plan?date={target_date}", headers=user(uid))
    assert res1.status_code == 200
    plan1 = res1.json()
    assert plan1["date"] == target_date
    assert plan1["cached"] is False
    assert len(plan1["events"]) >= 2

    # Verify Demo Scenario requirements:
    # Event A: 10:00 client meeting
    meeting_plan = next(e for e in plan1["events"] if "meeting" in e["title"].lower())
    assert meeting_plan["dress_code"] == "Business Formal"
    assert len(meeting_plan["outfit_item_ids"]) > 0
    assert len(meeting_plan["reasons"]) > 0

    # Event B: 19:00 wedding reception
    reception_plan = next(e for e in plan1["events"] if "wedding" in e["title"].lower())
    assert "Ethnic" in reception_plan["dress_code"]
    assert len(reception_plan["pack_list"]) > 0  # pack list for reception change
    assert any("garment bag" in item.lower() for item in reception_plan["pack_list"])

    # Verify cross-circle chip for reception plan
    assert reception_plan["repeat_check"]["chip_text"] is not None
    assert "Last worn with Office" in reception_plan["repeat_check"]["chip_text"]

    # 2. Second call: should return from Store cache (cached == True)
    res2 = client.get(f"/plan?date={target_date}", headers=user(uid))
    assert res2.status_code == 200
    plan2 = res2.json()
    assert plan2["cached"] is True
    assert plan2["events"][0]["event_id"] == plan1["events"][0]["event_id"]

    # 3. Third call with refresh=True: should bypass cache (cached == False)
    res3 = client.get(f"/plan?date={target_date}&refresh=true", headers=user(uid))
    assert res3.status_code == 200
    plan3 = res3.json()
    assert plan3["cached"] is False


def test_stubbed_planner_tools():
    # Prompt 5 occasion_poses stub (backward-compatible when called with ID only)
    poses = occasion_poses("event-client-meeting")
    assert poses["event_id"] == "event-client-meeting"
    assert poses["status"] == "stubbed_prompt_5"
    assert "standing" in poses
    assert "seated" in poses

    # Weather returns "unknown" when API unavailable
    weather = get_weather("Downtown", "2026-10-18")
    assert weather == "unknown"


def test_occasion_poses_with_event():
    from fitcore.schemas import CalendarEvent

    # Seated event keyword: meeting (2 hours = 120 minutes)
    meeting_ev = CalendarEvent(
        id="ev-exec-meeting",
        title="Quarterly Review Meeting",
        start_time="2026-10-20T10:00:00",
        end_time="2026-10-20T12:00:00",
        location="Boardroom",
        attendees=["board@corp.com"],
        circle="office",
        description="Review presentation and strategy discussion.",
    )
    seated_result = occasion_poses("ev-exec-meeting", event=meeting_ev)
    assert seated_result["event_id"] == "ev-exec-meeting"
    assert seated_result["status"] == "deterministic_estimate"
    assert seated_result["seated"] == 120.0
    assert seated_result["standing"] == 0.0

    # Standing event keyword: reception (3 hours = 180 minutes)
    reception_ev = CalendarEvent(
        id="ev-gala-reception",
        title="Charity Gala Reception",
        start_time="2026-10-20T19:00:00",
        end_time="2026-10-20T22:00:00",
        location="Grand Ballroom",
        attendees=["guests@gala.org"],
        circle="friends",
        description="Standing cocktail reception and mixer.",
    )
    standing_result = occasion_poses("ev-gala-reception", event=reception_ev)
    assert standing_result["event_id"] == "ev-gala-reception"
    assert standing_result["status"] == "deterministic_estimate"
    assert standing_result["standing"] == 180.0
    assert standing_result["seated"] == 0.0


def test_propose_circle_with_llm_and_fallback(monkeypatch):
    from unittest.mock import MagicMock
    from apilib.planner import propose_circle_with_llm

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "gemini-exp-custom")
    monkeypatch.setenv("LLM_BACKEND", "vertex")

    # 1. LLM returns valid proposal
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value.text = (
        '{"circle": "college", "confidence": 0.95, "reason": "Alumni reunion on campus"}'
    )
    circle = propose_circle_with_llm(
        "Alumni Reunion Dinner",
        attendees=["classof2020@university.edu"],
        client=mock_client,
    )
    assert circle == "college"

    # 2. LLM fails / raises exception -> fallback to deterministic keywords
    broken_client = MagicMock()
    broken_client.models.generate_content.side_effect = RuntimeError("Service connection error")
    fallback_circle = propose_circle_with_llm(
        "Family Wedding Reception",
        attendees=["cousins@family.net"],
        client=broken_client,
    )
    assert fallback_circle == "family"


def test_all_six_tools_in_isolation():
    from apilib.planner import (
        ensure_demo_wardrobe,
        get_events,
        get_wear_log,
        get_weather,
        log_outfit,
        occasion_poses,
        search_wardrobe,
    )

    store = MemoryStore()
    uid = "tool_tester"
    date_str = "2026-10-18"

    # Tool 1: get_events
    events = get_events(date_str, uid=uid, store=store)
    assert len(events) >= 2
    assert any("meeting" in e.title.lower() for e in events)

    # Tool 2: get_weather
    weather = get_weather("New Delhi", date_str)
    assert weather == "unknown"

    # Tool 3: search_wardrobe
    ensure_demo_wardrobe(store, uid)
    formal_items = search_wardrobe(uid=uid, store=store, formality_min=4)
    assert len(formal_items) >= 2
    ethnic_items = search_wardrobe(uid=uid, store=store, ethnic_or_western="ethnic")
    assert all(i.ethnic_or_western == "ethnic" for i in ethnic_items)

    # Tool 4: log_outfit
    logged = log_outfit(
        event_id="event-client-meeting",
        item_ids=["demo-shirt-white", "demo-trousers-navy"],
        circle="office",
        date_str=date_str,
        uid=uid,
        store=store,
    )
    assert len(logged) == 2

    # Tool 5: get_wear_log
    history = get_wear_log(item_id="demo-shirt-white", uid=uid, store=store)
    assert len(history) == 1
    assert history[0].circle == "office"

    # Tool 6: occasion_poses (stubbed until Prompt 5)
    poses = occasion_poses("event-wedding-reception")
    assert poses["event_id"] == "event-wedding-reception"
    assert poses["status"] == "stubbed_prompt_5"
    assert poses["standing"] == 60.0


def test_planner_agent_with_recorded_response(monkeypatch):
    import json
    from pathlib import Path
    from unittest.mock import MagicMock
    from apilib.planner import run_planner_agent

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "gemini-exp-custom")
    monkeypatch.setenv("LLM_BACKEND", "vertex")

    store = MemoryStore()
    uid = "shopper_agent_test"
    target_date = "2026-10-19"

    fixture_path = Path(__file__).resolve().parent / "fixtures" / "llm" / "planner_day_plan.json"
    recorded_json = fixture_path.read_text(encoding="utf-8")

    # Mock client returning recorded Gemini Flash response
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value.text = recorded_json

    agent_plan = run_planner_agent(uid=uid, date_str=target_date, store=store, client=mock_client)
    assert agent_plan.date == target_date
    assert len(agent_plan.events) == 2

    meeting = next(e for e in agent_plan.events if "meeting" in e.title.lower())
    assert meeting.dress_code == "Business Formal"
    assert "demo-shirt-white" in meeting.outfit_item_ids
    assert len(meeting.reasons) > 0

    reception = next(e for e in agent_plan.events if "wedding" in e.title.lower())
    assert "Ethnic" in reception.dress_code
    assert len(reception.pack_list) > 0
    assert reception.repeat_check.chip_text is not None
    assert "Last worn with Office" in reception.repeat_check.chip_text


def test_calendar_oauth_connect_and_state_binding(client, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/calendar/callback")

    uid = "calendar_user_1"
    res = client.get("/auth/calendar/connect", headers=user(uid))
    assert res.status_code == 200
    data = res.json()
    assert "auth_url" in data
    assert "state" in data
    assert "accounts.google.com" in data["auth_url"]
    assert "calendar.readonly" in data["auth_url"]
    assert f"state={data['state']}" in data["auth_url"]


def test_calendar_oauth_callback_security_and_storage(client, monkeypatch):
    from unittest.mock import MagicMock
    from apilib.calendar_auth import oauth_callback

    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/calendar/callback")

    uid = "calendar_user_2"
    # 1. Connect to generate state
    conn_res = client.get("/auth/calendar/connect", headers=user(uid))
    state = conn_res.json()["state"]

    # 2. Callback with mismatched user -> 403
    mismatch_res = client.get(f"/auth/calendar/callback?code=mock-code&state={state}", headers=user("different_user"))
    assert mismatch_res.status_code == 403

    # 3. Callback with invalid state -> 400
    bad_state_res = client.get("/auth/calendar/callback?code=mock-code&state=nonexistent-state", headers=user(uid))
    assert bad_state_res.status_code == 400

    # 4. Valid callback with mocked Google token exchange
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "access_token": "ya29.mock_access_token",
        "refresh_token": "1//mock_refresh_token",
        "expires_in": 3600,
        "scope": "https://www.googleapis.com/auth/calendar.readonly",
    }
    mock_session.post.return_value = mock_resp

    with patch("apilib.calendar_auth.requests", mock_session):
        success_res = client.get(f"/auth/calendar/callback?code=good-code&state={state}", headers=user(uid))
        assert success_res.status_code == 200
        data = success_res.json()
        assert data["status"] == "connected"
        assert "access_token" not in data  # Never expose tokens to client!
        assert "refresh_token" not in data

        # 5. Single-use state verification: re-using the same state must be rejected with 400
        reuse_res = client.get(f"/auth/calendar/callback?code=good-code&state={state}", headers=user(uid))
        assert reuse_res.status_code == 400

        # 6. Check status endpoint: connected without exposing tokens
        status_res = client.get("/auth/calendar/status", headers=user(uid))
        assert status_res.status_code == 200
        assert status_res.json()["connected"] is True
        assert "access_token" not in status_res.json()

        # 7. Disconnect endpoint
        disc_res = client.post("/auth/calendar/disconnect", headers=user(uid))
        assert disc_res.status_code == 200
        assert disc_res.json()["status"] == "disconnected"

        # 8. Re-check status: now disconnected
        status_after = client.get("/auth/calendar/status", headers=user(uid))
        assert status_after.json()["connected"] is False


def test_calendar_seven_days_retrieval_and_demo_fallback():
    from unittest.mock import MagicMock
    from apilib.calendar_auth import fetch_google_calendar_events

    store = MemoryStore()
    uid = "calendar_sync_user"

    # 1. Not connected -> returns None
    assert fetch_google_calendar_events(uid, store) is None

    # 2. Connected with valid token -> returns mapped 7-day events
    store.set("calendar_tokens", uid, {
        "access_token": "ya29.valid_token",
        "expires_at": time.time() + 3600,
        "scope": "calendar.readonly",
    })

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "items": [
            {
                "id": "gcal_event_1",
                "summary": "Quarterly Client Strategy Review",
                "start": {"dateTime": "2026-10-20T10:00:00Z"},
                "end": {"dateTime": "2026-10-20T11:00:00Z"},
                "location": "Boardroom Alpha",
                "attendees": [{"email": "client@enterprise.com"}],
                "description": "Executive quarterly business review",
            },
            {
                "id": "gcal_event_2",
                "summary": "Cousin Wedding Reception",
                "start": {"dateTime": "2026-10-22T19:00:00Z"},
                "end": {"dateTime": "2026-10-22T22:00:00Z"},
                "location": "The Leela Palace",
                "attendees": [{"email": "family@relatives.org"}],
                "description": "Family reception and dinner",
            },
        ]
    }
    mock_client.get.return_value = mock_resp

    events = fetch_google_calendar_events(uid, store, start_iso="2026-10-20T00:00:00Z", http_client=mock_client)
    assert events is not None
    assert len(events) == 2
    assert events[0].id == "gcal_event_1"
    assert events[0].circle == "office"
    assert events[1].circle == "family"


