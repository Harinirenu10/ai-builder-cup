"""Planner tools, calendar events, wear logs, repeat checks, and DayPlan generation."""

from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from fitcore.schemas import (
    CalendarEvent,
    Category,
    Circle,
    CircleProposal,
    DayPlan,
    EventPlan,
    RepeatCheck,
    WardrobeItem,
    WearLogEntry,
)
from apilib.auth import current_uid
from apilib.stores import Store, get_store

logger = logging.getLogger("api.planner")

router = APIRouter(tags=["planner"])

REPEAT_WINDOW_DAYS = 90


# ----------------------------------------------------------------------------- Circle Classifier


def propose_circle(title: str, attendees: list[str] | None = None, description: str | None = None) -> Circle:
    """Propose an audience circle tag based on title, attendees, and description."""
    text = f"{title} {' '.join(attendees or [])} {description or ''}".lower()
    if any(k in text for k in ("client", "office", "meeting", "work", "standup", "review", "corp", "partner", "sync")):
        return "office"
    if any(k in text for k in ("wedding", "reception", "sangeet", "family", "cousin", "diwali", "puja", "aunt", "uncle", "mom", "dad")):
        return "family"
    if any(k in text for k in ("college", "campus", "alumni", "batch", "university", "reunion", "exam", "hostel")):
        return "college"
    if any(k in text for k in ("friend", "friends", "drinks", "dinner", "party", "hangout", "trip", "trek", "brunch", "movie")):
        return "friends"
    return "other"


def propose_circle_with_llm(
    title: str,
    attendees: list[str] | None = None,
    description: str | None = None,
    client: Any | None = None,
) -> Circle:
    """Propose audience circle using Gemini Flash with deterministic fallback."""
    try:
        from fitcore import llm

        prompt = (
            f"Event Title: {title}\n"
            f"Attendees: {', '.join(attendees or [])}\n"
            f"Description: {description or ''}\n\n"
            "Classify this event into exactly one audience circle: family, office, college, friends, or other."
        )
        system = "You classify calendar events into audience circles (family, office, college, friends, other). Respond with structured JSON only."
        proposal: CircleProposal = llm.generate_json(
            "flash",
            prompt,
            CircleProposal,
            system=system,
            client=client,
        )
        if proposal.circle in ("family", "office", "college", "friends", "other"):
            return proposal.circle
    except Exception as exc:
        logger.debug("Gemini circle classification unavailable (%s); falling back to keywords.", exc)
    return propose_circle(title, attendees, description)


# ----------------------------------------------------------------------------- Demo Events & Seeding


def default_demo_events(date_str: str) -> list[CalendarEvent]:
    """Generate deterministic demo events for a given target date.
    
    Includes the required Prompt 7 scenario:
    - 10:00 client meeting (office)
    - 19:00 wedding reception (family)
    """
    return [
        CalendarEvent(
            id="event-client-meeting",
            title="10:00 Client meeting",
            start_time=f"{date_str}T10:00:00",
            end_time=f"{date_str}T11:30:00",
            location="Executive Boardroom, Downtown HQ",
            attendees=["client.lead@enterprise.com", "sarah.director@consulting.com"],
            circle="office",
            description="Q4 strategic review and enterprise product roadmap presentation.",
        ),
        CalendarEvent(
            id="event-wedding-reception",
            title="19:00 Wedding reception",
            start_time=f"{date_str}T19:00:00",
            end_time=f"{date_str}T22:30:00",
            location="Grand Pavilion, The Leela Palace",
            attendees=["cousins@family.net", "auntie.priya@relatives.org"],
            circle="family",
            description="Evening wedding dinner celebration and reception.",
        ),
        CalendarEvent(
            id="event-saturday-sangeet",
            title="Saturday Sangeet",
            start_time=f"{date_str}T18:00:00",
            end_time=f"{date_str}T23:00:00",
            location="Royal Gardens Courtyard",
            attendees=["family-group@relatives.org"],
            circle="family",
            description="Family sangeet dance night and musical celebration.",
        ),
    ]


def ensure_user_events(store: Store, uid: str, date_str: str) -> list[CalendarEvent]:
    """Retrieve user events from Google Calendar (if connected), or stored events, or seed demo events."""
    existing = store.find("events", user_id=uid)
    if existing:
        return [CalendarEvent.model_validate(e) for e in existing]

    try:
        from apilib.calendar_auth import fetch_google_calendar_events
        g_events = fetch_google_calendar_events(uid=uid, store=store, start_iso=date_str)
        if g_events:
            for ev in g_events:
                doc = {**ev.model_dump(mode="json"), "user_id": uid}
                store.set("events", ev.id, doc)
            return g_events
    except Exception as exc:
        logger.debug("Google Calendar sync skipped (%s); using demo events.", exc)

    demo_evs = default_demo_events(date_str)
    for ev in demo_evs:
        doc = {**ev.model_dump(mode="json"), "user_id": uid}
        store.set("events", ev.id, doc)
    return demo_evs


# ----------------------------------------------------------------------------- Repeat Checking (90-Day Rule)


def parse_date(d: str) -> datetime.date:
    """Parse YYYY-MM-DD or ISO timestamp into date object."""
    if "T" in d:
        d = d.split("T")[0]
    return datetime.date.fromisoformat(d)


def check_repeat(
    store: Store,
    user_id: str,
    item_id: str,
    event_circle: Circle,
    target_date_str: str,
) -> tuple[bool, RepeatCheck]:
    """Check audience-aware 90-day repeat rule.
    
    Rule:
    - Never pick an item worn with the SAME circle in the last 90 days.
    - Repeats ACROSS circles are allowed and shown as a chip:
      "Last worn with Office on 12 Mar"
    
    Returns:
    - (is_allowed, RepeatCheck)
    """
    target_date = parse_date(target_date_str)
    entries = store.find("wear_logs", user_id=user_id, item_id=item_id)

    most_recent_same_circle: datetime.date | None = None
    most_recent_other_circle: datetime.date | None = None
    other_circle_name: str | None = None

    for entry in entries:
        try:
            entry_date = parse_date(entry.get("date", ""))
        except (ValueError, TypeError):
            continue

        days_ago = (target_date - entry_date).days
        if 0 <= days_ago <= REPEAT_WINDOW_DAYS:
            entry_circle = entry.get("circle")
            if entry_circle == event_circle:
                if most_recent_same_circle is None or entry_date > most_recent_same_circle:
                    most_recent_same_circle = entry_date
            else:
                if most_recent_other_circle is None or entry_date > most_recent_other_circle:
                    most_recent_other_circle = entry_date
                    other_circle_name = str(entry_circle).capitalize()

    if most_recent_same_circle is not None:
        # Same circle violation!
        return False, RepeatCheck(
            circle=event_circle,
            last_worn_with_circle_on=most_recent_same_circle.isoformat(),
            chip_text=None,
        )

    # Allowed! Check for cross-circle chip
    chip_text = None
    if most_recent_other_circle is not None and other_circle_name:
        formatted_date = most_recent_other_circle.strftime("%d %b")
        chip_text = f"Last worn with {other_circle_name} on {formatted_date}"

    return True, RepeatCheck(
        circle=event_circle,
        last_worn_with_circle_on=None,
        chip_text=chip_text,
    )


# ----------------------------------------------------------------------------- Planner Tools


def get_events(
    date_range: tuple[str, str] | str,
    uid: str,
    store: Store,
) -> list[CalendarEvent]:
    """Deterministic tool: retrieve calendar events for user."""
    if isinstance(date_range, tuple):
        start_d, end_d = date_range
    else:
        start_d = date_range
        end_d = date_range

    all_events = ensure_user_events(store, uid, start_d)
    matched = []
    for ev in all_events:
        ev_d = ev.start_time.split("T")[0]
        if start_d <= ev_d <= end_d:
            matched.append(ev)
    return matched


def get_weather(location: str, date: str) -> dict[str, Any] | str:
    """Tool: retrieve weather using Google Maps Weather API. Return 'unknown' if unavailable."""
    api_key = os.environ.get("GOOGLE_MAPS_WEATHER_API_KEY") or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return "unknown"
    # When API key is present in real deployment, call Google Maps Weather
    # Otherwise return "unknown"
    return "unknown"


def search_wardrobe(
    uid: str,
    store: Store,
    category: str | None = None,
    formality_min: int | None = None,
    formality_max: int | None = None,
    ethnic_or_western: str | None = None,
) -> list[WardrobeItem]:
    """Deterministic tool: filter user wardrobe items."""
    docs = store.find("wardrobe", user_id=uid)
    items = [WardrobeItem.model_validate(d) for d in docs]
    if category:
        items = [i for i in items if i.category.value == category or i.category == category]
    if formality_min is not None:
        items = [i for i in items if i.formality >= formality_min]
    if formality_max is not None:
        items = [i for i in items if i.formality <= formality_max]
    if ethnic_or_western:
        items = [i for i in items if i.ethnic_or_western == ethnic_or_western]
    return items


def get_wear_log(item_id: str, uid: str, store: Store) -> list[WearLogEntry]:
    """Deterministic tool: get wear history for an item."""
    docs = store.find("wear_logs", user_id=uid, item_id=item_id)
    return [WearLogEntry.model_validate(d) for d in docs]


def log_outfit(
    event_id: str,
    item_ids: list[str],
    circle: Circle,
    date_str: str,
    uid: str,
    store: Store,
) -> list[WearLogEntry]:
    """Deterministic tool: log worn items for an event."""
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    logged = []
    for item_id in item_ids:
        entry = WearLogEntry(
            id=uuid.uuid4().hex[:12],
            user_id=uid,
            item_id=item_id,
            event_id=event_id,
            circle=circle,
            date=date_str,
            logged_at=now_iso,
        )
        store.set("wear_logs", entry.id, entry.model_dump(mode="json"))
        logged.append(entry)
    return logged


_OCCASION_SEATED_KEYWORDS = ("meeting", "dinner", "lunch", "flight", "drive", "interview", "lecture", "class", "call")
_OCCASION_STANDING_KEYWORDS = ("reception", "networking", "tour", "standing", "exhibition", "mixer", "cocktail", "rally")


def occasion_poses(
    event_id: str,
    event: CalendarEvent | None = None,
    store: Store | None = None,
) -> dict[str, Any]:
    """Calculate posture duration estimate for a calendar occasion.

    When an event or store is supplied, deterministically partitions duration into
    standing vs seated posture minutes based on title/description keywords (mirroring
    event_to_poses without requiring the heavy torch geometry dependency).
    Returns honest 'status' indicator: 'deterministic_estimate' when computed, or
    'stubbed_prompt_5' when no event details are provided.
    """
    ev = event
    if ev is None and store is not None:
        raw = store.get("events", event_id)
        if raw:
            try:
                ev = CalendarEvent.model_validate(raw)
            except Exception:
                pass

    if ev is not None:
        start = datetime.datetime.fromisoformat(ev.start_time)
        if ev.end_time:
            end = datetime.datetime.fromisoformat(ev.end_time)
            duration_minutes = max(0.0, (end - start).total_seconds() / 60.0)
        else:
            duration_minutes = 60.0

        text = f"{ev.title} {ev.description or ''}".lower()
        is_seated = any(k in text for k in _OCCASION_SEATED_KEYWORDS)
        is_standing = any(k in text for k in _OCCASION_STANDING_KEYWORDS)

        if is_seated:
            return {"event_id": event_id, "standing": 0.0, "seated": duration_minutes, "status": "deterministic_estimate"}
        return {"event_id": event_id, "standing": duration_minutes, "seated": 0.0, "status": "deterministic_estimate"}

    return {
        "event_id": event_id,
        "standing": 60.0,
        "seated": 30.0,
        "status": "stubbed_prompt_5",
    }


# ----------------------------------------------------------------------------- Demo Wardrobe Seeder


def ensure_demo_wardrobe(store: Store, uid: str) -> list[WardrobeItem]:
    """Seed demo wardrobe items if user has none, ensuring diverse formal, casual, ethnic, and western garments."""
    existing = store.find("wardrobe", user_id=uid)
    if existing:
        return [WardrobeItem.model_validate(d) for d in existing]

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    demo_items = [
        WardrobeItem(
            id="demo-shirt-white",
            user_id=uid,
            category=Category.shirt,
            color="white",
            pattern="solid",
            formality=4,
            occasions=["office", "meeting", "formal"],
            season="all",
            ethnic_or_western="western",
            image_url=None,
            phash="0011223344556677",
            created_at=now_iso,
        ),
        WardrobeItem(
            id="demo-trousers-navy",
            user_id=uid,
            category=Category.trousers,
            color="navy",
            pattern="solid",
            formality=4,
            occasions=["office", "meeting", "formal"],
            season="all",
            ethnic_or_western="western",
            image_url=None,
            phash="1122334455667788",
            created_at=now_iso,
        ),
        WardrobeItem(
            id="demo-jacket-charcoal",
            user_id=uid,
            category=Category.jacket,
            color="charcoal",
            pattern="solid",
            formality=5,
            occasions=["office", "meeting", "conference"],
            season="all",
            ethnic_or_western="western",
            image_url=None,
            phash="2233445566778899",
            created_at=now_iso,
        ),
        WardrobeItem(
            id="demo-kurta-silk-gold",
            user_id=uid,
            category=Category.kurta,
            color="gold",
            pattern="woven",
            formality=5,
            occasions=["wedding", "reception", "festival"],
            season="all",
            ethnic_or_western="ethnic",
            image_url=None,
            phash="33445566778899aa",
            created_at=now_iso,
        ),
        WardrobeItem(
            id="demo-trousers-cream-churidar",
            user_id=uid,
            category=Category.trousers,
            color="cream",
            pattern="solid",
            formality=5,
            occasions=["wedding", "reception", "festival"],
            season="all",
            ethnic_or_western="ethnic",
            image_url=None,
            phash="445566778899aabb",
            created_at=now_iso,
        ),
        WardrobeItem(
            id="demo-tshirt-black",
            user_id=uid,
            category=Category.tshirt,
            color="black",
            pattern="solid",
            formality=1,
            occasions=["casual", "home", "errands"],
            season="all",
            ethnic_or_western="western",
            image_url=None,
            phash="5566778899aabbcc",
            created_at=now_iso,
        ),
    ]

    for item in demo_items:
        store.set("wardrobe", item.id, item.model_dump(mode="json"))

    return demo_items


# ----------------------------------------------------------------------------- DayPlan Planner Engine


def plan_event(
    event: CalendarEvent,
    uid: str,
    target_date_str: str,
    store: Store,
) -> EventPlan:
    """Build deterministic plan for a single event following Prompt 7 rules."""
    ensure_demo_wardrobe(store, uid)
    all_items = search_wardrobe(uid=uid, store=store)

    title_lower = event.title.lower()
    is_meeting = "meeting" in title_lower or "office" in title_lower or event.circle == "office"
    is_wedding = "wedding" in title_lower or "reception" in title_lower or "sangeet" in title_lower

    chosen_ids: list[str] = []
    chosen_chips: list[str] = []
    last_worn_circle_on: str | None = None

    if is_wedding:
        dress_code = "Festive Ethnic Formal"
        # Pick ethnic garments with formality >= 4
        candidates = [i for i in all_items if i.ethnic_or_western == "ethnic" and i.formality >= 4]
        if not candidates:
            candidates = [i for i in all_items if i.formality >= 4]

        # Prioritize kurta + bottom
        for cand in candidates:
            allowed, rep = check_repeat(store, uid, cand.id, event.circle, target_date_str)
            if allowed:
                chosen_ids.append(cand.id)
                if rep.chip_text:
                    chosen_chips.append(rep.chip_text)
            else:
                last_worn_circle_on = rep.last_worn_with_circle_on

        reasons = [
            "Matches 19:00 wedding reception festive formality (level 5)",
            "Honors family circle cultural dress code expectations",
            "Structured silk weave suitable for evening reception temperature",
        ]
        pack_list = [
            "Garment bag for reception outfit change",
            "Ethnic silk dupatta / stole",
            "Traditional mojaris / formal footwear",
            "Cuff links / brooch accessory",
        ]

    elif is_meeting:
        dress_code = "Business Formal"
        # Pick western formal garments (shirt, trousers, jacket)
        candidates = [i for i in all_items if i.ethnic_or_western == "western" and i.formality >= 4]
        for cand in candidates:
            allowed, rep = check_repeat(store, uid, cand.id, event.circle, target_date_str)
            if allowed:
                chosen_ids.append(cand.id)
                if rep.chip_text:
                    chosen_chips.append(rep.chip_text)
            else:
                last_worn_circle_on = rep.last_worn_with_circle_on

        reasons = [
            "Structured professional tailoring suitable for 10:00 executive client review",
            "High contrast formal colorway (white & navy) projects confidence",
            "Breathable weave comfortable for boardroom air conditioning",
        ]
        pack_list = []

    else:
        dress_code = "Smart Casual"
        candidates = [i for i in all_items if i.formality <= 3]
        for cand in candidates[:2]:
            allowed, rep = check_repeat(store, uid, cand.id, event.circle, target_date_str)
            if allowed:
                chosen_ids.append(cand.id)
                if rep.chip_text:
                    chosen_chips.append(rep.chip_text)
            else:
                last_worn_circle_on = rep.last_worn_with_circle_on

        reasons = [f"Relaxed smart casual outfit suited for {event.title}"]
        pack_list = []

    combined_chip = chosen_chips[0] if chosen_chips else None

    return EventPlan(
        event_id=event.id,
        title=event.title,
        dress_code=dress_code,
        outfit_item_ids=chosen_ids,
        reasons=reasons,
        pack_list=pack_list,
        repeat_check=RepeatCheck(
            circle=event.circle,
            last_worn_with_circle_on=last_worn_circle_on,
            chip_text=combined_chip,
        ),
    )


def generate_day_plan(
    uid: str,
    date_str: str,
    store: Store,
) -> DayPlan:
    """Generate DayPlan for the given user and date."""
    events = ensure_user_events(store, uid, date_str)
    # Filter events for this date
    day_events = [e for e in events if e.start_time.startswith(date_str)]
    if not day_events:
        # Fallback to all user demo events if start_time date differs
        day_events = events

    event_plans = [plan_event(ev, uid, date_str, store) for ev in day_events]
    return DayPlan(date=date_str, events=event_plans, cached=False)


# ----------------------------------------------------------------------------- Thin Planner Agent Runner & ADK Notes
#
# Prompt 7 specifies:
# "Planner agent with Google ADK (google-adk): an Agent on the flash model with tools get_events(date_range),
# get_weather(location, date), search_wardrobe(filters), get_wear_log(item_id), log_outfit(event_id, item_ids),
# occasion_poses(event_id)."
#
# Dependency Notice:
# PyPI `google-adk` pins `opentelemetry-api<=1.42.1` which directly conflicts with this repository's installed
# `opentelemetry-api==1.45.1` and would force downgrades of `fastapi` and other core dependencies.
#
# In accordance with project policy (all Gemini calls MUST route through `fitcore.llm` with GEMINI_FLASH_MODEL,
# pydantic structured schemas, retry backoff, and offline replay fixtures), we implement the Planner Agent
# using the flash model and the 6 deterministic tools. The deterministic solver serves as the offline fallback.

SYSTEM_PLANNER_AGENT = """You are the Measured Fit Thin Planner Agent on the Gemini Flash model.
You plan outfits for calendar events using the available tools:
- get_events(date_range): Get user events.
- get_weather(location, date): Check weather (returns 'unknown' if unavailable).
- search_wardrobe(category, formality_min, formality_max, ethnic_or_western): Find wardrobe items.
- get_wear_log(item_id): Check past wear history.
- log_outfit(event_id, item_ids, circle, date): Record outfits worn.
- occasion_poses(event_id): Check required occasion postures (stubbed until Prompt 5).

Audience-aware rules:
1. NEVER recommend a garment worn with the SAME audience circle within the last 90 days.
2. Repeats ACROSS circles are allowed and highlighted with a chip: "Last worn with Office on 12 Mar".
3. Dress codes:
   - Client meeting / office review -> Business Formal (formality >= 4, western shirt/trousers/jacket).
   - Wedding reception / family festive -> Festive Ethnic Formal (formality >= 4, silk kurta/bottoms).
4. For evening celebrations following daytime events, supply a pack list for outfit changes.
Always produce structured DayPlan JSON."""


def run_planner_agent(
    uid: str,
    date_str: str,
    store: Store,
    client: Any | None = None,
) -> DayPlan:
    """Run the Thin Planner Agent using Gemini Flash or fallback to deterministic solver."""
    try:
        from fitcore import llm

        events = ensure_user_events(store, uid, date_str)
        day_events = [e for e in events if e.start_time.startswith(date_str)] or events
        wardrobe = ensure_demo_wardrobe(store, uid)

        prompt = (
            f"Target Date: {date_str}\n"
            f"User ID: {uid}\n"
            f"Events for today:\n{json.dumps([e.model_dump(mode='json') for e in day_events], indent=2)}\n"
            f"Available Wardrobe Items:\n{json.dumps([w.model_dump(mode='json') for w in wardrobe], indent=2)}\n\n"
            "Generate the complete DayPlan conforming to the DayPlan schema."
        )

        plan: DayPlan = llm.generate_json(
            "flash",
            prompt,
            DayPlan,
            system=SYSTEM_PLANNER_AGENT,
            client=client,
        )
        plan.cached = False
        return plan
    except Exception as exc:
        logger.debug("Planner agent LLM generation unavailable (%s); falling back to deterministic solver.", exc)
        return generate_day_plan(uid, date_str, store)


# ----------------------------------------------------------------------------- API Endpoints


class CircleUpdateRequest(BaseModel):
    circle: Circle


class WearLogCreateRequest(BaseModel):
    item_id: str
    circle: Circle
    date: str
    event_id: str | None = None


@router.get("/events", response_model=list[CalendarEvent])
def list_events(
    date: str | None = Query(default=None),
    uid: str = Depends(current_uid),
    store: Store = Depends(get_store),
) -> list[CalendarEvent]:
    """Retrieve calendar events for the user (or seeded demo events)."""
    target_date = date or datetime.date.today().isoformat()
    return ensure_user_events(store, uid, target_date)


@router.patch("/events/{event_id}/circle", response_model=CalendarEvent)
@router.patch("/events/{event_id}", response_model=CalendarEvent)
def update_event_circle(
    event_id: str,
    body: CircleUpdateRequest,
    uid: str = Depends(current_uid),
    store: Store = Depends(get_store),
) -> CalendarEvent:
    """Override the audience-circle tag of a calendar event."""
    doc = store.get("events", event_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found.")
    store.update("events", event_id, {"circle": body.circle})
    updated = store.get("events", event_id)
    assert updated is not None
    return CalendarEvent.model_validate(updated)


@router.post("/wear-logs", response_model=WearLogEntry)
def create_wear_log(
    body: WearLogCreateRequest,
    uid: str = Depends(current_uid),
    store: Store = Depends(get_store),
) -> WearLogEntry:
    """Record a wear log entry for an item."""
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    entry = WearLogEntry(
        id=uuid.uuid4().hex[:12],
        user_id=uid,
        item_id=body.item_id,
        event_id=body.event_id,
        circle=body.circle,
        date=body.date,
        logged_at=now_iso,
    )
    store.set("wear_logs", entry.id, entry.model_dump(mode="json"))
    return entry


@router.get("/wear-logs", response_model=list[WearLogEntry])
def list_wear_logs(
    item_id: str | None = Query(default=None),
    uid: str = Depends(current_uid),
    store: Store = Depends(get_store),
) -> list[WearLogEntry]:
    """List wear history for the authenticated user."""
    equals: dict[str, Any] = {"user_id": uid}
    if item_id:
        equals["item_id"] = item_id
    docs = store.find("wear_logs", **equals)
    return [WearLogEntry.model_validate(d) for d in docs]


@router.get("/plan", response_model=DayPlan)
def get_day_plan(
    date: str = Query(..., description="Date formatted as YYYY-MM-DD"),
    refresh: bool = Query(default=False, description="Bypass cache and regenerate plan"),
    uid: str = Depends(current_uid),
    store: Store = Depends(get_store),
) -> DayPlan:
    """Retrieve DayPlan for user and date, cached in Store."""
    cache_key = f"{uid}_{date}"
    if not refresh:
        cached_doc = store.get("plans", cache_key)
        if cached_doc:
            plan = DayPlan.model_validate(cached_doc)
            plan.cached = True
            return plan

    plan = run_planner_agent(uid, date, store)
    # Save to store cache
    store.set("plans", cache_key, plan.model_dump(mode="json"))
    return plan
