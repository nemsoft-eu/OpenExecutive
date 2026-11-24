"""The Schedule page: everything the Executive does without being asked, in
one list, soonest first.

``GET /schedule`` reads the stores that already hold that work and adds
nothing of its own:

- **follow_up**: the follow-ups and nudges it will send, and
- **standing**: the standing work people gave As me, both read through
  ``today._watching`` (the lines Home's Following up shows, with the same
  rules for who sees what), so Stop is ``POST /today/watching/stop``;
- **workflow**: custom workflows on a schedule (their pending
  ``dynamic_workflow`` row says when they run next);
- **watch**: the Watch list's watches that are on (suggestions waiting for
  a yes stay on the Watch list);
- **always**: what it always does on its own: the owner's morning brief,
  end-of-day digest and weekly review, and each department's check-in.

Each row says where it is managed (``href``). Web (``/schedule``) and the
phone (``app/schedule.tsx``) render this one list. Never raises: a store it
can't read leaves its rows out.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

ScheduleKind = Literal["follow_up", "standing", "workflow", "watch", "always"]
KINDS: tuple[ScheduleKind, ...] = ("follow_up", "standing", "workflow", "watch", "always")

# How many follow-ups the page lists (Home shows the first few).
_FOLLOW_UPS_MAX = 200
_PENDING_MAX = 200

# What it always does, by its scheduled kind: the row's name and how often.
_ALWAYS: dict[str, tuple[str, str]] = {
    "principal_brief_morning": ("Morning brief", "every day"),
    "principal_brief_eod": ("End-of-day digest", "every day"),
    "principal_weekly_review": ("Weekly review", "every week"),
}
_DEPT_CHECK_IN = "dept_cadence"
# Delegate's As me block, where standing work is given and stopped.
_AS_ME_HREF = "/delegate#as-me"


class ScheduleItem(BaseModel):
    """One row of the Schedule. ``at`` is when it acts next (ISO), or None;
    a watch says when it last checked (``last_checked``) instead.
    ``cadence`` is how often, in words ("every day", "once a week"); "" for
    a one-off. ``href`` opens where it is managed; ``stop_key`` (follow-ups
    and standing work) is what ``POST /today/watching/stop`` takes."""

    key: str
    kind: ScheduleKind
    title: str
    detail: str = ""
    at: str | None = None
    last_checked: str | None = None
    cadence: str = ""
    href: str | None = None
    stop_key: str | None = None


class ScheduleResponse(BaseModel):
    items: list[ScheduleItem] = Field(default_factory=list)
    # Rows per kind, every kind present (0 when none).
    counts: dict[str, int] = Field(default_factory=dict)


def _cadence_words(spec: str | None) -> str:
    """A cadence spec (``daily@09:00``, ``weekly@mon@09:00``,
    ``quarterly@01-09:00``) as "every day", "every week", "every quarter"."""
    head = (spec or "").split("@", 1)[0].strip().lower()
    return {"daily": "every day", "weekly": "every week", "quarterly": "every quarter"}.get(head, "")


def _follow_ups_and_standing(request: Request, caller: int | None) -> list[ScheduleItem]:
    from openexecutive.api.routes.today import _watching

    items: list[ScheduleItem] = []
    for line in _watching(request, caller, limit=_FOLLOW_UPS_MAX):
        standing = line.kind == "standing_job"
        items.append(ScheduleItem(
            key=line.key,
            kind="standing" if standing else "follow_up",
            title=line.title,
            detail=line.detail,
            at=line.at,
            cadence="",
            href=_AS_ME_HREF if standing else None,
            stop_key=line.key,
        ))
    return items


def _workflows() -> list[ScheduleItem]:
    from openexecutive.memory.episodic import list_scheduled_actions
    from openexecutive.workflows.dynamic_store import list_definitions

    try:
        defs = {d.name: d for d in list_definitions(active_only=True) if d.cadence}
        if not defs:
            return []
        rows = list_scheduled_actions(status="pending", limit=_PENDING_MAX, kinds=("dynamic_workflow",))
    except Exception:
        logger.warning("schedule: reading scheduled workflows failed", exc_info=True)
        return []
    items: list[ScheduleItem] = []
    seen: set[str] = set()
    for row in rows:
        defn = defs.get(row.channel_ref or "")
        if defn is None or defn.name in seen:
            continue
        seen.add(defn.name)
        items.append(ScheduleItem(
            key=f"workflow:{defn.name}",
            kind="workflow",
            title=" ".join(defn.title.split()) or defn.name,
            at=row.run_at,
            cadence=_cadence_words(defn.cadence),
            href=f"/jobs/{defn.name}",
        ))
    return items


def _watches(caller: int | None = None) -> list[ScheduleItem]:
    from openexecutive.monitoring import store as monitoring_store
    from openexecutive.monitoring.describe import check_minutes, checks_every, watch_title

    try:
        waiting = {w.id for w in monitoring_store.list_pending_suggestions()}
        # The team's watches, and the caller's own email watches.
        watches = [
            w for w in monitoring_store.list_watchlist(enabled_only=True, for_person=caller) if w.id not in waiting
        ]
    except Exception:
        logger.warning("schedule: reading the watch list failed", exc_info=True)
        return []
    items: list[ScheduleItem] = []
    for w in watches:
        at = None
        if w.last_polled_at:
            # When it checks next: its last check plus how often it checks.
            try:
                last = datetime.fromisoformat(w.last_polled_at)
                at = (last + timedelta(minutes=check_minutes(w.signal_type, w.cadence))).isoformat()
            except ValueError:
                at = None
        items.append(ScheduleItem(
            key=f"watch:{w.slug}",
            kind="watch",
            title=watch_title(w),
            at=at,
            last_checked=w.last_polled_at,
            cadence=checks_every(w.signal_type, w.cadence),
            href=f"/watchlist/{w.slug}",
        ))
    return items


def _always(*, owner: bool, on_team: bool) -> list[ScheduleItem]:
    from openexecutive.memory.episodic import list_scheduled_actions

    kinds = [*(_ALWAYS if owner else ()), *([_DEPT_CHECK_IN] if on_team else [])]
    if not kinds:
        return []
    try:
        rows = list_scheduled_actions(status="pending", limit=_PENDING_MAX, kinds=kinds)
    except Exception:
        logger.warning("schedule: reading what it always does failed", exc_info=True)
        return []
    departments: dict[str, tuple[str, str]] = {}
    if _DEPT_CHECK_IN in kinds:
        try:
            from openexecutive.departments import store as departments_store

            departments = {
                d.config.slug: (d.config.title, d.config.cadences.get("check_in", ""))
                for d in departments_store.list_departments()
            }
        except Exception:
            logger.warning("schedule: reading departments failed", exc_info=True)
    items: list[ScheduleItem] = []
    seen: set[str] = set()
    for row in rows:
        if row.kind == _DEPT_CHECK_IN:
            slug = row.department or row.channel_ref or ""
            if slug not in departments:
                continue
            key = f"always:{_DEPT_CHECK_IN}:{slug}"
            title, spec = departments[slug]
            name, cadence, href = f"{title} check-in", _cadence_words(spec), f"/departments/{slug}"
        else:
            key = f"always:{row.kind}"
            name, cadence = _ALWAYS[row.kind]
            href = "/settings/executive"
        if key in seen:
            continue
        seen.add(key)
        items.append(ScheduleItem(key=key, kind="always", title=name, at=row.run_at, cadence=cadence, href=href))
    return items


def _sort_key(item: ScheduleItem) -> tuple[int, float]:
    # Soonest first, by the moment (the stores don't all write the same ISO
    # form); rows with no next time, or one that doesn't parse, last, in the
    # order they came.
    if not item.at:
        return (1, 0.0)
    try:
        at = datetime.fromisoformat(item.at.replace("Z", "+00:00"))
    except ValueError:
        return (1, 0.0)
    if at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    return (0, at.timestamp())


def build_schedule(request: Request) -> ScheduleResponse:
    """Everything on the reader's schedule. The owner sees the team's
    follow-ups and their own (never one a teammate asked for), the standing
    work, the workflows, watches, briefs and check-ins; anyone else on the
    team sees the follow-ups they asked for, the team's workflows and
    watches and the departments' check-ins. Never raises."""
    from openexecutive.api.routes.chat import (
        _caller_is_principal_or_unclaimed,
        _resolve_caller_person_id,
    )
    from openexecutive.api.routes.people import caller_on_team

    try:
        caller = _resolve_caller_person_id(request)
        owner = _caller_is_principal_or_unclaimed(request)
        # The owner is on the team; so is everyone on an install with no
        # owner yet, as everywhere ``_caller_is_principal_or_unclaimed`` is.
        on_team = owner or caller_on_team(request)
    except Exception:
        logger.warning("schedule: reading the caller failed", exc_info=True)
        return ScheduleResponse(counts={k: 0 for k in KINDS})
    items = _follow_ups_and_standing(request, caller)
    if on_team:
        items += _workflows() + _watches(caller)
    items += _always(owner=owner, on_team=on_team)
    items.sort(key=_sort_key)
    counts: dict[str, int] = {k: 0 for k in KINDS}
    for item in items:
        counts[item.kind] += 1
    return ScheduleResponse(items=items, counts=counts)


@router.get("/schedule", response_model=ScheduleResponse, tags=["schedule"])
def get_schedule(request: Request) -> ScheduleResponse:
    """Everything the Executive does without being asked, soonest first,
    each with its kind, when it acts next (or last checked), how often and
    where it is managed. Follow-ups and standing work stop with ``POST
    /today/watching/stop``; the rest open their own page."""
    return build_schedule(request)
