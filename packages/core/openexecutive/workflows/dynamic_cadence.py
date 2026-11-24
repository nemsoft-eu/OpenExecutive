"""Scheduler glue for cadence-enabled dynamic workflows.

A dynamic workflow may carry a cadence string (``daily@HH:MM`` etc.). When it
does, the scheduler should auto-run it and DM the artifact to
``cadence_person_id``. This module owns enqueueing and cancelling those
``scheduled_actions`` rows; the actual run happens in
``scheduler/runner.py``'s ``kind == "dynamic_workflow"`` branch.

Row encoding (mirrors ``departments/cadence.py``):
- ``channel``                = ``"__internal__"`` (never reaches a human directly)
- ``channel_ref``            = the workflow ``name`` (what to run; like dept slug)
- ``assigned_to_person_id``  = ``cadence_person_id`` (who receives the artifact)
- ``kind``                   = ``"dynamic_workflow"``
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, tzinfo
from pathlib import Path

from openexecutive.memory import workspace_settings
from openexecutive.workflows.dynamic_models import DynamicWorkflowDef

logger = logging.getLogger(__name__)


_DOW_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def from_utc(spec: str, tz: tzinfo, now: datetime | None = None) -> str:
    """``spec`` written in UTC (as cadences were before they took the
    company's time zone), rewritten as the same moment in ``tz``: its next
    occurrence, read there. A weekly or quarterly day moves with it when the
    time crosses midnight (a quarterly day keeps its own number if that
    would leave its month). Anything it can't read stays as it is."""
    from openexecutive.departments.cadence import (
        _DAILY_RE,
        _QUARTERLY_RE,
        _WEEKLY_RE,
        _parse_cadence_spec,
    )

    nxt = _parse_cadence_spec(spec, now or datetime.now(UTC), UTC)
    if nxt is None:
        return spec
    local = nxt.astimezone(tz)
    clock = f"{local.hour:02d}:{local.minute:02d}"
    if _DAILY_RE.match(spec):
        return f"daily@{clock}"
    if _WEEKLY_RE.match(spec):
        return f"weekly@{_DOW_NAMES[local.weekday()]}@{clock}"
    quarterly = _QUARTERLY_RE.match(spec)
    if quarterly:
        day = local.day if local.month == nxt.month else int(quarterly.group(1))
        return f"quarterly@{day:02d}-{clock}"
    return spec


def schedule_dynamic_workflow_cadence(
    defn: DynamicWorkflowDef, *, after: datetime | None = None, db_path: Path | None = None
) -> int | None:
    """Enqueue the next occurrence of ``defn``'s cadence. Returns the row id.

    No-op (returns None) when the definition has no cadence, no
    cadence_person_id, or an unparseable spec.
    """
    if not defn.cadence or defn.cadence_person_id is None:
        return None

    from openexecutive.departments.cadence import _parse_cadence_spec
    from openexecutive.memory.episodic import insert_scheduled_action

    base = (after or datetime.now(UTC)).astimezone(UTC)
    # A cadence's times are the company's own (Settings), summer time and
    # all: "weekly@mon@09:00" is 9 AM there every Monday.
    run_at = _parse_cadence_spec(defn.cadence, base, workspace_settings.get_user_timezone())
    if run_at is None:
        logger.warning(
            "dynamic_cadence: invalid spec %r for workflow %r", defn.cadence, defn.name
        )
        return None

    try:
        action_id = insert_scheduled_action(
            run_at=run_at.isoformat(),
            channel="__internal__",
            channel_ref=defn.name,
            intent_text=f"Scheduled run of dynamic workflow: {defn.title}",
            kind="dynamic_workflow",
            assigned_to_person_id=defn.cadence_person_id,
            db_path=db_path,
        )
        logger.info(
            "dynamic_cadence: enqueued %r → %s (id=%d)",
            defn.name, run_at.isoformat(), action_id,
        )
        return action_id
    except Exception:
        logger.exception("dynamic_cadence: enqueue failed for %r", defn.name)
        return None


def cancel_cadence_rows(name: str, db_path: Path | None = None) -> int:
    """Cancel pending dynamic_workflow rows for ``name``. Returns count cancelled.

    Idempotent; safe to call when no rows or no table exist.
    """
    from openexecutive.memory.episodic import _get_conn, _resolve_db_path

    resolved = _resolve_db_path(db_path)
    if not resolved.exists():
        return 0
    cancelled = 0
    with _get_conn(resolved) as conn:
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='scheduled_actions'"
        ).fetchone()
        if table is None:
            return 0
        cur = conn.execute(
            "UPDATE scheduled_actions SET status = 'cancelled' "
            "WHERE kind = 'dynamic_workflow' AND status = 'pending' AND channel_ref = ?",
            (name,),
        )
        cancelled = cur.rowcount
    if cancelled:
        logger.info("dynamic_cadence: cancelled %d row(s) for %r", cancelled, name)
    return cancelled


def seed_workflow_cadences(db_path: Path | None = None) -> int:
    """Schedule the next run of every active workflow with a cadence that has
    none pending or running. A cadence chains its next run when one finishes,
    so a run that never finished here (a restart mid-run, a restored copy
    saved mid-run) would otherwise stop it for good. One of the recurring
    seeds (``scheduler.runner.seed_recurring_jobs``). Returns the count
    scheduled. Never raises."""
    from openexecutive.memory.episodic import _get_conn, _resolve_db_path
    from openexecutive.workflows.dynamic_store import list_definitions
    from openexecutive.workflows.failure_alerts import RERUN_SCOPE_PREFIX

    try:
        resolved = _resolve_db_path(db_path)
        if not resolved.exists():
            return 0
        with _get_conn(resolved) as conn:
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='scheduled_actions'"
            ).fetchone() is None:
                return 0
            # A queued Run again is a one-off that never chains, so it is not
            # the cadence's next run.
            scheduled = {
                str(row[0]) for row in conn.execute(
                    "SELECT DISTINCT channel_ref FROM scheduled_actions "
                    "WHERE kind = 'dynamic_workflow' AND status IN ('pending', 'running') "
                    "AND (scope_key IS NULL OR scope_key NOT LIKE ?)",
                    (f"{RERUN_SCOPE_PREFIX}%",),
                ).fetchall()
            }
        defs = list_definitions(active_only=True, db_path=db_path)
    except Exception:
        logger.warning("dynamic_cadence: couldn't read the workflows to seed", exc_info=True)
        return 0
    seeded = 0
    for defn in defs:
        if not defn.cadence or defn.name in scheduled:
            continue
        if schedule_dynamic_workflow_cadence(defn, db_path=db_path) is not None:
            seeded += 1
    return seeded


def retime_pending(db_path: Path | None = None) -> int:
    """After the company's time zone changes: each workflow with a run
    already scheduled gets it again at its time in the new zone. One that
    is off or has nothing scheduled is left alone. Returns the count
    scheduled again. Never raises."""
    from openexecutive.memory.episodic import _get_conn, _resolve_db_path
    from openexecutive.workflows.dynamic_store import get_definition

    moved = 0
    try:
        resolved = _resolve_db_path(db_path)
        if not resolved.exists():
            return 0
        with _get_conn(resolved) as conn:
            names = [
                str(row[0]) for row in conn.execute(
                    "SELECT DISTINCT channel_ref FROM scheduled_actions "
                    "WHERE kind = 'dynamic_workflow' AND status = 'pending'"
                ).fetchall()
            ]
    except Exception:
        logger.warning("dynamic_cadence: couldn't read the scheduled runs to re-time", exc_info=True)
        return 0
    for name in names:
        try:
            defn = get_definition(name)
            if defn is None or not defn.is_active or not defn.cadence:
                continue
            cancel_cadence_rows(name, db_path)
            if schedule_dynamic_workflow_cadence(defn, db_path=db_path) is not None:
                moved += 1
        except Exception:
            logger.warning("dynamic_cadence: couldn't re-time %r", name, exc_info=True)
    return moved
