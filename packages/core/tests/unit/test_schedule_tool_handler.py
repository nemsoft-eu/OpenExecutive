"""Unit tests for the schedule_followup tool handler.

Focuses on input validation, anti-spam guards, and timezone handling. The send-
tool handlers are exercised separately (they hit real network APIs in prod).
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from openexecutive.memory.episodic import initialize_db
from openexecutive.orchestrator.schedule_tools import (
    current_session,
    handle_schedule_followup,
)
from openexecutive.orchestrator.session import Session


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "episodic.db"
    initialize_db(db_path)
    monkeypatch.setattr("openexecutive.memory.episodic.DB_PATH", db_path)
    return db_path


@pytest.fixture()
def session_with_telegram_ref() -> Iterator[Session]:
    s = Session()
    token = current_session.set(s)
    yield s
    current_session.reset(token)


def _future_iso(seconds: int = 600) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


def _call(tool_input: dict) -> dict:
    return json.loads(asyncio.run(handle_schedule_followup(tool_input)))


def test_happy_path_returns_scheduled_id(session_with_telegram_ref: Session) -> None:
    payload = _call(
        {
            "run_at": _future_iso(),
            "channel": "telegram",
            "channel_ref": "42",
            "intent": "Remind user about the board prep.",
        }
    )
    assert payload["status"] == "scheduled"
    assert isinstance(payload["id"], int)
    assert payload["channel"] == "telegram"


def test_rejects_past_run_at(session_with_telegram_ref: Session) -> None:
    payload = _call(
        {
            "run_at": (datetime.now(UTC) - timedelta(seconds=60)).isoformat(),
            "channel": "telegram",
            "channel_ref": "42",
            "intent": "x",
        }
    )
    assert "must be in the future" in payload["error"]


def test_rejects_unparseable_run_at(session_with_telegram_ref: Session) -> None:
    payload = _call(
        {
            "run_at": "next tuesday-ish",
            "channel": "telegram",
            "channel_ref": "42",
            "intent": "x",
        }
    )
    assert "company's time zone" in payload["error"]


def test_rejects_unknown_channel(session_with_telegram_ref: Session) -> None:
    payload = _call(
        {
            "run_at": _future_iso(),
            "channel": "carrier_pigeon",
            "channel_ref": "42",
            "intent": "x",
        }
    )
    assert "unknown channel" in payload["error"]


def test_any_address_is_scheduled_its_send_meets_the_gate_when_it_fires(
    session_with_telegram_ref: Session,
) -> None:
    payload = _call(
        {
            "run_at": _future_iso(),
            "channel": "telegram",
            "channel_ref": "99",  # never came up in this chat
            "intent": "x",
        }
    )
    assert payload["status"] == "scheduled"


def test_rejects_beyond_horizon(session_with_telegram_ref: Session) -> None:
    far = (datetime.now(UTC) + timedelta(days=400)).isoformat()
    payload = _call(
        {
            "run_at": far,
            "channel": "telegram",
            "channel_ref": "42",
            "intent": "x",
        }
    )
    assert "days out" in payload["error"]


def test_accepts_z_suffix_utc(session_with_telegram_ref: Session) -> None:
    iso_z = (datetime.now(UTC) + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = _call(
        {
            "run_at": iso_z,
            "channel": "telegram",
            "channel_ref": "42",
            "intent": "x",
        }
    )
    assert payload["status"] == "scheduled"


def test_rejects_when_pending_cap_exceeded(
    session_with_telegram_ref: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_PENDING_PER_CHANNEL_REF", "1")
    first = _call(
        {
            "run_at": _future_iso(60),
            "channel": "telegram",
            "channel_ref": "42",
            "intent": "first",
        }
    )
    assert first["status"] == "scheduled"
    second = _call(
        {
            "run_at": _future_iso(120),
            "channel": "telegram",
            "channel_ref": "42",
            "intent": "second",
        }
    )
    assert "too many pending" in second["error"]


def test_a_follow_up_goes_to_the_speakers_own_address_or_anyone_elses(
    isolated_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A chat app that seeds no addresses (Google Chat): "check my inbox at 5
    # and tell me" goes to the speaker's own email.
    from openexecutive.people import registry as people_registry
    from openexecutive.people import store as people_store

    monkeypatch.setattr(people_store, "DB_PATH", isolated_db)
    people_store.initialize_db(isolated_db)
    people_registry.invalidate()
    me = people_store.upsert_person(full_name="Olivia Owner", is_principal=True, email="olivia@co.example")
    s = Session(caller_person_id=me)
    token = current_session.set(s)
    try:
        own = _call({"run_at": _future_iso(), "channel": "email", "channel_ref": "Olivia@co.example", "intent": "x"})
        assert own["status"] == "scheduled"
        other = _call({"run_at": _future_iso(), "channel": "email", "channel_ref": "eve@x.example", "intent": "x"})
        assert other["status"] == "scheduled"
    finally:
        current_session.reset(token)
        people_registry.invalidate()


def test_a_time_with_no_offset_is_the_companys_time(
    session_with_telegram_ref: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """"Tomorrow at noon" in New York is noon there, not noon UTC."""
    from zoneinfo import ZoneInfo

    zone = ZoneInfo("America/New_York")
    monkeypatch.setattr("openexecutive.memory.workspace_settings.get_user_timezone", lambda: zone)
    noon = (datetime.now(zone) + timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
    payload = _call(
        {
            "run_at": noon.replace(tzinfo=None).isoformat(),
            "channel": "telegram",
            "channel_ref": "42",
            "intent": "Follow up on the board prep.",
        }
    )
    assert payload["status"] == "scheduled"
    assert datetime.fromisoformat(payload["run_at"]) == noon


def test_a_repeating_follow_up_comes_back_until_cancelled(session_with_telegram_ref: Session) -> None:
    """'Nudge us every Monday from now on' was set once (Test lab): a
    follow-up can repeat, and each occurrence that finishes queues the next."""
    from openexecutive.memory import episodic

    payload = _call({
        "run_at": _future_iso(), "channel": "telegram", "channel_ref": "42",
        "intent": "Nudge the team about the open orders.", "repeat": "weekly@mon@09:00",
    })
    assert payload["status"] == "scheduled" and payload["repeat"] == "weekly@mon@09:00"
    first = episodic.get_scheduled_action(payload["id"])
    assert first is not None and first.repeat == "weekly@mon@09:00"

    assert episodic.mark_action_done(first.id)
    pending = [a for a in episodic.list_scheduled_actions(status="pending") if a.id != first.id]
    assert len(pending) == 1
    nxt = pending[0]
    assert nxt.repeat == "weekly@mon@09:00" and nxt.intent_text == first.intent_text
    assert datetime.fromisoformat(nxt.run_at) > datetime.now(UTC)

    # Cancelling the waiting one ends it: nothing queues after a cancel.
    episodic.cancel_scheduled_action(nxt.id)
    assert [a for a in episodic.list_scheduled_actions(status="pending")] == []


def test_an_occurrence_that_stands_down_keeps_the_repeat(session_with_telegram_ref: Session) -> None:
    """A run that finds it was already handled skips only that time."""
    from openexecutive.memory import episodic
    from openexecutive.scheduler import runner

    payload = _call({
        "run_at": _future_iso(), "channel": "telegram", "channel_ref": "42",
        "intent": "Nudge the team about the open orders.", "repeat": "weekly@mon@09:00",
    })
    first = episodic.get_scheduled_action(payload["id"])
    assert first is not None
    runner._skip_already_handled(first, "they talked about it this week")
    pending = episodic.list_scheduled_actions(status="pending")
    assert len(pending) == 1 and pending[0].id != first.id and pending[0].repeat == "weekly@mon@09:00"

    # A stop nobody would want repeated (the creator left) ends the series.
    nxt = pending[0]
    episodic.mark_action_cancelled(nxt.id, "the person who scheduled it is no longer on the team", ends_repeat=True)
    assert episodic.list_scheduled_actions(status="pending") == []


def _repeating(**kw: Any) -> int:
    from openexecutive.memory import episodic

    return episodic.insert_scheduled_action(
        run_at=_future_iso(), channel="telegram", channel_ref="42", intent_text="Daily nudge.",
        repeat="daily@09:00", **kw,
    )


def _pending_others(action_id: int) -> list[Any]:
    from openexecutive.memory import episodic

    return [a for a in episodic.list_scheduled_actions(status="pending") if a.id != action_id]


def test_a_repeat_that_gives_up_still_comes_back(session_with_telegram_ref: Session) -> None:
    import sqlite3

    from openexecutive.memory import episodic

    action_id = _repeating()
    # A retry (attempts below the max) queues nothing new.
    assert episodic.mark_action_failed_or_retry(action_id, "boom", max_attempts=3) == "pending"
    assert _pending_others(action_id) == []
    with sqlite3.connect(str(episodic.DB_PATH)) as conn:
        conn.execute("UPDATE scheduled_actions SET attempts = 3 WHERE id = ?", (action_id,))
    assert episodic.mark_action_failed_or_retry(action_id, "boom", max_attempts=3) == "failed"
    assert len(_pending_others(action_id)) == 1


def test_a_second_finish_queues_nothing(session_with_telegram_ref: Session) -> None:
    from openexecutive.memory import episodic

    action_id = _repeating()
    assert episodic.mark_action_done(action_id)
    assert not episodic.mark_action_done(action_id)
    assert len(_pending_others(action_id)) == 1


def test_cancelling_a_running_repeat_stops_what_comes_after(session_with_telegram_ref: Session) -> None:
    import sqlite3

    from openexecutive.memory import episodic

    action_id = _repeating()
    with sqlite3.connect(str(episodic.DB_PATH)) as conn:
        conn.execute("UPDATE scheduled_actions SET status = 'running' WHERE id = ?", (action_id,))
    assert episodic.cancel_scheduled_action(action_id) == "cancelled"
    assert episodic.mark_action_done(action_id)
    assert _pending_others(action_id) == []


def test_a_restore_drops_only_the_occurrence(session_with_telegram_ref: Session) -> None:
    import sqlite3

    from openexecutive.memory import episodic

    action_id = _repeating()
    with sqlite3.connect(str(episodic.DB_PATH)) as conn:
        conn.execute("UPDATE scheduled_actions SET run_at = '2020-01-01T00:00:00+00:00' WHERE id = ?", (action_id,))
    assert episodic.settle_restored_queue(episodic.DB_PATH) == 1
    (nxt,) = _pending_others(action_id)
    assert nxt.repeat == "daily@09:00" and datetime.fromisoformat(nxt.run_at) > datetime.now(UTC)


def test_a_one_time_follow_up_does_not_come_back(session_with_telegram_ref: Session) -> None:
    from openexecutive.memory import episodic

    payload = _call({"run_at": _future_iso(), "channel": "telegram", "channel_ref": "42", "intent": "Once."})
    assert payload["repeat"] == "none: it runs once"
    episodic.mark_action_done(payload["id"])
    assert episodic.list_scheduled_actions(status="pending") == []


def test_an_unreadable_repeat_is_sent_back(session_with_telegram_ref: Session) -> None:
    payload = _call({
        "run_at": _future_iso(), "channel": "telegram", "channel_ref": "42", "intent": "x",
        "repeat": "every monday",
    })
    assert "can't be read" in payload["error"]


@pytest.mark.parametrize("spec", ["daily@25:00", "weekly@mon@09:75"])
def test_a_time_of_day_that_doesnt_exist_is_sent_back(session_with_telegram_ref: Session, spec: str) -> None:
    payload = _call({
        "run_at": _future_iso(), "channel": "telegram", "channel_ref": "42", "intent": "x", "repeat": spec,
    })
    assert "can't be read" in payload["error"]
