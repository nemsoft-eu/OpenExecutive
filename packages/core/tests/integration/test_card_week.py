"""The card week bench: the simulated week's steps replayed with no model.

Each event in ``tests/fixtures/card_week.json`` is something that happened
on a test company in a simulated week: who asked, on which surface, at which
Approvals level, with which rules and training, and the
exact tool calls the model made. The bench feeds those calls to the real
agent loop (a scripted model stands in for the real one) over the real
gateway with a stub server behind it that records what would have gone out,
then checks what the gates, the card store and routing did: which steps ran,
which waited on a card, whose card it is, what an Approve tap does, and
what each open card shows: its panel, what it needs and its buttons, as
GET /today gives them to every app.

It catches a gate that lets a step through or holds one it shouldn't, a card
that goes to the wrong person, a step that is carded twice and a card that
shows the wrong words or buttons, with no model.
To add an event, copy a recorded turn's tool calls into the file (made-up
names and addresses only) and say what should happen.
"""
from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openexecutive.alerts import store as alerts_store
from openexecutive.api.routes import decisions as decisions_route
from openexecutive.audit import logger as audit_logger
from openexecutive.audit.logger import AuditLogger, log_event
from openexecutive.departments import registry as dept_registry
from openexecutive.departments import store as dept_store
from openexecutive.memory import episodic
from openexecutive.orchestrator import mcp_gateway as gw_module
from openexecutive.orchestrator import outside_mode
from openexecutive.orchestrator import take_the_lead as ttl
from openexecutive.orchestrator.schedule_tools import current_session, set_session
from openexecutive.orchestrator.session import Session
from openexecutive.people import registry as people_registry
from openexecutive.people import store as people_store

WEEK = json.loads((Path(__file__).parent.parent / "fixtures" / "card_week.json").read_text())
EXEC = WEEK["company"]["exec_email"]
_MODES = {"careful": outside_mode.ASK, "balanced": outside_mode.OUTSIDERS, "hands_off": outside_mode.TRUST}


@pytest.fixture(autouse=True)
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "week.db"
    for module in (people_store, episodic, dept_store, alerts_store):
        monkeypatch.setattr(module, "DB_PATH", path)
    episodic.initialize_db(path)
    people_store.initialize_db()
    dept_store.initialize_db()
    alerts_store.initialize_db(path)
    people_registry.invalidate()
    dept_registry.invalidate()
    monkeypatch.setattr(audit_logger, "_default_logger", AuditLogger(db_path=path))
    # Signed in on this server as the owner, for the Approve taps.
    monkeypatch.setenv("OE_LOCAL_LOGIN", "1")
    token = current_session.set(None)
    yield path
    current_session.reset(token)
    people_registry.invalidate()
    dept_registry.invalidate()


class _Week:
    """One event's company: its people, its level and what went out."""

    def __init__(self, db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.db = db
        self.ids: dict[str, int] = {}
        self.mode = outside_mode.OUTSIDERS
        self.reached: list[tuple[str, dict[str, Any]]] = []
        self.turns = 0
        for p in WEEK["company"]["people"]:
            self.ids[p["key"]] = int(people_store.upsert_person(
                full_name=p["name"], email=p["email"], role=p.get("role", ""),
                is_principal=p.get("owner", False), kind=p.get("kind", "team"),
            ))
        self.emails = {p["key"]: p["email"] for p in WEEK["company"]["people"]}
        self.contacts = {p["key"] for p in WEEK["company"]["people"] if p.get("kind") == "contact"}
        people_registry.invalidate()
        # The bench sets the company's level itself, as the owner does in
        # Settings (tests/conftest.py pins Careful for the rest of the suite).
        monkeypatch.setattr(outside_mode, "get_mode", lambda **_: self.mode)
        # The real gateway (its own recipient gates run) over a stub server
        # that records each call that reached it and says it was done. An
        # Approve tap carries out the held call through the same gateway.
        self.gateway = gw_module.MCPGateway()
        self.gateway._session = SimpleNamespace(call_tool=self._server)
        monkeypatch.setattr(gw_module, "_active_gateway", self.gateway)
        monkeypatch.setattr(gw_module, "get_settings", lambda: SimpleNamespace(exec_email_address=EXEC))

    async def _server(self, name: str, args: dict[str, Any]) -> Any:
        self.reached.append((name, args))
        sent = "send" in str(args.get("tool_name", ""))
        return SimpleNamespace(content=[SimpleNamespace(text="Email sent! Message ID: m1" if sent else "{}")])

    def sent(self) -> list[str]:
        return [args["tool_name"] for name, args in self.reached if name == "call_tool"]

    def cards(self) -> list[tuple[int, int | None]]:
        with sqlite3.connect(self.db) as conn:
            return [(r[0], r[1]) for r in conn.execute(
                "SELECT id, approver_person_id FROM decision_instances ORDER BY id")]

    def open_cards(self) -> list[tuple[int, int | None]]:
        """The cards on Today: open, and not joined into another."""
        with sqlite3.connect(self.db) as conn:
            return [(r[0], r[1]) for r in conn.execute(
                "SELECT id, routed_to_person_id FROM alerts WHERE status IN ('unread', 'read')"
                " AND superseded_by_alert_id IS NULL AND archived_at IS NULL ORDER BY id")]

    def views(self) -> list[dict[str, Any]]:
        """What the open cards show (words and buttons), as GET /today
        gives them to every app."""
        from openexecutive.api.routes import today as today_route

        today = today_route._build_today(include_private=True)
        return [_shown(p.view) for p in today.proposals if p.view is not None]

    def settle(self, settings: dict[str, Any]) -> None:
        self.mode = _MODES[settings.get("level", "balanced")]
        for rule in settings.get("rules", []):
            ttl.add_rule(ttl.SCOPE_COMPANY, rule["kind"], rule["value"], created_by="bench")
        if settings.get("training"):
            ttl.set_(ttl.SCOPE_EXECUTIVE, training=True, updated_by="bench")

    def session(self, turn: dict[str, Any]) -> Session:
        who = turn["who"]
        if turn["surface"] == "web":
            return Session(from_web_chat=True, caller_person_id=self.ids[who])
        # Mail into the Executive's inbox, from an address the mail provider
        # vouched for. A contact's mail is the owner's private business, as
        # the email poller marks it.
        return Session(
            session_id=f"email:{turn.get('thread', who)}", origin_channel="email",
            email_from=self.emails[who], email_sender_authenticated=True,
            private_to_principal=who in self.contacts,
        )

    def turn(self, turn: dict[str, Any]) -> list[str]:
        from openexecutive.alerts import pipeline
        from openexecutive.audit.context import get_active_ids, set_turn
        from openexecutive.orchestrator.executive import Executive

        script = _Script(_uses(turn["calls"]))
        session = self.session(turn)
        raised: list[tuple[Any, str | None, str | None]] = []

        def _schedule(event: Any) -> None:
            raised.append((event, *get_active_ids()))

        async def _go() -> None:
            with (
                patch("openexecutive.orchestrator.executive.get_provider", return_value=script),
                patch("openexecutive.orchestrator.executive.audit_log", log_event),
                patch.object(pipeline, "schedule_evaluation", _schedule),
                patch.object(pipeline, "_triage", _keep),
                set_session(session),
                # Bound for the whole turn, as the chat route and the email
                # poller bind it: the card store reads the turn from it.
                set_turn(session_id=session.session_id, turn_id=f"t-{self.turns}"),
            ):
                async for _ in Executive(mcp_gateway=self.gateway)._stream_agent_loop(
                    system_blocks=[],
                    messages=[{"role": "user", "content": turn.get("ask", "x")}],
                    model="claude-test",
                    workspace_mode="team",
                    principal_role_tag="",
                    turn_id=f"t-{self.turns}",
                ):
                    pass
                # A raised card is reviewed after the turn, bound to it
                # (alerts.pipeline.schedule_evaluation).
                for event, sid, tid in raised:
                    await pipeline._evaluate_in_turn(event, sid, tid)

        self.turns += 1

        asyncio.run(_go())
        return [
            _outcome(block["content"])
            for call in script.calls[1:]
            for block in call["messages"][-1]["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]

    def approve(self, step: dict[str, Any], where: str) -> None:
        app = FastAPI()
        app.include_router(decisions_route.router)
        cards = self.cards()
        assert cards, f"{where}: nothing to approve, no card yet"
        card_id = cards[-1][0]
        body = {"edits": {"allow": True}} if step.get("allow") else {}
        response = TestClient(app).post(f"/decisions/{card_id}/approve", json=body)
        assert response.status_code == 200, response.text


class _Script:
    """A model that answers with the recorded calls, one round each, then
    ends the turn."""

    def __init__(self, rounds: list[list[Any]]) -> None:
        self._finals = [SimpleNamespace(content=r, stop_reason="tool_use", usage=None) for r in rounds]
        self._finals.append(
            SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], stop_reason="end_turn", usage=None)
        )
        self.calls: list[dict[str, Any]] = []

    def messages_stream(self, **kwargs: Any) -> Any:
        # The loop keeps adding to the one list it passes: keep what this
        # call was given.
        self.calls.append({**kwargs, "messages": copy.deepcopy(kwargs.get("messages"))})
        final = self._finals.pop(0)

        class _Stream:
            async def __aenter__(self) -> _Stream:
                return self

            async def __aexit__(self, *_a: Any) -> None:
                return None

            def __aiter__(self) -> _Stream:
                return self

            async def __anext__(self) -> Any:
                raise StopAsyncIteration

            async def get_final_message(self) -> Any:
                return final

        return _Stream()


async def _keep(event: Any, _path: Path) -> Any:
    """The card review's verdict: keep it, as written. The review is the
    model's call; the bench checks what the card store does with a card."""
    from openexecutive.alerts.models import AlertChannel, AlertSeverity, TriageDecision

    return TriageDecision(
        alert=True, severity=AlertSeverity.MEDIUM, channels=[AlertChannel.WEB, AlertChannel.PERSISTED],
        headline=event.subject, body=event.body, dedup_key=f"{event.source}:{event.external_id}",
    )


def _uses(calls: list[dict[str, Any]]) -> list[list[Any]]:
    rounds = []
    for n, step in enumerate(calls, start=1):
        name, tool_input = step["tool"], step["input"]
        if step.get("mcp"):
            name, tool_input = "call_tool", {"name": step["tool"], "arguments": step["input"]}
        rounds.append([SimpleNamespace(type="tool_use", id=f"tu-{n}", name=name, input=tool_input)])
    return rounds


def _outcome(result: str) -> str:
    try:
        parsed = json.loads(result)
    except (TypeError, ValueError):
        return "ran"
    if isinstance(parsed, dict) and parsed.get("status") == "waiting_for_approval":
        return "card"
    if isinstance(parsed, dict) and parsed.get("error"):
        return "error"
    return "ran"


def _shown(view: Any) -> dict[str, Any]:
    """The part of a card's view the bench checks: what it needs, its panel
    and its buttons by place."""
    return {
        "kind": view.kind,
        "panel": view.panel,
        "need": view.need,
        "buttons": [f"{b.place}:{b.action}:{b.label}" for b in view.buttons],
    }


@pytest.mark.parametrize("event", WEEK["events"], ids=[e["id"] for e in WEEK["events"]])
def test_the_week(event: dict[str, Any], db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    week = _Week(db, monkeypatch)
    week.settle(event.get("settings", {}))
    for n, step in enumerate(event["steps"], start=1):
        cards_before, sent_before = len(week.cards()), len(week.sent())
        where = f"{event['id']} step {n}"
        expect = step.get("expect", {})
        if "turn" in step:
            assert "steps" in expect, f"{where}: a turn step needs expect.steps"
            outcomes = week.turn(step["turn"])
            assert outcomes == expect["steps"], where
        elif "approve" in step:
            week.approve(step, where)
        new_cards = [approver for _id, approver in week.cards()[cards_before:]]
        assert new_cards == [week.ids[k] for k in expect.get("cards_for", [])], where
        assert len(week.sent()) - sent_before == expect.get("sent", 0), where
        if "open_cards" in expect:
            assert [w for _id, w in week.open_cards()] == [
                week.ids.get(k) for k in expect["open_cards"]], where
        if "views" in expect:
            assert week.views() == expect["views"], where
