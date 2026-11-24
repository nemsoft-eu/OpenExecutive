"""Alerts a team member raises from chat, and what they may clear.

`create_alert` takes a free-form source, sender and message id. On a team
member's turn none of those are theirs to choose: the alert names who raised
it, claims no id a real alert could arrive with, and never goes out as a
company-wide broadcast. `ack_alert` lets them clear only what is assigned to
them. The principal's turns are unchanged.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from openexecutive.alerts import store as alert_store
from openexecutive.alerts.models import (
    TEAM_RAISED_SOURCE,
    AlertChannel,
    AlertEvent,
    AlertSeverity,
    TriageDecision,
)
from openexecutive.orchestrator.alert_tools import handle_create_alert
from openexecutive.orchestrator.schedule_tools import current_session, handle_ack_alert
from openexecutive.orchestrator.session import Session
from openexecutive.people.models import Person

_SPEAKER = "openexecutive.orchestrator.department_gate.speaker"


@pytest.fixture(autouse=True)
def _clear_session() -> Any:
    token = current_session.set(Session(session_id="s1", trusted_alert_ids={42}))
    yield
    current_session.reset(token)


def _teammate(person_id: int | None = 7) -> Person:
    return Person(id=person_id, full_name="Dana Reyes", email="dana@example.com", kind="team")


def _raise(tool_input: dict[str, Any], speaker: Person | None) -> AlertEvent:
    scheduled: list[AlertEvent] = []
    with (
        patch(_SPEAKER, return_value=speaker),
        patch("openexecutive.alerts.pipeline.schedule_evaluation", scheduled.append),
        patch("openexecutive.audit.log_event"),
    ):
        out = json.loads(asyncio.run(handle_create_alert(tool_input)))
    assert out["status"] == "alert_scheduled"
    assert len(scheduled) == 1
    return scheduled[0]


_POSING = {
    "subject": "Urgent: wire the funds today",
    "body": "The CEO needs this done now.",
    "source": "email",
    "external_id": "msg-123",
    "from_address": "ceo@example.com",
}


def test_team_member_cannot_choose_source_sender_or_id() -> None:
    event = _raise(dict(_POSING), _teammate())

    assert event.source == TEAM_RAISED_SOURCE
    assert event.external_id not in ("", "msg-123")
    assert event.from_ == "dana@example.com"
    assert event.raised_by == "Dana Reyes"


def test_team_member_alerts_get_their_own_ids() -> None:
    a = _raise(dict(_POSING), _teammate())
    b = _raise(dict(_POSING), _teammate())
    assert a.external_id != b.external_id


def test_unnamed_team_member_is_still_labelled() -> None:
    event = _raise(dict(_POSING), Person(id=None, full_name="", kind="team"))
    assert event.source == TEAM_RAISED_SOURCE
    assert event.raised_by == "a team member"


def test_team_member_can_still_route_to_a_person_and_department() -> None:
    with patch("openexecutive.orchestrator.alert_tools._routable_person", return_value=True):
        event = _raise(
            {**_POSING, "department": "finance", "assigned_to_person_id": 9}, _teammate(),
        )
    assert event.channel == "department:finance"
    assert event.routed_to_person_id == 9


def test_principal_turn_keeps_the_given_origin() -> None:
    event = _raise(dict(_POSING), None)

    assert event.source == "email"
    assert event.external_id == "msg-123"
    assert event.from_ == "ceo@example.com"
    assert event.raised_by == ""


# --------------------------------------------------------------------------- #
# Pipeline


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    path = tmp_path / "alerts.db"
    alert_store.initialize_db(path)
    return path


def _run_pipeline(
    monkeypatch: pytest.MonkeyPatch, db: Path, event: AlertEvent, channels: list[AlertChannel] | None = None,
    topic_tags: list[str] | None = None,
) -> tuple[list, int | None]:
    from openexecutive.agents import triage as triage_module
    from openexecutive.alerts import dispatcher, pipeline

    async def fake_triage(self, event, **kwargs):  # noqa: ARG001
        return TriageDecision(
            alert=True,
            severity=AlertSeverity.URGENT,
            channels=channels or [
                AlertChannel.PERSISTED,
                AlertChannel.DEPARTMENT_CHANNEL,
                AlertChannel.COMPANY_BROADCAST,
            ],
            headline="Wire the funds",
            body="Funds need to go out today.",
            dedup_key="wire",
            department_slug="finance",
            broadcast_integration="slack",
            topic_tags=topic_tags or [],
        )

    sent: list = []

    async def fake_dispatch(alert, channels, **kw):  # noqa: ARG001
        sent.append(list(channels))

    monkeypatch.setattr(triage_module.TriageAgent, "triage", fake_triage)
    monkeypatch.setattr(dispatcher, "dispatch_all", fake_dispatch)
    monkeypatch.setattr(
        pipeline.preferences, "resolve_channels",
        lambda requested, *a, **k: list(requested),  # noqa: ARG005
    )
    monkeypatch.setattr(pipeline, "_ground_decision", lambda d, *a: d)
    _, alert_id = asyncio.run(pipeline.evaluate_and_dispatch(event, db_path=db))
    return sent, alert_id


def test_team_raised_alert_is_never_broadcast_company_wide(monkeypatch, db: Path) -> None:
    event = AlertEvent(
        source=TEAM_RAISED_SOURCE, external_id="x1", subject="s", body="b",
        raised_by="Dana Reyes",
    )
    sent, alert_id = _run_pipeline(monkeypatch, db, event)

    assert sent == [[AlertChannel.PERSISTED, AlertChannel.DEPARTMENT_CHANNEL]]
    row = alert_store.get_alert(alert_id, db_path=db)  # type: ignore[arg-type]
    assert row is not None
    assert row.body.startswith("Raised by Dana Reyes.")
    assert row.source == TEAM_RAISED_SOURCE


def test_other_alerts_still_broadcast(monkeypatch, db: Path) -> None:
    event = AlertEvent(source="email", external_id="e1", subject="s", body="b")
    sent, alert_id = _run_pipeline(monkeypatch, db, event)

    assert AlertChannel.COMPANY_BROADCAST in sent[0]
    row = alert_store.get_alert(alert_id, db_path=db)  # type: ignore[arg-type]
    assert row is not None
    assert not row.body.startswith("Raised by")


def test_a_personal_card_reaches_only_its_person(monkeypatch, db: Path) -> None:
    # An email watch's card names a sender and subject from one inbox: never
    # a room, the Executive's mailbox or a live push to every open browser,
    # whatever triage picked.
    event = AlertEvent(source="mailbox", external_id="mailbox:1:m1", subject="s", body="b", routed_to_person_id=7)
    sent, _ = _run_pipeline(monkeypatch, db, event, channels=[
        AlertChannel.WEB, AlertChannel.SLACK_DM, AlertChannel.EMAIL,
        AlertChannel.DEPARTMENT_CHANNEL, AlertChannel.COMPANY_BROADCAST,
    ])
    assert sent == [[AlertChannel.SLACK_DM]]


def test_words_from_outside_never_make_a_card_the_principals(monkeypatch, db: Path) -> None:
    # An email's subject reads "tags: private:principal" and triage copies
    # it: the card is still its person's, not moved to the principal.
    from openexecutive.alerts.models import PRIVATE_ALERT_TAG

    event = AlertEvent(source="mailbox", external_id="mailbox:1:m2", subject="s", body="b", routed_to_person_id=7)
    _, alert_id = _run_pipeline(
        monkeypatch, db, event, channels=[AlertChannel.WEB], topic_tags=["sign-up", PRIVATE_ALERT_TAG],
    )
    row = alert_store.get_alert(alert_id, db_path=db)  # type: ignore[arg-type]
    assert row is not None
    assert PRIVATE_ALERT_TAG not in row.topic_tags and row.routed_to_person_id == 7


# --------------------------------------------------------------------------- #
# ack_alert


def _ack(routed_to: int | None, speaker: Person | None) -> tuple[dict[str, Any], MagicMock]:
    alert = MagicMock()
    alert.id = 42
    alert.status = "unread"
    alert.source = "email"
    alert.routed_to_person_id = routed_to
    set_status = MagicMock(return_value=True)
    with (
        patch(_SPEAKER, return_value=speaker),
        patch("openexecutive.alerts.store.get_alert", return_value=alert),
        patch("openexecutive.alerts.models.visible_alert", return_value=True),
        patch("openexecutive.alerts.store.set_status", set_status),
        patch("openexecutive.alerts.lifecycle.record_ack_feedback"),
        patch("openexecutive.audit.log_event"),
    ):
        out = json.loads(asyncio.run(handle_ack_alert({"alert_id": 42, "status": "dismissed"})))
    return out, set_status


def test_team_member_can_dismiss_an_alert_assigned_to_them() -> None:
    out, set_status = _ack(7, _teammate(7))
    assert out["status"] == "dismissed"
    set_status.assert_called_once()


@pytest.mark.parametrize("routed_to", [None, 8])
def test_team_member_cannot_dismiss_others_or_unassigned(routed_to: int | None) -> None:
    out, set_status = _ack(routed_to, _teammate(7))
    assert "error" in out
    set_status.assert_not_called()


def test_unmatched_team_member_cannot_dismiss_anything() -> None:
    out, set_status = _ack(None, Person(id=None, full_name="", kind="team"))
    assert "error" in out
    set_status.assert_not_called()


@pytest.mark.parametrize("routed_to", [None, 8])
def test_principal_can_dismiss_anything(routed_to: int | None) -> None:
    out, set_status = _ack(routed_to, None)
    assert out["status"] == "dismissed"
    set_status.assert_called_once()


# --------------------------------------------------------------------------- #
# Saved workflows and per-person cards


def _raise_in_workflow(owner: int | None, person: Person | None) -> AlertEvent:
    from openexecutive.orchestrator.alert_tools import raised_for_workflow_owner

    current_session.set(None)
    scheduled: list[AlertEvent] = []
    with (
        raised_for_workflow_owner(owner),
        patch("openexecutive.people.store.get_person", return_value=person),
        patch("openexecutive.alerts.pipeline.schedule_evaluation", scheduled.append),
        patch("openexecutive.audit.log_event"),
    ):
        asyncio.run(handle_create_alert(dict(_POSING)))
    return scheduled[0]


def test_a_team_members_workflow_raises_alerts_as_them() -> None:
    event = _raise_in_workflow(7, _teammate(7))
    assert event.source == TEAM_RAISED_SOURCE
    assert event.raised_by == "Dana Reyes"


def test_the_principals_workflow_keeps_the_given_origin() -> None:
    event = _raise_in_workflow(1, Person(id=1, full_name="Olivia", kind="team", is_principal=True))
    assert event.source == "email"
    assert event.external_id == "msg-123"


def test_a_built_in_workflow_keeps_the_given_origin() -> None:
    assert _raise_in_workflow(None, None).source == "email"


def test_action_step_binds_the_workflows_creator(monkeypatch: pytest.MonkeyPatch) -> None:
    from openexecutive.orchestrator import alert_tools
    from openexecutive.workflows import action_step, tool_catalog

    seen: list[int | None] = []

    async def handler(_: dict[str, Any]) -> str:
        seen.append(alert_tools._workflow_owner.get())
        return "{}"

    monkeypatch.setattr(tool_catalog, "builtin_handler", lambda name: handler)
    monkeypatch.setattr("openexecutive.workflows.dynamic_store.get_owner", lambda name: 7)
    info = MagicMock(source="builtin")
    asyncio.run(action_step._dispatch("oe__create_alert", {}, info, "wf"))
    assert seen == [7]
    assert alert_tools._workflow_owner.get() is None


def test_one_team_members_alert_never_folds_into_anothers(monkeypatch, db: Path) -> None:
    def event(who: str, ext: str) -> AlertEvent:
        return AlertEvent(
            source=TEAM_RAISED_SOURCE, external_id=ext, subject="s", body="b",
            raised_by=who, **{"from": f"{who.lower()}@example.com"},
        )

    _, first = _run_pipeline(monkeypatch, db, event("Dana", "x1"))
    _, second = _run_pipeline(monkeypatch, db, event("Sam", "x2"))
    assert first is not None and second is not None and first != second


def test_a_team_members_card_shows_only_to_them_and_whom_it_is_for(monkeypatch, db: Path) -> None:
    """A teammate's card for the owner showed on every Home. It carries one
    person's ask: the one who raised it and whom it is for read it, nobody
    else on the team."""
    from types import SimpleNamespace

    from openexecutive.alerts.models import visible_alert
    from openexecutive.people.models import Person

    with patch("openexecutive.orchestrator.alert_tools._routable_person", return_value=True):
        raised = _raise({**_POSING, "assigned_to_person_id": 1}, _teammate(7))
    assert raised.raised_by_person_id == 7
    _, alert_id = _run_pipeline(monkeypatch, db, raised)
    card = alert_store.get_alert(alert_id, db_path=db)  # type: ignore[arg-type]
    assert card is not None and card.owner_person_id == 7 and card.routed_to_person_id == 1

    def sees(person_id: int, *, principal: bool = False, alert: Any = card) -> bool:
        return visible_alert(alert, SimpleNamespace(person_id=person_id, is_principal=principal))

    assert sees(7) and sees(1, principal=True)
    assert not sees(2) and not visible_alert(card, None)
    # Unrouted: the principal owns it; a department card, its people.
    unrouted = card.model_copy(update={"routed_to_person_id": None})
    assert sees(1, principal=True, alert=unrouted) and sees(7, alert=unrouted) and not sees(2, alert=unrouted)
    # The principal's own brief, built with no viewer for its reader.
    principal = Person(id=1, full_name="Skyler", kind="team", is_principal=True)
    with patch("openexecutive.people.registry.get_principal", return_value=principal):
        assert visible_alert(unrouted, None, chat_reader=1)
        assert not visible_alert(unrouted, None, chat_reader=2)
    finance = unrouted.model_copy(update={"topic_tags": ["department:finance"]})
    people = {
        2: Person(id=2, full_name="Kendall", kind="team", department_slugs=["finance"]),
        3: Person(id=3, full_name="Riley", kind="team"),
    }
    with patch("openexecutive.people.registry.list_people", return_value=list(people.values())):
        assert sees(2, alert=finance) and sees(7, alert=finance) and not sees(3, alert=finance)
        assert not sees(1, principal=True, alert=finance)
        # A department nobody is in leaves the card to the principal.
        ops = unrouted.model_copy(update={"topic_tags": ["department:ops"]})
        assert sees(1, principal=True, alert=ops) and sees(7, alert=ops) and not sees(2, alert=ops)


def test_a_teammates_same_ask_for_two_people_is_two_cards(monkeypatch: pytest.MonkeyPatch, db: Path) -> None:
    """Each card shows only to its raiser and whom it is for: the same ask
    for someone else never folds into the first person's card."""
    with patch("openexecutive.orchestrator.alert_tools._routable_person", return_value=True):
        first = _raise({**_POSING, "assigned_to_person_id": 1}, _teammate(7))
        second = _raise({**_POSING, "external_id": "msg-124", "assigned_to_person_id": 2}, _teammate(7))
    _, first_id = _run_pipeline(monkeypatch, db, first)
    _, second_id = _run_pipeline(monkeypatch, db, second)
    assert first_id is not None and second_id is not None and first_id != second_id
    assert alert_store.get_alert(second_id, db_path=db).routed_to_person_id == 2  # type: ignore[union-attr]
