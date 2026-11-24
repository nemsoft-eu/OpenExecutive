"""create_alert called without the fields its schema requires.

A chat script or saved workflow calls the handler directly, past the tool
schema, so a missing subject or body has to come back as an error the caller
can fix rather than a KeyError.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

import pytest

from openexecutive.orchestrator.alert_tools import handle_create_alert


def _call(tool_input: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    scheduled: list[Any] = []
    with (
        patch("openexecutive.alerts.pipeline.schedule_evaluation", scheduled.append),
        patch("openexecutive.audit.log_event"),
    ):
        out = json.loads(asyncio.run(handle_create_alert(tool_input)))
    return out, scheduled


@pytest.mark.parametrize(
    ("tool_input", "named"),
    [
        ({"body": "Server is down"}, "subject"),
        ({"title": "Server down", "body": "Server is down"}, "subject"),
        ({"subject": "Server down"}, "body"),
        ({"subject": "  ", "body": ""}, "subject and body"),
    ],
)
def test_missing_field_is_named_and_nothing_is_raised(
    tool_input: dict[str, Any], named: str,
) -> None:
    out, scheduled = _call(tool_input)

    assert out["error"].startswith(f"create_alert needs {named}:")
    assert scheduled == []


def test_complete_input_still_schedules() -> None:
    out, scheduled = _call({"subject": "Server down", "body": "Server is down"})

    assert out["status"] == "alert_scheduled"
    assert len(scheduled) == 1


def _person_event() -> Any:
    from openexecutive.alerts.models import AlertEvent

    return AlertEvent.model_validate({"source": "email", "external_id": "m-3", "subject": "s", "body": "b"})


def test_a_second_ask_from_one_person_is_asked_which_card_it_joins() -> None:
    """With an open card from an earlier turn with the same person, nothing
    is raised until the turn says whether this is the same matter."""
    from openexecutive.orchestrator.alert_tools import _matter_choice

    with patch("openexecutive.alerts.one_card.open_cards_from", return_value=[(227, "Card 227", "Approve the plan?")]):
        out = _matter_choice(_person_event(), None)
    assert out is not None
    told = json.loads(out)
    assert told["status"] == "person_has_open_card"
    assert told["open_cards"] == [{"id": 227, "headline": "Card 227"}]
    assert "Approve the plan?" in told["message"]


def test_the_same_matter_joins_its_card_and_a_new_one_gets_its_own() -> None:
    from openexecutive.orchestrator.alert_tools import _matter_choice

    cards = [(227, "Card 227", ""), (230, "Card 230", "")]
    with patch("openexecutive.alerts.one_card.open_cards_from", return_value=cards):
        same = _person_event()
        assert _matter_choice(same, 230) is None and same.joins_alert_id == 230
        different = _person_event()
        assert _matter_choice(different, 0) is None and different.joins_alert_id == 0
        # An id that isn't one of their open cards starts a card too.
        stray = _person_event()
        assert _matter_choice(stray, 99) is None and stray.joins_alert_id is None
        # A message passed on as written is never asked.
        message = _person_event()
        message.as_written = True
        assert _matter_choice(message, None) is None


def test_no_open_card_raises_straight_away() -> None:
    from openexecutive.orchestrator.alert_tools import _matter_choice

    event = _person_event()
    with patch("openexecutive.alerts.one_card.open_cards_from", return_value=[]):
        assert _matter_choice(event, None) is None
    assert event.joins_alert_id is None


def test_a_passed_on_message_in_a_mail_turn_is_its_own_card() -> None:
    """``as_written`` (a message passed on to someone) never folds into the
    open card of the sender whose mail the turn is reading."""
    from types import SimpleNamespace

    from openexecutive.orchestrator.schedule_tools import current_session

    mail = SimpleNamespace(origin_channel="email", email_from="casey@example.com")
    token = current_session.set(mail)
    scheduled: list[Any] = []
    try:
        with (
            patch("openexecutive.alerts.pipeline.schedule_evaluation", scheduled.append),
            patch("openexecutive.audit.log_event"),
        ):
            out = json.loads(asyncio.run(handle_create_alert(
                {"subject": "From Sam", "body": "Standup moved to 10am"}, as_written=True,
            )))
    finally:
        current_session.reset(token)

    assert out["status"] == "alert_scheduled"
    (event,) = scheduled
    assert event.as_written is True
    assert event.dedup_hint == ""
    assert event.follow_up is False
