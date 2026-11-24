"""A team member's chat-turn sends and edits go through their department's
authority (orchestrator/department_gate.py)."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import openexecutive.audit as audit
from openexecutive.alerts import store as alerts_store
from openexecutive.api.routes import decisions as decisions_route
from openexecutive.departments import registry as dept_registry
from openexecutive.departments import store as dept_store
from openexecutive.departments.models import AuthorityLevel
from openexecutive.memory import decision_ledger as ledger
from openexecutive.memory import episodic
from openexecutive.orchestrator import department_gate as gate
from openexecutive.orchestrator import levels
from openexecutive.orchestrator import take_the_lead as ttl
from openexecutive.orchestrator.session import Session
from openexecutive.people import registry as people_registry
from openexecutive.people import store as people_store
from openexecutive.people.models import AuthorityScope, Person

pytestmark = pytest.mark.usefixtures("stored_department_levels")

SAM = "sam@co.example"
RIA = "ria@co.example"


@pytest.fixture(autouse=True)
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "episodic.db"
    for module in (episodic, people_store, alerts_store, dept_store):
        monkeypatch.setattr(module, "DB_PATH", path)
    monkeypatch.delenv("OE_PUBLIC_DEPLOYMENT", raising=False)
    monkeypatch.delenv("CALLER_ASSERTION_PUBLIC_KEYS", raising=False)
    episodic.initialize_db(path)
    people_store.initialize_db(path)
    alerts_store.initialize_db(path)
    dept_store.initialize_db(path)
    dept_store.seed_default_departments()
    people_registry.invalidate()
    dept_registry.invalidate()
    yield path
    people_registry.invalidate()
    dept_registry.invalidate()


@pytest.fixture(autouse=True)
def events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    seen: list[tuple[str, dict[str, Any]]] = []

    def _log(event_type: str, summary: str, **kw: Any) -> None:
        seen.append((event_type, {"summary": summary, **kw}))

    monkeypatch.setattr(audit, "log_event", _log)
    return seen


@pytest.fixture
def team() -> dict[str, Person]:
    owner_id = people_store.upsert_person(full_name="Olivia Owner", is_principal=True, email="olivia@co.example")
    people_store.set_authority_scope(owner_id, [AuthorityScope.WILDCARD])
    sam_id = people_store.upsert_person(full_name="Sam Seller", email=SAM, department_slugs=["marketing"])
    ria_id = people_store.upsert_person(
        full_name="Ria Ops", email=RIA, department_slugs=["operations"], slack_user_id="U_RIA",
    )
    people_registry.invalidate()
    people = {
        "owner": people_store.get_person(owner_id),
        "sam": people_store.get_person(sam_id),
        "ria": people_store.get_person(ria_id),
    }
    assert all(people.values())
    return people  # type: ignore[return-value]


def _level(slug: str, level: AuthorityLevel) -> None:
    dept_store.update_department(slug, authority_level=level)
    dept_registry.invalidate()


def _reload(person: Person) -> Person:
    people_registry.invalidate()
    fresh = people_store.get_person(person.id)  # type: ignore[arg-type]
    assert fresh is not None
    return fresh


# ── whose turn it is ──────────────────────────────────────────────────────────


def test_only_a_team_members_attended_turn_is_gated(team: dict[str, Person]) -> None:
    sam, owner = team["sam"], team["owner"]
    assert gate.speaker(Session(caller_person_id=sam.id)).id == sam.id  # type: ignore[union-attr]
    assert gate.speaker(Session(caller_person_id=owner.id)) is None
    assert gate.speaker(Session(caller_person_id=sam.id, unattended=True)) is None
    assert gate.speaker(Session(caller_person_id=None)) is None
    assert gate.speaker(None) is None
    contact = people_store.upsert_person(full_name="Dana Client", email="dana@else.example", kind="contact")
    assert gate.speaker(Session(caller_person_id=contact)) is None


def test_someone_signed_in_but_not_on_people_is_gated(team: dict[str, Person]) -> None:
    """Not the principal, so not ungated: an archived teammate still let in,
    or an allowlisted email, waits like someone in no department."""
    unknown = gate.speaker(Session(from_web_chat=True, web_caller_signed_in=True))
    assert unknown is not None and unknown.id is None
    # No sign-in at all on the web is the operator, as before.
    assert gate.speaker(Session(from_web_chat=True)) is None


@pytest.mark.parametrize("channel", ["google_chat", "telegram", "discord", "slack"])
def test_a_chat_channel_sender_on_no_people_entry_is_gated(team: dict[str, Person], channel: str) -> None:
    unknown = gate.speaker(Session(origin_channel=channel, caller_person_id=None))
    assert unknown is not None and unknown.id is None


@pytest.mark.parametrize("channel", ["google_chat", "telegram", "discord", "slack"])
def test_the_owner_on_a_chat_channel_is_not_gated(team: dict[str, Person], channel: str) -> None:
    owner = team["owner"]
    assert gate.speaker(Session(origin_channel=channel, caller_person_id=owner.id)) is None


def test_google_chat_gates_its_sender_by_their_people_entry(team: dict[str, Person]) -> None:
    """Google Chat names a sender on People as the turn's caller, as Slack
    does: the owner goes, a teammate gets their own department's authority."""
    owner, sam = team["owner"], team["sam"]
    assert gate.speaker(Session(origin_channel="google_chat", caller_person_id=owner.id)) is None
    teammate = gate.speaker(Session(origin_channel="google_chat", caller_person_id=sam.id))
    assert teammate is not None and teammate.id == sam.id


def test_email_turns_are_left_to_the_mailbox(team: dict[str, Person]) -> None:
    """Whoever writes in is never on People; the mailbox's own gates cover
    its replies."""
    assert gate.speaker(Session(origin_channel="email", caller_person_id=None)) is None


# ── the card a held step waits on ─────────────────────────────────────────────


def _send(text: str = "Can we move the demo to Friday?") -> dict[str, Any]:
    return {"user_id": "U_RIA", "text": text}


def _ask_as(person: Person, session_id: str = "s-1") -> Session:
    return Session(session_id=session_id, from_web_chat=True, web_caller_signed_in=True, caller_person_id=person.id)


def _hold_as(person: Person, reason: str = levels.OWN_MAIL, tool_input: dict[str, Any] | None = None) -> dict[str, Any]:
    return json.loads(levels.hold("send_slack_dm", tool_input or _send(), mcp=False, reason=reason,
                                  session=_ask_as(person)))


def test_there_is_no_department_gate_left(team: dict[str, Person]) -> None:
    """Approvals is the one gate: the department authority ranges no longer
    decide or wrap anything."""
    for gone in ("decide", "gated_handlers", "gated_call_tool", "GATED_TOOLS"):
        assert not hasattr(gate, gone), gone


def test_a_teammates_held_step_is_their_own_card(
    team: dict[str, Person], events: list[tuple[str, dict[str, Any]]],
) -> None:
    result = _hold_as(team["sam"])
    assert result["status"] == "waiting_for_approval" and "Do not retry" in result["message"]
    instance = ledger.get_decision_instance(result["decision_instance_id"])
    assert instance is not None
    assert instance.decision_class == gate.DECISION_CLASS
    # The level is theirs, so the card is theirs to approve.
    assert instance.approver_person_id == team["sam"].id
    assert instance.originating_session_id == "s-1"
    payload = json.loads(instance.proposed_payload_json)
    assert payload["tool"] == "send_slack_dm" and payload["requested_by_person_id"] == team["sam"].id
    # Who it goes to by name, never a chat app's own id for them.
    assert payload["details"] == [
        {"field": "To", "value": "Ria Ops"}, {"field": "Message", "value": _send()["text"]},
    ]
    [card] = alerts_store.list_alerts(limit=10)
    assert card.routed_to_person_id == team["sam"].id
    assert f"decision_instance:{instance.id}" in card.topic_tags
    assert [e for e, _ in events] == ["department_action_held"]
    assert events[0][1]["actor"] == f"person:{team['sam'].id}"
    # Asking again while it waits is the same card.
    assert _hold_as(team["sam"])["decision_instance_id"] == instance.id
    assert len(alerts_store.list_alerts(limit=10)) == 1


@pytest.mark.parametrize("reason", [levels.STRANGER, levels.NOT_OWNER, levels.READ_ONLY])
def test_what_only_the_owner_may_allow_goes_to_the_owner(team: dict[str, Person], reason: str) -> None:
    instance = ledger.get_decision_instance(_hold_as(team["sam"], reason)["decision_instance_id"])
    assert instance is not None and instance.approver_person_id == team["owner"].id


def test_a_step_reaching_outside_the_team_goes_to_the_owner(team: dict[str, Person]) -> None:
    held = levels.hold(
        "google_workspace__send_gmail_message", {"to": "dana@else.example", "subject": "Hi", "body": "Hi"},
        mcp=True, reason=levels.OWN_MAIL, session=_ask_as(team["sam"]),
    )
    instance = ledger.get_decision_instance(json.loads(held)["decision_instance_id"])
    assert instance is not None and instance.approver_person_id == team["owner"].id


# ── approving it ──────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(decisions_route.router)
    return TestClient(app)


def _hold(team: dict[str, Person]) -> int:
    return int(_hold_as(team["sam"])["decision_instance_id"])


def test_approving_does_the_exact_call_once(
    client: TestClient, team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
    events: list[tuple[str, dict[str, Any]]],
) -> None:
    done: list[dict[str, Any]] = []

    async def carry_out(payload: dict[str, Any], *, by_principal: bool, own_card: bool = False) -> str:
        done.append({**payload, "own_card": own_card})
        return json.dumps({"status": "sent"})

    monkeypatch.setattr(ttl, "carry_out", carry_out)
    decision_id = _hold(team)
    headers = {"x-caller-email": SAM}
    response = client.post(f"/decisions/{decision_id}/approve", json={}, headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == ledger.STATUS_APPROVED_UNCHANGED
    assert [(p["tool"], p["input"]) for p in done] == [("send_slack_dm", _send())]
    # Sam asked and Sam approved: their own card (carry_out decides what that
    # may relax: only a step that runs on their own device).
    assert done[0]["own_card"] is True
    assert client.post(f"/decisions/{decision_id}/approve", json={}, headers=headers).status_code == 409
    [card] = alerts_store.list_alerts(limit=10, status="ack")
    assert card.external_id == f"decision:{decision_id}"
    assert [e for e, _ in events] == ["department_action_held", "department_action_done"]


def test_someone_else_cant_approve_a_teammates_card(client: TestClient, team: dict[str, Person]) -> None:
    decision_id = _hold(team)
    headers = {"x-caller-email": RIA}
    assert client.post(f"/decisions/{decision_id}/approve", json={}, headers=headers).status_code == 404
    instance = ledger.get_decision_instance(decision_id)
    assert instance is not None and instance.status == ledger.STATUS_PROPOSED


def test_declining_is_recorded(
    client: TestClient, team: dict[str, Person], events: list[tuple[str, dict[str, Any]]],
) -> None:
    decision_id = _hold(team)
    headers = {"x-caller-email": SAM}
    assert client.post(f"/decisions/{decision_id}/reject", json={}, headers=headers).status_code == 200
    assert [e for e, _ in events][-1] == "department_action_declined"


def test_an_email_card_can_be_changed_before_approving(
    client: TestClient, team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
) -> None:
    done: list[dict[str, Any]] = []

    async def carry_out(payload: dict[str, Any], *, by_principal: bool, own_card: bool = False) -> str:
        done.append(payload)
        return json.dumps({"status": "sent"})

    monkeypatch.setattr(ttl, "carry_out", carry_out)
    mail = {"to": "dana@else.example", "subject": "Hi", "body": "Draft text"}
    held = levels.hold("google_workspace__send_gmail_message", mail, mcp=True, reason=levels.OWN_MAIL,
                       session=_ask_as(team["sam"]))
    decision_id = json.loads(held)["decision_instance_id"]
    instance = ledger.get_decision_instance(decision_id)
    assert instance is not None
    fields = json.loads(instance.proposed_payload_json)["fields"]
    assert [f["field"] for f in fields] == ["subject", "body"]
    owner = {"x-caller-email": "olivia@co.example"}
    # Who it goes to is never changed from the card.
    bad = client.post(f"/decisions/{decision_id}/approve", json={"edits": {"input": {"to": "x@else.example"}}},
                      headers=owner)
    assert bad.status_code == 422
    response = client.post(f"/decisions/{decision_id}/approve",
                           json={"edits": {"input": {"body": "Better text"}}}, headers=owner)
    assert response.status_code == 200, response.text
    assert [p["input"] for p in done] == [{**mail, "body": "Better text"}]
    # Recorded as changed, and what it says is what went out.
    after = ledger.get_decision_instance(decision_id)
    assert after is not None and after.status == "approved_with_edit"
    assert "Better text" in done[0]["summary"] and "Draft text" not in done[0]["summary"]


def _rename(n: int) -> dict[str, Any]:
    return {"file_id": f"f{n}", "name": f"2026 Tax - Lot {n}.pdf"}


def test_calls_held_together_are_one_card(team: dict[str, Person]) -> None:
    """Fourteen renames in one round: one card that lists them all."""
    from openexecutive.orchestrator.department_gate import Verdict

    verdict = Verdict(run=False, approver_person_id=team["owner"].id, reason="it read your own mail first")
    steps = [_rename(n) for n in range(1, 4)]
    decision_id = gate.hold(
        team["owner"], "google_workspace__update_drive_file", steps[0], verdict,
        mcp=True, held=True, more=steps[1:],
    )
    instance = ledger.get_decision_instance(decision_id)
    assert instance is not None
    payload = json.loads(instance.proposed_payload_json)
    assert payload["steps"] == steps and payload["input"] == steps[0]
    assert payload["fields"] == [] and "2 more like it" in payload["summary"]
    [card] = alerts_store.list_alerts(limit=10)
    assert all(f"Lot {n}" in card.body for n in range(1, 4))
    # The same calls still waiting are that card, not a new one.
    assert gate.hold(
        team["owner"], "google_workspace__update_drive_file", steps[0], verdict,
        mcp=True, held=True, more=steps[1:],
    ) == decision_id



def test_a_cards_title_says_what_it_does_and_who_asked_only_for_the_team(team: dict[str, Person]) -> None:
    """A reply to a contact's mail is titled by what it does, never as the
    contact's own ask; a teammate's held ask names them."""
    from openexecutive.orchestrator.department_gate import Verdict

    dana_id = people_store.upsert_person(full_name="Dana Price", email="dana@vendor.example", kind="contact")
    people_registry.invalidate()
    dana = people_store.get_person(dana_id)
    assert dana is not None
    verdict = Verdict(run=False, approver_person_id=team["owner"].id, reason="Logistics is on Careful")
    mail = {"to": "dana@vendor.example", "subject": "Re: Dock", "body": "Hi Dana, thanks for the heads-up."}
    gate.hold(dana, "google_workspace__send_gmail_message", mail, verdict, mcp=True)
    gate.hold(team["sam"], "google_workspace__send_gmail_message", {**mail, "subject": "Re: Dock 2"}, verdict, mcp=True)
    titles = sorted(a.headline for a in alerts_store.list_alerts(limit=10))
    assert titles == ["Email Dana Price", "Sam Seller asked: Email Dana Price"]
    assert not any("asked the Executive to" in t or "Hi Dana" in t for t in titles)


def test_a_card_for_several_calls_shows_every_one(team: dict[str, Person]) -> None:
    """The web card and the chat-app card show each call Approve runs, not
    only the first."""
    from openexecutive.api.routes.decisions import _served
    from openexecutive.integrations.channel_cards import card_text
    from openexecutive.orchestrator.department_gate import Verdict

    verdict = Verdict(run=False, approver_person_id=team["owner"].id, reason="it read your own mail first")
    steps = [_rename(n) for n in range(1, 4)]
    decision_id = gate.hold(
        team["owner"], "google_workspace__update_drive_file", steps[0], verdict,
        mcp=True, held=True, more=steps[1:],
    )
    instance = ledger.get_decision_instance(decision_id)
    assert instance is not None
    served = json.loads(_served(instance).proposed_payload_json)
    shown = " ".join(d["value"] for d in served["details"])
    assert all(f"Lot {n}" in shown for n in range(1, 4))
    assert {d["field"].split(".")[0] for d in served["details"]} == {"1", "2", "3"}
    text = card_text(decision_id)
    assert text is not None and all(f"Lot {n}" in text for n in range(1, 4))


def test_one_tap_carries_out_each_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from openexecutive.orchestrator import mcp_gateway

    calls: list[dict[str, Any]] = []

    class Gateway:
        async def call_tool(self, tool_input: dict[str, Any]) -> str:
            calls.append(tool_input)
            if tool_input["arguments"]["file_id"] == "f3":
                return json.dumps({"error": "no such file"})
            return "Renamed."

    monkeypatch.setattr(mcp_gateway, "get_active_gateway", lambda: Gateway())
    steps = [_rename(n) for n in range(1, 3)]
    payload = {"tool": "google_workspace__update_drive_file", "input": steps[0], "mcp": True,
               "held": True, "steps": steps}
    done = json.loads(asyncio.run(ttl.carry_out(payload, by_principal=True)))
    assert done["done"] == 2 and [c["arguments"] for c in calls] == steps
    # It stops at the first that fails: nothing after it runs.
    calls.clear()
    failed = asyncio.run(ttl.carry_out({**payload, "steps": [*steps, _rename(3), _rename(4)]}, by_principal=True))
    assert ttl.result_failed(failed)
    error = json.loads(failed)
    assert error["done"] == 2 and "step 3 didn't" in error["error"] and "rest didn't run" in error["error"]
    assert [c["arguments"]["file_id"] for c in calls] == ["f1", "f2", "f3"]


def test_a_step_that_raises_stops_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    from openexecutive.orchestrator import mcp_gateway

    calls: list[str] = []

    class Gateway:
        async def call_tool(self, tool_input: dict[str, Any]) -> str:
            calls.append(tool_input["arguments"]["file_id"])
            if tool_input["arguments"]["file_id"] == "f2":
                raise TimeoutError("timed out")
            return "Renamed."

    monkeypatch.setattr(mcp_gateway, "get_active_gateway", lambda: Gateway())
    steps = [_rename(n) for n in range(1, 4)]
    payload = {"tool": "google_workspace__update_drive_file", "input": steps[0], "mcp": True,
               "held": True, "steps": steps}
    out = json.loads(asyncio.run(ttl.carry_out(payload, by_principal=True)))
    assert out["done"] == 1 and "timed out" in out["error"] and calls == ["f1", "f2"]


# ── one card per ask ─────────────────────────────────────────────────────────

_GMAIL = "google_workspace__send_gmail_message"


def _reply(body: str, subject: str = "Re: 30 minutes Thursday?") -> dict[str, Any]:
    return {"to": "dana@northpeak.example", "subject": subject, "body": body}


def _hold_reply(owner: Person, body: str, subject: str = "Re: 30 minutes Thursday?") -> int:
    verdict = gate.Verdict(run=False, department="logistics", approver_person_id=owner.id, reason="Logistics is on Careful")
    return gate.hold(owner, _GMAIL, _reply(body, subject), verdict, mcp=True, session_id="email:1")


def test_a_newer_reply_on_the_same_thread_replaces_the_one_waiting(team: dict[str, Person]) -> None:
    owner = team["owner"]
    first = _hold_reply(owner, "Hi Dana, nothing is booked yet.")
    second = _hold_reply(owner, "Hi Dana, I can't confirm Thursday myself.", subject="RE: 30 minutes thursday?")
    assert ledger.get_decision_instance(first).status == ledger.STATUS_CLOSED_EXTERNALLY  # type: ignore[union-attr]
    assert ledger.get_decision_instance(second).status == ledger.STATUS_PROPOSED  # type: ignore[union-attr]
    live = [a for a in alerts_store.list_alerts(limit=10) if a.status == "unread"]
    assert [f"decision_instance:{second}" in a.topic_tags for a in live] == [True]
    # Another thread, or other people, is a reply of its own.
    third = _hold_reply(owner, "Hi Dana, Tuesday's delivery is on.", subject="Re: Delivery window")
    assert ledger.get_decision_instance(second).status == ledger.STATUS_PROPOSED  # type: ignore[union-attr]
    assert ledger.get_decision_instance(third).status == ledger.STATUS_PROPOSED  # type: ignore[union-attr]
    # Two new mails (no thread, no Re:) that share a subject are two asks.
    new_a = _hold_reply(owner, "Hi Dana, one question.", subject="Question")
    new_b = _hold_reply(owner, "Hi Dana, another question.", subject="Question")
    assert ledger.get_decision_instance(new_a).status == ledger.STATUS_PROPOSED  # type: ignore[union-attr]
    assert ledger.get_decision_instance(new_b).status == ledger.STATUS_PROPOSED  # type: ignore[union-attr]
    # Someone else's ask on the thread replaces nothing.
    verdict = gate.Verdict(run=False, department="logistics", approver_person_id=owner.id, reason="Careful")
    from openexecutive.orchestrator.schedule_tools import set_session

    with set_session(Session(session_id="chat:sam", caller_person_id=team["sam"].id)):
        theirs = gate.hold(team["sam"], _GMAIL, _reply("Hi Dana, Sam here."), verdict, mcp=True, session_id="email:2")
    assert ledger.get_decision_instance(second).status == ledger.STATUS_PROPOSED  # type: ignore[union-attr]
    assert ledger.get_decision_instance(theirs).status == ledger.STATUS_PROPOSED  # type: ignore[union-attr]


def test_a_replaced_reply_whose_card_was_opened_closes_with_it(team: dict[str, Person]) -> None:
    owner = team["owner"]
    first = _hold_reply(owner, "Hi Dana, nothing is booked yet.")
    card = alerts_store.get_alert_by_external("decision_scheduling", ledger.decision_alert_external_id(first))
    assert card is not None and card.id is not None
    alerts_store.set_status(card.id, "read")
    second = _hold_reply(owner, "Hi Dana, I can't confirm Thursday myself.")
    assert ledger.get_decision_instance(first).status == ledger.STATUS_CLOSED_EXTERNALLY  # type: ignore[union-attr]
    assert alerts_store.get_alert(card.id).status == "dismissed"  # type: ignore[union-attr]
    live = [a for a in alerts_store.list_alerts(limit=10) if a.status in ("unread", "read")]
    assert [f"decision_instance:{second}" in a.topic_tags for a in live] == [True]




# ── words said on the card ────────────────────────────────────────────────────


def _model(monkeypatch: pytest.MonkeyPatch, fields: dict[str, str], *, go: bool = False,
           said_go: bool | None = None) -> list[str]:
    from openexecutive.alerts import replan

    replan._CALLS.clear()
    replan._CALLER_CALLS.clear()
    turns: list[str] = []

    async def call(system: str, tool: dict[str, Any], turn: str, *, actor: str, max_tokens: int = 0) -> dict[str, Any]:
        turns.append(turn)
        if actor == "card_go":
            if said_go is None:
                raise replan.ReplanError("unavailable", "Couldn't read that just now. Try again.")
            return {"go": said_go}
        return {"fields": [{"field": k, "value": v} for k, v in fields.items()], "go": go, "reply": "Made it shorter."}

    monkeypatch.setattr(replan, "_call_tool", call)
    return turns


def _mail_card(team: dict[str, Person]) -> int:
    mail = {"to": "dana@else.example", "subject": "Hi", "body": "Draft text"}
    held = levels.hold("google_workspace__send_gmail_message", mail, mcp=True, reason=levels.OWN_MAIL,
                       session=_ask_as(team["sam"]))
    return int(json.loads(held)["decision_instance_id"])


def _carry_out(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    done: list[dict[str, Any]] = []

    async def carry_out(payload: dict[str, Any], *, by_principal: bool, own_card: bool = False) -> str:
        done.append(payload)
        return json.dumps({"status": "sent"})

    monkeypatch.setattr(ttl, "carry_out", carry_out)
    return done


OWNER = {"x-caller-email": "olivia@co.example"}


def test_words_on_a_held_step_rewrite_its_fields_and_approve_sends_them(
    client: TestClient, team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
) -> None:
    done = _carry_out(monkeypatch)
    turns = _model(monkeypatch, {"body": "Short text", "to": "x@else.example"})
    decision_id = _mail_card(team)
    res = client.post(f"/decisions/{decision_id}/say", json={"text": "make it shorter"}, headers=OWNER)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["reply"] == "Made it shorter." and body["approved"] is None
    # Only the step's own editable fields change: who it goes to never does.
    assert {f["field"]: f["value"] for f in body["fields"]} == {"subject": "Hi", "body": "Short text"}
    assert "<person_said>\nmake it shorter" in turns[0]
    # The card keeps what was said beside the step as filed; the card that
    # shows it sends it.
    served = json.loads(client.get(f"/decisions/{decision_id}", headers=OWNER).json()["proposed_payload_json"])
    assert served["input"]["body"] == "Draft text" and served["said_input"] == {"body": "Short text"}
    assert [f["value"] for f in served["fields"]] == ["Hi", "Draft text"]
    # What the card shows of the step is what its Approve sends.
    # A card that sends what was said shows it; a plain Approve's details
    # are the step as filed, which is what it carries out.
    with_said = " ".join(str(d.get("value", "")) for d in served["said_details"])
    assert "Short text" in with_said and "Draft text" not in with_said
    as_filed = " ".join(str(d.get("value", "")) for d in served["details"])
    assert "Draft text" in as_filed and "Short text" not in as_filed
    # A chat card shows it with what was said, which its Approve sends.
    from openexecutive.integrations import channel_cards

    in_chat = channel_cards.card_text(decision_id)
    assert in_chat is not None and "Short text" in in_chat and "Draft text" not in in_chat
    # Approve sends what was said as the edit, under the step's own rules.
    ok = client.post(f"/decisions/{decision_id}/approve", json={"edits": {"input": {"body": "Short text"}}},
                     headers=OWNER)
    assert ok.status_code == 200, ok.text
    assert done[0]["input"]["body"] == "Short text"
    after = ledger.get_decision_instance(decision_id)
    assert after is not None and after.status == "approved_with_edit"
    # What was said stays on the card, for them.
    from openexecutive.alerts import card_talk

    card = alerts_store.get_alert_by_external(ledger.DECISION_ALERT_SOURCE, f"decision:{decision_id}")
    assert card is not None
    talk = card_talk.read([card])[card.id]  # type: ignore[index]
    assert [(t.who, t.text) for t in talk.turns] == [("you", "make it shorter"), ("me", "Made it shorter.")]


def test_a_go_on_a_held_step_approves_it_at_once(
    client: TestClient, team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
) -> None:
    done = _carry_out(monkeypatch)
    _model(monkeypatch, {"body": "Short text"}, go=True, said_go=True)
    decision_id = _mail_card(team)
    seen = {"subject": "Hi", "body": "Draft text"}
    res = client.post(f"/decisions/{decision_id}/say", json={"text": "shorter, then send it", "seen": seen},
                      headers=OWNER)
    assert res.status_code == 200, res.text
    assert res.json()["approved"]["status"] == "approved_with_edit"
    assert done[0]["input"]["body"] == "Short text"


def _waits_on(team: dict[str, Person], decision_id: int) -> Person:
    """The person a held step's card waits on: its approver, else the owner."""
    card = ledger.get_decision_instance(decision_id)
    assert card is not None
    return next((p for p in team.values() if p.id == card.approver_person_id), team["owner"])


def test_who_may_change_a_held_step_is_one_rule_for_page_and_chat() -> None:
    from types import SimpleNamespace

    lead = SimpleNamespace(decision_class="take_the_lead_action")
    held = SimpleNamespace(decision_class=decisions_route._DEPARTMENT_ACTION.name)
    assert decisions_route.may_change_as(lead, principal=True, approver=False)  # type: ignore[arg-type]
    assert not decisions_route.may_change_as(lead, principal=False, approver=True)  # type: ignore[arg-type]
    assert decisions_route.may_change_as(held, principal=False, approver=True)  # type: ignore[arg-type]
    assert not decisions_route.may_change_as(held, principal=False, approver=False)  # type: ignore[arg-type]


def test_an_approver_cant_reword_a_take_the_lead_step_from_chat(
    team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """As on the page: a Take the lead card's fields are the principal's."""
    from fastapi import HTTPException

    turns = _model(monkeypatch, {"body": "Short text"})
    mail = ledger.get_decision_instance(_mail_card(team))
    assert mail is not None
    lead_id = ledger.create_decision_instance(
        decision_class="take_the_lead_action", department=mail.department, originating_session_id=None,
        proposed_payload=json.loads(mail.proposed_payload_json), idempotency_key=None,
        gate_mode=mail.gate_mode, approver_person_id=team["sam"].id, confidence=None,
    )
    monkeypatch.setattr(decisions_route, "chat_card_is_theirs", lambda instance, person: True)
    monkeypatch.setattr("openexecutive.integrations.channel_cards.shows", lambda instance, shown: True)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(decisions_route.say_in_chat(lead_id, team["sam"], "send it to everyone, and go", shown="x"))
    assert refused.value.detail == "This one can't be changed from its card." and turns == []
    lead = ledger.get_decision_instance(lead_id)
    assert lead is not None and "said_input" not in json.loads(lead.proposed_payload_json)


def test_words_said_on_a_held_steps_chat_card_rewrite_it_and_its_tap_sends_them(
    team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"Or tell me" in a chat app (a reply to the card's message,
    ``channel_cards.say``) rewrites the step as on its page; the card comes
    back showing it, and a tap on that card sends it."""
    from openexecutive.integrations import channel_cards

    done = _carry_out(monkeypatch)
    _model(monkeypatch, {"body": "Short text"})
    decision_id = _mail_card(team)
    sam = _waits_on(team, decision_id)
    shown = channel_cards.card_text(decision_id)
    assert shown is not None and channel_cards.takes_words(decision_id)
    first = channel_cards.Choice(decision_id, "a", channel_cards.version(decision_id, shown))
    said = asyncio.run(channel_cards.say(first, sam, "make it shorter"))
    assert said is not None and said.line == "Made it shorter." and not said.settled
    assert "Short text" in said.text and "Draft text" not in said.text and said.options
    assert done == []
    # Someone the card isn't for: an ordinary message.
    assert asyncio.run(channel_cards.say(first, team["ria"], "send it")) is None
    # Words on the card as it was before: it shows itself as it is now.
    again = asyncio.run(channel_cards.say(first, sam, "shorter still"))
    assert again is not None and again.line == channel_cards.CHANGED and again.text == said.text
    # A tap checked against the card before the words, landing after them,
    # does nothing: what is done is what the tapped card showed.
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as late:
        asyncio.run(decisions_route.resolve_in_chat(decision_id, sam, approve=True, shown=first.version))
    assert late.value.status_code == 409 and done == []
    # A tap on the card as it shows now sends what was said.
    tap = channel_cards.parse_choice(said.options[0][1])
    assert tap is not None
    asyncio.run(channel_cards.tap(tap, sam))
    assert done[0]["input"]["body"] == "Short text"


def test_a_go_said_on_a_held_steps_chat_card_carries_it_out(
    team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.integrations import channel_cards

    done = _carry_out(monkeypatch)
    _model(monkeypatch, {"body": "Short text"}, go=True, said_go=True)
    decision_id = _mail_card(team)
    shown = channel_cards.card_text(decision_id)
    assert shown is not None
    said = asyncio.run(channel_cards.say(
        channel_cards.Choice(decision_id, "a", channel_cards.version(decision_id, shown)), _waits_on(team, decision_id),
        "shorter, then send it",
    ))
    assert said is not None and said.settled and said.options == ()
    assert done[0]["input"]["body"] == "Short text"
    after = ledger.get_decision_instance(decision_id)
    assert after is not None and after.status == "approved_with_edit"


def test_a_go_that_fails_once_tried_says_so_and_closes_the_card(
    client: TestClient, team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def carry_out(payload: dict[str, Any], *, by_principal: bool, own_card: bool = False) -> str:
        raise TimeoutError

    monkeypatch.setattr(ttl, "carry_out", carry_out)
    _model(monkeypatch, {"body": "Short text"}, go=True, said_go=True)
    decision_id = _mail_card(team)
    res = client.post(f"/decisions/{decision_id}/say", json={"text": "shorter, then send it"}, headers=OWNER)
    assert res.status_code == 200, res.text
    body = res.json()
    # It may have gone out: never "try again", and the card shows it failed.
    assert "may not have gone through" in body["reply"]
    assert body["approved"]["status"] == ledger.STATUS_FAILED


def test_a_go_without_what_the_page_showed_is_judged_against_the_card(
    client: TestClient, team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _carry_out(monkeypatch)
    turns = _model(monkeypatch, {"body": "Short text"}, go=True, said_go=True)
    decision_id = _mail_card(team)
    res = client.post(f"/decisions/{decision_id}/say", json={"text": "shorter, then send it"}, headers=OWNER)
    assert res.status_code == 200, res.text
    # The step as the card showed it is what the rewrite is checked against.
    assert "<plan_before>" in turns[-1] and "Draft text" in turns[-1]


def test_a_go_past_the_hours_limit_says_so(
    client: TestClient, team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.alerts import replan

    done = _carry_out(monkeypatch)
    _model(monkeypatch, {"body": "Short text"}, go=True)

    async def spent(*args: Any, **kwargs: Any) -> bool:
        raise replan.ReplanError("too_many", "That's a lot of changes this hour. Try again later.")

    monkeypatch.setattr(replan, "said_go", spent)
    decision_id = _mail_card(team)
    res = client.post(f"/decisions/{decision_id}/say", json={"text": "shorter, then send it"}, headers=OWNER)
    assert res.status_code == 200, res.text
    assert res.json()["reply"] == "That's a lot of changes this hour. Try again later." and done == []


def test_someone_the_held_step_doesnt_wait_on_cant_reword_it(
    client: TestClient, team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = _model(monkeypatch, {"body": "Short text"})
    decision_id = _mail_card(team)
    # Ria is a teammate the card doesn't wait on: nothing is rewritten or kept.
    res = client.post(f"/decisions/{decision_id}/say", json={"text": "change the body"},
                      headers={"x-caller-email": RIA})
    assert res.status_code in (403, 404, 409) and turns == []
    instance = ledger.get_decision_instance(decision_id)
    assert instance is not None and "said_input" not in json.loads(instance.proposed_payload_json)


def test_a_go_that_cant_be_read_leaves_the_step_waiting(
    client: TestClient, team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
) -> None:
    done = _carry_out(monkeypatch)
    _model(monkeypatch, {"body": "Short text"}, go=True, said_go=None)
    decision_id = _mail_card(team)
    res = client.post(f"/decisions/{decision_id}/say", json={"text": "shorter, then send it"}, headers=OWNER)
    assert res.status_code == 200, res.text
    assert res.json()["approved"] is None and done == []
    instance = ledger.get_decision_instance(decision_id)
    assert instance is not None and instance.status == ledger.STATUS_PROPOSED
    # It doesn't say it's on it when the step still waits.
    assert res.json()["reply"] == "Here it is now. Tap Do it to go ahead."


def test_an_approve_sends_only_what_its_card_showed(
    client: TestClient, team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
) -> None:
    done = _carry_out(monkeypatch)
    _model(monkeypatch, {"body": "Short text"})
    decision_id = _mail_card(team)
    assert client.post(f"/decisions/{decision_id}/say", json={"text": "make it shorter"}, headers=OWNER).status_code == 200
    # A page or chat card that showed the step as filed sends it as filed,
    # never wording said after it was shown.
    ok = client.post(f"/decisions/{decision_id}/approve", json={}, headers=OWNER)
    assert ok.status_code == 200, ok.text
    assert done[0]["input"]["body"] == "Draft text"
    after = ledger.get_decision_instance(decision_id)
    assert after is not None and after.status == "approved_unchanged"
    # What was carried out shows as sent: what was said is in it, not beside it.
    final = json.loads(after.final_payload_json or "{}")
    assert "said_input" not in final



def test_words_the_step_cant_take_change_nothing(
    client: TestClient, team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
) -> None:
    done = _carry_out(monkeypatch)
    # A rewrite the step's own rules for a change turn down.
    _model(monkeypatch, {"body": "x" * (ttl.EDIT_MAX + 1)}, go=True, said_go=True)
    decision_id = _mail_card(team)
    res = client.post(f"/decisions/{decision_id}/say", json={"text": "make it much longer, send it"}, headers=OWNER)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["approved"] is None and done == []
    assert body["reply"].startswith("I couldn't change it that way") and body["reply"].endswith("Nothing changed.")
    # Nothing kept, so every way of approving still works.
    instance = ledger.get_decision_instance(decision_id)
    assert instance is not None and "said_input" not in json.loads(instance.proposed_payload_json)
    assert client.post(f"/decisions/{decision_id}/approve", json={}, headers=OWNER).status_code == 200

def test_a_go_on_a_long_email_is_judged_on_all_of_it(
    client: TestClient, team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
) -> None:
    done = _carry_out(monkeypatch)
    turns = _model(monkeypatch, {}, go=True, said_go=True)
    body = "Long text. " * 80
    mail = {"to": "dana@else.example", "subject": "Hi", "body": body}
    held = levels.hold("google_workspace__send_gmail_message", mail, mcp=True, reason=levels.OWN_MAIL,
                       session=_ask_as(team["sam"]))
    decision_id = int(json.loads(held)["decision_instance_id"])
    seen = {"subject": "Hi", "body": body.strip()}
    res = client.post(f"/decisions/{decision_id}/say", json={"text": "send it", "seen": seen}, headers=OWNER)
    assert res.status_code == 200, res.text
    # What they saw was the step as it goes: no "rewritten, not seen" part.
    assert "<plan_before>" not in turns[-1]
    assert res.json()["approved"] is not None and len(done) == 1


def test_a_field_too_long_to_rewrite_is_changed_by_hand(
    client: TestClient, team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.alerts import replan

    turns = _model(monkeypatch, {"body": "cut"})
    assert replan._FIELD_CHARS == ttl.EDIT_MAX
    mail = {"to": "dana@else.example", "subject": "Hi", "body": "x" * (ttl.EDIT_MAX + 1)}
    held = levels.hold("google_workspace__send_gmail_message", mail, mcp=True, reason=levels.OWN_MAIL,
                       session=_ask_as(team["sam"]))
    decision_id = int(json.loads(held)["decision_instance_id"])
    res = client.post(f"/decisions/{decision_id}/say", json={"text": "fix the greeting"}, headers=OWNER)
    assert res.status_code == 422 and turns == []


def test_words_on_a_held_step_are_for_the_person_it_waits_on(
    client: TestClient, team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _model(monkeypatch, {"text": "Hi there"})
    decision_id = _hold(team)
    assert client.post(f"/decisions/{decision_id}/say", json={"text": "say hi"},
                       headers={"x-caller-email": RIA}).status_code == 404
    res = client.post(f"/decisions/{decision_id}/say", json={"text": "say hi"}, headers={"x-caller-email": SAM})
    assert res.status_code == 200, res.text


def test_a_card_for_several_calls_takes_no_words(
    client: TestClient, team: dict[str, Person], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.orchestrator.department_gate import Verdict

    turns = _model(monkeypatch, {})
    verdict = Verdict(run=False, approver_person_id=team["owner"].id, reason="it read your own mail first")
    steps = [_rename(n) for n in range(1, 4)]
    decision_id = gate.hold(
        team["owner"], "google_workspace__update_drive_file", steps[0], verdict,
        mcp=True, held=True, more=steps[1:],
    )
    res = client.post(f"/decisions/{decision_id}/say", json={"text": "rename it"}, headers=OWNER)
    assert res.status_code == 409 and turns == []
