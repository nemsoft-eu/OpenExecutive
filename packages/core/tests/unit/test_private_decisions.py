"""A decision made in one person's own chat is theirs until they share it
(``memory.facts`` ``private_to_person_id``): only they read it, on the Memory
page, in their chats and in their briefs. Shared, it is the team's as before.
Decisions from a shared chat, a channel or a meeting are shared from the start.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openexecutive.api.routes import episodic as episodic_route
from openexecutive.delegation.settings import TurnDelegation
from openexecutive.memory import episodic, facts
from openexecutive.orchestrator import decision_tools, fact_tools
from openexecutive.orchestrator.schedule_tools import current_session
from openexecutive.people import registry as people_registry
from openexecutive.people import store as people_store

SAID = "We're going with the Q1 hire. We won't renew Acme."


@pytest.fixture(autouse=True)
def roster(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    path = tmp_path / "facts.db"
    for module in (people_store, episodic):
        monkeypatch.setattr(module, "DB_PATH", path)
    people_store.initialize_db()
    episodic.initialize_db(path)
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "openexecutive.audit.log_event",
        lambda event_type, summary, **kw: events.append({"event_type": event_type, "summary": summary, **kw}),
    )
    monkeypatch.setattr(fact_tools, "private_owner", fact_tools.own_chat_person)
    people_registry.invalidate()
    ids = SimpleNamespace(
        owner=people_store.upsert_person(full_name="Olivia Owner", is_principal=True,
                                         email="owner@northwind.test"),
        sam=people_store.upsert_person(full_name="Sam Lee", email="sam@northwind.test"),
        dana=people_store.upsert_person(full_name="Dana Roe", email="dana@northwind.test"),
        db=path, events=events,
    )
    people_registry.invalidate()
    token = current_session.set(None)
    yield ids
    current_session.reset(token)
    people_registry.invalidate()


def _speak(person_id: int | None, *, said: str = SAID, **surface: Any) -> None:
    session = SimpleNamespace(
        session_id="s-1", caller_person_id=person_id, origin_channel="", from_web_chat=True, web_caller_signed_in=True,
        unattended=False, private_to_principal=False, email_from="", company_profile=None,
        turn_delegation=TurnDelegation(speaker_text=said, session_id="s-1"),
    )
    for key, value in surface.items():
        setattr(session, key, value)
    current_session.set(session)  # type: ignore[arg-type]


def _decide(**payload: Any) -> dict[str, Any]:
    body = {"subject": "Q1 hire", "statement": "Going with the Q1 hire.",
            "source_quote": "We're going with the Q1 hire", "decision": True, **payload}
    return json.loads(asyncio.run(fact_tools.handle_remember_fact(body)))


def _client(monkeypatch: pytest.MonkeyPatch, roster: SimpleNamespace, who: str) -> TestClient:
    person = getattr(roster, who)
    monkeypatch.setattr(episodic_route, "_caller_is_principal", lambda request: who == "owner")
    monkeypatch.setattr(episodic_route, "_caller_id", lambda request: person)
    app = FastAPI()
    app.include_router(episodic_route.router)
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #


def test_a_decision_in_someones_own_chat_is_private_to_them(roster: SimpleNamespace) -> None:
    _speak(roster.sam)
    out = _decide()
    assert out["status"] == "ok" and out["private"] is True
    assert facts.list_facts() == [] and facts.decision_rows() == []
    assert facts.list_facts(reader=roster.dana) == []
    [mine] = facts.decision_rows(reader=roster.sam)
    assert mine.private_to_person_id == roster.sam and mine.status == "active"
    # In their chat, never in anyone else's or in a run for nobody.
    assert "Going with the Q1 hire." in facts.render_facts_for_prompt(reader=roster.sam)
    assert facts.render_facts_for_prompt(reader=roster.owner) == ""
    assert facts.render_facts_for_prompt() == ""
    assert facts.render_teammate_changes(datetime.now(UTC) - timedelta(days=1)) == ""
    # The audit row does not say what it was.
    [row] = [e for e in roster.events if e["event_type"] == "tool_invocation"]
    assert "Q1 hire" not in json.dumps(row, default=str)


def test_the_owners_own_chat_decision_is_private_too(roster: SimpleNamespace) -> None:
    _speak(roster.owner)
    assert _decide()["private"] is True
    assert facts.decision_rows(reader=roster.sam) == []
    assert len(facts.decision_rows(reader=roster.owner)) == 1


@pytest.mark.parametrize(
    "surface",
    [
        {"shared_chat": True},
        {"meeting_call": True},
        {"from_web_chat": False, "origin_channel": "slack", "session_id": "slack:C123:1.2"},
    ],
    ids=["shared-chat", "meeting", "channel"],
)
def test_a_decision_made_in_front_of_others_is_shared(
    roster: SimpleNamespace, surface: dict[str, Any]
) -> None:
    _speak(roster.owner, **surface)
    out = _decide()
    assert out["status"] == "ok" and "private" not in out
    [row] = facts.decision_rows()
    assert row.private_to_person_id is None


def test_a_fact_is_never_private(roster: SimpleNamespace) -> None:
    _speak(roster.sam)
    out = _decide(decision=False, subject="Cedar Court units", statement="We won't renew Acme.",
                  source_quote="We won't renew Acme")
    assert out["status"] == "ok" and "private" not in out
    assert len(facts.list_facts()) == 1


def test_a_solo_install_keeps_decisions_as_before(
    roster: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()
    # Re-point the stores the autouse fixture set, with the real owner rule.
    for module in (people_store, episodic):
        monkeypatch.setattr(module, "DB_PATH", roster.db)
    monkeypatch.setattr("openexecutive.audit.log_event", lambda *a, **k: None)
    monkeypatch.setattr(
        "openexecutive.memory.workspace_settings.effective_workspace_mode", lambda session=None: "solo"
    )
    people_registry.invalidate()
    _speak(roster.owner)
    assert "private" not in _decide()
    assert len(facts.decision_rows()) == 1


def test_private_and_team_rows_do_not_replace_each_other(roster: SimpleNamespace) -> None:
    team, _ = facts.record_fact(subject="Q1 hire", statement="No Q1 hire.", source_quote="", decision=True)
    _speak(roster.sam)
    _decide()
    _speak(roster.dana)
    _decide(statement="Going with the Q1 hire, in sales.")
    assert facts.get_fact(team.id).status == "active"  # type: ignore[union-attr]
    assert len(facts.decision_rows(reader=roster.sam)) == 2
    assert len(facts.decision_rows(reader=roster.dana)) == 2
    # The same person saying it again replaces their own.
    _speak(roster.sam)
    _decide(statement="Going with the Q1 hire, in March.")
    mine = [f for f in facts.decision_rows(reader=roster.sam) if f.private_to_person_id]
    assert [f.statement for f in mine] == ["Going with the Q1 hire, in March."]


def test_nobody_else_can_replace_or_forget_one_by_its_id(roster: SimpleNamespace) -> None:
    _speak(roster.sam)
    fact_id = _decide()["fact_id"]
    _speak(roster.dana)
    out = _decide(replaces_fact_id=fact_id, statement="No Q1 hire.")
    assert "no standing fact" in out["error"]
    _speak(roster.owner)
    out = json.loads(asyncio.run(fact_tools.handle_forget_fact(
        {"fact_id": fact_id, "rationale": "they said so", "source_quote": "We won't renew Acme"}
    )))
    assert "no standing fact" in out["error"]
    assert facts.get_fact(fact_id).status == "active"  # type: ignore[union-attr]


def test_an_edit_of_a_private_decision_stays_private(roster: SimpleNamespace) -> None:
    _speak(roster.sam)
    fact_id = _decide()["fact_id"]
    assert episodic.update_decision(fact_id, summary="Going with the Q1 hire in April.", reader=roster.sam)
    assert episodic.update_decision(fact_id, summary="x", reader=roster.dana) is False
    [mine] = facts.decision_rows(reader=roster.sam)
    assert mine.statement == "Going with the Q1 hire in April." and mine.private_to_person_id == roster.sam
    assert facts.decision_rows() == []


# --------------------------------------------------------------------------- #
# Sharing
# --------------------------------------------------------------------------- #


def test_sharing_makes_it_the_teams_and_taking_it_back_undoes_that(roster: SimpleNamespace) -> None:
    _speak(roster.sam)
    fact_id = _decide()["fact_id"]
    assert facts.share_decision(fact_id, shared=True, person_id=roster.dana) is None
    assert facts.share_decision(fact_id, shared=True, person_id=roster.owner, is_principal=True) is None
    shared = facts.share_decision(fact_id, shared=True, person_id=roster.sam)
    assert shared is not None and shared.private_to_person_id is None and shared.status == "active"
    assert "Going with the Q1 hire. (decided) (per Sam Lee)" in facts.render_facts_for_prompt()
    # Only the person who recorded it takes it back.
    assert facts.share_decision(fact_id, shared=False, person_id=roster.dana) is None
    assert facts.share_decision(fact_id, shared=False, person_id=roster.owner, is_principal=True) is None
    back = facts.share_decision(fact_id, shared=False, person_id=roster.sam)
    assert back is not None and back.private_to_person_id == roster.sam
    assert facts.render_facts_for_prompt() == ""


def test_a_teammates_share_replaces_the_owners_decision_at_once(roster: SimpleNamespace) -> None:
    owners, _ = facts.record_fact(subject="Q1 hire", statement="No Q1 hire.", source_quote="", decision=True)
    _speak(roster.sam)
    fact_id = _decide()["fact_id"]
    shared = facts.share_decision(fact_id, shared=True, person_id=roster.sam)
    assert shared is not None and shared.status == "active" and shared.private_to_person_id is None
    assert facts.get_fact(owners.id).status == "superseded"  # type: ignore[union-attr]
    assert "Going with the Q1 hire." in facts.render_facts_for_prompt()


def test_sharing_replaces_the_teams_decision_on_the_same_subject(roster: SimpleNamespace) -> None:
    old, _ = facts.record_fact(subject="Q1 hire", statement="No Q1 hire.", source_quote="", decision=True)
    _speak(roster.owner)
    fact_id = _decide()["fact_id"]
    facts.share_decision(fact_id, shared=True, person_id=roster.owner, is_principal=True)
    assert facts.get_fact(old.id).status == "superseded"  # type: ignore[union-attr]
    assert [f.id for f in facts.decision_rows()] == [fact_id]
    # Taking it back gives the team its own decision again.
    facts.share_decision(fact_id, shared=False, person_id=roster.owner, is_principal=True)
    assert [f.id for f in facts.decision_rows()] == [old.id]
    assert len(facts.decision_rows(reader=roster.owner)) == 2


# --------------------------------------------------------------------------- #
# The Memory page's routes, as two teammates and the owner
# --------------------------------------------------------------------------- #


def test_the_memory_page_lists_a_private_decision_for_its_person_alone(
    roster: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    team, _ = facts.record_fact(subject="Acme", statement="Not renewing Acme.", source_quote="", decision=True,
                                recorded_by_person_id=roster.owner)
    _speak(roster.sam)
    fact_id = _decide()["fact_id"]

    sam = _client(monkeypatch, roster, "sam").get("/memories/decisions").json()
    assert {d["id"]: (d["private"], d["can_share"], d["can_make_private"]) for d in sam} == {
        fact_id: (True, True, False), team.id: (False, False, False),
    }
    assert fact_id in {f["id"] for f in _client(monkeypatch, roster, "sam").get("/memories/facts").json()["facts"]}

    for who in ("dana", "owner"):
        client = _client(monkeypatch, roster, who)
        assert [d["id"] for d in client.get("/memories/decisions").json()] == [team.id]
        assert fact_id not in {f["id"] for f in client.get("/memories/facts").json()["facts"]}
        # Not theirs to change, remove, retire, edit or share, by its id either.
        assert client.patch(f"/memories/decisions/{fact_id}", json={"summary": "x"}).status_code in (403, 404)
        assert client.delete(f"/memories/decisions/{fact_id}").status_code in (403, 404)
        assert client.post(f"/memories/facts/{fact_id}/retire").status_code in (403, 404)
        assert client.post(f"/memories/facts/{fact_id}/edit", json={"statement": "x"}).status_code in (403, 404)
        assert client.post(f"/memories/decisions/{fact_id}/share", json={"shared": True}).status_code == 404
    assert facts.get_fact(fact_id).status == "active"  # type: ignore[union-attr]
    assert facts.get_fact(fact_id).private_to_person_id == roster.sam  # type: ignore[union-attr]

    owner = _client(monkeypatch, roster, "owner").get("/memories/decisions").json()
    assert owner[0]["can_make_private"] is True


def test_share_with_team_and_make_private_from_the_memory_page(
    roster: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _speak(roster.sam)
    fact_id = _decide()["fact_id"]
    sam = _client(monkeypatch, roster, "sam")
    out = sam.post(f"/memories/decisions/{fact_id}/share", json={"shared": True})
    assert out.status_code == 200
    assert (out.json()["private"], out.json()["can_make_private"]) == (False, True)
    assert [d["id"] for d in _client(monkeypatch, roster, "dana").get("/memories/decisions").json()] == [fact_id]
    assert any(e["event_type"] == "decision_shared" for e in roster.events)

    assert _client(monkeypatch, roster, "dana").post(
        f"/memories/decisions/{fact_id}/share", json={"shared": False}
    ).status_code == 404
    back = _client(monkeypatch, roster, "sam").post(f"/memories/decisions/{fact_id}/share", json={"shared": False})
    assert back.status_code == 200 and back.json()["private"] is True
    assert _client(monkeypatch, roster, "dana").get("/memories/decisions").json() == []


def test_its_person_edits_and_removes_their_own_private_decision(
    roster: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _speak(roster.sam)
    fact_id = _decide()["fact_id"]
    sam = _client(monkeypatch, roster, "sam")
    out = sam.patch(f"/memories/decisions/{fact_id}", json={"summary": "Going with the Q1 hire in May."})
    assert out.status_code == 200 and out.json()["private"] is True
    assert sam.delete(f"/memories/decisions/{out.json()['id']}").status_code == 204
    assert sam.get("/memories/decisions").json() == []
    # A shared decision stays the principal's to change.
    team, _ = facts.record_fact(subject="Acme", statement="Not renewing Acme.", source_quote="", decision=True)
    assert sam.patch(f"/memories/decisions/{team.id}", json={"summary": "x"}).status_code == 403


# --------------------------------------------------------------------------- #
# With no screen: outcomes, briefs, an older database
# --------------------------------------------------------------------------- #


def test_an_outcome_is_recorded_only_from_its_persons_own_chat(roster: SimpleNamespace) -> None:
    _speak(roster.owner)
    fact_id = _decide()["fact_id"]
    _speak(roster.owner, shared_chat=True)
    out = json.loads(asyncio.run(decision_tools.handle_record_decision_outcome(
        {"decision_id": fact_id, "outcome": "Worked.", "rationale": "they said it worked"}
    )))
    assert "no decision" in out["error"] and out["awaiting_outcome"] == []
    _speak(roster.owner)
    out = json.loads(asyncio.run(decision_tools.handle_record_decision_outcome(
        {"decision_id": fact_id, "outcome": "Worked.", "rationale": "they said it worked"}
    )))
    assert out["status"] == "ok"


def test_a_brief_reads_the_owners_private_decisions_only_when_it_goes_to_them_alone(
    roster: SimpleNamespace,
) -> None:
    from openexecutive.workflows.morning_brief import PRINCIPAL_DELIVERY

    _speak(roster.owner)
    _decide()
    _speak(roster.sam)
    _decide(statement="Going with the Q1 hire, in sales.")
    assert facts.brief_reader() is None
    assert facts.with_standing_facts("BRIEF", reader=facts.brief_reader()) == "BRIEF"
    token = PRINCIPAL_DELIVERY.set(True)
    try:
        assert facts.brief_reader() == roster.owner
        text = facts.with_standing_facts("BRIEF", reader=facts.brief_reader())
    finally:
        PRINCIPAL_DELIVERY.reset(token)
    assert "Going with the Q1 hire." in text and "in sales" not in text


def test_existing_decisions_stay_shared_after_the_upgrade(tmp_path: Path) -> None:
    old = tmp_path / "old.db"
    with sqlite3.connect(str(old)) as conn:
        conn.execute(
            "CREATE TABLE facts (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, "
            "subject TEXT NOT NULL, subject_key TEXT NOT NULL, statement TEXT NOT NULL, "
            "previous_statement TEXT NOT NULL DEFAULT '', source_quote TEXT NOT NULL DEFAULT '', "
            "source_channel TEXT NOT NULL DEFAULT '', session_id TEXT, turn_id TEXT, "
            "recorded_by_person_id INTEGER, created_at TEXT NOT NULL, "
            "status TEXT NOT NULL DEFAULT 'active', superseded_by INTEGER, retired_at TEXT, "
            "retired_reason TEXT NOT NULL DEFAULT '', decision INTEGER NOT NULL DEFAULT 0, "
            "department TEXT NOT NULL DEFAULT '', outcome TEXT NOT NULL DEFAULT '')"
        )
        conn.execute(
            "INSERT INTO facts (kind, subject, subject_key, statement, created_at, decision) "
            "VALUES ('fact', 'Acme', 'acme', 'Not renewing Acme.', '2026-09-01', 1)"
        )
    # Read before any write migrates it, with and without a reader.
    assert "Not renewing Acme." in facts.render_facts_for_prompt(db_path=old)
    assert "Not renewing Acme." in facts.render_facts_for_prompt(reader=7, db_path=old)
    # After the upgrade it is the team's: everyone reads it.
    [row] = facts.decision_rows(db_path=old)
    assert row.private_to_person_id is None
    assert [f.id for f in facts.decision_rows(reader=7, db_path=old)] == [row.id]


# --------------------------------------------------------------------------- #
# Found in review
# --------------------------------------------------------------------------- #


def test_an_edited_private_decision_is_shared_as_edited(
    roster: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _speak(roster.sam)
    fact_id = _decide()["fact_id"]
    sam = _client(monkeypatch, roster, "sam")
    edited = sam.patch(f"/memories/decisions/{fact_id}", json={"summary": "Hiring two people in Q1."}).json()
    row = facts.get_fact(edited["id"])
    assert row is not None and row.recorded_by_role == "teammate" and row.recorded_by_person_id == roster.sam
    assert row.private_to_person_id == roster.sam
    # The owner's "what changed" list never names it while it is private.
    from openexecutive.memory import changes

    listed = changes._facts(roster.owner, True, "2000-01-01T00:00:00+00:00", None, None)
    assert "Hiring two people" not in json.dumps(listed, default=str)
    out = sam.post(f"/memories/decisions/{edited['id']}/share", json={"shared": True})
    assert out.status_code == 200 and out.json()["waiting_for_approval"] is False
    assert "Hiring two people" in facts.render_facts_for_prompt()


def test_retiring_or_editing_a_private_decision_leaves_its_words_out_of_the_audit_log(
    roster: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _speak(roster.sam)
    fact_id = _decide()["fact_id"]
    sam = _client(monkeypatch, roster, "sam")
    edited = sam.post(f"/memories/facts/{fact_id}/edit", json={"statement": "Going with the Q1 hire in June."})
    assert edited.status_code == 200
    assert sam.post(f"/memories/facts/{edited.json()['id']}/retire").status_code == 200
    logged = json.dumps([e for e in roster.events if e["event_type"].startswith("fact_")], default=str)
    assert "fact_edited" in logged and "fact_retired" in logged
    assert "Q1 hire" not in logged


def test_a_proposal_nobody_reviewed_cannot_be_made_private_and_shared_into_use(
    roster: SimpleNamespace,
) -> None:
    waiting, _ = facts.record_fact(subject="Vendor", statement="Switching vendors.", source_quote="",
                                   decision=True, recorded_by_person_id=roster.owner)
    with facts._conn() as conn:
        conn.execute("UPDATE facts SET status='proposed' WHERE id=?", (waiting.id,))
    assert facts.share_decision(waiting.id, shared=False, person_id=roster.owner, is_principal=True) is None
    assert facts.get_fact(waiting.id).status == "proposed"  # type: ignore[union-attr]


def test_the_memory_page_count_leaves_private_decisions_out(roster: SimpleNamespace) -> None:
    _speak(roster.sam)
    _decide()
    days = episodic.count_activity_by_day(days=7, db_path=roster.db)
    assert sum(sum(v for k, v in d.items() if isinstance(v, int)) for d in days) == 0


def test_a_teammate_cannot_take_back_a_decision_the_owner_approved(
    roster: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    kept, _ = facts.record_fact(subject="Q1 hire", statement="No Q1 hire.", source_quote="", decision=True)
    _speak(roster.sam)
    fact_id = _decide()["fact_id"]
    facts.share_decision(fact_id, shared=True, person_id=roster.sam)
    with facts._conn() as conn:
        conn.execute("UPDATE facts SET approved_at='2026-09-29' WHERE id=?", (fact_id,))
    assert facts.get_fact(kept.id).status == "superseded"  # type: ignore[union-attr]
    # Approved, it is the owner's to keep or drop: the owner's old fact stays replaced.
    assert facts.share_decision(fact_id, shared=False, person_id=roster.sam) is None
    assert [f.id for f in facts.decision_rows()] == [fact_id]
    [listed] = _client(monkeypatch, roster, "sam").get("/memories/decisions").json()
    assert listed["can_make_private"] is False


def test_sharing_an_edited_private_decision_leaves_its_earlier_wording_out(
    roster: SimpleNamespace,
) -> None:
    _speak(roster.sam)
    fact_id = _decide()["fact_id"]
    assert episodic.update_decision(fact_id, summary="Going with the Q1 hire in April.", reader=roster.sam)
    [mine] = facts.decision_rows(reader=roster.sam)
    assert mine.previous_statement
    shared = facts.share_decision(mine.id, shared=True, person_id=roster.sam)
    assert shared is not None and shared.previous_statement == ""
    assert "Q1 hire in April" in facts.render_facts_for_prompt()
    assert "corrects" not in facts.render_facts_for_prompt()


def test_a_teammates_edit_stops_once_their_decision_is_shared(roster: SimpleNamespace) -> None:
    _speak(roster.sam)
    fact_id = _decide()["fact_id"]
    facts.share_decision(fact_id, shared=True, person_id=roster.sam)
    # The route checked it was theirs and private; it was shared before the write.
    assert episodic.update_decision(fact_id, summary="x", reader=roster.sam, own_only=True) is False
    [team] = facts.decision_rows()
    assert team.id == fact_id and team.recorded_by_role == "teammate"


@pytest.mark.parametrize(("scope", "reads_own"), [("p-sam", True), ("company", False)])
def test_a_teammates_own_header_reads_their_private_decisions(
    roster: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, scope: str, reads_own: bool
) -> None:
    from openexecutive.api.routes import today
    from openexecutive.briefing import narrative_cache

    class _Read(Exception):
        pass

    def render(*, reader: int | None = None, **_: Any) -> str:
        raise _Read(reader)

    monkeypatch.setattr(today, "_header_live", lambda include_private: ([], None, ""))
    monkeypatch.setattr(facts, "render_facts_for_prompt", render)
    scope = narrative_cache.COMPANY_SCOPE if scope == "company" else scope
    with pytest.raises(_Read) as read:
        today._header_context({}, SimpleNamespace(id=roster.sam), "day", scope, None)  # type: ignore[arg-type]
    assert read.value.args == (roster.sam if reads_own else None,)
