"""A card raised from one person's own chat stays with that person.

Someone asks for something in their own chat and a card waits for a tap. That
card shows to them and to whoever it went to for the OK: not to the rest of
the team under "Across the team", not in someone else's brief, and nobody
else can act on it by id. A card with no chat behind it (a department's run,
company mail, a watch) is the team's, as before.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openexecutive.alerts import lifecycle, own_chat
from openexecutive.alerts import store as alerts_store
from openexecutive.api.routes import today as today_route
from openexecutive.api.routes.alerts import router as alerts_router
from openexecutive.departments import registry as dept_registry
from openexecutive.departments import store as dept_store
from openexecutive.memory import decision_ledger, episodic, session_store
from openexecutive.orchestrator.artifact_records import viewer_for_person
from openexecutive.people import registry as people_registry
from openexecutive.people import store as people_store

OWNER = "jordan@example.com"
MATE = "sam@example.com"
OTHER = "riley@example.com"
_SECRET = "shared-s3cret"


@pytest.fixture()
def team(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    db = tmp_path / "own_chat_cards.db"
    for module in (episodic, dept_store, people_store, alerts_store):
        monkeypatch.setattr(module, "DB_PATH", db)
    from openexecutive.audit import logger as audit_logger

    monkeypatch.setattr(audit_logger, "_default_logger", audit_logger.AuditLogger(db_path=db))
    monkeypatch.setenv("BACKEND_SHARED_SECRET", _SECRET)
    dept_registry.invalidate()
    people_registry.invalidate()
    episodic.initialize_db(db)
    dept_store.initialize_db(db)
    people_store.initialize_db(db)
    alerts_store.initialize_db(db)
    ids = {
        "owner": people_store.upsert_person(
            full_name="Jordan Avery", is_principal=True, email=OWNER, kind="team", db_path=db,
        ),
        "mate": people_store.upsert_person(full_name="Sam Lee", email=MATE, kind="team", db_path=db),
        "other": people_store.upsert_person(full_name="Riley Cho", email=OTHER, kind="team", db_path=db),
    }
    people_registry.invalidate()
    now = datetime.now(UTC).isoformat()
    session_store.create_session("s-owner", "investor note", now, caller_person_id=ids["owner"], db_path=db)
    session_store.create_session("s-mate", "refund", now, caller_person_id=ids["mate"], db_path=db)
    yield ids  # type: ignore[misc]
    dept_registry.invalidate()
    people_registry.invalidate()


def _card(
    headline: str, *, session: str | None, approver: int | None, routed: int | None,
    decision_class: str = "department_action", department: str = "",
) -> int:
    decision = decision_ledger.create_decision_instance(
        decision_class=decision_class, department=department, originating_session_id=session,
        proposed_payload={"tool": "send_email"}, idempotency_key=None, gate_mode="approve",
        approver_person_id=approver, confidence=None,
    )
    alert = alerts_store.insert_alert(
        source=decision_ledger.DECISION_ALERT_SOURCE,
        external_id=decision_ledger.decision_alert_external_id(decision),
        severity="medium", headline=headline, body="waiting for an OK",
        topic_tags=[decision_ledger.decision_instance_tag(decision)],
        routed_to_person_id=routed,
    )
    assert alert is not None
    return alert


def _seen(person: int | None) -> set[str]:
    return {a.headline for a in lifecycle.list_live_alerts(viewer=viewer_for_person(person))}


def _today(person: int, *, principal: bool = False) -> set[str]:
    built = today_route._build_today(include_private=principal, viewer=viewer_for_person(person))
    return {p.headline for p in built.proposals}


def test_a_chat_card_shows_to_its_person_and_its_approver_only(team: dict[str, int]) -> None:
    _card("Email the investor", session="s-owner", approver=team["owner"], routed=team["owner"])
    _card("Refund order 12", session="s-mate", approver=team["mate"], routed=team["mate"])
    _card("Sam asks: pay the vendor", session="s-mate", approver=team["owner"], routed=team["owner"])
    _card("Ops check-in wants an OK", session=None, approver=team["mate"], routed=team["mate"],
          department="ops")
    _card("Reply to company mail", session="mail-thread-9", approver=team["mate"], routed=team["mate"])

    team_cards = {"Ops check-in wants an OK", "Reply to company mail"}
    assert _seen(team["owner"]) == team_cards | {"Email the investor", "Sam asks: pay the vendor"}
    assert _seen(team["mate"]) == team_cards | {"Refund order 12", "Sam asks: pay the vendor"}
    assert _seen(team["other"]) == team_cards
    assert _seen(None) == team_cards
    # A read for nobody in particular (the review, the reflection).
    assert {a.headline for a in lifecycle.list_live_alerts()} == team_cards


def test_today_keeps_them_out_of_across_the_team(team: dict[str, int]) -> None:
    _card("Email the investor", session="s-owner", approver=team["owner"], routed=team["owner"])
    _card("Refund order 12", session="s-mate", approver=team["mate"], routed=team["mate"])
    _card("Ops check-in wants an OK", session=None, approver=team["mate"], routed=team["mate"],
          department="ops")

    assert _today(team["owner"], principal=True) == {"Email the investor", "Ops check-in wants an OK"}
    assert _today(team["mate"]) == {"Refund order 12", "Ops check-in wants an OK"}
    assert _today(team["other"]) == {"Ops check-in wants an OK"}


def test_a_build_for_nobody_has_only_its_readers_chat_cards(team: dict[str, int]) -> None:
    _card("Email the investor", session="s-owner", approver=team["owner"], routed=team["owner"])
    _card("Refund order 12", session="s-mate", approver=team["mate"], routed=team["mate"])

    def built(**kw: object) -> set[str]:
        return {p.headline for p in today_route._build_today(**kw).proposals}  # type: ignore[arg-type]

    assert built() == set()
    assert built(include_private=True) == {"Email the investor"}
    assert built(chat_reader=team["mate"]) == {"Refund order 12"}


def test_a_chat_saved_before_callers_is_the_principals(team: dict[str, int]) -> None:
    session_store.create_session(
        "s-old", "old chat", datetime.now(UTC).isoformat(), db_path=episodic.DB_PATH,
    )
    _card("From an old chat", session="s-old", approver=None, routed=None)
    assert _seen(team["owner"]) == {"From an old chat"}
    assert _seen(team["mate"]) == set()


def test_deleting_the_chat_does_not_hand_the_card_to_the_team(team: dict[str, int]) -> None:
    _card("Refund order 12", session="s-mate", approver=team["mate"], routed=team["mate"])
    session_store.delete_session("s-mate", db_path=episodic.DB_PATH)
    assert _seen(team["mate"]) == {"Refund order 12"}
    assert _seen(team["other"]) == set()
    assert _seen(team["owner"]) == set()


def test_a_decision_tag_on_another_alert_is_just_text(team: dict[str, int]) -> None:
    card = _card("Refund order 12", session="s-mate", approver=team["mate"], routed=team["mate"])
    real = alerts_store.get_alert(card)
    assert real is not None
    tag = next(t for t in real.topic_tags if t.startswith("decision_instance:"))
    # An alert that only carries the tag neither hides itself nor unhides the card.
    alerts_store.insert_alert(
        source="email", external_id="msg-1", severity="medium", headline="Vendor wrote in",
        body="hello", topic_tags=[tag],
    )
    assert _seen(team["other"]) == {"Vendor wrote in"}
    assert _seen(team["mate"]) == {"Vendor wrote in", "Refund order 12"}


def test_the_held_step_itself_is_theirs_too(team: dict[str, int]) -> None:
    from openexecutive.api.routes.decisions import router as decisions_router

    _card("Refund order 12", session="s-mate", approver=team["mate"], routed=team["mate"])
    _card("Ops check-in wants an OK", session=None, approver=team["mate"], routed=team["mate"],
          department="ops")
    app = FastAPI()
    app.include_router(decisions_router)
    client = TestClient(app, client=("203.0.113.5", 50000))

    def listed(email: str) -> list[int]:
        r = client.get(
            "/decisions", params={"decision_class": "department_action"},
            headers={"x-api-key": _SECRET, "x-caller-email": email},
        )
        assert r.status_code == 200, r.text
        return sorted(d["id"] for d in r.json())

    def opened(email: str, decision: int) -> int:
        return client.get(
            f"/decisions/{decision}", headers={"x-api-key": _SECRET, "x-caller-email": email},
        ).status_code

    # department_action is the principal's and its approver's; the one from
    # Sam's own chat is not the principal's either.
    assert listed(OWNER) == [2]
    assert opened(OWNER, 1) == 404
    assert opened(OWNER, 2) == 200
    assert opened(MATE, 1) == 200
    assert opened(OTHER, 1) == 404
    r = client.post(
        "/decisions/1/reject", json={}, headers={"x-api-key": _SECRET, "x-caller-email": OWNER},
    )
    assert r.status_code == 404, r.text


def test_an_unreadable_ledger_leaves_a_card_with_its_routed_person(
    team: dict[str, int], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _card("Refund order 12", session="s-mate", approver=team["mate"], routed=team["mate"])

    def boom(*_a: object, **_k: object) -> dict[int, object]:
        raise RuntimeError("ledger down")

    monkeypatch.setattr(decision_ledger, "get_decision_instances", boom)
    assert _seen(team["mate"]) == {"Refund order 12"}
    assert _seen(team["other"]) == set()
    assert _seen(team["owner"]) == set()


def test_nobody_else_acts_on_it_by_id(team: dict[str, int]) -> None:
    alert = _card("Refund order 12", session="s-mate", approver=team["mate"], routed=team["mate"])
    app = FastAPI()
    app.include_router(alerts_router)
    client = TestClient(app, client=("203.0.113.5", 50000))

    def snooze(email: str) -> int:
        return client.post(
            f"/alerts/{alert}/snooze", json={"until": None},
            headers={"x-api-key": _SECRET, "x-caller-email": email},
        ).status_code

    assert snooze(OTHER) == 404
    assert snooze(OWNER) == 404
    assert snooze(MATE) == 200


def test_readers_leaves_team_cards_out(team: dict[str, int]) -> None:
    alert = _card("Ops check-in wants an OK", session=None, approver=team["mate"], routed=team["mate"],
                  department="ops")
    found = alerts_store.get_alert(alert)
    assert found is not None
    assert own_chat.readers([found]) == {}
    assert own_chat.keep_for([found], None) == [found]


def test_a_page_of_other_peoples_held_steps_does_not_hide_the_teams(team: dict[str, int]) -> None:
    from openexecutive.api.routes.decisions import router as decisions_router

    _card("Ops check-in wants an OK", session=None, approver=team["mate"], routed=team["mate"],
          department="ops")
    for n in (12, 13):
        _card(f"Refund order {n}", session="s-mate", approver=team["mate"], routed=team["mate"])
    app = FastAPI()
    app.include_router(decisions_router)
    client = TestClient(app, client=("203.0.113.5", 50000))
    r = client.get(
        "/decisions", params={"decision_class": "department_action", "limit": 1},
        headers={"x-api-key": _SECRET, "x-caller-email": OWNER},
    )
    assert r.status_code == 200, r.text
    # The two newest are from Sam's own chat; the team's older one still lists.
    assert [d["id"] for d in r.json()] == [1]
