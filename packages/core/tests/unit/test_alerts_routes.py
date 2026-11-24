"""Alert queue-grooming routes: ack feedback + mute, bulk-ack, reopen, review."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openexecutive.alerts import lifecycle
from openexecutive.alerts import store as alert_store
from openexecutive.api.routes import alerts as alerts_route
from openexecutive.memory import episodic
from openexecutive.monitoring import store as monitoring_store
from openexecutive.people import store as people_store


@pytest.fixture()
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "alerts.db"
    monkeypatch.setattr(alert_store, "DB_PATH", db_path)
    monkeypatch.setattr(episodic, "DB_PATH", db_path)
    # The principal check reads the roster; without this it reads
    # ./episodic_memory.db, which may lack a people table (403).
    monkeypatch.setattr(people_store, "DB_PATH", db_path)
    monkeypatch.setattr(lifecycle, "_ttl_settings", lambda: (3, 14))
    alert_store.initialize_db(db_path)
    episodic.initialize_db(db_path)
    monitoring_store.initialize_db(db_path)
    people_store.initialize_db(db_path)
    return db_path


@pytest.fixture()
def client(db: Path) -> TestClient:
    app = FastAPI()
    app.include_router(alerts_route.router)
    return TestClient(app)


def _insert(db: Path, headline: str, **kw: object) -> int:
    params: dict = dict(
        source="triage", external_id=f"ext-{headline}", severity="medium",
        headline=headline, body="b", db_path=db,
    )
    params.update(kw)
    aid = alert_store.insert_alert(**params)
    assert aid is not None
    return aid


def _watch(db: Path, slug: str) -> None:
    monitoring_store.insert_watchlist_item(
        slug=slug, signal_type="rss", target="https://example.com/feed", db_path=db,
    )


def test_single_ack_still_works_and_teaches_the_watch(client: TestClient, db: Path) -> None:
    _watch(db, "acme-news")
    aid = _insert(db, "acme raised", source="rss", topic_tags=["external:rss", "external:acme-news"])
    res = client.post(f"/alerts/{aid}/ack", json={"status": "dismissed"})
    assert res.status_code == 200 and res.json()["status"] == "dismissed"
    item = monitoring_store.get_watchlist_item_by_slug("acme-news", db_path=db)
    assert item is not None and item.dismiss_count == 1 and item.trust_score == pytest.approx(0.8)

    # Approving a later alert from the same watch recovers a little trust.
    aid2 = _insert(db, "acme again", source="rss", topic_tags=["external:rss", "external:acme-news"])
    client.post(f"/alerts/{aid2}/ack", json={"status": "ack"})
    item = monitoring_store.get_watchlist_item_by_slug("acme-news", db_path=db)
    assert item is not None and item.trust_score == pytest.approx(0.85)
    assert client.post("/alerts/9999/ack", json={"status": "ack"}).status_code == 404
    assert client.post(f"/alerts/{aid}/ack", json={"status": "expired"}).status_code == 422
    # Re-posting the same status is idempotent for the feedback loop.
    client.post(f"/alerts/{aid}/ack", json={"status": "dismissed"})
    client.post(f"/alerts/{aid}/ack", json={"status": "dismissed"})
    item = monitoring_store.get_watchlist_item_by_slug("acme-news", db_path=db)
    assert item is not None and item.dismiss_count == 1


def test_ack_with_mute_topic_adds_a_mute(client: TestClient, db: Path) -> None:
    aid = _insert(db, "newsletter", topic_tags=["external:rss", "external:acme-news"])
    res = client.post(f"/alerts/{aid}/ack", json={"status": "dismissed", "mute_topic": True})
    assert res.status_code == 200
    assert [m.pattern for m in alert_store.list_mutes(db_path=db)] == ["external:acme-news"]
    aid2 = _insert(db, "finance thing", topic_tags=["finance", "department:finance"])
    client.post(f"/alerts/{aid2}/ack", json={"status": "dismissed", "mute_topic": "finance"})
    assert sorted(m.pattern for m in alert_store.list_mutes(db_path=db)) == ["external:acme-news", "finance"]
    # An ack (not a dismiss) never mutes, even if asked.
    aid3 = _insert(db, "keep", topic_tags=["customer"])
    client.post(f"/alerts/{aid3}/ack", json={"status": "ack", "mute_topic": True})
    assert len(alert_store.list_mutes(db_path=db)) == 2
    # A broad pattern (substring-matched against every future alert) is refused.
    aid4 = _insert(db, "broad", topic_tags=["customer"])
    client.post(f"/alerts/{aid4}/ack", json={"status": "dismissed", "mute_topic": "e"})
    assert len(alert_store.list_mutes(db_path=db)) == 2


def test_bulk_ack_by_ids_and_by_age(client: TestClient, db: Path) -> None:
    a = _insert(db, "a")
    b = _insert(db, "b")
    old_mon = _insert(db, "old mon", source="rss", severity="low", topic_tags=["external:rss"])
    old_art = _insert(db, "old artifact", source="artifact", topic_tags=["artifact"])
    with sqlite3.connect(str(db)) as conn:
        for aid in (old_mon, old_art):
            conn.execute(
                "UPDATE alerts SET created_at=? WHERE id=?",
                ((datetime.now(UTC) - timedelta(days=10)).isoformat(), aid),
            )
        conn.commit()

    res = client.post("/alerts/bulk-ack", json={"status": "dismissed", "alert_ids": [a, b]})
    assert res.status_code == 200 and res.json() == {"count": 2}
    res = client.post(
        "/alerts/bulk-ack",
        json={"status": "dismissed", "older_than_days": 7, "category": "monitoring"},
    )
    assert res.json() == {"count": 1}
    assert alert_store.get_alert(old_mon, db_path=db).status == "dismissed"  # type: ignore[union-attr]
    # Exempt sources are never bulk-closed by an age sweep.
    assert alert_store.get_alert(old_art, db_path=db).status == "unread"  # type: ignore[union-attr]
    assert client.post("/alerts/bulk-ack", json={"status": "ack"}).status_code == 400
    assert client.post("/alerts/bulk-ack", json={"status": "read", "alert_ids": [a]}).status_code == 422
    # A 0-day sweep would close the whole company queue → rejected.
    assert client.post("/alerts/bulk-ack", json={"status": "dismissed", "older_than_days": 0}).status_code == 422
    assert client.post("/alerts/bulk-ack", json={"status": "ack", "alert_ids": []}).json() == {"count": 0}
    assert client.post("/alerts/bulk-ack", json={"status": "ack", "alert_ids": list(range(501))}).status_code == 422


def test_bulk_ack_teaches_the_watch_like_a_single_dismiss(client: TestClient, db: Path) -> None:
    _watch(db, "acme-news")
    ids = [
        _insert(db, f"n{i}", source="rss", severity="low", topic_tags=["external:rss", "external:acme-news"])
        for i in range(2)
    ]
    assert client.post("/alerts/bulk-ack", json={"status": "dismissed", "alert_ids": ids}).json() == {"count": 2}
    item = monitoring_store.get_watchlist_item_by_slug("acme-news", db_path=db)
    assert item is not None and item.dismiss_count == 2 and item.trust_score == pytest.approx(0.64)


def test_a_bulk_dismiss_leaves_a_suggestions_cards_suggestions_pending(client: TestClient, db: Path) -> None:
    # Housekeeping, not a no: only a single-card dismiss declines them.
    from openexecutive.monitoring.research import watch_policy as wp

    monitoring_store.insert_watchlist_item(
        slug="rss-s", signal_type="rss", target="https://s.com/feed",
        mode="dry_run", origin="research_proposed", db_path=db,
    )
    (item,) = monitoring_store.list_pending_suggestions(db_path=db)
    card = _insert(db, "w", source=wp.NUDGE_ALERT_SOURCE, body=wp.nudge_body([item]), topic_tags=["watchlist"])
    assert client.post("/alerts/bulk-ack", json={"status": "dismissed", "alert_ids": [card]}).json() == {"count": 1}
    assert [i.slug for i in monitoring_store.list_pending_suggestions(db_path=db)] == ["rss-s"]
    assert monitoring_store.list_declines(db_path=db) == []


def test_reopen_brings_a_closed_alert_back(
    client: TestClient, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.orchestrator.artifact_records import Viewer

    monkeypatch.setattr(alerts_route, "_caller_viewer", lambda request: Viewer(person_id=7))
    aid = _insert(db, "closed")
    alert_store.set_status(aid, "expired", db_path=db)
    alert_store.set_review(aid, verdict="likely_stale", note="x", db_path=db)
    res = client.post(f"/alerts/{aid}/reopen")
    assert res.status_code == 200
    assert res.json()["status"] == "unread" and res.json()["review_verdict"] == ""
    assert client.post(f"/alerts/{aid}/reopen").status_code == 409
    assert client.post("/alerts/9999/reopen").status_code == 404
    # `ack` and exempt sources are not Undo targets.
    acked = _insert(db, "approved")
    alert_store.set_status(acked, "ack", db_path=db)
    assert client.post(f"/alerts/{acked}/reopen").status_code == 409
    art = _insert(db, "doc", source="artifact", topic_tags=["artifact"], owner_person_id=7)
    alert_store.set_status(art, "dismissed", db_path=db)
    assert client.post(f"/alerts/{art}/reopen").status_code == 409


def test_someone_elses_drafted_artifact_is_not_found(
    client: TestClient, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.orchestrator.artifact_records import Viewer

    theirs = _insert(db, "their doc", source="artifact", topic_tags=["artifact"], owner_person_id=8)
    principals = _insert(db, "principal doc", source="artifact", topic_tags=["artifact"])
    # Not even the principal sees a teammate's document; nobody but the
    # principal sees one with no owner.
    for viewer, hidden in (
        (Viewer(person_id=1, is_principal=True), theirs),
        (Viewer(person_id=7), principals),
        (Viewer(person_id=None), theirs),
    ):
        monkeypatch.setattr(alerts_route, "_caller_viewer", lambda request, v=viewer: v)
        res = client.post(f"/alerts/{hidden}/ack", json={"status": "dismissed"})
        assert res.status_code == 404
    assert alert_store.get_alert(theirs, db_path=db).status == "unread"  # type: ignore[union-attr]
    monkeypatch.setattr(alerts_route, "_caller_viewer", lambda request: Viewer(person_id=8))
    res = client.post(f"/alerts/{theirs}/ack", json={"status": "dismissed"})
    assert res.status_code == 200 and res.json()["status"] == "dismissed"


def test_snooze_hides_a_card_until_then_and_undo_brings_it_back(
    client: TestClient, db: Path
) -> None:
    aid = _insert(db, "payout late")
    until = (datetime.now(UTC) + timedelta(hours=3)).isoformat()
    res = client.post(f"/alerts/{aid}/snooze", json={"until": until})
    assert res.status_code == 200 and res.json()["snoozed_until"] is not None
    assert aid not in {a.id for a in lifecycle.list_live_alerts(db_path=db)}

    res = client.post(f"/alerts/{aid}/snooze", json={"until": None})
    assert res.status_code == 200 and res.json()["snoozed_until"] is None
    assert aid in {a.id for a in lifecycle.list_live_alerts(db_path=db)}


def test_snooze_refuses_past_too_far_closed_and_roster_cards(
    client: TestClient, db: Path
) -> None:
    aid = _insert(db, "open")
    past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    far = (datetime.now(UTC) + timedelta(days=31)).isoformat()
    assert client.post(f"/alerts/{aid}/snooze", json={"until": past}).status_code == 422
    assert client.post(f"/alerts/{aid}/snooze", json={"until": far}).status_code == 422
    soon = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    closed = _insert(db, "closed")
    alert_store.set_status(closed, "dismissed", db_path=db)
    assert client.post(f"/alerts/{closed}/snooze", json={"until": soon}).status_code == 409
    roster = _insert(db, "who is this", source="roster_request")
    assert client.post(f"/alerts/{roster}/snooze", json={"until": soon}).status_code == 409
    assert client.post("/alerts/9999/snooze", json={"until": soon}).status_code == 404


def _as(monkeypatch: pytest.MonkeyPatch, person_id: int, *, principal: bool) -> None:
    from openexecutive.orchestrator.artifact_records import Viewer

    monkeypatch.setattr(alerts_route, "_caller_person_id", lambda request: person_id)
    monkeypatch.setattr(alerts_route, "_caller_is_principal", lambda request: principal)
    monkeypatch.setattr(alerts_route, "_caller_viewer", lambda request: Viewer(person_id=person_id, is_principal=principal))


def test_tidied_up_lists_only_what_was_on_your_needs_you(
    client: TestClient, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mine = _insert(db, "Send the board deck", routed_to_person_id=5)
    theirs = _insert(db, "Book the offsite", routed_to_person_id=6)
    unrouted = _insert(db, "Renew the domain")
    still_open = _insert(db, "Sign the lease", routed_to_person_id=5)
    for aid, reason in ((mine, "Sent on Tuesday."), (theirs, "Done."), (unrouted, "No longer needed.")):
        alert = alert_store.get_alert(aid, db_path=db)
        assert alert is not None and lifecycle.settle(alert, "resolved", reason, db_path=db)
    _as(monkeypatch, 5, principal=False)
    items = client.get("/alerts/tidied").json()["items"]
    assert [(i["alert_id"], i["reason"]) for i in items] == [(mine, "Sent on Tuesday.")]
    assert items[0]["tidied_at"]
    _as(monkeypatch, 1, principal=True)
    assert [i["alert_id"] for i in client.get("/alerts/tidied").json()["items"]] == [unrouted]
    assert still_open not in [i["alert_id"] for i in client.get("/alerts/tidied").json()["items"]]


def test_undo_puts_a_settled_waiting_step_back_and_it_waits_again(
    client: TestClient, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.api.routes import decisions
    from openexecutive.memory import decision_ledger

    told: list[tuple[int, bool]] = []
    monkeypatch.setattr(decisions, "run_after_lapse", lambda i, *, lapsed, db_path=None: told.append((i, lapsed)))
    iid = decision_ledger.create_decision_instance(
        decision_class="take_the_lead_action", department="", originating_session_id=None,
        proposed_payload={"tool": "send"}, idempotency_key=None, gate_mode="propose",
        approver_person_id=None, confidence=None, db_path=db,
    )
    aid = _insert(
        db, "Send the launch note", source="decision_scheduling",
        external_id=decision_ledger.decision_alert_external_id(iid),
        topic_tags=[decision_ledger.decision_instance_tag(iid)],
    )
    alert = alert_store.get_alert(aid, db_path=db)
    assert alert is not None and lifecycle.settle(alert, "dismissed", "No longer needed.", db_path=db)
    step = decision_ledger.get_decision_instance(iid, db_path=db)
    assert step is not None and step.status == decision_ledger.STATUS_AUTO_NO_RESPONSE
    _as(monkeypatch, 1, principal=True)
    assert [i["alert_id"] for i in client.get("/alerts/tidied").json()["items"]] == [aid]
    res = client.post(f"/alerts/{aid}/reopen")
    assert res.status_code == 200 and res.json()["status"] == "unread"
    assert res.json()["tidied_reason"] == "" and res.json()["tidied_at"] is None
    step = decision_ledger.get_decision_instance(iid, db_path=db)
    assert step is not None and step.status == decision_ledger.STATUS_PROPOSED
    assert client.get("/alerts/tidied").json()["items"] == []
    # Its class closed it out on the lapse and opened it again on Undo.
    assert told == [(iid, True), (iid, False)]
    # A step someone answered since stays answered.
    assert lifecycle.settle(alert_store.get_alert(aid, db_path=db), "dismissed", "x", db_path=db)  # type: ignore[arg-type]
    decision_ledger.unlapse(iid, db_path=db)
    decision_ledger.mark_resolved(iid, status=decision_ledger.STATUS_REJECTED, db_path=db)
    assert client.post(f"/alerts/{aid}/reopen").status_code == 409


def test_home_counts_what_was_tidied_today_and_this_week(db: Path) -> None:
    from openexecutive.api.routes import today

    fresh = _insert(db, "Fresh")
    older = _insert(db, "Older")
    gone = _insert(db, "Long gone")
    for aid in (fresh, older, gone):
        alert = alert_store.get_alert(aid, db_path=db)
        assert alert is not None and lifecycle.settle(alert, "resolved", "Done.", db_path=db)
    now = datetime.now(UTC)
    alert_store.stamp_tidied([older], "Done.", at=(now - timedelta(days=3)).isoformat(), db_path=db)
    alert_store.stamp_tidied([gone], "Done.", at=(now - timedelta(days=9)).isoformat(), db_path=db)
    assert today._tidied_counts(None, True) == (1, 2)


def test_undo_never_brings_back_a_step_asked_for_again_since(
    client: TestClient, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.memory import decision_ledger

    def ask() -> int:
        return decision_ledger.create_decision_instance(
            decision_class="take_the_lead_action", department="", originating_session_id=None,
            proposed_payload={"tool": "send", "to": "lee"}, idempotency_key="send-lee", gate_mode="propose",
            approver_person_id=None, confidence=None, db_path=db,
        )

    first = ask()
    aid = _insert(
        db, "Send Lee the note", source="decision_scheduling",
        external_id=decision_ledger.decision_alert_external_id(first),
        topic_tags=[decision_ledger.decision_instance_tag(first)],
    )
    alert = alert_store.get_alert(aid, db_path=db)
    assert alert is not None and lifecycle.settle(alert, "dismissed", "No longer needed.", db_path=db)
    ask()  # the same step, asked for again
    _as(monkeypatch, 1, principal=True)
    assert client.post(f"/alerts/{aid}/reopen").status_code == 409
    step = decision_ledger.get_decision_instance(first, db_path=db)
    assert step is not None and step.status == decision_ledger.STATUS_CLOSED_EXTERNALLY


def _waiting_card(db: Path, headline: str, *, key: str | None = None, payload: dict | None = None) -> tuple[int, int]:
    from openexecutive.memory import decision_ledger

    iid = decision_ledger.create_decision_instance(
        decision_class="take_the_lead_action", department="", originating_session_id=None,
        proposed_payload=payload or {"tool": "send", "to": headline}, idempotency_key=key, gate_mode="propose",
        approver_person_id=None, confidence=None, db_path=db,
    )
    aid = _insert(
        db, headline, source="decision_scheduling",
        external_id=decision_ledger.decision_alert_external_id(iid),
        topic_tags=[decision_ledger.decision_instance_tag(iid)],
    )
    with sqlite3.connect(str(db)) as conn:
        conn.execute("UPDATE alerts SET created_at=? WHERE id=?", ((datetime.now(UTC) - timedelta(days=90)).isoformat(), aid))
        conn.commit()
    return iid, aid


def test_the_sweep_lets_an_old_waiting_step_lapse_with_its_card(db: Path) -> None:
    from openexecutive.memory import decision_ledger

    waiting, waiting_card = _waiting_card(db, "Send the old note")
    running, running_card = _waiting_card(db, "Book the room")
    answered, answered_card = _waiting_card(db, "Email Pat")
    assert decision_ledger.claim_for_execution(running, db_path=db)
    assert decision_ledger.mark_resolved(answered, decision_ledger.STATUS_REJECTED, db_path=db)
    assert lifecycle.expire_stale_alerts(db_path=db) == 2
    card = alert_store.get_alert(waiting_card, db_path=db)
    assert card is not None and card.status == "expired" and card.tidied_reason.startswith("No change in")
    step = decision_ledger.get_decision_instance(waiting, db_path=db)
    assert step is not None and step.status == decision_ledger.STATUS_AUTO_NO_RESPONSE
    # A running step keeps its card; one answered elsewhere still lets it go.
    assert alert_store.get_alert(running_card, db_path=db).status == "unread"  # type: ignore[union-attr]
    assert alert_store.get_alert(answered_card, db_path=db).status == "expired"  # type: ignore[union-attr]


def test_undo_never_brings_back_a_step_asked_again_under_its_key(
    client: TestClient, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.memory import decision_ledger

    first, card = _waiting_card(db, "Book Lee", key="book-lee", payload={"title": "Sync", "note": "v1"})
    alert = alert_store.get_alert(card, db_path=db)
    assert alert is not None and lifecycle.settle(alert, "dismissed", "No longer needed.", db_path=db)
    decision_ledger.create_decision_instance(
        decision_class="take_the_lead_action", department="", originating_session_id=None,
        proposed_payload={"title": "Sync", "note": "v2"}, idempotency_key="book-lee", gate_mode="propose",
        approver_person_id=None, confidence=None, db_path=db,
    )
    _as(monkeypatch, 1, principal=True)
    assert client.post(f"/alerts/{card}/reopen").status_code == 409
    step = decision_ledger.get_decision_instance(first, db_path=db)
    assert step is not None and step.status == decision_ledger.STATUS_CLOSED_EXTERNALLY


def test_tidied_up_finds_your_cards_past_a_busy_week_for_the_team(db: Path) -> None:
    mine = _insert(db, "Mine", routed_to_person_id=5)
    alert_store.stamp_tidied([mine], "Done.", db_path=db)
    alert_store.set_status(mine, "resolved", db_path=db)
    for n in range(3):
        other = _insert(db, f"Team {n}", routed_to_person_id=6)
        alert_store.set_status(other, "resolved", db_path=db)
        alert_store.stamp_tidied([other], "Done.", db_path=db)
    since = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    assert [a.id for a in alert_store.list_tidied(since, reader=5, unrouted=False, limit=2, db_path=db)] == [mine]
    assert alert_store.list_tidied(since, reader=None, unrouted=True, db_path=db) == []


def test_a_teammates_own_ask_shows_on_their_tidied_up(
    client: TestClient, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.alerts.models import TEAM_RAISED_SOURCE

    ask = _insert(db, "Sam asks about pilot pricing", source=TEAM_RAISED_SOURCE, owner_person_id=7)
    alert = alert_store.get_alert(ask, db_path=db)
    assert alert is not None and lifecycle.settle(alert, "dismissed", "Pricing is set.", db_path=db)
    _as(monkeypatch, 7, principal=False)
    assert [i["alert_id"] for i in client.get("/alerts/tidied").json()["items"]] == [ask]
    _as(monkeypatch, 8, principal=False)
    assert client.get("/alerts/tidied").json()["items"] == []


def test_a_step_tag_on_any_other_card_is_just_text(db: Path) -> None:
    from openexecutive.memory import decision_ledger

    iid, _ = _waiting_card(db, "Send the real note")
    forged = _insert(db, "Mail that says decision", topic_tags=[decision_ledger.decision_instance_tag(iid)])
    alert = alert_store.get_alert(forged, db_path=db)
    assert alert is not None and decision_ledger.instance_id_of(alert) is None
    assert lifecycle.settle(alert, "dismissed", "No longer needed.", db_path=db)
    step = decision_ledger.get_decision_instance(iid, db_path=db)
    assert step is not None and step.status == decision_ledger.STATUS_PROPOSED


def test_a_lapsed_department_step_is_audited_as_the_executives(
    client: TestClient, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive import audit
    from openexecutive.memory import decision_ledger

    rows: list[tuple[str, str | None]] = []
    monkeypatch.setattr(audit, "log_event", lambda event_type, summary, **kw: rows.append((event_type, kw.get("actor"))))
    iid = decision_ledger.create_decision_instance(
        decision_class="department_action", department="sales", originating_session_id=None,
        proposed_payload={"tool": "send", "summary": "Email the quote", "department": "sales"},
        idempotency_key=None, gate_mode="propose", approver_person_id=None, confidence=None, db_path=db,
    )
    card = _insert(
        db, "Email the quote", source="decision_scheduling",
        external_id=decision_ledger.decision_alert_external_id(iid),
        topic_tags=[decision_ledger.decision_instance_tag(iid)],
    )
    alert = alert_store.get_alert(card, db_path=db)
    assert alert is not None and lifecycle.settle(alert, "expired", "No change in 14 days.", db_path=db)
    assert ("department_action_lapsed", "executive") in rows
    _as(monkeypatch, 1, principal=True)
    assert client.post(f"/alerts/{card}/reopen").status_code == 200
    assert ("department_action_waiting", "user") in rows


def test_the_one_who_raised_an_ask_sees_it_tidied_wherever_it_was_routed(
    client: TestClient, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.alerts.models import TEAM_RAISED_SOURCE

    ask = _insert(db, "Sam asks Dana about pricing", source=TEAM_RAISED_SOURCE, owner_person_id=7, routed_to_person_id=9)
    alert = alert_store.get_alert(ask, db_path=db)
    assert alert is not None and lifecycle.settle(alert, "dismissed", "Pricing is set.", db_path=db)
    for person in (7, 9):
        _as(monkeypatch, person, principal=False)
        assert [i["alert_id"] for i in client.get("/alerts/tidied").json()["items"]] == [ask]
