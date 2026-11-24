"""Tests for alert pipeline rules: preferences, mute checks, dispatch dispatch."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from openexecutive.alerts import store as alert_store
from openexecutive.alerts.models import (
    AlertChannel,
    AlertEvent,
    AlertSeverity,
    TriageDecision,
    UserPreferences,
)
from openexecutive.alerts.preferences import (
    matches_mute,
    resolve_channels,
    save_preferences,
)
from openexecutive.alerts.store import add_mute, initialize_db, insert_alert, list_alerts


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    db_path = tmp_path / "alerts.db"
    initialize_db(db_path)
    return db_path


def test_resolve_channels_intersects_with_enabled() -> None:
    prefs = UserPreferences(
        severity_threshold=AlertSeverity.LOW,
        channels_enabled=[AlertChannel.WEB, AlertChannel.PERSISTED],
    )
    out = resolve_channels(
        [AlertChannel.WEB, AlertChannel.SLACK_DM, AlertChannel.EMAIL],
        AlertSeverity.HIGH,
        prefs,
    )
    assert AlertChannel.WEB in out
    assert AlertChannel.SLACK_DM not in out
    assert AlertChannel.PERSISTED in out


def test_resolve_channels_drops_below_threshold() -> None:
    prefs = UserPreferences(severity_threshold=AlertSeverity.HIGH)
    out = resolve_channels(
        [AlertChannel.WEB, AlertChannel.PERSISTED],
        AlertSeverity.MEDIUM,
        prefs,
    )
    assert out == [AlertChannel.PERSISTED]


def test_resolve_channels_quiet_hours_suppresses_non_urgent() -> None:
    prefs = UserPreferences(
        severity_threshold=AlertSeverity.LOW,
        quiet_hours_start="00:00",
        quiet_hours_end="23:59",
        quiet_hours_tz="UTC",
    )
    out = resolve_channels(
        [AlertChannel.WEB, AlertChannel.PERSISTED],
        AlertSeverity.HIGH,
        prefs,
    )
    assert out == [AlertChannel.PERSISTED]


def test_resolve_channels_quiet_hours_passes_urgent_through() -> None:
    prefs = UserPreferences(
        severity_threshold=AlertSeverity.LOW,
        quiet_hours_start="00:00",
        quiet_hours_end="23:59",
        quiet_hours_tz="UTC",
    )
    out = resolve_channels(
        [AlertChannel.WEB, AlertChannel.SLACK_DM, AlertChannel.PERSISTED],
        AlertSeverity.URGENT,
        prefs,
    )
    assert AlertChannel.WEB in out
    assert AlertChannel.SLACK_DM in out


def test_resolve_channels_always_includes_persisted() -> None:
    prefs = UserPreferences(
        severity_threshold=AlertSeverity.LOW,
        channels_enabled=[AlertChannel.WEB],  # PERSISTED missing on purpose
    )
    out = resolve_channels(
        [AlertChannel.WEB],
        AlertSeverity.HIGH,
        prefs,
    )
    assert AlertChannel.PERSISTED in out


def test_matches_mute_substring_case_insensitive() -> None:
    assert matches_mute(["customer", "churn"], ["CHURN"])
    assert matches_mute(["hiring-pipeline"], ["hiring"])
    assert not matches_mute(["finance"], ["legal"])
    assert not matches_mute([], ["anything"])
    assert not matches_mute(["a"], [])


def test_save_and_reload_preferences(db: Path) -> None:
    prefs = UserPreferences(
        severity_threshold=AlertSeverity.HIGH,
        quiet_hours_start="22:00",
        quiet_hours_end="07:00",
        channels_enabled=[AlertChannel.WEB, AlertChannel.PERSISTED],
    )
    save_preferences(prefs, db_path=db)

    from openexecutive.alerts.preferences import get_preferences

    loaded = get_preferences(db_path=db)
    assert loaded.severity_threshold == AlertSeverity.HIGH
    assert loaded.quiet_hours_start == "22:00"
    assert loaded.channels_enabled == [AlertChannel.WEB, AlertChannel.PERSISTED]


def test_evaluate_and_dispatch_persists_alert_and_publishes(monkeypatch, db: Path) -> None:
    """End-to-end pipeline test with TriageAgent stubbed."""
    from openexecutive.agents import triage as triage_module
    from openexecutive.alerts import pipeline, sse_bus

    async def fake_triage(self, event, **kwargs):  # noqa: ARG001 - signature match
        return TriageDecision(
            alert=True,
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.WEB, AlertChannel.PERSISTED],
            headline="Stub alert",
            body="A real signal",
            suggested_action="Reply within an hour",
            topic_tags=["customer"],
            dedup_key="stub-key",
        )

    monkeypatch.setattr(triage_module.TriageAgent, "triage", fake_triage)

    async def runner() -> tuple:
        queue = sse_bus.subscribe()
        try:
            decision, alert_id = await pipeline.evaluate_and_dispatch(
                AlertEvent(source="email", external_id="msg-x", subject="hi", body="world"),
                db_path=db,
            )
            received: list[dict] = []
            while not queue.empty():
                received.append(queue.get_nowait())
            return decision, alert_id, received
        finally:
            sse_bus.unsubscribe(queue)

    decision, alert_id, received = asyncio.run(runner())

    assert decision.alert is True
    assert alert_id is not None

    persisted = list_alerts(db_path=db)
    assert len(persisted) == 1
    assert persisted[0].headline == "Stub alert"
    assert persisted[0].channels_delivered == ["web", "persisted"]

    assert any(e.get("type") == "alert" for e in received)


def test_evaluate_post_mute_suppresses(monkeypatch, db: Path) -> None:
    """A muted topic produced by the LLM is suppressed after the call."""
    from openexecutive.agents import triage as triage_module
    from openexecutive.alerts import pipeline

    add_mute("hiring", db_path=db)

    async def fake_triage(self, event, **kwargs):  # noqa: ARG001
        return TriageDecision(
            alert=True,
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.WEB, AlertChannel.PERSISTED],
            headline="New senior eng candidate",
            body="...",
            topic_tags=["hiring"],
            dedup_key="cand-1",
        )

    monkeypatch.setattr(triage_module.TriageAgent, "triage", fake_triage)

    decision, alert_id = asyncio.run(
        pipeline.evaluate_and_dispatch(
            AlertEvent(source="email", external_id="m-mute", subject="cand", body="b"),
            db_path=db,
        )
    )

    assert decision.alert is False
    assert "muted" in decision.reason_if_suppressed
    assert alert_id is None
    assert list_alerts(db_path=db) == []


def test_in_quiet_hours_wraparound_window() -> None:
    """A 22:00 → 07:00 window correctly straddles midnight."""
    from openexecutive.alerts.preferences import _in_quiet_hours

    prefs = UserPreferences(
        quiet_hours_start="22:00",
        quiet_hours_end="07:00",
        quiet_hours_tz="UTC",
    )
    # 23:00 UTC — inside.
    assert _in_quiet_hours(prefs, now=datetime(2030, 1, 1, 23, 0, tzinfo=UTC))
    # 03:00 UTC — inside.
    assert _in_quiet_hours(prefs, now=datetime(2030, 1, 1, 3, 0, tzinfo=UTC))
    # 12:00 UTC — outside.
    assert not _in_quiet_hours(prefs, now=datetime(2030, 1, 1, 12, 0, tzinfo=UTC))


# --------------------------------------------------------------------------- #
# Coalescing, routing, dedup hint (alert lifecycle)
# --------------------------------------------------------------------------- #


def _stub_triage(monkeypatch, *, severity=AlertSeverity.MEDIUM, dedup_key="model-key",
                 tags=None):
    from openexecutive.agents import triage as triage_module

    async def fake_triage(self, event, **kwargs):  # noqa: ARG001
        return TriageDecision(
            alert=True,
            severity=severity,
            channels=[AlertChannel.PERSISTED],
            headline="Stub",
            body=f"body for {event.external_id}",
            topic_tags=list(tags or ["customer"]),
            dedup_key=dedup_key or f"key-{event.external_id}",
        )

    monkeypatch.setattr(triage_module.TriageAgent, "triage", fake_triage)


def _count_dispatches(monkeypatch) -> list:
    from openexecutive.alerts import dispatcher

    calls: list = []

    async def fake_dispatch(alert, channels, **kw):  # noqa: ARG001
        calls.append(alert.id)

    monkeypatch.setattr(dispatcher, "dispatch_all", fake_dispatch)
    return calls


def test_pipeline_routes_person_only_from_the_explicit_field(monkeypatch, db: Path) -> None:
    from openexecutive.alerts import pipeline

    _stub_triage(monkeypatch, dedup_key=None)
    _count_dispatches(monkeypatch)

    async def run():
        _, a = await pipeline.evaluate_and_dispatch(
            AlertEvent(source="email", external_id="f", body="x", routed_to_person_id=4),
            db_path=db,
        )
        _, b = await pipeline.evaluate_and_dispatch(
            AlertEvent(source="email", external_id="u", body="x", user="person:9"),
            db_path=db,
        )
        _, c = await pipeline.evaluate_and_dispatch(
            AlertEvent(source="email", external_id="n", body="x", user="U123"),
            db_path=db,
        )
        return a, b, c

    a, b, c = asyncio.run(run())
    assert alert_store.get_alert(a, db_path=db).routed_to_person_id == 4  # type: ignore[union-attr]
    # `event.user` is a sender-controlled display name on some integrations —
    # a "person:9" there must never pick the routing target.
    assert alert_store.get_alert(b, db_path=db).routed_to_person_id is None  # type: ignore[union-attr]
    assert alert_store.get_alert(c, db_path=db).routed_to_person_id is None  # type: ignore[union-attr]


def test_pipeline_replay_of_same_external_id_stays_a_noop(monkeypatch, db: Path) -> None:
    """A webhook retry (same source + external_id) must not coalesce into a
    second occurrence — it is the same event, not a repeat of the situation."""
    from openexecutive.alerts import pipeline

    _stub_triage(monkeypatch, dedup_key="same-key")
    calls = _count_dispatches(monkeypatch)

    async def run():
        _, a = await pipeline.evaluate_and_dispatch(
            AlertEvent(source="email", external_id="msg-1", body="x"), db_path=db,
        )
        _, b = await pipeline.evaluate_and_dispatch(
            AlertEvent(source="email", external_id="msg-1", body="x"), db_path=db,
        )
        return a, b

    a, b = asyncio.run(run())
    assert a is not None and b is None
    row = alert_store.get_alert(a, db_path=db)
    assert row is not None and row.occurrence_count == 1 and calls == [a]


def test_pipeline_adds_department_tag_from_channel(monkeypatch, db: Path) -> None:
    from openexecutive.alerts import pipeline

    _stub_triage(monkeypatch, tags=["finance"])
    _count_dispatches(monkeypatch)

    async def run():
        _, aid = await pipeline.evaluate_and_dispatch(
            AlertEvent(source="email", external_id="d", body="x", department="Finance"),
            db_path=db,
        )
        return aid

    aid = asyncio.run(run())
    row = alert_store.get_alert(aid, db_path=db)
    assert row is not None and row.topic_tags == ["finance", "department:finance"]


def test_pipeline_strips_collection_tags_from_triage(monkeypatch, db: Path) -> None:
    """Only an extension's own code files into its collection
    (orchestrator/extensions.py): a tag the triage model was talked into
    never reaches the stored alert."""
    from openexecutive.alerts import pipeline

    _stub_triage(monkeypatch, tags=["finance", "collection:widgets"])
    _count_dispatches(monkeypatch)

    async def run():
        _, aid = await pipeline.evaluate_and_dispatch(
            AlertEvent(source="artifact", external_id="c", body="x"),
            db_path=db,
        )
        return aid

    aid = asyncio.run(run())
    row = alert_store.get_alert(aid, db_path=db)
    assert row is not None and row.topic_tags == ["finance"]


def test_pipeline_coalesces_same_key_without_dispatch_and_redispatches_on_escalation(
    monkeypatch, db: Path
) -> None:
    from openexecutive.alerts import pipeline

    calls = _count_dispatches(monkeypatch)

    async def run():
        _stub_triage(monkeypatch, severity=AlertSeverity.LOW, dedup_key="k")
        _, first = await pipeline.evaluate_and_dispatch(
            AlertEvent(source="stock", external_id="day1", body="x"), db_path=db,
        )
        _, second = await pipeline.evaluate_and_dispatch(
            AlertEvent(source="stock", external_id="day2", body="x"), db_path=db,
        )
        _stub_triage(monkeypatch, severity=AlertSeverity.URGENT, dedup_key="k")
        _, third = await pipeline.evaluate_and_dispatch(
            AlertEvent(source="stock", external_id="day3", body="crash"), db_path=db,
        )
        return first, second, third

    first, second, third = asyncio.run(run())
    assert first is not None and second is None and third == first
    row = alert_store.get_alert(first, db_path=db)
    assert row is not None
    assert row.occurrence_count == 3
    assert row.severity == "urgent"
    assert row.body == "body for day3"
    # One dispatch for the insert, none for the quiet repeat, one for the escalation.
    assert calls == [first, first]
    assert len(alert_store.list_alerts(db_path=db)) == 1


def test_pipeline_dedup_hint_overrides_model_key(monkeypatch, db: Path) -> None:
    from openexecutive.alerts import pipeline

    _count_dispatches(monkeypatch)

    async def run():
        _stub_triage(monkeypatch, dedup_key="model-a")
        _, a = await pipeline.evaluate_and_dispatch(
            AlertEvent(source="rss", external_id="1", body="x", dedup_hint="watch:acme-news"),
            db_path=db,
        )
        _stub_triage(monkeypatch, dedup_key="model-b")
        _, b = await pipeline.evaluate_and_dispatch(
            AlertEvent(source="rss", external_id="2", body="x", dedup_hint="watch:acme-news"),
            db_path=db,
        )
        return a, b

    a, b = asyncio.run(run())
    assert a is not None and b is None
    row = alert_store.get_alert(a, db_path=db)
    assert row is not None and row.dedup_key == "watch:acme-news" and row.occurrence_count == 2


@pytest.mark.parametrize("private", [False, True])
def test_pipeline_a_follow_up_from_one_sender_joins_the_first_card(
    monkeypatch, db: Path, private: bool,
) -> None:
    from openexecutive.alerts import pipeline

    _count_dispatches(monkeypatch)

    async def run():
        ids = []
        for n in ("1", "2"):
            _stub_triage(monkeypatch, dedup_key=f"model-{n}")
            _, aid = await pipeline.evaluate_and_dispatch(
                AlertEvent(
                    source="email", external_id=n, body="x", private=private,
                    dedup_hint="email-from:casey@example.com", follow_up=True,
                ),
                db_path=db,
            )
            ids.append(aid)
        return ids

    first, second = asyncio.run(run())
    assert first is not None and second is None
    row = alert_store.get_alert(first, db_path=db)
    assert row is not None and row.occurrence_count == 2
    assert row.body == "body for 1\n\nThen: body for 2"


def test_triage_text_naming_people_or_figures_not_in_the_event_is_dropped(
    monkeypatch, db: Path,
) -> None:
    """Triage rewrites the event into the text that ships; a colleague or a
    figure the event never mentioned must not reach the stored alert."""
    from openexecutive.agents import triage as triage_module
    from openexecutive.alerts import pipeline
    from openexecutive.briefing import grounding

    monkeypatch.setattr(grounding, "profile_sources", lambda: [])
    monkeypatch.setattr(grounding, "grounding_mode", lambda: "enforce")
    audits: list[dict] = []
    monkeypatch.setattr(
        "openexecutive.audit.log_event",
        lambda event_type, summary, **kw: audits.append({"type": event_type, **kw}),
    )

    async def fake_triage(self, event, **kwargs):  # noqa: ARG001
        return TriageDecision(
            alert=True,
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.PERSISTED],
            headline="Marcus Lee: lease renewal",
            body="Dana wants to renew 48 units by Friday. Marcus Lee manages the site.",
            suggested_action="Call Marcus Lee",
            dedup_key="lease",
        )

    monkeypatch.setattr(triage_module.TriageAgent, "triage", fake_triage)
    event = AlertEvent(
        source="email", external_id="m-lease", subject="Lease renewal",
        body="Hi — can we renew all 48 units by Friday? Dana", **{"from": "Dana Whitfield"},
    )
    decision, alert_id = asyncio.run(pipeline.evaluate_and_dispatch(event, db_path=db))

    assert alert_id is not None
    [stored] = list_alerts(db_path=db)
    assert stored.headline == "Lease renewal"
    assert stored.body == "Dana wants to renew 48 units by Friday."
    assert stored.suggested_action == ""
    assert decision.body == stored.body
    [row] = [a for a in audits if a["type"] == "grounding"]
    assert "Marcus Lee" in row["details"]["items"]


def test_grounded_triage_text_is_stored_unchanged(monkeypatch, db: Path) -> None:
    from openexecutive.agents import triage as triage_module
    from openexecutive.alerts import pipeline
    from openexecutive.briefing import grounding

    monkeypatch.setattr(grounding, "profile_sources", lambda: [])

    async def fake_triage(self, event, **kwargs):  # noqa: ARG001
        return TriageDecision(
            alert=True, severity=AlertSeverity.HIGH, channels=[AlertChannel.PERSISTED],
            headline="Dana Whitfield: renew 48 units", body="Renewal of 48 units due Friday.",
            suggested_action="Reply to Dana", dedup_key="lease-ok",
        )

    monkeypatch.setattr(triage_module.TriageAgent, "triage", fake_triage)
    event = AlertEvent(
        source="email", external_id="m-ok", subject="Lease renewal",
        body="Can we renew all 48 units by Friday?", **{"from": "Dana Whitfield"},
    )
    asyncio.run(pipeline.evaluate_and_dispatch(event, db_path=db))
    [stored] = list_alerts(db_path=db)
    assert (stored.headline, stored.body, stored.suggested_action) == (
        "Dana Whitfield: renew 48 units", "Renewal of 48 units due Friday.", "Reply to Dana",
    )


def test_as_written_message_skips_triage_and_keeps_its_words(monkeypatch, db: Path) -> None:
    """A message passed on to one person keeps its subject and body: the
    triage model (which sees other cards) never rewrites it."""
    from openexecutive.agents import triage as triage_module
    from openexecutive.alerts import dispatcher, pipeline

    async def no_triage(self, event, **kwargs):  # noqa: ARG001
        raise AssertionError("triage must not run for a passed-on message")

    monkeypatch.setattr(triage_module.TriageAgent, "triage", no_triage)
    sent: list[list[AlertChannel]] = []

    async def fake_dispatch(alert, channels, **kw):  # noqa: ARG001
        sent.append(list(channels))

    monkeypatch.setattr(dispatcher, "dispatch_all", fake_dispatch)
    event = AlertEvent(
        source="team_member", external_id="m-1",
        subject="From Sam: the shared folder won't open",
        body="From Sam: the shared folder won't open for me.",
        raised_by="Sam Lee", routed_to_person_id=1, as_written=True,
    )

    _, alert_id = asyncio.run(pipeline.evaluate_and_dispatch(event, db_path=db))

    assert alert_id is not None
    row = list_alerts(db_path=db)[0]
    assert row.headline == "From Sam: the shared folder won't open"
    assert row.body == "Raised by Sam Lee. From Sam: the shared folder won't open for me."
    assert row.routed_to_person_id == 1
    assert sent and AlertChannel.WEB in sent[0]


def test_as_written_same_message_folds_into_open_card(monkeypatch, db: Path) -> None:
    from openexecutive.alerts import dispatcher, pipeline

    async def fake_dispatch(alert, channels, **kw):  # noqa: ARG001
        return None

    monkeypatch.setattr(dispatcher, "dispatch_all", fake_dispatch)

    def ev(ext: str) -> AlertEvent:
        return AlertEvent(source="executive", external_id=ext, subject="Hi", body="Hi there", as_written=True)

    _, first = asyncio.run(pipeline.evaluate_and_dispatch(ev("a"), db_path=db))
    _, second = asyncio.run(pipeline.evaluate_and_dispatch(ev("b"), db_path=db))
    assert first is not None
    assert second is None
    assert len(list_alerts(db_path=db)) == 1


def test_only_a_teammates_own_message_keeps_to_its_own_card(monkeypatch, db: Path) -> None:
    """A message the Executive writes to someone joins the open card on its
    topic (one card per topic); a teammate's own message stays its own card."""
    from openexecutive.alerts import dispatcher, one_card, pipeline
    from openexecutive.audit.context import set_turn

    async def fake_dispatch(alert, channels, **kw):  # noqa: ARG001
        return None

    monkeypatch.setattr(dispatcher, "dispatch_all", fake_dispatch)
    with set_turn(session_id="s", turn_id="t-dock"):
        with one_card.about(raised=True):
            question = insert_alert(
                source="department", external_id="", severity="medium",
                headline="Dock door 4 on Dec 3: yes or no?", body="Northpeak needs an answer.",
                routed_to_person_id=1, db_path=db,
            )
        own = AlertEvent(
            source="executive", external_id="m-own", subject="Dana needs a yes or no on door 4",
            body="Dana needs a yes or no on door 4 today.", routed_to_person_id=1,
            user="person:1", as_written=True,
        )
        _, joined = asyncio.run(pipeline.evaluate_and_dispatch(own, db_path=db))
        teammate = AlertEvent(
            source="executive", external_id="m-jordan", subject="From Jordan: door 4 is free",
            body="From Jordan: door 4 is free on Dec 3.", routed_to_person_id=1,
            user="person:1", raised_by="Jordan Lee", as_written=True,
        )
        _, kept = asyncio.run(pipeline.evaluate_and_dispatch(teammate, db_path=db))
    assert question is not None and joined is not None and kept is not None
    by_id = {a.id: a for a in list_alerts(db_path=db)}
    assert by_id[joined].superseded_by_alert_id == question
    assert by_id[kept].superseded_by_alert_id is None and by_id[kept].status == "unread"


def test_as_written_same_text_to_two_people_is_two_cards(monkeypatch, db: Path) -> None:
    from openexecutive.alerts import dispatcher, pipeline

    async def fake_dispatch(alert, channels, **kw):  # noqa: ARG001
        return None

    monkeypatch.setattr(dispatcher, "dispatch_all", fake_dispatch)

    def ev(ext: str, person: int) -> AlertEvent:
        return AlertEvent(
            source="executive", external_id=ext, subject="Standup moved", body="Standup moved to 10am",
            routed_to_person_id=person, user=f"person:{person}", as_written=True,
        )

    _, first = asyncio.run(pipeline.evaluate_and_dispatch(ev("a", 1), db_path=db))
    _, second = asyncio.run(pipeline.evaluate_and_dispatch(ev("b", 2), db_path=db))
    assert first is not None and second is not None
    assert {a.routed_to_person_id for a in list_alerts(db_path=db)} == {1, 2}



class _Chat:
    """A chat session as ``current_session`` holds one."""

    def __init__(self, session_id: str, speaker: int, *, mail_thread: str = "") -> None:
        self.session_id = session_id
        self.caller_person_id = speaker
        self.origin_channel = "email" if mail_thread else ""
        self.email_from = "ann@example.com" if mail_thread else ""
        self.email_thread_id = mail_thread


def _pass_on(
    monkeypatch, db: Path, chat: _Chat, turn: str, ext: str, subject: str, body: str, raised_by: str = "Sam Lee",
) -> int | None:
    from openexecutive.alerts import pipeline
    from openexecutive.audit.context import set_turn
    from openexecutive.orchestrator.schedule_tools import current_session

    token = current_session.set(chat)
    try:
        with set_turn(session_id=chat.session_id, turn_id=turn):
            event = AlertEvent(
                source="message", external_id=ext, subject=subject, body=body,
                routed_to_person_id=1, user="person:1", raised_by=raised_by, as_written=True,
            )
            return asyncio.run(pipeline.evaluate_and_dispatch(event, db_path=db))[1]
    finally:
        current_session.reset(token)


def test_messages_from_one_chat_to_one_person_are_one_card(monkeypatch, db: Path) -> None:
    """A teammate who sends a note, then a correction, then a fix from the
    same chat leaves their reader one card, newest last, and each still
    reaches them: never three cards that read alike and contradict."""
    from openexecutive.alerts import dispatcher

    sent: list[int | None] = []

    async def fake_dispatch(alert, channels, **kw):  # noqa: ARG001
        sent.append(alert.id)

    monkeypatch.setattr(dispatcher, "dispatch_all", fake_dispatch)
    chat = _Chat("chat-sam", 2)
    first = _pass_on(monkeypatch, db, chat, "t1", "m1", "Sharing one file doesn't work", "Sharing one file doesn't work.")
    second = _pass_on(monkeypatch, db, chat, "t2", "m2", "Correction: it works now", "Correction: it works now.")
    third = _pass_on(monkeypatch, db, chat, "t3", "m3", "It works from the shared drive", "It works from the shared drive only.")
    rows = {a.id: a for a in list_alerts(db_path=db)}
    assert rows[second].superseded_by_alert_id == first
    assert rows[third].superseded_by_alert_id == first
    open_cards = [a for a in rows.values() if a.status == "unread"]
    assert [a.id for a in open_cards] == [first]
    body = open_cards[0].body
    assert body.startswith("Raised by Sam Lee. Sharing one file doesn't work.")
    assert body.index("Correction: it works now.") < body.index("It works from the shared drive only.")
    assert body.count("Raised by Sam Lee.") == 1
    assert sent == [first, first, first]


def test_messages_from_another_chat_are_their_own_card(monkeypatch, db: Path) -> None:
    from openexecutive.alerts import dispatcher

    async def fake_dispatch(alert, channels, **kw):  # noqa: ARG001
        return None

    monkeypatch.setattr(dispatcher, "dispatch_all", fake_dispatch)
    first = _pass_on(monkeypatch, db, _Chat("chat-a", 2), "t1", "m1", "Launch moved", "Launch moved to Friday.")
    second = _pass_on(monkeypatch, db, _Chat("chat-b", 2), "t2", "m2", "Invoice paid", "The invoice is paid.")
    rows = {a.id: a for a in list_alerts(db_path=db)}
    assert rows[first].superseded_by_alert_id is None and rows[second].superseded_by_alert_id is None


def test_a_passed_on_message_never_joins_a_card_on_its_mail_thread(monkeypatch, db: Path) -> None:
    """On a turn answering mail, a passed-on message keeps out of the card
    the Executive raised on that mail thread: only other passed-on messages
    share its conversation."""
    from openexecutive.alerts import dispatcher, one_card
    from openexecutive.audit.context import set_turn
    from openexecutive.orchestrator.schedule_tools import current_session

    async def fake_dispatch(alert, channels, **kw):  # noqa: ARG001
        return None

    monkeypatch.setattr(dispatcher, "dispatch_all", fake_dispatch)
    chat = _Chat("mail-session", 2, mail_thread="T1")
    token = current_session.set(chat)
    try:
        with set_turn(session_id=chat.session_id, turn_id="t1"), one_card.about(raised=True):
            raised = insert_alert(
                source="email", external_id="e1", severity="medium",
                headline="Ann wants the contract back", body="Ann asked for the signed contract.",
                routed_to_person_id=1, db_path=db,
            )
    finally:
        current_session.reset(token)
    passed = _pass_on(monkeypatch, db, chat, "t2", "m1", "From Sam: contract is in Drive", "From Sam: contract is in Drive.")
    rows = {a.id: a for a in list_alerts(db_path=db)}
    assert rows[passed].superseded_by_alert_id is None
    assert rows[raised].body == "Ann asked for the signed contract."


def test_a_teammates_message_never_joins_a_card_from_its_own_turn(monkeypatch, db: Path) -> None:
    """A teammate's message joins only by its chat: a card the Executive
    raised on the same turn keeps its own words."""
    from openexecutive.alerts import dispatcher, one_card
    from openexecutive.audit.context import set_turn
    from openexecutive.orchestrator.schedule_tools import current_session

    async def fake_dispatch(alert, channels, **kw):  # noqa: ARG001
        return None

    monkeypatch.setattr(dispatcher, "dispatch_all", fake_dispatch)
    chat = _Chat("chat-sam", 2)
    token = current_session.set(chat)
    try:
        with set_turn(session_id=chat.session_id, turn_id="t1"), one_card.about(raised=True):
            raised = insert_alert(
                source="executive", external_id="x1", severity="medium",
                headline="Budget review is Friday", body="The budget review moved to Friday.",
                routed_to_person_id=1, db_path=db,
            )
    finally:
        current_session.reset(token)
    passed = _pass_on(monkeypatch, db, chat, "t1", "m1", "From Sam: I'll bring the deck", "From Sam: I'll bring the deck.")
    rows = {a.id: a for a in list_alerts(db_path=db)}
    assert rows[passed].superseded_by_alert_id is None
    assert rows[raised].superseded_by_alert_id is None
    assert rows[raised].body == "The budget review moved to Friday."


def test_a_name_with_a_period_is_said_once_on_the_joined_card(monkeypatch, db: Path) -> None:
    from openexecutive.alerts import dispatcher

    async def fake_dispatch(alert, channels, **kw):  # noqa: ARG001
        return None

    monkeypatch.setattr(dispatcher, "dispatch_all", fake_dispatch)
    chat = _Chat("chat-dr", 2)
    first = _pass_on(monkeypatch, db, chat, "t1", "m1", "Lab is closed", "Lab is closed today.", "Dr. Smith")
    _pass_on(monkeypatch, db, chat, "t2", "m2", "Lab opens at noon", "Lab opens at noon.", "Dr. Smith")
    body = {a.id: a for a in list_alerts(db_path=db)}[first].body
    assert body.count("Raised by Dr. Smith.") == 1
    assert "Smith. Lab opens" not in body and body.endswith("Lab opens at noon.")


def test_a_teammates_message_never_joins_the_executives_message_from_the_same_chat(monkeypatch, db: Path) -> None:
    from openexecutive.alerts import dispatcher, pipeline
    from openexecutive.audit.context import set_turn
    from openexecutive.orchestrator.schedule_tools import current_session

    async def fake_dispatch(alert, channels, **kw):  # noqa: ARG001
        return None

    monkeypatch.setattr(dispatcher, "dispatch_all", fake_dispatch)
    chat = _Chat("shared-chat", 2)
    token = current_session.set(chat)
    try:
        with set_turn(session_id=chat.session_id, turn_id="t1"):
            own = AlertEvent(
                source="message", external_id="x1", subject="Standup moved", body="Standup moved to 10.",
                routed_to_person_id=1, user="person:1", as_written=True,
            )
            _, executives = asyncio.run(pipeline.evaluate_and_dispatch(own, db_path=db))
    finally:
        current_session.reset(token)
    teammates = _pass_on(monkeypatch, db, chat, "t2", "m1", "From Sam: I'll be late", "From Sam: I'll be late.")
    rows = {a.id: a for a in list_alerts(db_path=db)}
    assert rows[teammates].superseded_by_alert_id is None
    assert rows[executives].body == "Standup moved to 10."


def test_a_teammates_message_card_is_not_offered_for_the_executive_to_join(monkeypatch, db: Path) -> None:
    from openexecutive.alerts import dispatcher, one_card

    async def fake_dispatch(alert, channels, **kw):  # noqa: ARG001
        return None

    monkeypatch.setattr(dispatcher, "dispatch_all", fake_dispatch)
    _pass_on(monkeypatch, db, _Chat("chat-sam", 2), "t1", "m1", "From Sam: running late", "From Sam: running late.")
    assert one_card.open_cards_from("person:2", turn="t2", reader="1", db_path=db) == []


def test_a_step_the_grounding_rewrote_loses_its_buttons_words(monkeypatch: pytest.MonkeyPatch) -> None:
    from openexecutive.alerts import pipeline
    from openexecutive.alerts.models import TriageDecision
    from openexecutive.briefing import grounding

    event = AlertEvent(source="email", external_id="g-1", subject="Renewal", body="Dana asked about renewal")
    decision = TriageDecision(
        alert=True, headline="Renewal", body="b", suggested_action="Email Dana and Sam",
        action_label="Email Dana and Sam",
    )

    def rewrite(headline: str, body: str, action: str, **_: object) -> tuple[str, str, str, bool]:
        return headline, body, "Email Dana", True

    monkeypatch.setattr(grounding, "ground_alert_text", rewrite)
    grounded = pipeline._ground_decision(decision, event, [])
    assert (grounded.suggested_action, grounded.action_label) == ("Email Dana", "")

    def keep(headline: str, body: str, action: str, **_: object) -> tuple[str, str, str, bool]:
        return headline, "other body", action, True

    monkeypatch.setattr(grounding, "ground_alert_text", keep)
    assert pipeline._ground_decision(decision, event, []).action_label == "Email Dana and Sam"


def _paced(monkeypatch):  # type: ignore[no-untyped-def]
    """The pacing with a fake clock: a sleep moves it on and yields."""
    from openexecutive.alerts import pipeline as ap

    clock = [1000.0]
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        clock[0] += seconds
        await real_sleep(0)

    monkeypatch.setattr(ap.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ap.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(ap, "_recent_event_ts", ap.deque(maxlen=ap._RATE_LIMIT_PER_MIN))
    monkeypatch.setattr(ap, "_waiting", 0)
    return ap, clock


def test_a_burst_past_the_minutes_cap_waits_its_turn(monkeypatch) -> None:
    # Four email watches catching up at once: 80 evaluations in a minute.
    # The cap paces them into the next minute; every one is evaluated.
    ap, clock = _paced(monkeypatch)

    async def burst() -> list[bool]:
        return list(await asyncio.gather(*(ap._wait_for_slot() for _ in range(ap._RATE_LIMIT_PER_MIN + 20))))

    assert all(asyncio.run(burst()))
    assert clock[0] >= 1060.0


def _all_at_once(monkeypatch, *, bounded: bool) -> tuple[list[bool], int]:  # type: ignore[no-untyped-def]
    """Every event arrives within the minute, past the cap and what may wait."""
    ap, clock = _paced(monkeypatch)
    extra = 25
    total = ap._RATE_LIMIT_PER_MIN + ap._MAX_WAITING + extra
    real_sleep = asyncio.sleep

    async def run() -> list[bool]:
        arrived = asyncio.Event()

        async def held_sleep(seconds: float) -> None:
            await arrived.wait()
            clock[0] += seconds
            await real_sleep(0)

        monkeypatch.setattr(ap.asyncio, "sleep", held_sleep)
        tasks = [asyncio.ensure_future(ap._wait_for_slot(bounded=bounded)) for _ in range(total)]
        await real_sleep(0)
        arrived.set()
        return list(await asyncio.gather(*tasks))

    ran = asyncio.run(run())
    assert ap._waiting == 0
    return ran, extra


def test_a_flood_past_what_may_wait_is_still_dropped(monkeypatch) -> None:
    # A looping bot: past the minute's cap and what may wait for the next
    # minutes, the rest are dropped, so the triage spend stays capped.
    ran, extra = _all_at_once(monkeypatch, bounded=True)
    assert ran.count(False) == extra


def test_a_source_that_limits_itself_always_waits(monkeypatch) -> None:
    # Nine email watches catching up at once (20 each a poll): past what may
    # wait, they still wait, as their source has already read past them.
    ran, _ = _all_at_once(monkeypatch, bounded=False)
    assert all(ran)
