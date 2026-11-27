"""Email watches: the email a person can see, on the watch list
(monitoring/mailbox.py, monitoring/sources/mailbox.py).

"Tell me every time a sign-up email comes in" (11-19 1:1): match by sender,
by subject words or both, every email, until turned off, and only its owner
sees it. "Watch for Robin's email, ping me if nothing by Wednesday" (Test lab
11-21): it watches the Executive's inbox too, and pings on the day."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openexecutive.alerts.models import PERSONAL_SOURCES, AlertSeverity
from openexecutive.alerts.store import initialize_db as initialize_alerts_db
from openexecutive.briefing.ranking import watch_slug_of
from openexecutive.delegation import gmail as gm
from openexecutive.memory.episodic import initialize_db as initialize_episodic_db
from openexecutive.monitoring import describe, mailbox
from openexecutive.monitoring import store as ms
from openexecutive.monitoring.models import SOURCE_KIND_MAILBOX, Signal
from openexecutive.monitoring.pipeline import _signal_to_alert_event
from openexecutive.monitoring.sources import get_source_for_kind
from openexecutive.monitoring.sources import mailbox as mailbox_source
from openexecutive.orchestrator import watchlist_tools as wt

OWNER = 7
OTHER = 8


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "mailbox_watch.db"
    monkeypatch.setattr("openexecutive.memory.episodic.DB_PATH", db_path)
    monkeypatch.setattr("openexecutive.alerts.store.DB_PATH", db_path)
    initialize_episodic_db(db_path)
    initialize_alerts_db(db_path)
    ms.initialize_db(db_path)
    return db_path


@pytest.fixture(autouse=True)
def _no_audit(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def record(event: str, summary: str, **kwargs: Any) -> None:
        rows.append({"event": event, **kwargs})

    monkeypatch.setattr(wt, "audit_log", record)
    return rows


def _email_watch(db: Path, *, owner: int = OWNER, trigger: dict[str, Any] | None = None, slug: str = "email-sign-ups") -> Any:
    new_id = ms.insert_watchlist_item(
        slug=slug,
        signal_type=SOURCE_KIND_MAILBOX,
        target="sign-up emails",
        config=mailbox.owner_config({}, owner),
        trigger=trigger if trigger is not None else {"keywords": ["sign up", "signed up"]},
        cadence="real_time",
        db_path=db,
    )
    item = ms.get_watchlist_item(new_id, db_path=db)
    assert item is not None
    return item


# --------------------------------------------------------------------------- #
# What matches
# --------------------------------------------------------------------------- #


def test_an_email_matches_by_subject_word_without_a_sender() -> None:
    trigger = {"keywords": ["sign up"]}
    assert mailbox.matches(trigger, from_addr="noreply@app.example", from_name="App", subject="New sign up: Dana")
    assert mailbox.matches(trigger, from_addr="a@b.example", from_name="", subject="NEW SIGN UP")
    assert not mailbox.matches(trigger, from_addr="a@b.example", from_name="", subject="Invoice")


def test_an_email_matches_by_sender_and_both_means_both() -> None:
    assert mailbox.matches({"from": ["pat@acme.example"]}, from_addr="PAT@acme.example", from_name="Pat", subject="hi")
    both = {"from": ["pat@acme.example"], "keywords": ["contract"]}
    assert mailbox.matches(both, from_addr="pat@acme.example", from_name="Pat", subject="The contract")
    assert not mailbox.matches(both, from_addr="pat@acme.example", from_name="Pat", subject="Lunch")
    assert not mailbox.matches(both, from_addr="sam@acme.example", from_name="Sam", subject="The contract")


def test_a_keyword_matches_whole_words_only() -> None:
    # "AI newsletters": every Gmail sender is not an AI newsletter, nor is
    # "Patch notes" from Pat.
    ai = {"keywords": ["AI"]}
    assert not mailbox.matches(ai, from_addr="jane@gmail.com", from_name="Jane", subject="Lunch")
    assert mailbox.matches(ai, from_addr="news@x.example", from_name="X", subject="This week in AI")
    assert not mailbox.matches({"keywords": ["pat"]}, from_addr="a@b.example", from_name="", subject="Patch notes")
    assert not mailbox.matches({"keywords": ["hr"]}, from_addr="a@b.example", from_name="", subject="Chrome, three")
    # Punctuation is a space, and an address's parts are words.
    assert mailbox.matches({"keywords": ["sign up"]}, from_addr="a@b.example", from_name="", subject="Sign-up: Dana")
    assert mailbox.matches({"keywords": ["stripe"]}, from_addr="billing@stripe.com", from_name="", subject="Receipt")


def test_an_empty_trigger_matches_nothing() -> None:
    assert not mailbox.matches({}, from_addr="a@b.example", from_name="A", subject="anything")


def test_the_trigger_keeps_real_addresses_and_plain_words() -> None:
    cleaned = mailbox.clean_trigger({
        "from": ["Pat@Acme.example", "not an address", "pat@acme.example"],
        "keywords": ["  sign   up ", "Sign up", ""],
        "only_big": True,
    })
    assert cleaned == {"from": ["pat@acme.example"], "keywords": ["sign up"]}
    assert mailbox.clean_trigger({"keywords": "sign up"}) == {"keywords": ["sign up"]}
    assert mailbox.clean_trigger("nope") == {}


# --------------------------------------------------------------------------- #
# Reading the inbox
# --------------------------------------------------------------------------- #


@dataclass
class _Mailbox:
    messages: list[Any]
    asked: list[datetime]

    # Ids a real client lists but drops (the person's own mail, or mail
    # deleted since it was listed), so the page comes back short.
    dropped: frozenset[str] = frozenset()

    async def received_since(self, after: datetime, *, before: datetime | None = None, limit: int = 25) -> list[Any]:
        if before is None:
            self.asked.append(after)
        newest_first = sorted(
            (m for m in self.messages if before is None or datetime.fromisoformat(m.received_at) < before),
            key=lambda m: m.received_at, reverse=True,
        )
        return [m for m in newest_first[:limit] if m.id not in self.dropped]


def _person(**kw: Any) -> Any:
    return SimpleNamespace(id=OWNER, email="jordan@acme.example", archived=False, is_principal=True, **kw)


def _patch_mailbox(monkeypatch: pytest.MonkeyPatch, box: _Mailbox, *, status: str = "connected") -> None:
    async def gmail_status(email: str, **_: Any) -> str:
        return status

    monkeypatch.setattr(gm, "gmail_status", gmail_status)
    monkeypatch.setattr(gm, "gmail_for", lambda email: box)
    monkeypatch.setattr("openexecutive.people.store.get_person", lambda pid: _person() if pid == OWNER else None)
    monkeypatch.setattr(mailbox_source, "observe", lambda item, note: None)


def test_every_matching_email_since_the_watch_was_set_is_its_own_signal(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _email_watch(db)
    set_at = datetime.fromisoformat(item.created_at)
    later = (set_at + timedelta(minutes=3)).isoformat()
    before = (set_at - timedelta(hours=2)).isoformat()
    box = _Mailbox([
        gm.MailMessage(id="m1", thread_id="t1", from_addr="noreply@app.example", from_name="App",
                       subject="New sign up: Dana", received_at=later, text="Dana's phone is 555"),
        gm.MailMessage(id="m2", thread_id="t2", from_addr="noreply@app.example", from_name="App",
                       subject="Someone signed up", received_at=later),
        gm.MailMessage(id="m3", thread_id="t3", from_addr="ar@vendor.example", from_name="AR",
                       subject="Invoice 12", received_at=later),
        gm.MailMessage(id="m0", thread_id="t0", from_addr="noreply@app.example", from_name="App",
                       subject="Old sign up", received_at=before),
    ], [])
    _patch_mailbox(monkeypatch, box)

    signals = asyncio.run(get_source_for_kind(SOURCE_KIND_MAILBOX).poll(item, db_path=db))  # type: ignore[union-attr]

    assert [s.source_external_id for s in signals] == ["m1", "m2"]
    first = signals[0]
    assert first.dedup_key == f"mailbox:{item.id}:m1"
    assert first.severity_hint == AlertSeverity.HIGH
    assert first.normalized_summary == "App: New sign up: Dana"
    # The email's text never leaves the mailbox.
    assert first.raw_payload == {} and "555" not in json.dumps(first.model_dump(mode="json"))
    # Read from when it was set, the first time.
    assert box.asked == [set_at]


def test_a_poll_reads_back_page_by_page_to_where_it_last_read(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A day disconnected, then 130 emails: the newest page alone would miss
    # the sign-up that came first, and so would stopping at a short page.
    item = _email_watch(db)
    set_at = datetime.fromisoformat(item.created_at)
    mail = [
        gm.MailMessage(id=f"n{i}", thread_id=f"t{i}", from_addr="news@shop.example", from_name="Shop",
                       subject="Sale", received_at=(set_at + timedelta(minutes=10 + i)).isoformat())
        for i in range(130)
    ]
    mail.append(gm.MailMessage(id="first", thread_id="tf", from_addr="noreply@app.example", from_name="App",
                               subject="New sign up", received_at=(set_at + timedelta(minutes=1)).isoformat()))
    # One of the newest is their own note to self: the first page is short.
    _patch_mailbox(monkeypatch, _Mailbox(mail, [], dropped=frozenset({"n129"})))

    signals = asyncio.run(mailbox_source.MailboxSource().poll(item, db_path=db))

    assert [s.source_external_id for s in signals] == ["first"]


def test_a_backlog_comes_through_oldest_first_a_poll_at_a_time(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Back after a day away to 25 sign-ups: each poll tells at most PER_POLL
    # (one triage call each), oldest first, and the next poll picks up where
    # it stopped, never skipping one.
    item = _email_watch(db)
    set_at = datetime.fromisoformat(item.created_at)
    _patch_mailbox(monkeypatch, _Mailbox([
        gm.MailMessage(id=f"s{i:02d}", thread_id=f"t{i}", from_addr="noreply@app.example", from_name="App",
                       subject="New sign up", received_at=(set_at + timedelta(minutes=i)).isoformat())
        for i in range(25)
    ], []))
    told: list[str] = []
    for _ in range(3):
        fresh = ms.get_watchlist_item(item.id, db_path=db)
        assert fresh is not None
        signals = asyncio.run(mailbox_source.MailboxSource().poll(fresh, db_path=db))
        for signal in signals:
            ms.insert_signal(signal, db_path=db)  # the pipeline records each
        told.append(",".join(s.source_external_id for s in signals))
    assert told[0] == ",".join(f"s{i:02d}" for i in range(mailbox_source.PER_POLL))
    assert told[1] == ",".join(f"s{i:02d}" for i in range(mailbox_source.PER_POLL, 25))
    assert told[2] == ""  # the overlap re-reads only what was already told


def test_emails_a_tick_never_recorded_are_read_again(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Back a day later to 5 sign-ups from then; the tick is cut off (a
    # restart) before it records them. The next poll reads from where it
    # read before, not from a day later, and moves on once they are recorded.
    item = _email_watch(db)
    set_at = datetime.fromisoformat(item.created_at)
    day_later = set_at + timedelta(days=1)
    box = _Mailbox([
        gm.MailMessage(id=f"y{i}", thread_id=f"t{i}", from_addr="noreply@app.example", from_name="App",
                       subject="New sign up", received_at=(set_at + timedelta(minutes=i)).isoformat())
        for i in range(5)
    ], [])
    _patch_mailbox(monkeypatch, box)
    monkeypatch.setattr(mailbox_source, "datetime", SimpleNamespace(
        now=lambda tz=None: day_later, fromisoformat=datetime.fromisoformat,
    ))

    def poll() -> list[Signal]:
        fresh = ms.get_watchlist_item(item.id, db_path=db)
        assert fresh is not None
        return asyncio.run(mailbox_source.MailboxSource().poll(fresh, db_path=db))

    assert [s.source_external_id for s in poll()] == [f"y{i}" for i in range(5)]
    again = poll()  # nothing was recorded
    assert box.asked == [set_at, set_at] and len(again) == 5
    for signal in again:
        ms.insert_signal(signal, db_path=db)
    assert poll() == []
    assert box.asked[-1] == day_later - timedelta(minutes=10)
    row = ms.get_watchlist_item(item.id, db_path=db)
    assert row is not None and row.config_json.get(mailbox_source.PENDING_KEY) is None


def test_turned_back_on_it_reads_from_then_not_from_when_it_was_turned_off(db: Path) -> None:
    item = _email_watch(db)
    key = mailbox.READ_THROUGH_KEY
    weeks_ago = (datetime.now(UTC) - timedelta(days=14)).isoformat()
    ms.set_watchlist_config_key(item.id, key, weeks_ago, db_path=db)

    def read_through() -> datetime:
        row = ms.get_watchlist_item(item.id, db_path=db)
        assert row is not None
        return datetime.fromisoformat(row.config_json[key])

    # On, and changed while on: where it read to stays (it catches up).
    ms.update_watchlist_fields(item.id, {"enabled": 1}, db_path=db)
    assert read_through().isoformat() == weeks_ago
    # Off, then on again: the mail while it was off is not news.
    for turn_on in (
        lambda: ms.update_watchlist_fields(item.id, {"enabled": 1}, db_path=db),
        lambda: ms.set_enabled(item.id, True, db_path=db),
    ):
        ms.set_watchlist_config_key(item.id, key, weeks_ago, db_path=db)
        ms.set_enabled(item.id, False, db_path=db)
        turn_on()
        assert datetime.now(UTC) - read_through() < timedelta(minutes=1)


def test_turned_back_on_it_tells_nothing_from_while_it_was_off(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Off from 09:00 to 10:00; a sign-up at 09:55 is not news at 10:00, not
    # even inside the overlap a later poll re-reads.
    item = _email_watch(db)
    ms.set_enabled(item.id, False, db_path=db)
    ms.set_enabled(item.id, True, db_path=db)
    row = ms.get_watchlist_item(item.id, db_path=db)
    assert row is not None
    on_at = datetime.fromisoformat(row.config_json[mailbox.READ_FROM_KEY])
    box = _Mailbox([
        gm.MailMessage(id="off", thread_id="t1", from_addr="noreply@app.example", from_name="App",
                       subject="New sign up", received_at=(on_at - timedelta(minutes=5)).isoformat()),
        gm.MailMessage(id="on", thread_id="t2", from_addr="noreply@app.example", from_name="App",
                       subject="New sign up", received_at=(on_at + timedelta(seconds=1)).isoformat()),
    ], [])
    _patch_mailbox(monkeypatch, box)
    signals = asyncio.run(mailbox_source.MailboxSource().poll(row, db_path=db))
    assert [s.source_external_id for s in signals] == ["on"]
    assert box.asked == [on_at]


def test_a_later_poll_reads_from_a_little_before_the_last_read_that_worked(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _email_watch(db)
    box = _Mailbox([], [])
    _patch_mailbox(monkeypatch, box)
    asyncio.run(mailbox_source.MailboxSource().poll(item, db_path=db))
    item = ms.get_watchlist_item(item.id, db_path=db)
    read = datetime.fromisoformat(item.config_json[mailbox_source.READ_THROUGH_KEY])
    # Polls that couldn't read (inbox disconnected for hours) don't move it on.
    _patch_mailbox(monkeypatch, box, status="needs_reconnect")
    later = item.model_copy(update={
        "created_at": (read - timedelta(days=1)).isoformat(),
        "last_polled_at": (read + timedelta(hours=2)).isoformat(),
    })
    asyncio.run(mailbox_source.MailboxSource().poll(later, db_path=db))
    assert ms.get_watchlist_item(item.id, db_path=db).config_json[mailbox_source.READ_THROUGH_KEY] == read.isoformat()
    _patch_mailbox(monkeypatch, box)
    asyncio.run(mailbox_source.MailboxSource().poll(later, db_path=db))
    assert box.asked[-1] == read - timedelta(minutes=10)


def test_an_unconnected_inbox_is_not_read(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    item = _email_watch(db)
    box = _Mailbox([gm.MailMessage(id="m1", thread_id="t1", subject="sign up")], [])
    _patch_mailbox(monkeypatch, box, status="not_configured")
    notes: list[str] = []
    monkeypatch.setattr(mailbox_source, "observe", lambda item, note: notes.append(note))
    assert asyncio.run(mailbox_source.MailboxSource().poll(item, db_path=db)) == []
    assert box.asked == [] and notes == [mailbox_source.NOT_CONNECTED]


# --------------------------------------------------------------------------- #
# Who sees it, who it tells
# --------------------------------------------------------------------------- #


def test_only_its_owner_sees_an_email_watch(db: Path) -> None:
    ms.insert_watchlist_item(slug="stock-nwt", signal_type="stock", target="NWT", db_path=db)
    _email_watch(db)

    def slugs(**kw: Any) -> list[str]:
        return [i.slug for i in ms.list_watchlist(db_path=db, **kw)]

    assert slugs() == ["stock-nwt"]
    assert slugs(for_person=OTHER) == ["stock-nwt"]
    assert slugs(for_person=OWNER) == ["stock-nwt", "email-sign-ups"]
    assert slugs(every_watch=True) == ["stock-nwt", "email-sign-ups"]


def test_an_email_watch_never_shares_a_name() -> None:
    # Stored with a random ending: no one else's watch runs into it, and its
    # name can't be found by trying names.
    first, second = mailbox.own_slug("email-sign-ups"), mailbox.own_slug("email-sign-ups")
    assert first.startswith("email-sign-ups-") and first != second
    long = mailbox.own_slug("y" * 61)
    assert len(long) == 61 and long.startswith("y" * 54 + "-")


def test_company_wide_reads_leave_its_signals_out(db: Path) -> None:
    item = _email_watch(db)
    ms.insert_signal(Signal(
        watchlist_id=item.id, source_kind=SOURCE_KIND_MAILBOX, source_external_id="m1",
        captured_at=datetime.now(UTC).isoformat(), normalized_summary="App: New sign up",
        provenance_url="https://mail.google.com/", severity_hint=AlertSeverity.HIGH,
        dedup_key=f"mailbox:{item.id}:m1",
    ), db_path=db)
    assert ms.list_recent_signals(db_path=db) == []
    assert len(ms.list_signals_for_watchlist(item.id, db_path=db)) == 1


def test_every_email_is_its_own_personal_card_for_the_owner(db: Path) -> None:
    item = _email_watch(db)
    signal = Signal(
        watchlist_id=item.id, source_kind=SOURCE_KIND_MAILBOX, source_external_id="m1",
        captured_at=datetime.now(UTC).isoformat(), normalized_summary="App: New sign up",
        provenance_url="https://mail.google.com/", severity_hint=AlertSeverity.HIGH,
        dedup_key=f"mailbox:{item.id}:m1",
    )
    event = _signal_to_alert_event(signal, item)
    assert event.dedup_hint == "watch:email-sign-ups#m1"
    assert watch_slug_of(SimpleNamespace(
        source=event.source, dedup_key=event.dedup_hint, topic_tags=["external:email-sign-ups"],
    )) == "email-sign-ups"
    assert event.routed_to_person_id == OWNER
    assert event.source in PERSONAL_SOURCES
    # Set to tell someone else, or a department: it still tells only its
    # owner (text in an email it read can't send their mail to anyone).
    told = item.model_copy(update={"route_to_person_id": OTHER, "route_to_department": "sales"})
    event = _signal_to_alert_event(signal, told)
    assert (event.routed_to_person_id, event.department) == (OWNER, "")


def test_its_audit_rows_are_its_owners() -> None:
    item = SimpleNamespace(signal_type=SOURCE_KIND_MAILBOX, config_json={"mailbox_of": OWNER})
    assert mailbox.audit_privacy(item) == {"private": True, "private_to_person": OWNER}  # type: ignore[arg-type]
    other = SimpleNamespace(signal_type="stock", config_json={"mailbox_of": OWNER})
    assert mailbox.audit_privacy(other) == {}  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Words
# --------------------------------------------------------------------------- #


def test_it_says_what_it_watches_in_plain_words() -> None:
    assert describe.trigger_line(SOURCE_KIND_MAILBOX, {"keywords": ["sign up"]}) == (
        "An email that mentions “sign up” reaches the Executive or you"
    )
    assert describe.trigger_line(SOURCE_KIND_MAILBOX, {"from": ["pat@acme.example", "sam@acme.example"]}) == (
        "pat@acme.example or sam@acme.example emails the Executive or you"
    )
    # Checked every few minutes, not every 15.
    assert describe.checks_every(SOURCE_KIND_MAILBOX, "real_time") == "every 5 minutes"


# --------------------------------------------------------------------------- #
# The Executive's tools
# --------------------------------------------------------------------------- #


def _speaker(monkeypatch: pytest.MonkeyPatch, person_id: int | None, *, status: str = "connected") -> None:
    monkeypatch.setattr(wt, "_speaker_id", lambda: person_id)

    def get_person(pid: int, **_: Any) -> Any:
        return SimpleNamespace(id=pid, email="jordan@acme.example", archived=False, is_principal=pid == PRINCIPAL) if pid == person_id else None

    async def gmail_status(email: str, **_: Any) -> str:
        return status

    monkeypatch.setattr("openexecutive.people.store.get_person", get_person)
    monkeypatch.setattr(gm, "gmail_status", gmail_status)
    monkeypatch.setattr(wt, "_scan_soon", lambda: None)
    monkeypatch.setattr("openexecutive.delegation.lockdown.outside_reach_refusal", lambda *a, **k: None)


def _add(**kw: Any) -> dict[str, Any]:
    body = {"slug": "email-sign-ups", "signal_type": "mailbox", "target": "sign-up emails", **kw}
    return json.loads(asyncio.run(wt.handle_add_watchlist_entry(body)))


def test_the_executive_watches_the_speakers_own_inbox(
    db: Path, monkeypatch: pytest.MonkeyPatch, _no_audit: list[dict[str, Any]],
) -> None:
    _speaker(monkeypatch, OWNER)
    out = _add(trigger={"keywords": ["sign up"]}, config={"mailbox_of": OTHER})
    assert out["ok"] is True, out
    assert out["slug"].startswith("email-sign-ups-")  # mailbox.own_slug
    row = ms.get_watchlist_item_by_slug(out["slug"], db_path=db)
    assert row is not None
    assert row.config_json["mailbox_of"] == OWNER and row.config_json["enrich"] is False
    assert row.trigger_json == {"keywords": ["sign up"]}
    assert row.cadence == "real_time"
    assert _no_audit[-1]["private"] is True and _no_audit[-1]["private_to_person"] == OWNER


def test_an_email_watch_needs_a_sender_or_a_word(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _speaker(monkeypatch, OWNER)
    assert "error" in _add(trigger={})
    assert ms.list_watchlist(db_path=db, every_watch=True) == []


def test_without_their_own_inbox_it_watches_the_executives(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test lab 11-21: "No watch was set" for everyone without Act as me. Now
    it is set, and the reply says what it watches: the Executive's inbox for
    the principal, who reads it; a teammate's own inbox."""
    _speaker(monkeypatch, OWNER, status="not_configured")
    out = _add(trigger={"from": ["robin@hale.example"]})
    assert out["ok"] is True and out["watches"].startswith("nothing yet")
    assert "emails you." in out["note"] and "connected" in out["note"]
    _speaker(monkeypatch, OWNER)
    both = _add(trigger={"from": ["robin@hale.example"]})
    assert both["watches"] == "their own inbox" and "emails your inbox" in both["note"]
    words = wt._watch_words(False, None, principal=True)
    assert words["watches"] == "the Executive's inbox" and "emails me." in words["note"]
    words = wt._watch_words(True, None, principal=True)
    assert words["watches"] == "the Executive's inbox and their own" and "emails me or your inbox" in words["note"]


def test_no_one_asking_sets_no_watch(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _speaker(monkeypatch, None)
    assert "error" in _add(trigger={"from": ["robin@hale.example"]})
    assert ms.list_watchlist(db_path=db, every_watch=True) == []


def test_if_nothing_by_a_day_is_kept_and_said(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _speaker(monkeypatch, OWNER, status="not_configured")
    day = (datetime.now(UTC) + timedelta(days=4)).date()
    out = _add(trigger={"from": ["robin@hale.example"]}, quiet_by=day.isoformat())
    assert out["quiet_by"] == day.isoformat() and "I'll ping you that morning" in out["note"]
    row = ms.get_watchlist_item_by_slug(out["slug"], db_path=db)
    assert row is not None and mailbox.quiet_by(row) == day
    assert describe.quiet_line(row) == f"Ping if nothing by {day:%a %b} {day.day}"
    assert "error" in _add(trigger={"from": ["robin@hale.example"]}, quiet_by="2020-01-01")
    # "Wait another day": the day moves on, and so does when matches count from.
    later = day + timedelta(days=1)
    tuned = json.loads(asyncio.run(wt.handle_tune_watchlist_entry({"slug": out["slug"], "quiet_by": later.isoformat()})))
    assert tuned["ok"] is True
    row = ms.get_watchlist_item_by_slug(out["slug"], db_path=db)
    assert row is not None and mailbox.quiet_by(row) == later
    cleared = json.loads(asyncio.run(wt.handle_tune_watchlist_entry({"slug": out["slug"], "quiet_by": None})))
    assert cleared["ok"] is True
    row = ms.get_watchlist_item_by_slug(out["slug"], db_path=db)
    assert row is not None and mailbox.quiet_by(row) is None and describe.quiet_line(row) == ""


def test_someone_else_cant_see_or_change_an_email_watch(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _email_watch(db)
    _speaker(monkeypatch, OTHER)
    listed = json.loads(asyncio.run(wt.handle_list_watchlist({"enabled_only": False})))
    assert listed["count"] == 0
    assert "error" in json.loads(asyncio.run(wt.handle_tune_watchlist_entry({"slug": "email-sign-ups", "enabled": False})))
    assert "error" in json.loads(asyncio.run(wt.handle_remove_watchlist_entry({"slug": "email-sign-ups"})))
    assert ms.get_watchlist_item_by_slug("email-sign-ups", db_path=db) is not None

    _speaker(monkeypatch, OWNER)
    listed = json.loads(asyncio.run(wt.handle_list_watchlist({"enabled_only": False})))
    assert [w["slug"] for w in listed["watchlist"]] == ["email-sign-ups"]
    assert listed["watchlist"][0]["tells_you_when"] == (
        "An email that mentions “sign up” or “signed up” reaches the Executive or you"
    )
    tuned = json.loads(asyncio.run(wt.handle_tune_watchlist_entry({"slug": "email-sign-ups", "trigger": {"from": ["x"]}})))
    assert "error" in tuned  # no real address and no word: nothing would ever match


def test_research_never_proposes_an_email_watch() -> None:
    out = json.loads(asyncio.run(wt.handle_propose_watch(
        {"slug": "email-x", "signal_type": "mailbox", "target": "x"}, [],
    )))
    assert "error" in out


def test_an_inbox_watch_can_be_set_after_their_mail_was_read(db: Path) -> None:
    # "Check Dana's email and tell me when she replies": the read shuts the
    # watch list's outside fetches, not a watch on their own inbox.
    from openexecutive.delegation import lockdown

    assert not lockdown.mail_touched_withholds("add_watchlist_entry", {"signal_type": "mailbox", "slug": "x"})
    assert lockdown.mail_touched_withholds("add_watchlist_entry", {"signal_type": "rss", "slug": "x"})
    _email_watch(db)
    ms.insert_watchlist_item(slug="rss-acme", signal_type="rss", target="https://acme.example/feed", db_path=db)
    assert not lockdown.mail_touched_withholds("tune_watchlist_entry", {"slug": "email-sign-ups", "enabled": False})
    assert lockdown.mail_touched_withholds("tune_watchlist_entry", {"slug": "rss-acme", "enabled": False})
    assert lockdown.mail_touched_withholds("tune_watchlist_entry", "not a dict")


def test_the_executives_own_row_of_an_email_watch_call_is_the_speakers_alone(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatch row the Executive writes for every tool call quotes the
    call's input and result: for a watch on the speaker's own inbox, who the
    email is from and what it says. That row is theirs alone, like the rows
    the watch writes itself; a company watch's row stays the company's."""
    from openexecutive.orchestrator import artifact_records
    from openexecutive.orchestrator.executive import _artifact_row_owner, _private_tool_row

    _speaker(monkeypatch, OWNER)
    monkeypatch.setattr(artifact_records, "current_viewer", lambda: SimpleNamespace(person_id=OWNER))

    def row(name: str, tool_input: dict[str, Any], result: str) -> tuple[bool, int | None]:
        return _private_tool_row(name, tool_input, result), _artifact_row_owner(name, tool_input, result)

    add_input = {"slug": "email-offers", "signal_type": "mailbox", "target": "offers",
                 "trigger": {"from": ["recruiter@acme.example"], "keywords": ["offer"]}}
    added = asyncio.run(wt.handle_add_watchlist_entry(add_input))
    slug = json.loads(added)["slug"]
    assert row("add_watchlist_entry", add_input, added) == (True, OWNER)
    # Before the handler ran (a failed call's row): the input alone says so.
    assert row("add_watchlist_entry", add_input, None) == (True, OWNER)

    tune_input = {"slug": slug, "trigger": {"keywords": ["offer letter"]}}
    tuned = asyncio.run(wt.handle_tune_watchlist_entry(tune_input))
    assert row("tune_watchlist_entry", tune_input, tuned) == (True, OWNER)

    listed = asyncio.run(wt.handle_list_watchlist({}))
    assert row("list_watchlist", {}, listed) == (True, OWNER)

    removed = asyncio.run(wt.handle_remove_watchlist_entry({"slug": slug}))
    assert json.loads(removed)["ok"] is True
    assert row("remove_watchlist_entry", {"slug": slug}, removed) == (True, OWNER)

    # A company watch: the row stays the company's.
    ms.insert_watchlist_item(slug="news-acme", signal_type="query", target="acme", db_path=db)
    assert row("list_watchlist", {}, asyncio.run(wt.handle_list_watchlist({}))) == (False, None)
    company = {"slug": "news-acme", "notes": "launches only"}
    assert row("tune_watchlist_entry", company, asyncio.run(wt.handle_tune_watchlist_entry(company))) == (False, None)


def test_a_sender_never_sets_or_changes_a_watch_on_the_owners_inbox(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Their own mail read never holds a watch on their own inbox, but a turn
    answering mail from someone not in People is the sender asking: an email
    saying "turn off the fraud-alert watch" changes nothing."""
    from openexecutive.delegation import lockdown
    from openexecutive.orchestrator import levels, outside_content

    item = _email_watch(db)
    _speaker(monkeypatch, OWNER)
    monkeypatch.setattr(lockdown, "mail_turn_withholds", lambda name, tool_input=None: False)
    monkeypatch.setattr(outside_content, "read_outside", lambda: True)
    add = {"slug": "email-fraud", "signal_type": "mailbox", "target": "fraud alerts",
           "trigger": {"keywords": ["fraud"]}}
    tune = {"slug": item.slug, "enabled": False}

    monkeypatch.setattr(outside_content, "answers_off_team_mail", lambda: False)
    assert lockdown.outside_reach_reason("add_watchlist_entry", add) is None
    assert lockdown.outside_reach_reason("tune_watchlist_entry", tune) is None

    monkeypatch.setattr(outside_content, "answers_off_team_mail", lambda: True)
    assert lockdown.outside_reach_reason("add_watchlist_entry", add) == levels.STRANGER
    assert lockdown.outside_reach_reason("tune_watchlist_entry", tune) == levels.STRANGER


# --------------------------------------------------------------------------- #
# The Executive's inbox
# --------------------------------------------------------------------------- #

PRINCIPAL = 1


def _executive_inbox(monkeypatch: pytest.MonkeyPatch, *, own: str = "not_configured") -> list[str]:
    """No own inbox for the owner; the principal is someone else."""
    box = _Mailbox([], [])
    _patch_mailbox(monkeypatch, box, status=own)
    monkeypatch.setattr(
        "openexecutive.people.store.find_principal_person", lambda **_: SimpleNamespace(id=PRINCIPAL),
    )
    monkeypatch.setattr("openexecutive.monitoring.pipeline.scan_soon", lambda *a, **k: True)
    notes: list[str] = []
    monkeypatch.setattr(mailbox_source, "observe", lambda item, note: notes.append(note))
    return notes


async def _proven() -> bool:
    return True


def _robin(**kw: Any) -> int:
    args: dict[str, Any] = {
        "message_id": "x1", "thread_id": "th1", "from_addr": "Robin@Hale.example", "from_name": "Robin Hale",
        "subject": "December rent, $480 due Tue Dec 1", "private": False, "mail_of": None, "proven": _proven, **kw,
    }
    return asyncio.run(mailbox.note_executive_mail(**args))


def test_an_email_to_the_executive_is_told_to_whoever_watches_for_it(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    notes = _executive_inbox(monkeypatch)
    item = _email_watch(db, trigger={"from": ["robin@hale.example"]}, slug="robin-email")
    ms.mark_polled(item.id, datetime.now(UTC), db_path=db)
    # Robin's email as this teammate's own mail with the Executive (mail_of).
    assert _robin(mail_of=OWNER) == 1
    assert _robin(message_id="x2", from_addr="pat@acme.example", mail_of=OWNER) == 0  # not from Robin
    # Due at once, whatever its cadence: it has something to tell.
    fresh = ms.get_watchlist_item(item.id, db_path=db)
    assert fresh is not None and fresh.last_polled_at is None
    signals = asyncio.run(mailbox_source.MailboxSource().poll(fresh, db_path=db))
    assert [s.normalized_summary for s in signals] == ["Robin Hale: December rent, $480 due Tue Dec 1"]
    assert signals[0].raw_payload == {"mail_thread": "th1", "mail_from": "robin@hale.example"}
    assert signals[0].dedup_key == f"mailbox:{item.id}:exec:x1" and notes == []
    # Its card names the mail thread, so it joins the turn's card on it.
    event = _signal_to_alert_event(signals[0], fresh)
    assert (event.mail_thread, event.mail_from, event.routed_to_person_id) == ("th1", "robin@hale.example", OWNER)
    # Once told, it is forgotten and never told again.
    ms.insert_signal(signals[0], db_path=db)
    assert asyncio.run(mailbox_source.MailboxSource().poll(fresh, db_path=db)) == []
    assert ms.executive_mail_for(item.id, db_path=db) == []
    assert notes == [mailbox_source.NOT_CONNECTED]


def test_only_email_its_owner_could_already_see_is_told(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Privacy hides, never stops: the watch is set either way; it tells only
    what its owner already sees elsewhere (the audit log, the chats)."""
    _executive_inbox(monkeypatch)
    mine = _email_watch(db, trigger={"from": ["robin@hale.example"]}, slug="robin-mine")
    principals = _email_watch(db, owner=PRINCIPAL, trigger={"from": ["robin@hale.example"]}, slug="robin-principal")
    # Mail private to the principal (one they forwarded): theirs alone.
    assert _robin(message_id="p1", private=True) == 1
    assert [m["message_id"] for m in ms.executive_mail_for(principals.id, db_path=db)] == ["p1"]
    assert ms.executive_mail_for(mine.id, db_path=db) == []
    # A teammate's own mail: only that teammate's.
    assert _robin(message_id="t1", mail_of=OWNER) == 1
    assert [m["message_id"] for m in ms.executive_mail_for(mine.id, db_path=db)] == ["t1"]
    # Any other mail to the Executive (the principal's own, a contact's, a
    # teammate's nothing proved): the principal, who reads its inbox.
    assert _robin(message_id="c1") == 1
    assert [m["message_id"] for m in ms.executive_mail_for(mine.id, db_path=db)] == ["t1"]
    assert mailbox.seen_by(OTHER, private=False, mail_of=None, principal_id=PRINCIPAL) is False
    assert mailbox.seen_by(PRINCIPAL, private=False, mail_of=None, principal_id=PRINCIPAL) is True


def test_a_sender_nothing_proved_joins_no_card(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A forged From with a real thread's References must not add its subject
    to the card already open on that thread: it is told on a card of its own."""
    _executive_inbox(monkeypatch)
    item = _email_watch(db, owner=PRINCIPAL, trigger={"from": ["robin@hale.example"]}, slug="robin-email")

    async def unproven() -> bool:
        return False

    async def broken() -> bool:
        raise RuntimeError("no proof")

    assert _robin(message_id="f1", proven=unproven) == 1
    assert _robin(message_id="f2", proven=broken) == 1
    assert _robin(message_id="f3", proven=None) == 1
    assert _robin(message_id="ok") == 1
    kept = {m["message_id"]: m["thread_id"] for m in ms.executive_mail_for(item.id, db_path=db)}
    assert kept == {"f1": "", "f2": "", "f3": "", "ok": "th1"}
    calls: list[int] = []

    async def counted() -> bool:
        calls.append(1)
        return True

    # Asked only when a watch matches.
    assert _robin(message_id="n1", from_addr="pat@acme.example", proven=counted) == 0 and calls == []


def test_nothing_by_the_day_pings_that_morning_once(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from zoneinfo import ZoneInfo

    from openexecutive.alerts.models import NEEDS_CHOICE

    _executive_inbox(monkeypatch)
    zone = ZoneInfo("America/New_York")
    monkeypatch.setattr("openexecutive.memory.workspace_settings.get_user_timezone", lambda *a, **k: zone)
    set_at = datetime(2026, 11, 21, 15, 0, tzinfo=UTC)
    day = set_at.date() + timedelta(days=4)
    item = _email_watch(db, trigger={"from": ["robin@hale.example"]}, slug="robin-email")
    assert item.id is not None
    ms.set_watchlist_config_key(item.id, "display_label", "Robin Hale's email", db_path=db)
    for key, value in mailbox.quiet_config(day, set_at).items():
        ms.set_watchlist_config_key(item.id, key, value, db_path=db)
    item = ms.get_watchlist_item(item.id, db_path=db)
    assert item is not None
    morning = datetime(day.year, day.month, day.day, 9, 0, tzinfo=zone)
    assert not mailbox.quiet_ping_due(item, morning - timedelta(minutes=1), zone, db_path=db)
    assert mailbox.quiet_ping_due(item, morning, zone, db_path=db)
    # The day after still pings (the scan may have been down); later never.
    assert mailbox.quiet_ping_due(item, morning + timedelta(days=1, hours=14), zone, db_path=db)
    assert not mailbox.quiet_ping_due(item, morning + timedelta(days=2), zone, db_path=db)

    monkeypatch.setattr(
        mailbox_source, "datetime",
        SimpleNamespace(now=lambda tz=None: morning.astimezone(UTC), fromisoformat=datetime.fromisoformat),
    )
    signals = asyncio.run(mailbox_source.MailboxSource().poll(item, db_path=db))
    assert [s.normalized_summary for s in signals] == ["No email from Robin Hale yet"]
    assert signals[0].raw_payload["needs"] == NEEDS_CHOICE
    assert signals[0].raw_payload["choices"] == ["Ask Robin Hale", mailbox_source.WAIT_A_DAY]
    event = _signal_to_alert_event(signals[0], item)
    assert event.needs == NEEDS_CHOICE and event.choices == ["Ask Robin Hale", "Wait another day"]
    # Once a day: told, it doesn't ping again.
    ms.insert_signal(signals[0], db_path=db)
    assert asyncio.run(mailbox_source.MailboxSource().poll(item, db_path=db)) == []


def test_an_email_that_came_but_was_never_told_still_means_something_came(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Robin's email is recorded on Tuesday (a watch in dry run tells
    nothing); Wednesday morning brings no "nothing came" ping."""
    from zoneinfo import ZoneInfo

    _executive_inbox(monkeypatch)
    zone = ZoneInfo("America/New_York")
    set_at = datetime(2026, 11, 21, 15, 0, tzinfo=UTC)
    day = set_at.date() + timedelta(days=4)
    item = _email_watch(db, trigger={"from": ["robin@hale.example"]}, slug="robin-email")
    assert item.id is not None
    for key, value in mailbox.quiet_config(day, set_at).items():
        ms.set_watchlist_config_key(item.id, key, value, db_path=db)
    item = ms.get_watchlist_item(item.id, db_path=db)
    assert item is not None and item.last_fired_at is None
    morning = datetime(day.year, day.month, day.day, 9, 0, tzinfo=zone)
    ms.insert_signal(Signal(
        watchlist_id=item.id, source_kind=SOURCE_KIND_MAILBOX, source_external_id="exec:x1",
        captured_at=(set_at + timedelta(days=1)).isoformat(), normalized_summary="Robin Hale: rent",
        provenance_url="https://app.example/", dedup_key=f"mailbox:{item.id}:exec:x1",
    ), db_path=db)
    assert not mailbox.quiet_ping_due(item, morning, zone, db_path=db)


def test_mail_that_comes_on_the_day_is_told_without_the_nothing_came_ping(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Robin writes at 9:03 on the day; the 9:05 poll tells that, not "no
    email from Robin yet" as well."""
    from zoneinfo import ZoneInfo

    _executive_inbox(monkeypatch)
    zone = ZoneInfo("America/New_York")
    monkeypatch.setattr("openexecutive.memory.workspace_settings.get_user_timezone", lambda *a, **k: zone)
    set_at = datetime(2026, 11, 21, 15, 0, tzinfo=UTC)
    day = set_at.date() + timedelta(days=4)
    item = _email_watch(db, trigger={"from": ["robin@hale.example"]}, slug="robin-email")
    assert item.id is not None
    for key, value in mailbox.quiet_config(day, set_at).items():
        ms.set_watchlist_config_key(item.id, key, value, db_path=db)
    assert _robin(mail_of=OWNER) == 1
    item = ms.get_watchlist_item(item.id, db_path=db)
    assert item is not None
    at = datetime(day.year, day.month, day.day, 9, 5, tzinfo=zone)
    monkeypatch.setattr(
        mailbox_source, "datetime",
        SimpleNamespace(now=lambda tz=None: at.astimezone(UTC), fromisoformat=datetime.fromisoformat),
    )
    signals = asyncio.run(mailbox_source.MailboxSource().poll(item, db_path=db))
    assert [s.normalized_summary for s in signals] == ["Robin Hale: December rent, $480 due Tue Dec 1"]


def _both_inboxes(monkeypatch: pytest.MonkeyPatch, messages: list[Any]) -> None:
    """The owner's own inbox is connected and holds ``messages``."""
    _patch_mailbox(monkeypatch, _Mailbox(messages, []))
    monkeypatch.setattr(
        "openexecutive.people.store.find_principal_person", lambda **_: SimpleNamespace(id=PRINCIPAL),
    )
    monkeypatch.setattr("openexecutive.monitoring.pipeline.scan_soon", lambda *a, **k: True)


def _rent(item: Any, *, id: str = "own1", header: str = "<RENT-1@hale.example>") -> Any:
    at = (datetime.fromisoformat(item.created_at) + timedelta(minutes=1)).isoformat()
    return gm.MailMessage(
        id=id, thread_id="ot1", from_addr="robin@hale.example", from_name="Robin Hale",
        subject="December rent, $480 due Tue Dec 1", received_at=at, message_id_header=header,
    )


def _header(value: str) -> Any:
    async def read() -> str:
        return value

    return read


def _poll_and_record(item_id: int, db: Path) -> list[Signal]:
    fresh = ms.get_watchlist_item(item_id, db_path=db)
    assert fresh is not None
    signals = asyncio.run(mailbox_source.MailboxSource().poll(fresh, db_path=db))
    for signal in signals:
        ms.insert_signal(signal, db_path=db)
    return signals


def test_an_email_to_them_and_the_executive_is_told_once_from_the_executives_copy(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Robin writes to the owner and cc's the Executive: one card, on the
    thread, and the next poll reads on past it rather than telling it again."""
    item = _email_watch(db, trigger={"from": ["robin@hale.example"]}, slug="robin-email")
    assert item.id is not None
    _both_inboxes(monkeypatch, [_rent(item)])
    assert _robin(mail_of=OWNER, header_id=_header("<rent-1@hale.example>")) == 1

    first = _poll_and_record(item.id, db)
    assert [s.source_external_id for s in first] == ["exec:x1"]
    assert first[0].raw_payload["mail_thread"] == "th1"
    assert _poll_and_record(item.id, db) == []
    fresh = ms.get_watchlist_item(item.id, db_path=db)
    assert fresh is not None and mailbox_source._read_through(fresh, db_path=db) is not None


def test_an_email_read_in_their_own_inbox_first_is_not_told_again_from_the_executives(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _email_watch(db, trigger={"from": ["robin@hale.example"]}, slug="robin-email")
    assert item.id is not None
    _both_inboxes(monkeypatch, [_rent(item)])
    assert [s.source_external_id for s in _poll_and_record(item.id, db)] == ["own1"]
    # The Executive's copy arrives a minute later.
    assert _robin(mail_of=OWNER, header_id=_header("<rent-1@hale.example>")) == 1
    assert _poll_and_record(item.id, db) == []
    assert ms.executive_mail_for(item.id, db_path=db) == []


def test_two_emails_with_the_same_sender_and_subject_are_both_told(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _email_watch(db, trigger={"from": ["robin@hale.example"]}, slug="robin-email")
    assert item.id is not None
    _both_inboxes(monkeypatch, [_rent(item, header="<rent-2@hale.example>")])
    assert _robin(mail_of=OWNER, header_id=_header("<rent-1@hale.example>")) == 1
    assert sorted(s.source_external_id for s in _poll_and_record(item.id, db)) == ["exec:x1", "own1"]


def test_nothing_by_today_said_after_the_morning_waits_for_the_next_morning(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At 2 PM: "ping me if nothing from Robin by today". No ping at 2:01."""
    from zoneinfo import ZoneInfo

    _executive_inbox(monkeypatch)
    zone = ZoneInfo("America/New_York")
    set_at = datetime(2026, 11, 24, 14, 0, tzinfo=zone)
    item = _email_watch(db, trigger={"from": ["robin@hale.example"]}, slug="robin-email")
    assert item.id is not None
    for key, value in mailbox.quiet_config(set_at.date(), set_at.astimezone(UTC)).items():
        ms.set_watchlist_config_key(item.id, key, value, db_path=db)
    item = ms.get_watchlist_item(item.id, db_path=db)
    assert item is not None
    assert not mailbox.quiet_ping_due(item, set_at + timedelta(minutes=1), zone, db_path=db)
    next_morning = datetime(2026, 11, 25, 9, 0, tzinfo=zone)
    assert not mailbox.quiet_ping_due(item, next_morning - timedelta(minutes=1), zone, db_path=db)
    assert mailbox.quiet_ping_due(item, next_morning, zone, db_path=db)


def test_an_email_to_the_executive_while_the_watch_was_off_is_not_told(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _executive_inbox(monkeypatch)
    item = _email_watch(db, trigger={"from": ["robin@hale.example"]}, slug="robin-email")
    assert item.id is not None
    assert _robin(mail_of=OWNER) == 1
    ms.set_enabled(item.id, False, db_path=db)
    ms.set_enabled(item.id, True, db_path=db)
    fresh = ms.get_watchlist_item(item.id, db_path=db)
    assert fresh is not None
    assert asyncio.run(mailbox_source.MailboxSource().poll(fresh, db_path=db)) == []
    assert ms.executive_mail_for(item.id, db_path=db) == []


def test_a_teammate_watch_that_sees_no_inbox_never_says_nothing_came(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from zoneinfo import ZoneInfo

    _executive_inbox(monkeypatch)
    monkeypatch.setattr(
        "openexecutive.people.store.get_person",
        lambda pid: SimpleNamespace(**{**vars(_person()), "is_principal": False}) if pid == OWNER else None,
    )
    zone = ZoneInfo("America/New_York")
    monkeypatch.setattr("openexecutive.memory.workspace_settings.get_user_timezone", lambda *a, **k: zone)
    set_at = datetime(2026, 11, 21, 15, 0, tzinfo=UTC)
    day = set_at.date() + timedelta(days=4)
    item = _email_watch(db, trigger={"from": ["robin@hale.example"]}, slug="robin-email")
    assert item.id is not None
    for key, value in mailbox.quiet_config(day, set_at).items():
        ms.set_watchlist_config_key(item.id, key, value, db_path=db)
    item = ms.get_watchlist_item(item.id, db_path=db)
    assert item is not None
    morning = datetime(day.year, day.month, day.day, 9, 0, tzinfo=zone)
    monkeypatch.setattr(
        mailbox_source, "datetime",
        SimpleNamespace(now=lambda tz=None: morning.astimezone(UTC), fromisoformat=datetime.fromisoformat),
    )
    assert asyncio.run(mailbox_source.MailboxSource().poll(item, db_path=db)) == []
