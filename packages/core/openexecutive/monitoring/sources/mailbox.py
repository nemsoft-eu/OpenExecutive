"""Email-watch adapter: the email a person can see (``monitoring.mailbox``).

Watchlist row shape:

  - ``signal_type``: ``"mailbox"``
  - ``target``: what it watches for in plain words ("sign-up emails",
    "Pat's reply"); the watch's name.
  - ``config_json``: ``{"mailbox_of": <People id>, "enrich": false}``, and
    ``quiet_by`` / ``quiet_since`` for "if nothing by".
  - ``trigger_json``: ``{"from": [addresses], "keywords": [words]}``.

Each poll tells, at HIGH, so the person hears about it on their own channel
as well as on a card:

- the emails that reached the Executive and matched since the last poll
  (``mailbox.note_executive_mail`` kept them), each on the card of the mail
  thread when the Executive's handling of it made one for the same person;
- when the person's own inbox is connected, the mail that arrived there
  since the watch was set or since the last poll (a little overlap; the
  dedup key makes it once per email), every tab, one signal per email that
  matches;
- once, the morning of its "nothing by" day, that nothing came, with a
  choice to ask the sender or wait another day.

A signal carries the sender's name and the subject only, never the email's
text.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openexecutive.alerts.models import AlertSeverity
from openexecutive.monitoring import mailbox, store
from openexecutive.monitoring.models import SOURCE_KIND_MAILBOX, Signal, WatchlistItem
from openexecutive.monitoring.sources.base import observe

logger = logging.getLogger(__name__)

# Mail read per page, and pages per poll: a poll reads back page by page to
# where it last read (one more page to see there is nothing older), so a busy inbox or a day disconnected loses nothing,
# up to this many emails at once.
_READ_LIMIT = 50
_MAX_PAGES = 20
# New emails told per poll, oldest first. Each is a card and a triage call,
# and triage takes about a minute's worth at once; the rest wait for the
# next poll (the read point stops at the last one told), so a backlog after
# a day disconnected comes through in order, a few minutes apart.
PER_POLL = 20
# Re-read this far before the last poll: mail the provider listed a little
# late is still caught (the dedup key makes a re-read email a no-op).
_OVERLAP = timedelta(minutes=10)
_SUMMARY_MAX = 200
NOT_CONNECTED = "Nothing new matched. Your own inbox isn't connected, so only email to the Executive is checked."
NO_OWNER = "This watch has no one to tell."
NOTHING_NEW = "Nothing new matched."
# The ping's choices (``alerts.models.NEEDS_CHOICE``).
WAIT_A_DAY = "Wait another day"
READ_THROUGH_KEY = mailbox.READ_THROUGH_KEY
# ``config_json`` key: where the last poll that told emails read to, and the
# dedup keys of those emails. It becomes the read point only once all of
# them are recorded, so a tick cut off before it recorded them (a restart, a
# locked database) reads them again rather than skipping them.
PENDING_KEY = "read_pending"


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _read_through(item: WatchlistItem, *, db_path: Path | None) -> datetime | None:
    """Where the last read that worked read to (``READ_THROUGH_KEY``), moved
    on to the last poll's point (``PENDING_KEY``) once every email that poll
    told is recorded."""
    config = item.config_json or {}
    read = _parse(str(config.get(READ_THROUGH_KEY) or ""))
    pending = config.get(PENDING_KEY)
    if not isinstance(pending, dict):
        return read
    through = _parse(str(pending.get("through") or ""))
    keys = [str(k) for k in pending.get("keys") or []]
    if through is None or len(store.recorded_dedup_keys(keys, db_path=db_path)) < len(set(keys)):
        return read
    return through if read is None else max(read, through)


def _start(item: WatchlistItem, now: datetime) -> datetime:
    """The oldest mail it may tell: from when the watch was set, or when it
    was last turned back on (``mailbox.READ_FROM_KEY``)."""
    created = _parse(item.created_at) or now
    turned_on = _parse(str((item.config_json or {}).get(mailbox.READ_FROM_KEY) or ""))
    return max(created, turned_on) if turned_on is not None else created


def _since(item: WatchlistItem, now: datetime, read: datetime | None) -> datetime:
    """From when to read: a little before the last read that worked
    (``read``), else from its start (``_start``), never before that. Not
    the last poll: a poll that couldn't read (inbox not connected, the mail
    service down) must not move it on, or the mail that came meanwhile is
    skipped."""
    start = _start(item, now)
    return max(start, read - _OVERLAP) if read is not None else start


async def _read_back_to(client: Any, since: datetime) -> list[Any]:
    """Every inbox email since ``since``: the newest page, then the page
    before its oldest email, and so on until a page has nothing new (up to
    ``_MAX_PAGES``). Not until a page comes back short: the clients drop the
    person's own mail and mail deleted since it was listed, so a short page
    can still have older mail behind it. A page boundary is read twice (the
    mail services count in whole seconds); each email is kept once."""
    received: list[Any] = []
    seen: set[str] = set()
    before: datetime | None = None
    for _ in range(_MAX_PAGES):
        page = await client.received_since(since, before=before, limit=_READ_LIMIT)
        new = [m for m in page if m.id not in seen]
        seen.update(m.id for m in new)
        received += new
        stamps = [t for m in page if (t := _parse(getattr(m, "received_at", "") or "")) is not None]
        if not new or not stamps:
            break
        before = min(stamps) + timedelta(seconds=1)
    else:
        logger.warning("email watch: more than %d emails since %s; read the newest", _READ_LIMIT * _MAX_PAGES, since)
    return received


def _dedup_key(watch_id: int, message: Any) -> str:
    """An own-inbox email's key: its Message-ID's (``_rfc_key``), the same
    as the Executive's copy of it, so one email sent to both is told once
    whichever copy is read first; else its own id."""
    rfc = _rfc_key(watch_id, str(getattr(message, "message_id_header", "") or ""))
    return rfc or f"mailbox:{watch_id}:{message.id}"


def _rfc_key(watch_id: int, header: str) -> str:
    """The key of the email whose Message-ID header is ``header`` ("" for
    none): the same in every mailbox it reached."""
    value = header.strip().strip("<>").strip().lower()
    if not value:
        return ""
    return f"mailbox:{watch_id}:rfc:{hashlib.sha256(value.encode()).hexdigest()[:32]}"


def summary_of(message: Any) -> str:
    """Who it is from and its subject, cleaned: one line, no links or
    addresses (``delegation.reminders.clean_text``)."""
    from openexecutive.delegation.reminders import clean_text

    who = clean_text(str(getattr(message, "from_name", "") or "")) or "Someone"
    subject = clean_text(str(getattr(message, "subject", "") or ""))
    line = f"{who}: {subject}" if subject else f"{who} (no subject)"
    return line[:_SUMMARY_MAX]


def _executive_dedup_key(watch_id: int, message_id: str) -> str:
    return f"mailbox:{watch_id}:exec:{message_id}"


def _quiet_dedup_key(watch_id: int, day: str) -> str:
    return f"mailbox:{watch_id}:quiet:{day}"


def _app_link() -> str:
    from openexecutive.config import get_settings

    return get_settings().ui_base_url.rstrip("/") + "/"


def executive_signals(item: WatchlistItem, now: datetime, *, db_path: Path | None) -> list[Signal]:
    """The emails that reached the Executive and matched, not told yet,
    oldest first (``PER_POLL`` at most; the rest wait for the next poll).
    Those already told are forgotten."""
    assert item.id is not None
    # Only mail from since it was set or last turned back on (``_start``):
    # what came while it was off is forgotten, as in their own inbox.
    start = _start(item, now)
    rows = store.executive_mail_for(item.id, db_path=db_path)
    store.drop_executive_mail(
        item.id, [m["message_id"] for m in rows if (_parse(m["received_at"]) or now) < start], db_path=db_path,
    )
    kept = [m for m in rows if (_parse(m["received_at"]) or now) >= start]
    keys = {
        m["message_id"]: _rfc_key(item.id, m.get("rfc_id", "")) or _executive_dedup_key(item.id, m["message_id"])
        for m in kept
    }
    told = store.recorded_dedup_keys(list(keys.values()), db_path=db_path)
    store.drop_executive_mail(item.id, [mid for mid, key in keys.items() if key in told], db_path=db_path)
    signals: list[Signal] = []
    for mail in [m for m in kept if keys[m["message_id"]] not in told][:PER_POLL]:
        message = SimpleNamespace(from_name=mail["from_name"] or mail["from_addr"], subject=mail["subject"])
        signals.append(Signal(
            watchlist_id=item.id,
            source_kind=SOURCE_KIND_MAILBOX,
            source_external_id=f"exec:{mail['message_id']}",
            captured_at=now.isoformat(),
            published_at=mail["received_at"],
            normalized_summary=summary_of(message),
            # The mail thread it came on: the card joins the one the
            # Executive's handling of that email made for the same person.
            raw_payload={"mail_thread": mail["thread_id"], "mail_from": mail["from_addr"]},
            provenance_url=_app_link(),
            severity_hint=AlertSeverity.HIGH,
            dedup_key=keys[mail["message_id"]],
        ))
    return signals


def quiet_signal(item: WatchlistItem, now: datetime, *, db_path: Path | None) -> Signal | None:
    """The "nothing came" ping, once, the morning of the watch's "nothing
    by" day when nothing has matched since it was set
    (``mailbox.quiet_ping_due``)."""
    from openexecutive.alerts.models import NEEDS_CHOICE
    from openexecutive.memory.workspace_settings import get_user_timezone

    assert item.id is not None
    day = mailbox.quiet_by(item)
    if day is None or not mailbox.quiet_ping_due(item, now, get_user_timezone(), db_path=db_path):
        return None
    key = _quiet_dedup_key(item.id, day.isoformat())
    if store.recorded_dedup_keys([key], db_path=db_path):
        return None
    who = mailbox.who_from(item)
    return Signal(
        watchlist_id=item.id,
        source_kind=SOURCE_KIND_MAILBOX,
        source_external_id=f"quiet:{day.isoformat()}",
        captured_at=now.isoformat(),
        published_at=now.isoformat(),
        normalized_summary=f"No email from {who} yet" if who else "No email yet for " + (item.target or "this watch"),
        raw_payload={
            "needs": NEEDS_CHOICE,
            "question": "You asked me to ping you if nothing came by today. Want me to write and ask?",
            "choices": [f"Ask {who}" if who else "Write and ask", WAIT_A_DAY],
        },
        provenance_url=_app_link(),
        severity_hint=AlertSeverity.HIGH,
        dedup_key=key,
    )


class MailboxSource:
    kind: str = SOURCE_KIND_MAILBOX
    # The old one-off email watch checked every 5 minutes; so does this.
    default_poll_interval_minutes: int = 5
    # Only mail from after the watch was set is read, so nothing older can
    # replay: no baseline needed.
    seed_on_first_poll: bool = False

    async def poll(self, item: WatchlistItem, *, db_path: Path | None = None) -> list[Signal]:
        from openexecutive.delegation.gmail import gmail_status, normalize_email
        from openexecutive.people.store import get_person

        owner = mailbox.owner_id(item)
        person = get_person(owner) if owner is not None else None
        if person is None or person.archived or item.id is None:
            observe(item, NO_OWNER)
            return []
        now = datetime.now(UTC)
        signals = executive_signals(item, now, db_path=db_path)
        email = normalize_email(person.email)
        connected = await gmail_status(email) == "connected"
        if connected:
            # One email sent to them and the Executive both is told once,
            # by its Message-ID (``_dedup_key``): read in the same poll, from
            # the Executive's copy, on the card of its thread. Its key is
            # still recorded, so the read point moves past it.
            told = {s.dedup_key for s in signals}
            signals += [
                s for s in await self._own_inbox(item, email, now, db_path=db_path)
                if s.dedup_key not in told
            ]
        # Something came this poll, so it isn't the day nothing came. A
        # teammate's watch with no inbox of theirs sees no outside mail at
        # all (``mailbox.seen_by``), so it can't say nothing came either.
        blind = not connected and not getattr(person, "is_principal", False)
        quiet = None if signals or blind else quiet_signal(item, now, db_path=db_path)
        if quiet is not None:
            signals.append(quiet)
        if not signals:
            observe(item, NOTHING_NEW if connected else NOT_CONNECTED)
        return signals

    async def _own_inbox(
        self, item: WatchlistItem, email: str, now: datetime, *, db_path: Path | None,
    ) -> list[Signal]:
        """The matching mail in the person's own inbox not told yet."""
        from openexecutive.delegation.gmail import gmail_for, mailbox_link, normalize_email

        assert item.id is not None
        read = _read_through(item, db_path=db_path)
        since = _since(item, now, read)
        start = _start(item, now)
        received = await _read_back_to(gmail_for(email), since)
        found: list[tuple[datetime, Any]] = []
        for message in received:
            arrived = _parse(getattr(message, "received_at", "") or "")
            if arrived is not None and arrived < start:
                continue
            if mailbox.matches(
                item.trigger_json,
                from_addr=normalize_email(getattr(message, "from_addr", "") or ""),
                from_name=str(getattr(message, "from_name", "") or ""),
                subject=str(getattr(message, "subject", "") or ""),
            ):
                found.append((arrived or now, message))
        # Only what isn't told yet (the overlap re-reads a few), oldest first.
        told = store.recorded_dedup_keys([_dedup_key(item.id, m) for _, m in found], db_path=db_path)
        new = sorted(((at, m) for at, m in found if _dedup_key(item.id, m) not in told), key=lambda f: f[0])
        # Read: the next poll starts from here, or, when more is new than one
        # poll tells, from the last email it told (see PER_POLL). With
        # nothing to tell, at once; else once what it tells is recorded.
        read_to = new[PER_POLL - 1][0] if len(new) > PER_POLL else now
        if new:
            if read is not None:
                store.set_watchlist_config_key(item.id, READ_THROUGH_KEY, read.isoformat(), db_path=db_path)
            store.set_watchlist_config_key(item.id, PENDING_KEY, {
                "through": read_to.isoformat(),
                "keys": [_dedup_key(item.id, m) for _, m in new[:PER_POLL]],
            }, db_path=db_path)
        else:
            store.set_watchlist_config_key(item.id, READ_THROUGH_KEY, read_to.isoformat(), db_path=db_path)
            store.set_watchlist_config_key(item.id, PENDING_KEY, None, db_path=db_path)
        signals: list[Signal] = []
        for arrived, message in new[:PER_POLL]:
            link = mailbox_link(email, thread_id=message.thread_id or None, message_id=message.id)
            signals.append(Signal(
                watchlist_id=item.id,
                source_kind=SOURCE_KIND_MAILBOX,
                source_external_id=message.id,
                captured_at=now.isoformat(),
                published_at=arrived.isoformat(),
                normalized_summary=summary_of(message),
                raw_payload={},
                provenance_url=link if link.startswith("https://") else "https://mail.google.com/",
                severity_hint=AlertSeverity.HIGH,
                dedup_key=_dedup_key(item.id, message),
            ))
        return signals

    def matches_trigger(self, signal: Signal, item: WatchlistItem) -> bool:
        # Matched in poll, where the sender and subject are at hand.
        return True
