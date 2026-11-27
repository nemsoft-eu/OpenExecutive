"""Email watches: the email a person can see, on the watch list.

"Tell me every time a sign-up email comes in", "ping me if Pat replies": a
watch of kind ``mailbox`` (``SOURCE_KIND_MAILBOX``) is set up by one person
(``mailbox_of`` in its config, their People id) and tells them about every
email that matches, until they turn it off. It reads two inboxes:

- the Executive's own: every email that reaches the Executive is checked
  against the watches as it is handled (``note_executive_mail``, from the
  mail handling every inbound email goes through), and the watch's next poll
  tells its owner. Only email its owner could already see counts
  (``seen_by``): the principal's, or a teammate's own mail with the
  Executive, never anyone else's. A sender nothing proved tells the watch but
  joins no card (its thread is left off);
- the person's own, through Act as me's mailbox client
  (``delegation.gmail.gmail_for``, Gmail or Outlook), whenever it is
  connected for them.

Its trigger says what matches:

- ``from``: sender addresses; an email from one of them matches;
- ``keywords``: words; an email whose subject or sender names one matches.

With both, an email must meet both. With neither, it matches nothing (a
watch on every email is what the inbox itself is).

"If nothing comes by Wednesday, ping me": ``quiet_by`` (a date) makes the
poll ping its owner that morning when nothing has matched since it was set
(``quiet_ping_due``), once, with a choice to ask the sender or wait a day.

Private, like the mail it reads: only its owner sees it on the watch list
and in the Executive's tools (``shown_to``), its cards go to them and no one
else, whoever the watch was set to tell (``alerts.models.PERSONAL_SOURCES``,
``monitoring.pipeline``), and its audit rows are private to them
(``audit_privacy``). Privacy hides it from everyone else; it never stops it.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from openexecutive.monitoring.models import SOURCE_KIND_MAILBOX, WatchlistItem

logger = logging.getLogger(__name__)

OWNER_KEY = "mailbox_of"
FROM_KEY = "from"
KEYWORDS_KEY = "keywords"
# ``config_json`` key: when the last read that worked began
# (``sources.mailbox``); the next read starts a little before it.
READ_THROUGH_KEY = "read_through"
# ``config_json`` key: when it was last turned back on. Nothing that came
# before is read, as nothing from before the watch was set is.
READ_FROM_KEY = "read_from"
# ``config_json`` keys: the day to ping on when nothing has matched
# (``YYYY-MM-DD``), and when that was set (nothing before it counts). Each
# day pings once: its ping's dedup key is recorded (``sources.mailbox``).
QUIET_BY_KEY = "quiet_by"
QUIET_SINCE_KEY = "quiet_since"
# The ping goes out the morning of that day, the company's time.
QUIET_PING_AT = time(9, 0)
# How far ahead a "nothing by" day may be.
QUIET_MAX_DAYS = 90
MAX_SENDERS = 10
MAX_KEYWORDS = 10
_KEYWORD_MAX_CHARS = 60
# The longest watch name (``monitoring.validation.WATCHLIST_SLUG_RE``).
_SLUG_MAX = 61
_ADDRESS_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_NON_WORD_RE = re.compile(r"[\W_]+")


def is_mailbox(item: WatchlistItem) -> bool:
    return item.signal_type == SOURCE_KIND_MAILBOX


def owner_id(item: WatchlistItem) -> int | None:
    """The People id whose inbox this watch reads, or None (not an email
    watch, or a row with no owner, which reads nothing)."""
    if not is_mailbox(item):
        return None
    raw = (item.config_json or {}).get(OWNER_KEY)
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


def owner_config(config: dict[str, Any], person_id: int) -> dict[str, Any]:
    """``config`` for a new email watch set up by ``person_id``. Matches are
    the person's own words, so the company-relevance read is off for it."""
    return {**config, OWNER_KEY: int(person_id), "enrich": False}


def clean_quiet_by(raw: object, *, today: date) -> date | None | str:
    """The "nothing by" day as given (``YYYY-MM-DD``): the date, None for
    none (empty or null), or the reason it can't be used."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        day = date.fromisoformat(str(raw).strip()[:10])
    except ValueError:
        return "quiet_by must be a date, YYYY-MM-DD"
    if day < today:
        return "quiet_by is a day that has passed; give today or a later day"
    if day > today + timedelta(days=QUIET_MAX_DAYS):
        return f"quiet_by is more than {QUIET_MAX_DAYS} days away"
    return day


def quiet_config(day: date | None, now: datetime) -> dict[str, Any]:
    """The ``config_json`` keys for a "nothing by" ``day`` set at ``now``
    (None clears it): matches count from now."""
    if day is None:
        return {QUIET_BY_KEY: None, QUIET_SINCE_KEY: None}
    return {QUIET_BY_KEY: day.isoformat(), QUIET_SINCE_KEY: now.isoformat()}


def quiet_by(item: WatchlistItem) -> date | None:
    raw = (item.config_json or {}).get(QUIET_BY_KEY)
    try:
        return date.fromisoformat(str(raw)) if raw else None
    except ValueError:
        return None


def _parse(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value)) if value else None
    except ValueError:
        return None
    if parsed is None:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def quiet_ping_due(item: WatchlistItem, now: datetime, zone: Any, *, db_path: Path | None = None) -> bool:
    """Whether to ping the owner that nothing came: its "nothing by" day's
    morning (``zone``'s time) has come (the next morning for a day set after
    its own morning, so "nothing by today" said at 2 PM waits for the night),
    no later than the day after that (a scan that was down, or a watch
    turned off over the day, never pings weeks late), and nothing has
    matched since the day was set: no email of it recorded, told or not.
    (Whether it pinged for that day already is the caller's: the ping's
    dedup key.)"""
    from openexecutive.monitoring import store

    day = quiet_by(item)
    if day is None or item.id is None:
        return False
    since = _parse((item.config_json or {}).get(QUIET_SINCE_KEY)) or _parse(item.created_at)
    due = datetime.combine(day, QUIET_PING_AT, tzinfo=zone)
    if since is not None and since >= due:
        due = datetime.combine(since.astimezone(zone).date() + timedelta(days=1), QUIET_PING_AT, tzinfo=zone)
    local = now.astimezone(zone)
    if local < due or local.date() > due.date() + timedelta(days=1):
        return False
    if since is None:
        return True
    return not store.signals_since(
        item.id, since.astimezone(UTC).isoformat(), excluding=f"mailbox:{item.id}:quiet:", db_path=db_path,
    )


def who_from(item: WatchlistItem) -> str:
    """Whom the watch waits on, for the ping's words: its name without
    "'s email" ("Robin Hale's email" -> "Robin Hale"), else empty."""
    name = str((item.config_json or {}).get("display_label") or item.target or "").strip()
    for ending in ("\u2019s email", "'s email", "\u2019s emails", "'s emails"):
        if name.lower().endswith(ending):
            return name[: -len(ending)].strip()
    return ""


def seen_by(person_id: int, *, private: bool, mail_of: int | None, principal_id: int | None) -> bool:
    """Whether the person ``person_id`` could already see an email that
    reached the Executive, as everywhere else in the app (the audit log, the
    chats): a teammate's own (``mail_of``, theirs, proven) only that teammate;
    every other email, private to the principal (``private``) or not, only
    the principal, who reads the Executive's inbox."""
    if mail_of is not None and not private:
        return person_id == mail_of
    return principal_id is not None and person_id == principal_id


async def note_executive_mail(
    *, message_id: str, thread_id: str, from_addr: str, from_name: str, subject: str,
    private: bool, mail_of: int | None, proven: Callable[[], Awaitable[bool | None]] | None = None,
    received_at: datetime | None = None, header_id: Callable[[], Awaitable[str]] | None = None,
) -> int:
    """Check one email the Executive got against every email watch and keep
    it for each watch it matches whose owner could see it (``seen_by``): the
    watch's next poll tells them. ``proven`` says whether its sender is who
    it says, asked only when a watch matches: an email nothing proved is told
    without its thread, so a forged one never joins a card. ``header_id``
    reads its Message-ID, also only on a match: the same email in the
    watcher's own inbox is then told once (``sources.mailbox``). Returns how many
    watches it matched. Never raises: the email is handled whatever happens
    here."""
    try:
        kept = _watches_for(
            from_addr=from_addr, from_name=from_name, subject=subject, private=private, mail_of=mail_of,
        )
        if not kept or not message_id:
            return 0
        sure = False
        if proven is not None and thread_id:
            try:
                sure = await proven() is True
            except Exception:  # noqa: BLE001 - unproven is the safe reading
                logger.warning("email watch: couldn't tell who sent message=%s", message_id, exc_info=True)
        rfc_id = ""
        if header_id is not None:
            try:
                rfc_id = await header_id()
            except Exception:  # noqa: BLE001 - without it, only a second copy is the risk
                logger.warning("email watch: couldn't read the Message-ID of message=%s", message_id, exc_info=True)
        _keep(
            kept, message_id=message_id, thread_id=thread_id if sure else "", from_addr=from_addr,
            from_name=from_name, subject=subject, received_at=received_at, rfc_id=rfc_id,
        )
        return len(kept)
    except Exception:  # noqa: BLE001 - a watch missing one email is the worst case
        logger.warning("email watch: couldn't check an email against the watches", exc_info=True)
        return 0


def _watches_for(
    *, from_addr: str, from_name: str, subject: str, private: bool, mail_of: int | None,
) -> list[WatchlistItem]:
    """The enabled email watches this email matches whose owner could see it."""
    from openexecutive.monitoring import store
    from openexecutive.people.store import find_principal_person

    if not from_addr:
        return []
    watches = [
        w for w in store.list_watchlist(enabled_only=True, signal_type=SOURCE_KIND_MAILBOX, every_watch=True)
        if w.id is not None and owner_id(w) is not None
    ]
    if not watches:
        return []
    sender = from_addr.strip().lower()
    principal = find_principal_person()
    principal_id = principal.id if principal is not None else None
    return [
        w for w in watches
        if matches(w.trigger_json, from_addr=sender, from_name=from_name, subject=subject)
        and seen_by(owner_id(w) or 0, private=private, mail_of=mail_of, principal_id=principal_id)
    ]


def _keep(
    watches: list[WatchlistItem], *, message_id: str, thread_id: str, from_addr: str, from_name: str,
    subject: str, received_at: datetime | None, rfc_id: str = "",
) -> None:
    from openexecutive.monitoring import store
    from openexecutive.monitoring.pipeline import scan_soon

    at = (received_at or datetime.now(UTC)).isoformat()
    for watch in watches:
        assert watch.id is not None
        store.add_executive_mail(
            watch.id, message_id=message_id, thread_id=thread_id, from_addr=from_addr.strip().lower(),
            from_name=from_name, subject=subject, received_at=at, rfc_id=rfc_id,
        )
        store.poll_soon(watch.id)
    scan_soon()


def shown_to(item: WatchlistItem, person_id: int | None) -> bool:
    """Whether the person ``person_id`` (None: nobody in particular, a
    surface everyone reads) may see this watch. Every watch but an email
    watch is the company's; an email watch is its owner's alone."""
    if not is_mailbox(item):
        return True
    return person_id is not None and owner_id(item) == person_id


def own_slug(slug: str) -> str:
    """The name a new email watch is stored under: the name asked for plus
    a random ending, so no one else's watch name ever runs into it, and
    what someone else is watching their inbox for can't be found by trying
    names."""
    import secrets

    suffix = f"-{secrets.token_hex(3)}"
    return f"{slug[: _SLUG_MAX - len(suffix)]}{suffix}"


def audit_privacy(item: WatchlistItem) -> dict[str, Any]:
    """The ``audit.log_event`` keywords for a row about this watch: an email
    watch's rows name a sender and a subject, so they are its owner's
    alone; every other watch's rows are the company's (no keywords)."""
    if not is_mailbox(item):
        return {}
    return {"private": True, "private_to_person": owner_id(item)}


def _words(values: object, *, cap: int, each: int) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for v in values:
        word = " ".join(v.split())[:each] if isinstance(v, str) else ""
        if word and word.lower() not in {o.lower() for o in out}:
            out.append(word)
    return out[:cap]


def clean_trigger(raw: object) -> dict[str, Any]:
    """The trigger as stored: real addresses, lowercased, and plain words."""
    if not isinstance(raw, dict):
        return {}
    senders = [a.lower() for a in _words(raw.get(FROM_KEY), cap=MAX_SENDERS, each=254) if _ADDRESS_RE.match(a)]
    keywords = _words(raw.get(KEYWORDS_KEY), cap=MAX_KEYWORDS, each=_KEYWORD_MAX_CHARS)
    out: dict[str, Any] = {}
    if senders:
        out[FROM_KEY] = senders
    if keywords:
        out[KEYWORDS_KEY] = keywords
    return out


def matches(trigger: dict[str, Any], *, from_addr: str, from_name: str, subject: str) -> bool:
    """Whether an email meets the watch's trigger (see the module doc)."""
    senders = [str(a).lower() for a in trigger.get(FROM_KEY) or []]
    keywords = [str(k).lower() for k in trigger.get(KEYWORDS_KEY) or [] if str(k).strip()]
    if not senders and not keywords:
        return False
    if senders and (from_addr or "").strip().lower() not in senders:
        return False
    if keywords:
        # Whole words only, punctuation as a space: "ai" is not in "gmail",
        # "sign up" is in "Sign-up:", and "stripe" is in "billing@stripe.com".
        haystack = f" {_plain_words(' '.join((subject or '', from_name or '', from_addr or '')))} "
        if not any(f" {k} " in haystack for k in map(_plain_words, keywords) if k):
            return False
    return True


def _plain_words(text: str) -> str:
    """``text`` lowercased, with every run of non-word characters one space."""
    return " ".join(_NON_WORD_RE.split(text.lower())).strip()


__all__ = [
    "QUIET_BY_KEY",
    "QUIET_SINCE_KEY",
    "clean_quiet_by",
    "note_executive_mail",
    "quiet_by",
    "quiet_config",
    "quiet_ping_due",
    "seen_by",
    "who_from",
    "FROM_KEY",
    "KEYWORDS_KEY",
    "OWNER_KEY",
    "READ_FROM_KEY",
    "READ_THROUGH_KEY",
    "audit_privacy",
    "clean_trigger",
    "own_slug",
    "is_mailbox",
    "matches",
    "owner_config",
    "owner_id",
    "shown_to",
]
