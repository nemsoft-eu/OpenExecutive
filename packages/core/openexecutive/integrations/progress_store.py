"""The ``progress_updates`` table: what other agents reported, and whether it
was relayed.

Each row is one accepted update, filed under the source its token belongs
to. :func:`admit` decides, in one ``BEGIN IMMEDIATE`` transaction, whether an
update is a repeat, whether its source is over the daily cap, and whether it
may be relayed now, and stores it; two concurrent submits therefore cannot
both slip under a cap between the check and the insert. Rows older than
``MAX_AGE_DAYS`` are pruned on insert, and there is no row cap, so the
24-hour repeat window is never cut short.

Unlike the other small tables, a failed write raises
(:class:`StoreUnavailable`): an update that was not stored must not be sent.
Reads never raise and return nothing on failure. The table is company data:
a factory reset, a blank-slot switch and a fixture load clear it.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, NamedTuple

from openexecutive.integrations.progress_guard import CleanUpdate

logger = logging.getLogger(__name__)

TABLE = "progress_updates"
MAX_AGE_DAYS = 90
DUPLICATE_WINDOW = timedelta(hours=24)
RELAY_STATES = ("pending", "sent", "queued")

Outcome = Literal["stored", "duplicate", "over_limit"]


class StoreUnavailable(Exception):
    """The update could not be checked or stored; nothing may be sent."""


class Admission(NamedTuple):
    """What :func:`admit` decided. ``id`` is the new row's, or the first
    row's for a duplicate, and None only when ``outcome`` is
    ``over_limit``. ``relay_claimed`` is true when the row was counted
    against its source's relay cap and may be relayed now."""

    id: int | None
    outcome: Outcome
    relay_claimed: bool


@dataclass(frozen=True)
class ProgressUpdate:
    id: int
    source: str
    topic: str
    status: str
    summary: str
    ask: str | None
    link: str | None
    relay_state: str
    created_at: str


def content_hash(update: CleanUpdate) -> str:
    """sha256 over the fields that make two updates the same report. The
    guard rejects control characters, so the separator cannot occur in a
    field."""
    parts = (update.topic_key, update.status, update.summary, update.ask or "", update.link or "")
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _connect(db_path: Path | None) -> sqlite3.Connection:
    from openexecutive.memory.episodic import _resolve_db_path

    # autocommit, so admit can open its own BEGIN IMMEDIATE transaction
    conn = sqlite3.connect(str(_resolve_db_path(db_path)), timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE} ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  source TEXT NOT NULL,"
        "  topic TEXT NOT NULL,"
        "  topic_key TEXT NOT NULL,"
        "  status TEXT NOT NULL,"
        "  summary TEXT NOT NULL,"
        "  ask TEXT,"
        "  link TEXT,"
        "  content_hash TEXT NOT NULL,"
        "  idempotency_key TEXT,"
        "  relay_claimed INTEGER NOT NULL DEFAULT 0,"
        "  relay_state TEXT NOT NULL DEFAULT 'pending',"
        "  relayed_at TEXT,"
        "  created_at TEXT NOT NULL"
        ")"
    )
    conn.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{TABLE}_key ON {TABLE}(source, idempotency_key)"
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_source ON {TABLE}(source, created_at)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_hash ON {TABLE}(source, content_hash)")
    return conn


def _first_duplicate(
    conn: sqlite3.Connection, source: str, key: str | None, digest: str, since: str
) -> int | None:
    if key is not None:
        row = conn.execute(
            f"SELECT id FROM {TABLE} WHERE source = ? AND idempotency_key = ?",  # noqa: S608 — constant table name
            (source, key),
        ).fetchone()
        if row is not None:
            return int(row["id"])
    row = conn.execute(
        f"SELECT id FROM {TABLE} WHERE source = ? AND content_hash = ? AND created_at >= ? "  # noqa: S608 — constant table name
        "ORDER BY id LIMIT 1",
        (source, digest, since),
    ).fetchone()
    return int(row["id"]) if row is not None else None


def _count(conn: sqlite3.Connection, source: str, since: str, *, claimed_only: bool) -> int:
    claimed = " AND relay_claimed = 1" if claimed_only else ""
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM {TABLE} WHERE source = ? AND created_at >= ?{claimed}",  # noqa: S608 — constant table name and fixed clause
        (source, since),
    ).fetchone()
    return int(row["n"])


def admit(
    source: str,
    update: CleanUpdate,
    *,
    max_per_day: int | None = None,
    relay_max_per_hour: int | None = None,
    relay_max_per_day: int | None = None,
    now: datetime | None = None,
    db_path: Path | None = None,
) -> Admission:
    """Check and store one update from ``source``, atomically: a repeat
    (same idempotency key, or the same content within 24 hours) returns the
    first row's id; a source at its daily cap stores nothing; otherwise the
    row is stored, claiming a relay when the source is under both relay
    caps. The caps default to the ``PROGRESS_*`` settings. Raises
    :class:`StoreUnavailable` when the database can't be used."""
    if max_per_day is None or relay_max_per_hour is None or relay_max_per_day is None:
        from openexecutive.config import get_settings

        settings = get_settings()
        if max_per_day is None:
            max_per_day = settings.progress_max_per_source_per_day
        if relay_max_per_hour is None:
            relay_max_per_hour = settings.progress_relay_max_per_source_per_hour
        if relay_max_per_day is None:
            relay_max_per_day = settings.progress_relay_max_per_source_per_day
    when = now or datetime.now(UTC)
    digest = content_hash(update)
    day_ago = (when - timedelta(days=1)).isoformat()
    try:
        conn = _connect(db_path)
    except (sqlite3.Error, OSError) as exc:
        raise StoreUnavailable(str(exc)) from exc
    try:
        try:
            conn.execute("BEGIN IMMEDIATE")
            first = _first_duplicate(
                conn, source, update.idempotency_key, digest, (when - DUPLICATE_WINDOW).isoformat()
            )
            if first is not None:
                conn.execute("ROLLBACK")
                return Admission(first, "duplicate", False)
            if _count(conn, source, day_ago, claimed_only=False) >= max_per_day:
                conn.execute("ROLLBACK")
                return Admission(None, "over_limit", False)
            claimed = (
                _count(conn, source, (when - timedelta(hours=1)).isoformat(), claimed_only=True)
                < relay_max_per_hour
                and _count(conn, source, day_ago, claimed_only=True) < relay_max_per_day
            )
            cur = conn.execute(
                f"INSERT INTO {TABLE} (source, topic, topic_key, status, summary, ask, link, "  # noqa: S608 — constant table name
                "content_hash, idempotency_key, relay_claimed, relay_state, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                (
                    source,
                    update.topic,
                    update.topic_key,
                    update.status,
                    update.summary,
                    update.ask,
                    update.link,
                    digest,
                    update.idempotency_key,
                    1 if claimed else 0,
                    when.isoformat(),
                ),
            )
            cutoff = (when - timedelta(days=MAX_AGE_DAYS)).isoformat()
            conn.execute(f"DELETE FROM {TABLE} WHERE created_at < ?", (cutoff,))  # noqa: S608 — constant table name
            conn.execute("COMMIT")
        except sqlite3.IntegrityError:
            # the same key landed between the lookup and the insert (another
            # writer outside this transaction): the first row wins
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            row = conn.execute(
                f"SELECT id FROM {TABLE} WHERE source = ? AND idempotency_key = ?",  # noqa: S608 — constant table name
                (source, update.idempotency_key),
            ).fetchone()
            if row is None:
                raise
            return Admission(int(row["id"]), "duplicate", False)
        if cur.lastrowid is None:
            raise StoreUnavailable("the insert returned no row id")
        return Admission(int(cur.lastrowid), "stored", claimed)
    except (sqlite3.Error, OSError) as exc:
        # closing the connection below rolls back whatever was left open
        raise StoreUnavailable(str(exc)) from exc
    finally:
        conn.close()


def still_pending(update_id: int, source: str, digest: str, *, db_path: Path | None = None) -> bool:
    """Whether the row ``admit`` stored for this update is still here and not
    yet relayed. Every company swap (fixture load or unload, factory reset,
    slot switch) wipes or replaces this table, so False means the live
    company is no longer the one the update was posted to; matching the
    source and content hash too keeps another database's row with the same
    id from passing. False on any error, so a failed check stops a relay."""
    try:
        conn = _connect(db_path)
        try:
            row = conn.execute(
                f"SELECT 1 FROM {TABLE} WHERE id = ? AND source = ? AND content_hash = ? "  # noqa: S608 — constant table name
                "AND relay_state = 'pending'",
                (update_id, source, digest),
            ).fetchone()
        finally:
            conn.close()
    except Exception:
        logger.warning("progress_store: couldn't check update #%s", update_id, exc_info=True)
        return False
    return row is not None


def set_relay_state(
    update_id: int, state: str, *, now: datetime | None = None, db_path: Path | None = None
) -> None:
    """Record how the relay of ``update_id`` ended. Logs and returns when
    the write fails: the update was already sent, so there is nothing to
    undo."""
    if state not in RELAY_STATES:
        raise ValueError(f"unknown relay state {state!r}")
    relayed_at = None if state == "pending" else (now or datetime.now(UTC)).isoformat()
    try:
        conn = _connect(db_path)
        try:
            conn.execute(
                f"UPDATE {TABLE} SET relay_state = ?, relayed_at = ? WHERE id = ?",  # noqa: S608 — constant table name
                (state, relayed_at, update_id),
            )
        finally:
            conn.close()
    except Exception:
        logger.warning("progress_store: couldn't record relay state %s for #%s", state, update_id, exc_info=True)


def latest_per_topic(
    limit: int = 8,
    max_age_days: int = 30,
    *,
    now: datetime | None = None,
    db_path: Path | None = None,
) -> list[ProgressUpdate]:
    """The newest update for each source and topic, newest first, at most
    ``limit`` of them and none older than ``max_age_days``. Empty when there
    are none or the read fails."""
    since = ((now or datetime.now(UTC)) - timedelta(days=max_age_days)).isoformat()
    try:
        conn = _connect(db_path)
        try:
            rows = conn.execute(
                f"SELECT id, source, topic, status, summary, ask, link, relay_state, created_at "  # noqa: S608 — constant table name
                f"FROM {TABLE} WHERE id IN ("
                f"SELECT MAX(id) FROM {TABLE} WHERE created_at >= ? GROUP BY source, topic_key"
                ") ORDER BY created_at DESC, id DESC LIMIT ?",
                (since, max(0, limit)),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        logger.warning("progress_store: couldn't read the latest updates", exc_info=True)
        return []
    return [
        ProgressUpdate(
            id=int(r["id"]),
            source=str(r["source"]),
            topic=str(r["topic"]),
            status=str(r["status"]),
            summary=str(r["summary"]),
            ask=r["ask"],
            link=r["link"],
            relay_state=str(r["relay_state"]),
            created_at=str(r["created_at"]),
        )
        for r in rows
    ]
