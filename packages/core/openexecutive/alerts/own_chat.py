"""Whose a card is when a held step raised it from one person's own chat.

A card that waits for a tap (Approve, a question) and came out of someone's
own chat is that person's: it and its decision show to them and to the person
it went to for the OK, and to nobody else on the team. Every other card is
the team's, as before: one from a department's own run, a watch, company
mail, a workflow.

"Their own chat" is a saved conversation (``memory.session_store``): its
caller, or the principal for one saved before callers were recorded, the same
rule the Activity feed uses (``api.routes.today._Owners.of_session``). A step
with no saved conversation behind it (a scheduled run, inbound mail) is not
one. Whose chat it was is written down when the step is held
(``note_decision``), so deleting the conversation later changes nothing.

Only a decision's own card counts (the alert the ledger files for it:
``DECISION_ALERT_SOURCE`` with the decision's external id). A
``decision_instance:`` tag on any other alert is just text.
"""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The readers of a card nobody could be worked out for: only the person it
# is routed to keeps it, so a failed lookup never shows it to the team.
_NOBODY: frozenset[int] = frozenset()

_TABLE = """
    CREATE TABLE IF NOT EXISTS decision_chat_owners (
        decision_id INTEGER PRIMARY KEY,
        person_id INTEGER NOT NULL
    )
"""


def _path(db_path: Path | None) -> Path:
    from openexecutive.memory import episodic

    return db_path or episodic.DB_PATH


def _principal() -> int | None:
    from openexecutive.people import registry as people_registry

    principal = people_registry.get_principal()
    return principal.id if principal is not None else None


def _chat_owner(session_id: str | None, db_path: Path | None = None) -> int | None:
    """The person whose saved chat ``session_id`` is, or None for no chat."""
    if not session_id:
        return None
    from openexecutive.memory import session_store

    found, caller = session_store.get_session_owner(session_id, db_path=_path(db_path))
    if not found:
        return None
    return caller if caller is not None else _principal()


def note_decision(decision_id: int, session_id: str | None, db_path: Path | None = None) -> None:
    """Write down whose chat held step ``decision_id`` came from, as it is
    held. Never raises: a decision with nothing written is read from its
    conversation instead (``decision_readers``). A failed write is tried
    once more, since nothing else keeps the card with its person after the
    conversation is deleted."""
    for attempt in (1, 2):
        try:
            owner = _chat_owner(session_id, db_path)
            break
        except Exception:
            if attempt == 2:
                logger.error("own_chat: couldn't read whose chat held step %s came from", decision_id, exc_info=True)
                return
    if owner is not None:
        keep_with(decision_id, owner, db_path)


def keep_with(decision_id: int, person_id: int, db_path: Path | None = None) -> bool:
    """Keep held step ``decision_id`` with ``person_id`` (and its approver)
    alone, whatever conversation it came from: a card only its own person
    answers, wherever they asked. The first owner written stays. Never
    raises; a failed write is tried once more, and False says it failed."""
    for attempt in (1, 2):
        try:
            with closing(sqlite3.connect(_path(db_path), timeout=10)) as conn, conn:
                conn.execute(_TABLE)
                conn.execute(
                    "INSERT OR IGNORE INTO decision_chat_owners (decision_id, person_id) VALUES (?, ?)",
                    (decision_id, person_id),
                )
            return True
        except Exception:
            if attempt == 2:
                logger.error(
                    "own_chat: couldn't note whose held step %s is", decision_id, exc_info=True
                )
    return False


def _noted(decision_ids: list[int], db_path: Path | None = None) -> dict[int, int]:
    path = _path(db_path)
    if not decision_ids or not path.exists():
        return {}
    marks = ",".join("?" for _ in decision_ids)
    with closing(sqlite3.connect(path)) as conn:
        # A read: it never makes the table, which would take the write lock.
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='decision_chat_owners'"
        ).fetchone() is None:
            return {}
        rows = conn.execute(
            f"SELECT decision_id, person_id FROM decision_chat_owners WHERE decision_id IN ({marks})",  # noqa: S608
            decision_ids,
        ).fetchall()
    return {int(d): int(p) for d, p in rows}


def decision_readers(instances: Iterable[Any]) -> dict[int, frozenset[int]]:
    """Decision id → the only people who may see that held step. A decision
    that is not in the result is the team's. Never raises: one it can't work
    out is nobody's (its card stays with the person it is routed to)."""
    instances = [i for i in instances if i is not None]
    if not instances:
        return {}
    try:
        noted = _noted([int(i.id) for i in instances])
    except Exception:
        logger.warning("own_chat: couldn't read whose chats held steps came from", exc_info=True)
        return {int(i.id): _NOBODY for i in instances}
    out: dict[int, frozenset[int]] = {}
    for inst in instances:
        try:
            owner = noted.get(int(inst.id))
            if owner is None:
                owner = _chat_owner(getattr(inst, "originating_session_id", None))
        except Exception:
            logger.warning("own_chat: couldn't read a held step's chat", exc_info=True)
            out[int(inst.id)] = _NOBODY
            continue
        if owner is None:
            continue
        out[int(inst.id)] = frozenset(
            p for p in (owner, getattr(inst, "approver_person_id", None)) if p is not None
        )
    return out


def decision_shown_to(instance: Any, reader: int | None) -> bool:
    """Whether ``reader`` (a People id; None is nobody in particular) may see
    or decide the held step ``instance``."""
    only = decision_readers([instance]).get(int(instance.id))
    return only is None or (reader is not None and reader in only)


def _decision_id(alert: Any) -> int | None:
    """The decision ``alert`` is the card of, or None
    (``decision_ledger.instance_id_of``)."""
    from openexecutive.memory.decision_ledger import instance_id_of

    return instance_id_of(alert)


def readers(alerts: Iterable[Any]) -> dict[int, frozenset[int]]:
    """Alert id → the only people who may see that card. An alert that is
    not in the result is the team's. Never raises: a card it can't work out
    goes to the person it is routed to alone."""
    cards: dict[int, int] = {}
    for alert in alerts:
        decision = _decision_id(alert)
        if decision is not None and getattr(alert, "id", None) is not None:
            cards[int(alert.id)] = decision
    if not cards:
        return {}
    try:
        from openexecutive.memory.decision_ledger import get_decision_instances

        instances = get_decision_instances(sorted(set(cards.values())))
    except Exception:
        logger.warning("own_chat: couldn't read the cards' decisions", exc_info=True)
        return dict.fromkeys(cards, _NOBODY)
    by_decision = decision_readers(instances.values())
    return {alert: by_decision[decision] for alert, decision in cards.items() if decision in by_decision}


def shown_to(alert: Any, reader: int | None, allowed: dict[int, frozenset[int]]) -> bool:
    """Whether ``reader`` (a People id; None is nobody in particular, a
    surface the whole team reads) may see ``alert``."""
    only = allowed.get(getattr(alert, "id", None) or 0)
    if only is None:
        return True
    if reader is None:
        return False
    return reader in only or getattr(alert, "routed_to_person_id", None) == reader


def alert_shown_to(alert: Any, reader: int | None) -> bool:
    """``shown_to`` for one card on its own."""
    return shown_to(alert, reader, readers([alert]))


def keep_for(alerts: list[Any], reader: int | None) -> list[Any]:
    """``alerts`` without the cards that are someone else's."""
    allowed = readers(alerts)
    if not allowed:
        return alerts
    return [a for a in alerts if shown_to(a, reader, allowed)]
