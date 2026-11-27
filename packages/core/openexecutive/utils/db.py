"""How a store opens its SQLite database.

Dozens of stores each carried their own copy of the same context manager:
connect, name the columns, yield, commit, close. The copies drifted in ways
that meant nothing (a ``busy_timeout`` pragma that repeats the connect
timeout) and in ways that did (foreign keys on or off), and nothing said
which was which.

``opener`` is that context manager, once. A store keeps only what is its
own: where its database is (``resolve``, read on every call so a test that
points the store somewhere else is honoured) and what it needs switched on.

    _get_conn = db.opener(_resolve_db_path, wal=True)

    with _get_conn() as conn:
        conn.execute(...)
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Protocol


def company_path(db_path: Path | None = None) -> Path:
    """Where a store's tables live when it isn't told: the company's own
    database (``memory.episodic.DB_PATH``), read on every call so a test or
    a deployment that moves it is honoured."""
    if db_path is not None:
        return db_path
    from openexecutive.memory import episodic

    return Path(episodic.DB_PATH)


class Opener(Protocol):
    def __call__(self, db_path: Path | None = None) -> AbstractContextManager[sqlite3.Connection]: ...


def opener(
    resolve: Callable[[Path | None], Path],
    *,
    wal: bool = False,
    foreign_keys: bool = False,
    schema: str | None = None,
) -> Opener:
    """A store's ``with ... as conn``: rows by column name, committed when the
    block ends cleanly, always closed.

    ``wal``: the database is written from more than one process, so readers
    must not block the writer. ``foreign_keys``: the store's tables cascade.
    ``schema``: the store's own ``CREATE ... IF NOT EXISTS`` script, run
    before the block so a first use needs no separate set-up call.
    """

    @contextmanager
    def open_db(db_path: Path | None = None) -> Generator[sqlite3.Connection, None, None]:
        # sqlite3 waits 5 seconds on another writer by default and sets the
        # connection's busy timeout to match, so neither is repeated here.
        conn = sqlite3.connect(str(resolve(db_path)))
        try:
            conn.row_factory = sqlite3.Row
            if wal:
                conn.execute("PRAGMA journal_mode=WAL")
            if foreign_keys:
                conn.execute("PRAGMA foreign_keys = ON")
            if schema is not None:
                conn.executescript(schema)
            yield conn
            conn.commit()
        finally:
            conn.close()

    return open_db


__all__ = ["Opener", "company_path", "opener"]
