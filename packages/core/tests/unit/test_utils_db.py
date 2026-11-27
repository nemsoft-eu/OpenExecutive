"""How a store opens its database (``utils.db.opener``)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from openexecutive.utils import db


def test_a_block_commits_on_a_clean_exit_and_names_columns(tmp_path: Path) -> None:
    default = tmp_path / "store.db"
    open_db = db.opener(lambda p: p if p is not None else default)

    with open_db() as conn:
        conn.execute("CREATE TABLE t (name TEXT)")
        conn.execute("INSERT INTO t VALUES ('a')")
    with open_db() as conn:
        assert conn.execute("SELECT name FROM t").fetchone()["name"] == "a"


def test_a_block_that_raises_keeps_nothing_and_closes(tmp_path: Path) -> None:
    open_db = db.opener(lambda p: p or tmp_path / "store.db")
    with open_db() as conn:
        conn.execute("CREATE TABLE t (name TEXT)")

    held: list[sqlite3.Connection] = []
    with pytest.raises(RuntimeError), open_db() as conn:
        held.append(conn)
        conn.execute("INSERT INTO t VALUES ('a')")
        raise RuntimeError("boom")
    with pytest.raises(sqlite3.ProgrammingError):
        held[0].execute("SELECT 1")
    with open_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0


def test_the_path_is_read_on_every_call_and_a_given_one_wins(tmp_path: Path) -> None:
    where = {"path": tmp_path / "one.db"}
    open_db = db.opener(lambda p: p if p is not None else where["path"])

    with open_db() as conn:
        conn.execute("CREATE TABLE one (x)")
    where["path"] = tmp_path / "two.db"
    with open_db() as conn:
        assert conn.execute("SELECT name FROM sqlite_master").fetchall() == []
    with open_db(tmp_path / "one.db") as conn:
        assert [r["name"] for r in conn.execute("SELECT name FROM sqlite_master")] == ["one"]


def test_what_a_store_switches_on(tmp_path: Path) -> None:
    resolve = lambda p: p or tmp_path / "store.db"  # noqa: E731

    with db.opener(resolve)() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 0
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    with db.opener(resolve, foreign_keys=True)() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with db.opener(resolve, wal=True)() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    with db.opener(resolve, schema="CREATE TABLE IF NOT EXISTS s (x);")() as conn:
        assert conn.execute("SELECT COUNT(*) FROM s").fetchone()[0] == 0


def test_a_store_defaults_to_the_companys_database_as_it_is_now(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.memory import episodic

    monkeypatch.setattr(episodic, "DB_PATH", tmp_path / "company.db")
    assert db.company_path() == tmp_path / "company.db"
    assert db.company_path(None) == tmp_path / "company.db"
    assert db.company_path(tmp_path / "other.db") == tmp_path / "other.db"


def test_a_failed_set_up_still_closes_the_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Switching WAL on can fail (another writer holds the file); the
    connection must not be left open when it does."""
    closed: list[bool] = []

    class Conn:
        row_factory = None

        def execute(self, sql: str) -> None:
            raise sqlite3.OperationalError("database is locked")

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(sqlite3, "connect", lambda path: Conn())
    open_db = db.opener(lambda p: tmp_path / "x.db", wal=True)
    with pytest.raises(sqlite3.OperationalError), open_db():
        pass
    assert closed == [True]
