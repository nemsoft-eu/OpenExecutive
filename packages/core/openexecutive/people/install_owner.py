"""Who owns this install, whatever company it has loaded.

The principal is a People row, and People belongs to the live company: loading
a simulator company or switching to a client slot puts that company's own
principal on the roster, and a factory reset empties it. The person who runs
the install stays the same throughout, so their sign-in address is kept here,
in one instance-level row that a reset, a fixture load, an unload and a client
switch all leave as it is (``memory.company_tables.COMPANY_INDEPENDENT_TABLES``,
``clients.slots._GLOBAL_TABLES``).

It is written when the principal, signed in by their own address, changes the
company's saved data while their own company is live (``chat.principal_writes``).
``chat._caller_is_principal_or_unclaimed`` reads it in two cases only: while a
simulator company or a client is live, and while the roster has no principal.
With their own company live and a principal on it, that principal alone
decides, so handing the principal to someone else takes effect at once.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

TABLE = "install_owner"
_SCHEMA = f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        email TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
"""


def _db_path() -> Path:
    from openexecutive.memory import episodic

    return episodic.DB_PATH


def initialize_db(db_path: Path | None = None) -> None:
    """Create the table (idempotent)."""
    path = db_path or _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(path), timeout=10)) as conn, conn:
        conn.execute(_SCHEMA)


def get(db_path: Path | None = None) -> str | None:
    """The owner's sign-in address, lowercased, or None when none is kept."""
    path = db_path or _db_path()
    if not path.exists():
        return None
    with closing(sqlite3.connect(str(path))) as conn:
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)
        ).fetchone() is None:
            return None
        row = conn.execute(f"SELECT email FROM {TABLE} WHERE id = 1").fetchone()  # noqa: S608
    return str(row[0]) if row and row[0] else None


def remember(email: str, db_path: Path | None = None) -> None:
    """Keep ``email`` as the owner's address."""
    clean = (email or "").strip().lower()
    if not clean:
        return
    path = db_path or _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(path), timeout=10)) as conn, conn:
        conn.execute(_SCHEMA)
        conn.execute(
            f"INSERT INTO {TABLE} (id, email, updated_at) VALUES (1, ?, ?)"  # noqa: S608
            " ON CONFLICT(id) DO UPDATE SET email = excluded.email, updated_at = excluded.updated_at",
            (clean, datetime.now(UTC).isoformat()),
        )


def away_from_own_company() -> bool:
    """Whether the live company is not the owner's own: a simulator company
    or a client slot is loaded. Fails closed: an error reads as away, which
    only ever lets the kept owner in, never anyone else."""
    try:
        from openexecutive.cli.fixture_loader import _fixture_active_sentinel
        from openexecutive.clients.slots import get_active_client
        from openexecutive.config import get_settings

        settings = get_settings()
        return _fixture_active_sentinel(settings).exists() or get_active_client(settings) is not None
    except Exception:
        logger.warning("install_owner: couldn't tell which company is live", exc_info=True)
        return True


def from_own_company_copy() -> str | None:
    """The principal's address in the copy of the owner's own company saved
    before a simulator company or a client was loaded (``_user_backup/
    state.db``), lowercased, or None. Used only while away with no owner
    kept: an install that had a company loaded before the owner was first
    kept still knows who to let switch back."""
    try:
        from openexecutive.cli.fixture_loader import _user_backup_dir
        from openexecutive.config import get_settings

        copy = _user_backup_dir(get_settings()) / "state.db"
        if not copy.exists():
            return None
        with closing(sqlite3.connect(f"file:{copy}?mode=ro", uri=True)) as conn:
            row = conn.execute(
                "SELECT email FROM people WHERE is_principal = 1 AND email IS NOT NULL AND email != ''"
            ).fetchone()
        return str(row[0]).strip().lower() if row else None
    except Exception:
        logger.warning("install_owner: couldn't read the saved copy of the company", exc_info=True)
        return None
