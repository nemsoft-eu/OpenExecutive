"""Fixture load / unload / reset must clear the external-monitoring tables.

Before this change the ``watchlist`` and ``external_signals`` tables were
never wiped by the fixture loader. A research scan (or onboarding) would
populate the watchlist with AI/competitor monitors, and those rows then
leaked into every subsequent fixture load and into the restored user state
on unload — the "clean" fixture inherited the previous company's monitors.

Every company swap now wipes them through
``memory.company_tables.wipe_per_company_tables``, which deletes children
before parents from the database's own foreign keys
(``external_signals.watchlist_id`` → ``watchlist.id``) under
``PRAGMA foreign_keys=ON``. That the three swap paths really run it, against a
row in every table, is ``test_company_tables.py``.
"""
from __future__ import annotations

import inspect
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from openexecutive.cli.fixture_loader import _apply_state_from_source
from openexecutive.memory.company_tables import wipe_per_company_tables


@pytest.fixture
def _episodic_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the episodic DB at tmp and create episodic + monitoring schema."""
    from openexecutive.memory import episodic
    from openexecutive.monitoring import store as monitoring_store

    db_path = tmp_path / "episodic.db"
    monkeypatch.setattr(episodic, "DB_PATH", db_path)
    episodic.initialize_db(db_path)
    monitoring_store.initialize_db(db_path)
    return db_path


def test_delete_clears_watchlist_and_signals_in_fk_order(
    _episodic_db: Path,
) -> None:
    """A watchlist row plus a signal that references it must both be wiped.

    Deleting ``external_signals`` before ``watchlist`` keeps the FK satisfied;
    the registry lists ``external_signals`` first, but the wipe orders from the
    foreign keys, so a reordered registry still passes rather than raising an
    IntegrityError.
    """
    from openexecutive.monitoring import store as ms
    from openexecutive.monitoring.models import Signal

    wl_id = ms.insert_watchlist_item(
        slug="tesla-fsd",
        signal_type="rss",
        target="https://example.com/feed",
        db_path=_episodic_db,
    )
    ms.insert_signal(
        Signal(
            watchlist_id=wl_id,
            source_kind="rss",
            source_external_id="evt-1",
            captured_at=datetime.now(UTC).isoformat(),
            normalized_summary="something happened",
            provenance_url="https://example.com/evt-1",
            dedup_key="evt-1",
        ),
        db_path=_episodic_db,
    )

    with sqlite3.connect(str(_episodic_db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0] == 1
        assert (
            conn.execute("SELECT COUNT(*) FROM external_signals").fetchone()[0] == 1
        )

    cleared = wipe_per_company_tables(_episodic_db)

    assert {t: cleared[t] for t in ("external_signals", "watchlist")} == {
        "external_signals": 1, "watchlist": 1,
    }
    with sqlite3.connect(str(_episodic_db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0] == 0
        assert (
            conn.execute("SELECT COUNT(*) FROM external_signals").fetchone()[0] == 0
        )


def test_apply_state_from_source_resets_research_skip_gate() -> None:
    """The shared load/unload path must reset the research skip-if-unchanged gate.

    Fixture load wipes profile/initiatives/watchlist but preserves audit_log;
    without clearing the research run-history rows the seeded
    watchlist_research_scan skips itself (the "no findings on a seeded run"
    bug). Behavioural coverage of the helper lives in
    test_executive_research_scheduler.py; here we guard the wiring on the heavy
    load path (same cheap source-grep guard used for the monitoring tables)."""
    src = inspect.getsource(_apply_state_from_source)
    assert "clear_research_run_history" in src, (
        "_apply_state_from_source no longer calls clear_research_run_history — "
        "a fixture reload within 24h will skip its seeded research scan and "
        "surface zero findings."
    )
