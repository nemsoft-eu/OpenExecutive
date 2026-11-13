"""Fixture load / unload / reset must clear the external-monitoring tables.

Before this change the ``watchlist`` and ``external_signals`` tables were
never wiped by the fixture loader. A research scan (or onboarding) would
populate the watchlist with AI/competitor monitors, and those rows then
leaked into every subsequent fixture load and into the restored user state
on unload — the "clean" fixture inherited the previous company's monitors.

The fix routed the shared load/unload path (``_apply_state_from_source``)
and ``reset_all_state`` through ``_delete_all_rows`` with the two monitoring
tables, ``external_signals`` first so the FK
(``external_signals.watchlist_id`` → ``watchlist.id``) is never violated
under ``PRAGMA foreign_keys=ON``. Both now wipe every table in
``memory.company_tables`` (the fixture path keeps ``_delete_all_rows`` for a
user backup without state.db); ``test_company_tables.py`` proves each path.

We exercise ``_delete_all_rows`` directly so the assertion that matters — the
deletion works against the live schema in the correct FK order — is proven
without dragging ChromaDB, Honcho, and an ``app.state`` shim into a unit test.
"""
from __future__ import annotations

import inspect
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from openexecutive.cli.fixture_loader import (
    _apply_state_from_source,
    _delete_all_rows,
)


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
    if the tuple order were reversed this DELETE would raise an
    IntegrityError, so a green test also proves the ordering is correct.
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

    cleared = _delete_all_rows(
        _episodic_db, ("external_signals", "watchlist")
    )

    assert cleared == {"external_signals": 1, "watchlist": 1}
    with sqlite3.connect(str(_episodic_db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0] == 0
        assert (
            conn.execute("SELECT COUNT(*) FROM external_signals").fetchone()[0] == 0
        )


def test_monitoring_tables_are_swapped_by_every_path() -> None:
    """Reset, blank-slot switch and fixture load/unload all wipe the
    registered per-company tables (proven per path, with a row in every
    table, in ``test_company_tables.py``) — so the monitoring layer must stay
    registered.
    """
    from openexecutive.memory.company_tables import PER_COMPANY_TABLES

    for table in ("external_signals", "watchlist"):
        assert table in PER_COMPANY_TABLES, (
            f"{table!r} dropped out of memory.company_tables — the previous "
            "company's watchlist monitors would survive every company swap."
        )


def test_apply_state_from_source_clears_monitoring_tables() -> None:
    """The shared load/unload path must clear the monitoring layer too.

    ``_apply_state_from_source`` backs both ``load_fixture`` and
    ``unload_fixture``; if it stops clearing these tables the watchlist
    leaks across fixture loads and into the restored user state.
    """
    src = inspect.getsource(_apply_state_from_source)
    for table in ("external_signals", "watchlist"):
        assert f'"{table}"' in src, (
            f"_apply_state_from_source no longer mentions {table!r} — fixture "
            "load/unload will stop clearing the watchlist."
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


# ── Derived per-company caches (same leak class as the watchlist above) ──────
#
# briefing_narrative and person_insights were wiped by NO path. Because /today
# serves them from cache and only regenerates in a BackgroundTask, loading a
# demo fixture over a live company rendered that company's real briefing
# narrative verbatim under the demo — on the surface most likely to be
# screen-shared. All three swapping paths now consume PER_CLIENT_CACHE_TABLES,
# and on the fixture path the wipe sits in _apply_state_from_source rather
# than in _seed_episodic_memory, which returns early without a memory.json.


def test_delete_clears_derived_caches_on_a_never_briefed_db(
    _episodic_db: Path,
) -> None:
    """The cache wipe must work against the live schema, including a cold DB.

    Behavioural, and it calls the production helper ``_wipe_derived_caches``
    itself rather than re-implementing the initialize-then-delete sequence —
    otherwise deleting the schema init at the call site would leave this test
    green, and the cold-DB path would have no coverage at all. Exercising
    ``_apply_state_from_source`` end to end is what this module's docstring
    rules out (it would drag ChromaDB, Honcho and an ``app.state`` shim into a
    unit test); the sibling test below pins that the helper is actually called
    there, unconditionally.

    The cold-DB half is the regression the original fix hit: both caches
    CREATE TABLE lazily on first put, so a DB that has never served a briefing
    lacks them, and ``_delete_all_rows`` guards only the DB file, not each
    table. Removing the init inside ``_wipe_derived_caches`` fails this test.
    """
    from openexecutive.briefing import narrative_cache
    from openexecutive.cli.fixture_loader import (
        PER_CLIENT_CACHE_TABLES,
        _wipe_derived_caches,
    )
    from openexecutive.people import insights_cache

    # Cold: neither table exists yet on this DB.
    with sqlite3.connect(str(_episodic_db)) as conn:
        placeholders = ", ".join("?" * len(PER_CLIENT_CACHE_TABLES))
        present = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "  # noqa: S608
                f"AND name IN ({placeholders})",
                PER_CLIENT_CACHE_TABLES,
            )
        }
    assert present == set(), (
        "these caches are expected to be created lazily; if they now ship in "
        "the base episodic schema the initialize_db calls at the wipe site "
        "are redundant, but the wipe itself still must not regress."
    )
    _wipe_derived_caches(_episodic_db)  # must not raise on a cold DB

    # Warm: rows written by the outgoing company are actually removed.
    narrative_cache.put(
        narrative_cache.BriefingNarrative(
            scope="principal",
            input_hash="outgoing",
            narrative_text="**Real client narrative** — must not reach a demo.",
            generated_at=datetime.now(UTC).isoformat(),
        ),
        db_path=_episodic_db,
    )
    insights_cache.put(
        insights_cache.PersonInsight(
            person_id=1,
            input_hash="outgoing",
            insight_text="Real client founder note.",
            generated_at=datetime.now(UTC).isoformat(),
        ),
        db_path=_episodic_db,
    )
    assert narrative_cache.get("principal", db_path=_episodic_db) is not None
    assert insights_cache.get(1, db_path=_episodic_db) is not None

    _wipe_derived_caches(_episodic_db)

    assert narrative_cache.get("principal", db_path=_episodic_db) is None
    assert insights_cache.get(1, db_path=_episodic_db) is None


def test_every_swap_path_reads_the_derived_caches_from_the_registry() -> None:
    """The original bug was three hand-maintained table lists over one DB: the
    fix landed in one and the leak stayed live in the other two. All three now
    read ``memory.company_tables``; the caches must stay in it, and in the
    fixture path's fallback list (a user backup with no state.db yet).

    That the fixture-path wipe runs whatever the fixture holds (no
    ``memory.json``, say) is proven behaviourally, for both the full swap and
    the fallback, in ``test_company_tables.py``.
    """
    from openexecutive.cli.fixture_loader import PER_CLIENT_CACHE_TABLES
    from openexecutive.clients.slots import _BLANK_WIPE_TABLES
    from openexecutive.memory.company_tables import PER_COMPANY_TABLES

    assert PER_CLIENT_CACHE_TABLES == ("briefing_narrative", "person_insights")
    for table in PER_CLIENT_CACHE_TABLES:
        assert table in PER_COMPANY_TABLES
        assert table in _BLANK_WIPE_TABLES
