"""A factory reset empties every table in the company DB except the
company-independent ones (``memory.company_tables``) and what the caller
keeps, so a table added later is wiped without anyone listing it."""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from openexecutive.cli import fixture_loader
from openexecutive.departments import registry as dept_registry
from openexecutive.departments import store as dept_store
from openexecutive.knowledge import review_store as rs_mod
from openexecutive.knowledge.store import ChromaDBStore
from openexecutive.memory import episodic, honcho_client
from openexecutive.memory import workspace_settings as ws
from openexecutive.people import registry as people_registry
from openexecutive.people import store as people_store


@pytest.fixture()
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "company.db"
    for mod in (episodic, dept_store, people_store, rs_mod):
        monkeypatch.setattr(mod, "DB_PATH", path)
    monkeypatch.setattr(ws, "_configured_timezone", lambda: ws.ZoneInfo("UTC"))
    monkeypatch.setattr("openexecutive.audit.log_event", lambda *a, **k: None)

    async def _noop(workspace_id: str | None = None) -> None:
        return None

    monkeypatch.setattr(honcho_client, "delete_workspace_and_reset_client", _noop)
    episodic.initialize_db(path)
    dept_store.initialize_db(path)
    people_store.initialize_db(path)
    dept_store.seed_default_departments(db_path=path)
    people_store.upsert_person(full_name="Pat Owner", is_principal=True)
    with sqlite3.connect(str(path)) as conn:
        # A table no reset code has ever heard of.
        conn.execute("CREATE TABLE brand_new_feature (note TEXT)")
        conn.execute("INSERT INTO brand_new_feature VALUES ('old company')")
        conn.execute("INSERT OR IGNORE INTO app_migrations (name, applied_at) VALUES ('kept-marker', 'now')")
    dept_registry.invalidate()
    people_registry.invalidate()
    yield path
    dept_registry.invalidate()
    people_registry.invalidate()


def _settings(tmp_path: Path) -> Any:
    profile_path = tmp_path / "company" / "profile.yaml"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text("name: Old Co\n")
    return type("S", (), {
        "vector_store_path": tmp_path / "chroma",
        "company_profile_path": profile_path,
        "honcho_workspace_id": "openexec",
    })()


def _reset(tmp_path: Path, keep: frozenset[str] = frozenset()) -> dict[str, Any]:
    with patch.object(ChromaDBStore, "delete_company_docs", lambda self: None), \
         patch.object(ChromaDBStore, "delete_documents", lambda self, **kw: None):
        return asyncio.run(fixture_loader.reset_all_state(_settings(tmp_path), keep=keep))


def _rows(db: Path, table: str) -> int:
    with sqlite3.connect(str(db)) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])  # noqa: S608


def test_reset_empties_every_table_but_the_bookkeeping(db: Path, tmp_path: Path) -> None:
    attached = tmp_path / "company" / "conversation_files" / "s1" / "notes.txt"
    attached.parent.mkdir(parents=True)
    attached.write_text("old company")
    result = _reset(tmp_path)
    assert not attached.exists()
    assert _rows(db, "brand_new_feature") == 0
    assert result["cleared"]["brand_new_feature"] == 1
    # Only the principal is written back; everyone else is gone.
    assert _rows(db, "people") == 1
    assert _rows(db, "departments") == 0
    assert _rows(db, "app_migrations") >= 1


def test_reset_keeps_the_principal_and_their_addresses(db: Path, tmp_path: Path) -> None:
    owner_id = people_store.find_principal_person().id
    people_store.upsert_person(
        full_name="Pat Owner", role="CEO", is_principal=True, email="pat@example.com", person_id=owner_id,
    )
    people_store.add_person_email(owner_id, "pat.home@example.com")
    people_store.upsert_person(full_name="Sam Teammate", email="sam@example.com")
    _reset(tmp_path)
    people_registry.invalidate()
    people = people_store.list_people()
    assert [p.full_name for p in people] == ["Pat Owner"]
    owner = people[0]
    assert owner.is_principal is True
    assert owner.email == "pat@example.com"
    assert owner.role == "CEO"
    assert owner.email_aliases == ["pat.home@example.com"]


def test_reset_with_no_principal_leaves_people_empty(db: Path, tmp_path: Path) -> None:
    with sqlite3.connect(str(db)) as conn:
        conn.execute("DELETE FROM people")
    people_registry.invalidate()
    _reset(tmp_path)
    assert _rows(db, "people") == 0


def test_reset_keeps_the_install_owner(db: Path, tmp_path: Path) -> None:
    from openexecutive.people import install_owner

    install_owner.remember("pat@example.com", db)
    _reset(tmp_path)
    assert install_owner.get(db) == "pat@example.com"


def test_reset_drops_a_loaded_companys_principal(db: Path, tmp_path: Path) -> None:
    """With a simulator company live and no owner kept yet, its principal is
    not written back as the install's own."""
    owner_id = people_store.find_principal_person().id
    people_store.upsert_person(
        full_name="Fixture Boss", is_principal=True, email="boss@fixture.test", person_id=owner_id,
    )
    people_registry.invalidate()
    sentinel = fixture_loader._fixture_active_sentinel(_settings(tmp_path))
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("acme")
    _reset(tmp_path)
    assert _rows(db, "people") == 0


def test_reset_keeps_the_principal_after_a_handover(db: Path, tmp_path: Path) -> None:
    """The kept owner follows the principal on the owner's own company."""
    from openexecutive.people import install_owner

    install_owner.remember("pat@example.com", db)
    owner_id = people_store.find_principal_person().id
    people_store.upsert_person(
        full_name="Alex New", is_principal=True, email="alex@example.com", person_id=owner_id,
    )
    people_registry.invalidate()
    _reset(tmp_path)
    people_registry.invalidate()
    assert [p.email for p in people_store.list_people()] == ["alex@example.com"]
    assert install_owner.get(db) == "alex@example.com"


def test_reset_keeps_the_tables_the_caller_names(db: Path, tmp_path: Path) -> None:
    ws.restore_workspace_settings(ws.WorkspaceSettings(mode="solo", timezone="Asia/Seoul"))
    _reset(tmp_path, keep=frozenset({"people", "workspace_settings"}))
    assert _rows(db, "people") == 1
    assert _rows(db, "brand_new_feature") == 0
    assert _rows(db, "departments") == 0
    # Kept as it was, not put back to the defaults.
    assert ws.get_workspace() == ws.WorkspaceSettings(mode="solo", timezone="Asia/Seoul")


def test_keeps_cover_the_one_time_markers() -> None:
    # facts_meta is not here on purpose: the wipe empties ``decisions`` with
    # it, so its copy-once marker re-runs over nothing.
    from openexecutive.memory.company_tables import COMPANY_INDEPENDENT_TABLES

    for table in ("app_migrations", "monitoring_migrations", "review_meta",
                  "dynamic_workflows_meta", "departments_meta", "delegation_upgrades"):
        assert table in COMPANY_INDEPENDENT_TABLES


def test_reset_empties_the_queue_and_seeds_the_recurring_jobs_again(db: Path, tmp_path: Path) -> None:
    episodic.insert_scheduled_action(
        run_at="2099-01-01T00:00:00+00:00", channel="email", channel_ref="pat@example.com",
        intent_text="Call the bank", kind="reminder",
    )
    # An open loop sits on the internal channel but is the old company's.
    episodic.insert_scheduled_action(
        run_at="2099-01-01T00:00:00+00:00", channel="__internal__", channel_ref="",
        intent_text="Send Northwind the proposal", kind="open_loop",
    )
    _reset(tmp_path)
    with sqlite3.connect(str(db)) as conn:
        kinds = {row[0] for row in conn.execute("SELECT kind FROM scheduled_actions")}
    assert "reminder" not in kinds
    assert "open_loop" not in kinds
    assert "principal_brief_morning" in kinds


def test_one_broken_seed_does_not_stop_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    from openexecutive.scheduler import runner

    ran: list[str] = []
    monkeypatch.setattr(runner, "seed_principal_briefs", lambda: ran.append("briefs"))
    monkeypatch.setattr(runner, "_RECURRING_SEEDS", (
        ("missing", None, "openexecutive.no_such_module", "seed"),
        ("principal briefs", None, "openexecutive.scheduler.runner", "seed_principal_briefs"),
        ("off", "never_on", "openexecutive.no_such_module", "seed"),
    ))
    monkeypatch.setattr("openexecutive.config.get_settings", lambda: type("S", (), {"never_on": False})())
    runner.seed_recurring_jobs()
    assert ran == ["briefs"]


def test_sweep_empties_a_search_table_without_breaking_it(tmp_path: Path) -> None:
    path = tmp_path / "fts.db"
    with sqlite3.connect(str(path)) as conn:
        conn.execute("CREATE VIRTUAL TABLE notes_fts USING fts5(body)")
        conn.execute("INSERT INTO notes_fts (body) VALUES ('old company plan')")
        conn.execute("CREATE VIRTUAL TABLE older_fts USING fts4(body)")
        conn.execute("INSERT INTO older_fts (body) VALUES ('old company plan')")
        # An ordinary table that only shares the prefix is still swept.
        conn.execute("CREATE TABLE notes_fts_log (note TEXT)")
        conn.execute("INSERT INTO notes_fts_log VALUES ('old company')")
    from openexecutive.memory.company_tables import wipe_per_company_tables

    wipe_per_company_tables(path)
    with sqlite3.connect(str(path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM notes_fts_log").fetchone()[0] == 0
        conn.execute("INSERT INTO notes_fts (body) VALUES ('new company plan')")
        hits = conn.execute("SELECT body FROM notes_fts WHERE notes_fts MATCH 'new'").fetchall()
        conn.execute("INSERT INTO older_fts (body) VALUES ('new company plan')")
        older = conn.execute("SELECT body FROM older_fts WHERE older_fts MATCH 'company'").fetchall()
    assert hits == [("new company plan",)]
    assert older == [("new company plan",)]


def test_reset_forgets_the_old_agent_overrides(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.agents import overrides

    monkeypatch.setattr(overrides, "DB_PATH", db)
    overrides.initialize_overrides_db(db)
    overrides.set_override("finance", instructions="Old company voice.", instructions_set=True)
    assert overrides.get_override("finance") is not None  # now in memory
    _reset(tmp_path)
    assert overrides.get_override("finance") is None
    overrides.invalidate_cache()


def test_every_recurring_seed_points_at_real_code() -> None:
    """The seeds are named as strings, so a rename would only show as a log
    line after a reset. Each must import, be callable, and use a real flag."""
    import importlib

    from openexecutive.config import Settings
    from openexecutive.scheduler import runner

    for name, flag, module, func in runner._RECURRING_SEEDS:
        assert callable(getattr(importlib.import_module(module), func)), name
        assert flag is None or flag in Settings.model_fields, name
