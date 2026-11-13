"""Every table the app creates is classified, and every company swap empties
every per-company table.

``memory.company_tables`` is the one list the three company-swapping paths
read: a blank client-slot switch, the factory reset, and fixture load/unload.
Each used to keep its own hand-written list, and a table added to one and not
the others leaked one company's rows into the next. Two guards keep that shut:

- The source scan below finds every ``CREATE TABLE`` in the package and fails
  on a table that is neither per-company nor on the company-independent list,
  so a new table needs a decision before it ships.
- The path tests put one row in every registered table and run each path
  for real, then look for those rows. Rows a path re-seeds (the principal's
  briefs, the shipped knowledge's review rows) are new rows, not the seeded
  ones, so they do not count as a leak.
"""
from __future__ import annotations

import asyncio
import importlib
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import openexecutive
from openexecutive.cli import fixture_loader
from openexecutive.memory.company_tables import (
    COMPANY_INDEPENDENT_TABLES,
    MIGRATION_MARKER_TABLES,
    PER_COMPANY_TABLES,
    swap_tables,
    wipe_tables,
)

PACKAGE = Path(openexecutive.__file__).parent

# ``CREATE TABLE [IF NOT EXISTS] name (`` — or an f-string ``{CONSTANT}`` name,
# resolved from the module that defines it. A statement split across string
# literals ends the first one right after the name.
_CREATE = re.compile(
    r"CREATE\s+(?:VIRTUAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"`]?"
    r"(\{?[A-Za-z_][A-Za-z0-9_]*\}?)[\"`]?\s*(?:\(|AS\b|[\"']\s*$)",
    re.IGNORECASE | re.MULTILINE,
)

# Registered but created by nothing any more: kept so upgraded installs that
# still have them are wiped (they carry candidate PII).
_LEGACY_TABLES = {
    "onboarding_tasks",
    "onboarding_plans",
    "onboarding_templates",
    "offers",
    "candidates",
    "engagements",
}


def _created_tables() -> dict[str, set[str]]:
    """Every table name the package's source creates, with the files that do."""
    found: dict[str, set[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        for match in _CREATE.finditer(path.read_text(encoding="utf-8")):
            name = match.group(1)
            if name.startswith("{"):
                rel = path.relative_to(PACKAGE.parent).with_suffix("")
                module = importlib.import_module(".".join(rel.parts))
                name = getattr(module, name.strip("{}"))
            found.setdefault(name, set()).add(str(path.relative_to(PACKAGE)))
    return found


def test_the_scan_finds_tables_created_both_ways() -> None:
    """Positive control: a literal name and an f-string constant both resolve,
    so an empty unclassified set below means something."""
    created = _created_tables()
    assert "chat_messages" in created  # literal
    assert "department_notes" in created  # f"... {TABLE} ("
    assert "delegation_settings" in created  # f"... {SETTINGS_TABLE} ("
    assert len(created) > 100


def test_every_created_table_is_classified() -> None:
    created = _created_tables()
    unclassified = {
        name: sorted(files)
        for name, files in created.items()
        if name not in PER_COMPANY_TABLES and name not in COMPANY_INDEPENDENT_TABLES
    }
    assert not unclassified, (
        "These tables are in neither memory.company_tables.PER_COMPANY_TABLES "
        "nor COMPANY_INDEPENDENT_TABLES. A table holding anything about one "
        "company goes in PER_COMPANY_TABLES (every company swap then wipes it); "
        f"otherwise add it to COMPANY_INDEPENDENT_TABLES with the reason: {unclassified}"
    )


def test_the_registry_has_no_overlap_duplicates_or_stale_names() -> None:
    assert len(PER_COMPANY_TABLES) == len(set(PER_COMPANY_TABLES))
    assert not set(PER_COMPANY_TABLES) & set(COMPANY_INDEPENDENT_TABLES)
    assert set(MIGRATION_MARKER_TABLES) <= set(COMPANY_INDEPENDENT_TABLES)
    assert all(reason.strip() for reason in COMPANY_INDEPENDENT_TABLES.values())
    created = set(_created_tables())
    # A name nothing creates is a typo or a removed table — except the legacy
    # tables kept on purpose.
    assert set(PER_COMPANY_TABLES) - created == _LEGACY_TABLES
    assert set(COMPANY_INDEPENDENT_TABLES) <= created


# --------------------------------------------------------------------------- #
# wipe_tables / swap_tables
# --------------------------------------------------------------------------- #


def test_wipe_tables_deletes_children_before_parents(tmp_path: Path) -> None:
    db = tmp_path / "fk.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE child (id INTEGER PRIMARY KEY, p INTEGER REFERENCES parent(id))")
        conn.execute("INSERT INTO parent VALUES (1)")
        conn.execute("INSERT INTO child VALUES (1, 1)")
    # Parent listed first: a plain DELETE in this order raises under
    # foreign_keys=ON.
    assert wipe_tables(db, ("parent", "child", "missing")) == {"child": 1, "parent": 1}


def test_wipe_tables_never_creates_a_database(tmp_path: Path) -> None:
    db = tmp_path / "absent.db"
    assert wipe_tables(db) == {}
    assert not db.exists()


def test_swap_tables_restores_rows_creating_a_missing_table(tmp_path: Path) -> None:
    saved, live = tmp_path / "saved.db", tmp_path / "live.db"
    with sqlite3.connect(saved) as conn:
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
        conn.execute("CREATE TABLE gone (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE marks (name TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO notes VALUES (7, 'kept')")
        conn.execute("INSERT INTO gone VALUES (1)")
        conn.execute("INSERT INTO marks VALUES ('old-sweep')")
    with sqlite3.connect(live) as conn:
        # The live table gained a column since the copy was saved.
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT, extra TEXT DEFAULT 'x')")
        conn.execute("INSERT INTO notes (id, body) VALUES (7, 'outgoing')")
        conn.execute("CREATE TABLE marks (name TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO marks VALUES ('new-sweep')")
    cleared, restored = swap_tables(live, ("notes", "gone", "not_saved"), saved, restore_also=("marks",))
    assert cleared == {"notes": 1, "marks": 1}
    assert restored == {"notes": 1, "gone": 1, "marks": 1}
    with sqlite3.connect(live) as conn:
        assert conn.execute("SELECT * FROM notes").fetchall() == [(7, "kept", "x")]
        assert conn.execute("SELECT * FROM gone").fetchall() == [(1,)]
        assert conn.execute("SELECT * FROM marks").fetchall() == [("old-sweep",)]


def test_swap_tables_without_a_copy_only_wipes(tmp_path: Path) -> None:
    live = tmp_path / "live.db"
    with sqlite3.connect(live) as conn:
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE marks (name TEXT)")
        conn.execute("INSERT INTO notes VALUES (1)")
        conn.execute("INSERT INTO marks VALUES ('kept')")
    assert swap_tables(live, ("notes",), restore_also=("marks",)) == ({"notes": 1}, {})
    with sqlite3.connect(live) as conn:
        assert conn.execute("SELECT * FROM marks").fetchall() == [("kept",)]


def test_swap_tables_is_all_or_nothing(tmp_path: Path) -> None:
    """A restore that fails leaves the tables as they were, not wiped."""
    saved, live = tmp_path / "saved.db", tmp_path / "live.db"
    with sqlite3.connect(saved) as conn:
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
        conn.execute("INSERT INTO notes VALUES (1, NULL)")
    with sqlite3.connect(live) as conn:
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT NOT NULL)")
        conn.execute("INSERT INTO notes VALUES (1, 'live')")
    with pytest.raises(sqlite3.IntegrityError):
        swap_tables(live, ("notes",), saved)
    with sqlite3.connect(live) as conn:
        assert conn.execute("SELECT * FROM notes").fetchall() == [(1, "live")]


def test_swap_tables_refuses_a_missing_copy(tmp_path: Path) -> None:
    live = tmp_path / "live.db"
    with sqlite3.connect(live) as conn:
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO notes VALUES (1)")
    with pytest.raises(FileNotFoundError):
        swap_tables(live, ("notes",), tmp_path / "absent.db")
    with sqlite3.connect(live) as conn:
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1


def test_swap_tables_refuses_a_copy_holding_none_of_the_tables(tmp_path: Path) -> None:
    """An empty or truncated copy would turn the restore into a wipe."""
    saved, live = tmp_path / "saved.db", tmp_path / "live.db"
    with sqlite3.connect(saved) as conn:
        conn.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
    with sqlite3.connect(live) as conn:
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO notes VALUES (1)")
    with pytest.raises(ValueError):
        swap_tables(live, ("notes",), saved)
    with sqlite3.connect(live) as conn:
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1


# --------------------------------------------------------------------------- #
# The three swap paths, against a row in every table
# --------------------------------------------------------------------------- #


class _FakeStore:
    RESEARCH_COLLECTION = "recent_research"
    COMPANY_COLLECTION = "company_docs"
    BUILTIN_COLLECTION = "builtin_knowledge"

    def __init__(self, **_kw: object) -> None: ...
    def delete_company_docs(self) -> None: ...
    def delete_documents(self, **_kw: object) -> None: ...
    def delete_notion_docs(self) -> None: ...
    def delete_drive_docs(self) -> None: ...
    def delete_onedrive_docs(self) -> None: ...
    def delete_confluence_docs(self) -> None: ...
    def delete_attachment_docs(self) -> None: ...


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Every store on one temp DB, every registered table created, and the
    vector, Honcho and skills layers stubbed (side effects, not under test)."""
    from openexecutive.alerts import store as alerts_store
    from openexecutive.audit import logger as audit_logger
    from openexecutive.clients import slots
    from openexecutive.departments import registry as dept_registry
    from openexecutive.departments import store as dept_store
    from openexecutive.fixtures import store as fixtures_store
    from openexecutive.integrations import telegram_chats
    from openexecutive.knowledge import review_store
    from openexecutive.memory import episodic, honcho_client
    from openexecutive.people import registry as people_registry
    from openexecutive.people import store as people_store

    db = tmp_path / "episodic.db"
    for mod in (
        episodic, people_store, dept_store, alerts_store, review_store,
        fixtures_store, telegram_chats, audit_logger,
    ):
        monkeypatch.setattr(mod, "DB_PATH", db)
    monkeypatch.setattr("openexecutive.audit.log_event", lambda *a, **k: None)
    monkeypatch.setattr("openexecutive.knowledge.store.ChromaDBStore", _FakeStore)
    monkeypatch.setattr(
        "openexecutive.knowledge.skills_index.sync_builtin_skill_index", lambda *_a, **_k: None
    )
    for name in ("notion_sync", "drive_sync", "onedrive_sync", "confluence_sync"):
        monkeypatch.setattr(f"openexecutive.knowledge.{name}.reset_local_state", lambda **_k: None)

    async def _noop(*_a: object, **_k: object) -> None:
        return None

    async def _no_vector(_settings: Any, _app_state: Any) -> int:
        return 0

    monkeypatch.setattr(honcho_client, "delete_workspace_and_reset_client", _noop)
    monkeypatch.setattr(honcho_client, "set_active_workspace_id", lambda *a, **k: None)
    monkeypatch.setattr(honcho_client, "clear_active_workspace_id", lambda *a, **k: None)
    monkeypatch.setattr(honcho_client, "get_active_workspace_id", lambda: "openexec")
    monkeypatch.setattr(slots, "_rebuild_vector_state", _no_vector)
    monkeypatch.setattr(slots, "_set_honcho_client_workspace", lambda _slug: None)

    company = tmp_path / "company"
    company.mkdir()
    (company / "profile.yaml").write_text("name: My Co\n")
    fixture = tmp_path / "fixtures" / "demo"
    fixture.mkdir(parents=True)
    # Only a profile: the swap must not depend on memory.json or people.yaml.
    (fixture / "profile.yaml").write_text("name: Demo Co\n")
    monkeypatch.setattr(fixture_loader, "FIXTURES_ROOT", tmp_path / "fixtures")

    # The real schemas the app creates at boot and on restore, then each
    # table-defining module's own schema hook (most create theirs lazily on
    # first use) ...
    episodic.initialize_db(db)
    slots._ensure_schemas()
    audit_logger.AuditLogger(db_path=db)
    _create_lazy_schemas(db)
    # ... and a stand-in for any table still missing: what is under test is
    # that the path empties it, whatever its columns.
    with sqlite3.connect(db) as conn:
        existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in (*PER_COMPANY_TABLES, *COMPANY_INDEPENDENT_TABLES):
            if table not in existing:
                conn.execute(f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY, body TEXT)')
    dept_registry.invalidate()
    people_registry.invalidate()
    yield type("Env", (), {
        "db": db,
        "settings": type("S", (), {
            "vector_store_path": tmp_path / "chroma",
            "company_profile_path": company / "profile.yaml",
            "mcp_servers_config_path": company / "mcp_servers.json",
            "honcho_workspace_id": "openexec",
        })(),
    })()
    dept_registry.invalidate()
    people_registry.invalidate()


def _create_lazy_schemas(db: Path) -> None:
    """Best effort: call the schema hook of every module that creates a table.

    The hooks have no common name, so try the ones the package uses. A hook
    that wants other arguments is skipped and its tables get the stand-in.
    """
    modules = {f for files in _created_tables().values() for f in files}
    for rel in sorted(modules):
        module = importlib.import_module("openexecutive." + rel[:-3].replace("/", "."))
        path_hooks = [
            name
            for name in vars(module)
            if re.fullmatch(r"initiali[sz]e\w*|init_\w+|ensure_ready|_connect|_conn", name)
        ]
        for name in path_hooks:
            hook = getattr(module, name)
            if not callable(hook) or isinstance(hook, type):
                continue
            for args in ((db,), ()):
                try:
                    result = hook(*args)
                    if isinstance(result, sqlite3.Connection):
                        result.close()
                    elif hasattr(result, "__enter__"):
                        with result:
                            pass
                except Exception:
                    continue
                break
        for name in ("ensure_schema", "_ensure", "_ensure_schema"):
            hook = getattr(module, name, None)
            if callable(hook):
                conn = sqlite3.connect(db)
                try:
                    hook(conn)
                    conn.commit()
                except Exception:
                    pass
                finally:
                    conn.close()


# Columns a swap path reads through a validating model (the snapshot dumps
# departments and people): a value the model accepts.
_VALID: dict[tuple[str, str], object] = {
    ("department_goals", "period_type"): "quarter",
    ("department_goals", "status"): "on_track",
}


def _value(table: str, table_sql: str, column: str, decl_type: str) -> object:
    """A value for ``column``: what its CHECK allows when the DDL pins one
    (``x IN (0, 1)``, ``id = 1``), else a marker no default or re-seed uses."""
    if (table, column) in _VALID:
        return _VALID[(table, column)]
    pinned = re.search(
        rf"\b{column}\b\s*(?:IN\s*\(\s*|=\s*)('[^']*'|-?\d+)", table_sql, re.IGNORECASE
    )
    if pinned:
        literal = pinned.group(1)
        return literal.strip("'") if literal.startswith("'") else int(literal)
    t = decl_type.upper()
    if "INT" in t:
        return 424242
    if any(k in t for k in ("REAL", "FLOA", "DOUB")):
        return 4.25
    return "leak-marker"


def _seed_every_table(db: Path) -> dict[str, tuple[Any, ...]]:
    """One row in every registered table; return each table's row.

    Foreign keys are off for the insert only. Integer keys all take the same
    value, so a child row really references its parent and the wipe's delete
    order is exercised.
    """
    seeded: dict[str, tuple[Any, ...]] = {}
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        ddl = dict(conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall())
        for table in (*PER_COMPANY_TABLES, *COMPANY_INDEPENDENT_TABLES):
            cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            if not cols:
                continue  # a legacy table nothing creates
            if table in PER_COMPANY_TABLES:
                # Company-independent rows (the migration markers among them)
                # stay: the app's own markers keep its sweeps from re-running
                # over the seeded rows.
                conn.execute(f'DELETE FROM "{table}"')  # noqa: S608
            names = ", ".join(f'"{c[1]}"' for c in cols)
            marks = ", ".join("?" for _ in cols)
            conn.execute(
                f'INSERT INTO "{table}" ({names}) VALUES ({marks})',  # noqa: S608
                [_value(table, ddl[table], c[1], c[2]) for c in cols],
            )
            seeded[table] = conn.execute(
                f'SELECT * FROM "{table}" WHERE rowid = last_insert_rowid()'  # noqa: S608
            ).fetchone()
        conn.commit()
    finally:
        conn.close()
    return seeded


def _surviving(db: Path, seeded: dict[str, tuple[Any, ...]]) -> set[str]:
    """The tables still holding their seeded row."""
    with sqlite3.connect(db) as conn:
        return {
            table
            for table, row in seeded.items()
            if row in conn.execute(f'SELECT * FROM "{table}"').fetchall()  # noqa: S608
        }


def test_blank_slot_switch_empties_every_per_company_table(env: Any) -> None:
    from openexecutive.clients import slots

    seeded = _seed_every_table(env.db)
    slot = env.settings.company_profile_path.parent / "_client_slots" / "fresh"
    slot.mkdir(parents=True)
    asyncio.run(slots._restore_slot_state(env.settings, slot))

    survivors = _surviving(env.db, seeded)
    assert survivors & set(PER_COMPANY_TABLES) == set()
    assert survivors == set(COMPANY_INDEPENDENT_TABLES) & set(seeded)


def test_factory_reset_empties_every_per_company_table(env: Any) -> None:
    seeded = _seed_every_table(env.db)
    asyncio.run(fixture_loader.reset_all_state(env.settings))

    survivors = _surviving(env.db, seeded)
    assert survivors & set(PER_COMPANY_TABLES) == set()
    assert survivors == set(COMPANY_INDEPENDENT_TABLES) & set(seeded)


def test_fixture_load_swaps_every_table_and_unload_restores_it(env: Any) -> None:
    seeded = _seed_every_table(env.db)
    swapped = set(seeded) & set(PER_COMPANY_TABLES)
    kept = set(seeded) & set(COMPANY_INDEPENDENT_TABLES)

    summary = asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert summary["auto_snapshot_taken"] is True
    survivors = _surviving(env.db, seeded)
    assert survivors & swapped == set()
    assert survivors == kept

    asyncio.run(fixture_loader.unload_fixture(env.settings))
    # The user's own row is back in every table the load emptied.
    assert _surviving(env.db, seeded) == swapped | kept


def test_a_backup_without_a_database_copy_keeps_the_narrower_wipe(env: Any) -> None:
    """A backup from before ``state.db``, while a fixture is loaded: the live
    tables the backup does not carry hold the user's only copy, so a load must
    not wipe them. The derived caches still go, though the fixture has no
    memory.json."""
    seeded = _seed_every_table(env.db)
    backup = env.settings.company_profile_path.parent / "_user_backup"
    backup.mkdir()
    (backup / "profile.yaml").write_text("name: My Co\n")
    (backup / ".fixture_active").write_text("other")

    asyncio.run(fixture_loader.load_fixture("demo", env.settings))

    assert not (backup / fixture_loader.USER_STATE_DB).exists()
    survivors = _surviving(env.db, seeded)
    assert {"chat_messages", "department_notes", "delegation_settings"} <= survivors
    assert not survivors & {"briefing_narrative", "person_insights", "watchlist"}


def test_a_reused_backup_gets_a_fresh_database_copy(env: Any) -> None:
    """Unload keeps ``_user_backup``. The next load from the user's own state
    must refresh its database copy, or a second unload would roll the user
    back to the first load."""
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    with sqlite3.connect(env.db) as conn:
        conn.execute("INSERT INTO department_notes (department_slug, kind, body, created_at) VALUES ('finance', 'note', 'new', '2026-11-13')")

    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    with sqlite3.connect(env.db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM department_notes").fetchone()[0] == 0
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    with sqlite3.connect(env.db) as conn:
        assert conn.execute("SELECT body FROM department_notes").fetchall() == [("new",)]


def test_unload_with_nothing_loaded_keeps_the_users_newer_rows(env: Any) -> None:
    """An unload while the live state is already the user's own (the backup
    kept from an earlier cycle) must not roll it back to the backup's copy."""
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    with sqlite3.connect(env.db) as conn:
        conn.execute("INSERT INTO department_notes (department_slug, kind, body, created_at) VALUES ('finance', 'note', 'newer', '2026-11-13')")

    asyncio.run(fixture_loader.unload_fixture(env.settings))
    with sqlite3.connect(env.db) as conn:
        assert conn.execute("SELECT body FROM department_notes").fetchall() == [("newer",)]


def test_no_table_outside_the_registry_references_a_per_company_table(env: Any) -> None:
    """The fixture swap runs with foreign keys off, so a reference from a
    table it does not swap would be left dangling rather than cascade."""
    with sqlite3.connect(env.db) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        crossing = {
            (table, ref[2])
            for table in tables
            if table not in PER_COMPANY_TABLES
            for ref in conn.execute(f'PRAGMA foreign_key_list("{table}")')
            if ref[2] in PER_COMPANY_TABLES
        }
    assert crossing == set()


def test_a_failed_load_retried_keeps_the_users_copy(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A load that fails after the wipe leaves no fixture sentinel, so the
    live state looks like the user's again. The retry must not copy that
    half-wiped database over the backup."""
    seeded = _seed_every_table(env.db)
    real = fixture_loader._seed_departments
    calls = {"n": 0}

    def _fail_once(path: Path) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("database is locked")
        return real(path)

    monkeypatch.setattr(fixture_loader, "_seed_departments", _fail_once)
    with pytest.raises(RuntimeError):
        asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert fixture_loader.swap_in_progress(env.settings)

    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert not fixture_loader.swap_in_progress(env.settings)
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    assert set(seeded) & set(PER_COMPANY_TABLES) <= _surviving(env.db, seeded)


def test_a_failed_first_slot_activation_retried_keeps_the_users_copy(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.clients import slots

    seeded = _seed_every_table(env.db)
    # An earlier fixture cycle left a backup to reuse.
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    asyncio.run(slots.create_client_slot(env.settings, display_name="Fresh", slug="fresh", source="blank"))

    async def _fail(_settings: Any, _app_state: Any) -> int:
        raise RuntimeError("chroma down")

    monkeypatch.setattr(slots, "_rebuild_vector_state", _fail)
    with pytest.raises(RuntimeError):
        asyncio.run(slots.activate_client_slot(env.settings, "fresh"))

    async def _no_vector(_settings: Any, _app_state: Any) -> int:
        return 0

    monkeypatch.setattr(slots, "_rebuild_vector_state", _no_vector)
    asyncio.run(slots.activate_client_slot(env.settings, "fresh"))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    assert set(seeded) & set(PER_COMPANY_TABLES) <= _surviving(env.db, seeded)


def test_fixture_load_drops_the_cached_council_overrides(env: Any) -> None:
    from openexecutive.agents import overrides

    overrides.set_override("finance", model="the-users-model", model_set=True, db_path=env.db)
    assert overrides.get_override("finance", db_path=env.db) is not None
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert overrides.get_override("finance", db_path=env.db) is None


def test_the_users_own_upload_stays_withheld_during_a_demo(
    env: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wipe drops the upload's review row; with no row a document is
    retrievable. It comes back as pending, as after a factory reset."""
    builtin = tmp_path / "builtin"
    (builtin / "finance").mkdir(parents=True)
    (builtin / "finance" / "my-upload.md").write_text("# mine\n")
    monkeypatch.setattr("openexecutive.knowledge.loader.BUILTIN_KNOWLEDGE_PATH", builtin)

    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    with sqlite3.connect(env.db) as conn:
        assert conn.execute(
            "SELECT status FROM review_items WHERE filename = 'my-upload.md'"
        ).fetchall() == [("pending",)]


def test_a_swap_that_changed_nothing_leaves_no_marker(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The marker blocks every later refresh and snapshot, so a swap that
    rolled back must not set it."""
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))

    def _locked(*_a: object, **_k: object) -> Any:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("openexecutive.memory.company_tables.swap_tables", _locked)
    with pytest.raises(sqlite3.OperationalError):
        asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert not fixture_loader.swap_in_progress(env.settings)


def test_create_from_current_refuses_an_unfinished_switch(env: Any) -> None:
    from openexecutive.clients import slots

    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    fixture_loader.begin_swap(env.settings)
    with pytest.raises(slots.ClientSlotConflictError):
        asyncio.run(slots.create_client_slot(env.settings, display_name="Mine", slug="mine", source="current"))
    assert not (env.settings.company_profile_path.parent / "_client_slots" / "mine").exists()


def test_a_failed_refresh_aborts_the_first_activation(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a current copy, unload would roll the user back to the old one."""
    from openexecutive.clients import slots

    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    seeded = _seed_every_table(env.db)
    asyncio.run(slots.create_client_slot(env.settings, display_name="Fresh", slug="fresh", source="blank"))

    def _disk_full(_backup_dir: Path) -> None:
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(fixture_loader, "_save_user_state_db", _disk_full)
    with pytest.raises(sqlite3.OperationalError):
        asyncio.run(slots.activate_client_slot(env.settings, "fresh"))
    assert slots.get_active_client(env.settings) is None
    assert set(seeded) & set(PER_COMPANY_TABLES) <= _surviving(env.db, seeded)


def test_a_failed_database_copy_still_leaves_the_snapshot(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The YAML artifacts land first, so the legacy path can still restore."""

    def _disk_full(_backup_dir: Path) -> None:
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(fixture_loader, "_save_user_state_db", _disk_full)
    with pytest.raises(sqlite3.OperationalError):
        fixture_loader.snapshot_user_state(env.settings)
    backup = env.settings.company_profile_path.parent / "_user_backup"
    assert (backup / "profile.yaml").exists()
    assert not (backup / fixture_loader.USER_STATE_DB).exists()


def test_fixture_load_drops_the_cached_people(env: Any) -> None:
    from openexecutive.people import registry, store

    store.upsert_person(full_name="Ada Mine", email="ada@mine.example", db_path=env.db)
    assert [p.full_name for p in registry.list_people()] == ["Ada Mine"]
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert registry.list_people() == []
