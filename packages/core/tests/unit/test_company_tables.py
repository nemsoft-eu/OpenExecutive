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
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import openexecutive
from openexecutive.cli import fixture_loader
from openexecutive.memory.company_tables import (
    COMPANY_INDEPENDENT_TABLES,
    FIXTURE_SWAP_TABLES,
    KEPT_BY_FIXTURE_SWAP,
    MIGRATION_MARKER_TABLES,
    PER_COMPANY_TABLES,
    SwapRolledBack,
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
    # A table the fixture swap keeps is still per-company (reset and a blank
    # slot wipe it), each with its reason.
    assert set(KEPT_BY_FIXTURE_SWAP) <= set(PER_COMPANY_TABLES)
    assert all(reason.strip() for reason in KEPT_BY_FIXTURE_SWAP.values())
    assert set(FIXTURE_SWAP_TABLES) == set(PER_COMPANY_TABLES) - set(KEPT_BY_FIXTURE_SWAP)


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
    with pytest.raises(SwapRolledBack) as raised:
        swap_tables(live, ("notes",), saved)
    assert isinstance(raised.value.__cause__, sqlite3.IntegrityError)
    with sqlite3.connect(live) as conn:
        assert conn.execute("SELECT * FROM notes").fetchall() == [(1, "live")]


def test_swap_tables_refuses_a_missing_copy(tmp_path: Path) -> None:
    live = tmp_path / "live.db"
    with sqlite3.connect(live) as conn:
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO notes VALUES (1)")
    with pytest.raises(SwapRolledBack) as raised:
        swap_tables(live, ("notes",), tmp_path / "absent.db")
    assert isinstance(raised.value.__cause__, FileNotFoundError)
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
    with pytest.raises(SwapRolledBack) as raised:
        swap_tables(live, ("notes",), saved)
    assert isinstance(raised.value.__cause__, ValueError)
    with sqlite3.connect(live) as conn:
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1


def test_swap_tables_holds_the_write_lock_from_the_copy_to_the_wipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row another connection commits between the copy and the wipe would be
    in neither: wiped from the live database and missing from the copy."""
    from openexecutive.memory import company_tables

    live, copy = tmp_path / "live.db", tmp_path / "copy.db"
    with sqlite3.connect(live) as conn:
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
        conn.execute("INSERT INTO notes (body) VALUES ('mine')")
    real_copy = company_tables._copy_database
    late_write: list[str] = []

    def _copy_then_write(db: Path, target: Path) -> None:
        real_copy(db, target)
        other = sqlite3.connect(live, timeout=0.1)
        try:
            other.execute("INSERT INTO notes (body) VALUES ('late')")
            other.commit()
            late_write.append("committed")
        except sqlite3.OperationalError as exc:
            late_write.append(str(exc))
        finally:
            other.close()

    monkeypatch.setattr(company_tables, "_copy_database", _copy_then_write)
    assert swap_tables(live, ("notes",), save_to=copy) == ({"notes": 1}, {})
    assert late_write == ["database is locked"]
    with sqlite3.connect(copy) as conn:
        assert conn.execute("SELECT body FROM notes").fetchall() == [("mine",)]
    assert not (tmp_path / "copy.db.tmp").exists()


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
    swapped = set(seeded) & set(FIXTURE_SWAP_TABLES)
    kept = set(seeded) & (set(COMPANY_INDEPENDENT_TABLES) | set(KEPT_BY_FIXTURE_SWAP))

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
    """The fixture swap and the blank-slot wipe run with foreign keys off, so a
    reference from a table they do not swap would be left dangling rather than
    cascade."""
    with sqlite3.connect(env.db) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        crossing = {
            (table, ref[2])
            for swapped in (set(PER_COMPANY_TABLES), set(FIXTURE_SWAP_TABLES))
            for table in tables
            if table not in swapped
            for ref in conn.execute(f'PRAGMA foreign_key_list("{table}")')
            if ref[2] in swapped
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
    # audit_log is not here: entering a blank slot wipes it and unload keeps
    # the live one, as on main (KEPT_BY_FIXTURE_SWAP).
    assert set(seeded) & set(FIXTURE_SWAP_TABLES) <= _surviving(env.db, seeded)


def _note(db: Path, body: str) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO department_notes (department_slug, kind, body, created_at) "
            "VALUES ('finance', 'note', ?, '2026-11-13')",
            (body,),
        )


def _notes(db: Path) -> set[str]:
    with sqlite3.connect(db) as conn:
        return {r[0] for r in conn.execute("SELECT body FROM department_notes")}


def _user_backup_db(env: Any) -> Path:
    return env.settings.company_profile_path.parent / "_user_backup" / fixture_loader.USER_STATE_DB


@pytest.mark.parametrize(
    "failure",
    [sqlite3.OperationalError("database is locked"), SwapRolledBack("database is locked")],
    ids=["unknown-outcome", "rolled-back"],
)
def test_a_failed_swap_after_parking_a_client_keeps_the_users_copy(
    env: Any, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """Parking drops the client's sentinel. If the swap then fails, the live
    state is still the client's, and the retry must not copy it over the
    user's backup, even when the swap provably changed nothing."""
    from openexecutive.clients import slots
    from openexecutive.memory import company_tables

    seeded = _seed_every_table(env.db)
    asyncio.run(slots.create_client_slot(env.settings, display_name="Fresh", slug="fresh", source="blank"))
    # The first activation snapshots the user's state into _user_backup.
    asyncio.run(slots.activate_client_slot(env.settings, "fresh"))
    _note(env.db, "the client's")

    real = company_tables.swap_tables
    calls: list[int] = []

    def _locked_once(*a: Any, **k: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise failure
        return real(*a, **k)

    monkeypatch.setattr(company_tables, "swap_tables", _locked_once)
    with pytest.raises(type(failure)):
        asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert slots.get_active_client(env.settings) is None
    assert fixture_loader.swap_in_progress(env.settings)

    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert "the client's" not in _notes(_user_backup_db(env))
    assert set(seeded) & set(FIXTURE_SWAP_TABLES) <= _surviving(_user_backup_db(env), seeded)

    asyncio.run(fixture_loader.unload_fixture(env.settings))
    assert "the client's" not in _notes(env.db)
    assert set(seeded) & set(FIXTURE_SWAP_TABLES) <= _surviving(env.db, seeded)


def test_a_crash_right_after_the_swap_commits_keeps_the_users_copy(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tables are wiped once the swap commits; a process dying right then
    must still leave the swap marked, or the retried load would copy the
    wiped database over the user's only good copy."""
    from openexecutive.memory import company_tables

    seeded = _seed_every_table(env.db)
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    real = company_tables.swap_tables

    def _commit_then_die(*a: Any, **k: Any) -> Any:
        real(*a, **k)
        raise RuntimeError("process killed")

    monkeypatch.setattr(company_tables, "swap_tables", _commit_then_die)
    with pytest.raises(RuntimeError):
        asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    monkeypatch.setattr(company_tables, "swap_tables", real)

    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert set(seeded) & set(FIXTURE_SWAP_TABLES) <= _surviving(_user_backup_db(env), seeded)
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    assert set(seeded) & set(FIXTURE_SWAP_TABLES) <= _surviving(env.db, seeded)


def test_a_failure_to_mark_the_swap_keeps_the_users_copy(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writing the marker failing once (a full disk, say) must not leave
    wiped tables behind an absent marker."""
    seeded = _seed_every_table(env.db)
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    real = fixture_loader.begin_swap
    calls: list[int] = []

    def _fail_once(settings: Any) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise OSError("No space left on device")
        real(settings)

    monkeypatch.setattr(fixture_loader, "begin_swap", _fail_once)
    # Marked before the first row changes, so the failure rolls the swap back.
    with pytest.raises(SwapRolledBack) as raised:
        asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert isinstance(raised.value.__cause__, OSError)
    assert set(seeded) & set(PER_COMPANY_TABLES) <= _surviving(env.db, seeded)

    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    assert set(seeded) & set(FIXTURE_SWAP_TABLES) <= _surviving(env.db, seeded)


def test_a_crash_right_after_a_first_activations_wipe_keeps_the_users_copy(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.clients import slots

    seeded = _seed_every_table(env.db)
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    asyncio.run(slots.create_client_slot(env.settings, display_name="Fresh", slug="fresh", source="blank"))
    real = slots._wipe_per_client_tables

    def _wipe_then_die() -> None:
        real()
        raise RuntimeError("process killed")

    monkeypatch.setattr(slots, "_wipe_per_client_tables", _wipe_then_die)
    with pytest.raises(RuntimeError):
        asyncio.run(slots.activate_client_slot(env.settings, "fresh"))
    monkeypatch.setattr(slots, "_wipe_per_client_tables", real)

    asyncio.run(slots.activate_client_slot(env.settings, "fresh"))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    assert set(seeded) & set(FIXTURE_SWAP_TABLES) <= _surviving(env.db, seeded)


def test_a_rolled_back_first_activation_leaves_no_marker(env: Any) -> None:
    from openexecutive.clients import slots

    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    asyncio.run(slots.create_client_slot(env.settings, display_name="Fresh", slug="fresh", source="blank"))
    holder = sqlite3.connect(env.db, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(SwapRolledBack):
            asyncio.run(slots.activate_client_slot(env.settings, "fresh"))
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert not fixture_loader.swap_in_progress(env.settings)


def test_a_rolled_back_first_activation_keeps_an_earlier_swaps_marker(env: Any) -> None:
    """A load that died after its wipe left the marker over demo rows. A first
    activation that then rolls back must not drop it, or the next load would
    copy the demo rows over the user's backup."""
    from openexecutive.clients import slots

    _note(env.db, "the user's")
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    asyncio.run(slots.create_client_slot(env.settings, display_name="Fresh", slug="fresh", source="blank"))
    # The interrupted load: tables wiped, marker set, no fixture sentinel.
    fixture_loader.begin_swap(env.settings)
    with sqlite3.connect(env.db) as conn:
        conn.execute("DELETE FROM department_notes")
    _note(env.db, "demo row")

    holder = sqlite3.connect(env.db, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(SwapRolledBack):
            asyncio.run(slots.activate_client_slot(env.settings, "fresh"))
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert fixture_loader.swap_in_progress(env.settings)

    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert _notes(_user_backup_db(env)) == {"the user's"}


def test_a_reused_backup_restores_the_users_current_profile_and_docs(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The backup's database copy is refreshed on every load from the user's
    own state; its profile and docs must be too, or unload pairs fresh rows
    with an older profile."""

    async def _ingest(**_k: Any) -> int:
        return 1

    monkeypatch.setattr("openexecutive.knowledge.loader.ingest_file", _ingest)
    company = env.settings.company_profile_path.parent
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    env.settings.company_profile_path.write_text("name: My Co Renamed\n")
    (company / "docs" / "plan.md").write_text("# the new plan\n")
    _note(env.db, "new")

    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    assert "My Co Renamed" in env.settings.company_profile_path.read_text()
    assert (company / "docs" / "plan.md").read_text() == "# the new plan\n"
    assert "new" in _notes(env.db)


def test_a_first_activation_from_a_reused_backup_refreshes_its_profile(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.clients import slots

    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    env.settings.company_profile_path.write_text("name: My Co Renamed\n")
    asyncio.run(slots.create_client_slot(env.settings, display_name="Fresh", slug="fresh", source="blank"))
    asyncio.run(slots.activate_client_slot(env.settings, "fresh"))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    assert "My Co Renamed" in env.settings.company_profile_path.read_text()


def test_fixture_load_and_unload_keep_the_audit_log(env: Any) -> None:
    """The operator's record of real actions, a demo's included, as on main.
    Only the research run rows that would make a seeded scan skip go."""
    from openexecutive.monitoring.research.scheduler import EVENT_RAN

    def _audit(event_type: str, summary: str) -> None:
        with sqlite3.connect(env.db) as conn:
            conn.execute(
                "INSERT INTO audit_log (ts, event_type, summary) VALUES ('2026-11-13T09:00:00', ?, ?)",
                (event_type, summary),
            )

    def _rows() -> set[tuple[str, str]]:
        with sqlite3.connect(env.db) as conn:
            return set(conn.execute("SELECT event_type, summary FROM audit_log").fetchall())

    _audit("tool_call", "before the demo")
    _audit(EVENT_RAN, "the user's last scan")
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert ("tool_call", "before the demo") in _rows()
    _audit("tool_call", "during the demo")
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    rows = _rows()
    assert {("tool_call", "before the demo"), ("tool_call", "during the demo")} <= rows
    assert not any(event == EVENT_RAN for event, _ in rows)


def test_a_row_written_while_the_load_copies_the_users_state_is_not_lost(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another thread or process committing between the backup's copy and the
    wipe: its row must either wait for the swap or end up in the copy."""
    from openexecutive.memory import company_tables

    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    outcomes: list[str] = []

    def _then_write(copy: Any) -> Any:
        def wrapped(*a: Any, **k: Any) -> None:
            copy(*a, **k)
            other = sqlite3.connect(env.db, timeout=0.1)
            try:
                other.execute(
                    "INSERT INTO department_notes (department_slug, kind, body, created_at) "
                    "VALUES ('finance', 'note', 'written meanwhile', '2026-11-13')"
                )
                other.commit()
                outcomes.append("committed")
            except sqlite3.OperationalError:
                outcomes.append("waited")
            finally:
                other.close()

        return wrapped

    monkeypatch.setattr(fixture_loader, "_save_user_state_db", _then_write(fixture_loader._save_user_state_db))
    monkeypatch.setattr(company_tables, "_copy_database", _then_write(company_tables._copy_database))
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    # The reused backup's copy is taken under the swap's lock: the writer waits.
    assert outcomes == ["waited"]


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


def test_a_swap_that_changed_nothing_leaves_no_marker(env: Any, tmp_path: Path) -> None:
    """The marker blocks every later refresh and snapshot, so a swap that
    provably rolled back (here: the database is locked) must not leave it."""
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))

    holder = sqlite3.connect(env.db, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(SwapRolledBack):
            asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert not fixture_loader.swap_in_progress(env.settings)


class _Killed(BaseException):
    """The process dying: not an ``Exception``, so nothing handles it."""


class _Intercepted:
    """A connection whose statements go through ``on(conn, sql, params)``."""

    def __init__(self, conn: sqlite3.Connection, on: Any) -> None:
        self._conn, self._on = conn, on

    def execute(self, sql: str, *params: Any) -> Any:
        return self._on(self._conn, sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


def _intercept(monkeypatch: pytest.MonkeyPatch, on: Any) -> None:
    """Route the statements ``swap_tables`` runs through ``on``: the real
    function against real SQLite, failing exactly where ``on`` says. ``on``
    None puts the real module back."""
    from types import SimpleNamespace

    from openexecutive.memory import company_tables

    if on is None:
        monkeypatch.setattr(company_tables, "sqlite3", sqlite3)
        return

    def _connect(*a: Any, **k: Any) -> Any:
        return _Intercepted(sqlite3.connect(*a, **k), on)

    monkeypatch.setattr(company_tables, "sqlite3", SimpleNamespace(connect=_connect))


def _reused_backup(env: Any) -> dict[str, tuple[Any, ...]]:
    """The user's rows in every table, behind a backup an earlier cycle left."""
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    return _seed_every_table(env.db)


def test_a_commit_that_may_have_landed_keeps_the_marker(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """COMMIT reporting an error after SQLite already ended the transaction:
    the tables may be wiped, so this is not a rollback, and the marker must
    keep the retry from copying the wiped tables over the user's backup."""
    seeded = _reused_backup(env)

    def _on(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> Any:
        result = conn.execute(sql, *params)
        if sql == "COMMIT":
            raise sqlite3.OperationalError("disk I/O error")
        return result

    _intercept(monkeypatch, _on)
    with pytest.raises(sqlite3.OperationalError, match="no transaction is active"):
        asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert fixture_loader.swap_in_progress(env.settings)
    assert not _surviving(env.db, seeded) & set(FIXTURE_SWAP_TABLES)
    _intercept(monkeypatch, None)

    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    assert set(seeded) & set(FIXTURE_SWAP_TABLES) <= _surviving(env.db, seeded)


def test_a_refused_commit_rolls_back_and_drops_the_marker(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """COMMIT failing with the transaction still open (a reader holding the
    lock): the ROLLBACK provably undoes the wipe."""
    seeded = _reused_backup(env)

    def _on(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> Any:
        if sql == "COMMIT":
            raise sqlite3.OperationalError("database is locked")
        return conn.execute(sql, *params)

    _intercept(monkeypatch, _on)
    with pytest.raises(SwapRolledBack):
        asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert not fixture_loader.swap_in_progress(env.settings)
    assert set(seeded) & set(PER_COMPANY_TABLES) <= _surviving(env.db, seeded)


def test_a_swap_sqlite_already_rolled_back_drops_the_marker(env: Any) -> None:
    """A statement error after which SQLite has rolled the transaction back
    itself (here a trigger's RAISE(ROLLBACK), as a full disk does): nothing
    changed, so the marker goes rather than refuse every later snapshot."""
    seeded = _reused_backup(env)
    with sqlite3.connect(env.db) as conn:
        conn.execute(
            "CREATE TRIGGER no_delete BEFORE DELETE ON department_notes "
            "BEGIN SELECT RAISE(ROLLBACK, 'disk full'); END"
        )
    with pytest.raises(SwapRolledBack):
        asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert not fixture_loader.swap_in_progress(env.settings)
    assert set(seeded) & set(PER_COMPANY_TABLES) <= _surviving(env.db, seeded)


def test_a_kill_mid_swap_keeps_the_marker(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Killed after the first DELETE: the swap was marked before it, and the
    marker stays, since nothing proves the tables unchanged."""
    seeded = _reused_backup(env)
    deletes: list[str] = []

    def _on(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> Any:
        if sql.startswith("DELETE"):
            deletes.append(sql)
            if len(deletes) == 2:
                raise _Killed
        return conn.execute(sql, *params)

    _intercept(monkeypatch, _on)
    with pytest.raises(_Killed):
        asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert fixture_loader.swap_in_progress(env.settings)
    _intercept(monkeypatch, None)

    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    assert set(seeded) & set(FIXTURE_SWAP_TABLES) <= _surviving(env.db, seeded)


def test_a_kill_while_copying_a_reused_backup_keeps_the_users_newer_rows(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The backup's copy is retaken at the start of the swap. Marked before
    it, a kill during the copy would leave the marker over a stale copy while
    the live rows are still the user's: the retry would skip the refresh and
    unload would restore the stale copy, losing what was written since."""
    from openexecutive.memory import company_tables

    _reused_backup(env)
    _note(env.db, "written since the last load")
    real_copy = company_tables._copy_database

    def _killed(_db: Path, target: Path) -> None:
        target.with_name(target.name + ".tmp").write_bytes(b"half a copy")
        raise _Killed

    monkeypatch.setattr(company_tables, "_copy_database", _killed)
    with pytest.raises(_Killed):
        asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert not fixture_loader.swap_in_progress(env.settings)
    assert "written since the last load" in _notes(env.db)
    monkeypatch.setattr(company_tables, "_copy_database", real_copy)

    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    assert "written since the last load" in _notes(env.db)


def test_a_failed_first_activation_snapshot_aborts_it(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the snapshot the blank slot's wipe would destroy the user's
    only copy (a full disk, a lock)."""
    from openexecutive.clients import slots

    seeded = _seed_every_table(env.db)
    asyncio.run(slots.create_client_slot(env.settings, display_name="Fresh", slug="fresh", source="blank"))

    def _disk_full(_backup_dir: Path) -> None:
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(fixture_loader, "_save_user_state_db", _disk_full)
    with pytest.raises(sqlite3.OperationalError):
        asyncio.run(slots.activate_client_slot(env.settings, "fresh"))
    assert slots.get_active_client(env.settings) is None
    assert not fixture_loader.swap_in_progress(env.settings)
    assert set(seeded) & set(PER_COMPANY_TABLES) <= _surviving(env.db, seeded)


def test_a_failed_first_load_snapshot_aborts_it(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A snapshot that could not even create the backup: the load would take
    the narrower wipe with nothing to unload back to."""
    seeded = _seed_every_table(env.db)

    def _no_backup(_settings: Any, **_k: Any) -> None:
        raise PermissionError("company/_user_backup")

    monkeypatch.setattr(fixture_loader, "snapshot_user_state", _no_backup)
    with pytest.raises(PermissionError):
        asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert fixture_loader.get_fixture_status(env.settings)["active_fixture"] is None
    assert set(seeded) & set(PER_COMPANY_TABLES) <= _surviving(env.db, seeded)


def test_an_empty_install_still_switches(env: Any) -> None:
    """Nothing to snapshot is not a failed snapshot: an install with no
    database and no profile yet still activates a slot and loads a fixture."""
    from openexecutive.clients import slots

    env.db.unlink()
    env.settings.company_profile_path.unlink()
    asyncio.run(slots.create_client_slot(env.settings, display_name="Fresh", slug="fresh", source="blank"))
    asyncio.run(slots.activate_client_slot(env.settings, "fresh"))
    assert slots.get_active_client(env.settings) == "fresh"

    shutil.rmtree(env.settings.company_profile_path.parent / "_user_backup")
    (env.settings.company_profile_path.parent / ".active_client").unlink(missing_ok=True)
    env.db.unlink(missing_ok=True)
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert fixture_loader.get_fixture_status(env.settings)["active_fixture"] == "demo"


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("profile.yaml", "name: [unclosed\n"),
        ("people.yaml", "people:\n  - just a name\n"),
        ("people.yaml", "people:\n  - full_name: Ada\n    response_sla_hours: soon\n"),
        ("memory.json", '{"decisions": [{"summary": {"nested": true}}]}'),
        ("memory.json", '{"scheduled_actions": [{"run_at": 30}]}'),
        ("departments.yaml", "departments:\n  - slug: ops\n    title: Ops\n    goals: [q3]\n"),
        ("memory.json", "null"),
        ("memory.json", '{"decisions": null}'),
        ("people.yaml", "people:\n  - full_name: Ada\n    role:\n"),
        ("people.yaml", "people:\n  - full_name: Ada\n    department_slugs:\n"),
        ("departments.yaml", "departments:\n  - slug: ops\n    title: Ops\n    charter: {mission: [a, b]}\n"),
        ("departments.yaml", "departments:\n  - slug: ops\n    title: Ops\n    cadences: {weekly: 2026-01-05}\n"),
        ("memory.json", '{"advice_given": [{"timestamp": null}]}'),
        ("people.yaml", "people:\n  - full_name: Ada\n    department_slugs: [2026-01-05]\n"),
    ],
    ids=[
        "profile", "people-row", "people-sla", "memory-value", "memory-run-at", "department-goal",
        "memory-null", "memory-null-list", "people-null-role", "people-null-slugs",
        "charter-mission", "cadence-date", "advice-null-stamp", "people-date-slug",
    ],
)
def test_a_malformed_fixture_fails_before_anything_changes(
    env: Any, name: str, content: str
) -> None:
    """Parsed after the wipe, a bad file left the company half-replaced, the
    swap marked and no fixture recorded."""
    seeded = _reused_backup(env)
    (fixture_loader.FIXTURES_ROOT / "demo" / name).write_text(content)
    with pytest.raises(Exception):  # noqa: B017 — yaml, pydantic or ValueError
        asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert not fixture_loader.swap_in_progress(env.settings)
    assert fixture_loader.get_fixture_status(env.settings)["active_fixture"] is None
    assert set(seeded) & set(PER_COMPANY_TABLES) <= _surviving(env.db, seeded)


def test_a_malformed_slot_profile_fails_before_anything_changes(env: Any) -> None:
    from openexecutive.clients import slots

    seeded = _reused_backup(env)
    asyncio.run(slots.create_client_slot(env.settings, display_name="Fresh", slug="fresh", source="blank"))
    slot = env.settings.company_profile_path.parent / "_client_slots" / "fresh"
    (slot / "profile.yaml").write_text("name: [unclosed\n")
    with pytest.raises(Exception):  # noqa: B017 — the YAML parser's own error
        asyncio.run(slots.activate_client_slot(env.settings, "fresh"))
    assert slots.get_active_client(env.settings) is None
    assert not fixture_loader.swap_in_progress(env.settings)
    assert set(seeded) & set(PER_COMPANY_TABLES) <= _surviving(env.db, seeded)


def test_a_malformed_seed_slot_fails_before_anything_changes(env: Any) -> None:
    from openexecutive.clients import slots

    seeded = _reused_backup(env)
    asyncio.run(slots.create_client_slot(env.settings, display_name="Fresh", slug="fresh", source="blank"))
    slot = env.settings.company_profile_path.parent / "_client_slots" / "fresh"
    (slot / "people.yaml").write_text("people:\n  - just a name\n")
    with pytest.raises(ValueError):
        asyncio.run(slots.activate_client_slot(env.settings, "fresh"))
    assert slots.get_active_client(env.settings) is None
    assert not fixture_loader.swap_in_progress(env.settings)
    assert set(seeded) & set(PER_COMPANY_TABLES) <= _surviving(env.db, seeded)


def test_a_malformed_fixture_fails_before_parking_the_active_client(env: Any) -> None:
    """Parking leaves client mode; a fixture that cannot load must not."""
    from openexecutive.clients import slots

    asyncio.run(slots.create_client_slot(env.settings, display_name="Fresh", slug="fresh", source="blank"))
    asyncio.run(slots.activate_client_slot(env.settings, "fresh"))
    (fixture_loader.FIXTURES_ROOT / "demo" / "profile.yaml").write_text("name: [unclosed\n")
    with pytest.raises(Exception):  # noqa: B017 — the YAML parser's own error
        asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert slots.get_active_client(env.settings) == "fresh"
    assert not fixture_loader.swap_in_progress(env.settings)


def test_a_failed_create_from_current_snapshot_leaves_no_slot(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entering client mode past a failed snapshot left unload nothing, or
    only the narrower legacy copy, to restore the user's company from."""
    from openexecutive.clients import slots

    def _disk_full(_backup_dir: Path) -> None:
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(fixture_loader, "_save_user_state_db", _disk_full)
    with pytest.raises(sqlite3.OperationalError):
        asyncio.run(slots.create_client_slot(env.settings, display_name="Mine", slug="mine", source="current"))
    assert slots.get_active_client(env.settings) is None
    assert not (env.settings.company_profile_path.parent / "_client_slots" / "mine").exists()


def test_the_seed_check_accepts_what_the_seeders_accept(tmp_path: Path) -> None:
    """Empty sections, unread keys of any shape and nulls where the seeders
    allow them are not refusals."""
    (tmp_path / "people.yaml").write_text(
        "people:\n  - full_name: Ada\n    email:\n    aliases: [a, b]\n"
    )
    (tmp_path / "departments.yaml").write_text(
        "departments:\n  - slug: ops\n    title: Ops\n    headcount:\n    charter:\n"
        "    kpis: [x]\n    goals:\n"
    )
    (tmp_path / "memory.json").write_text('{"decisions": [{"created_at": 5, "extra": [1]}]}')
    fixture_loader.check_seed_files(tmp_path)
    (tmp_path / "people.yaml").write_text('people: ""\n')
    fixture_loader.check_seed_files(tmp_path)


def test_generated_and_dumped_seed_files_pass_the_seed_check(env: Any, tmp_path: Path) -> None:
    """Positive controls: a fully populated generated fixture, and the user's
    own backup YAML that an unload without state.db seeds from."""
    from openexecutive.fixtures import generator
    from openexecutive.people import store as people_store

    bundle = generator.FixtureBundle.model_validate({
        "profile": {"name": "Gen Co"},
        "people": [{"full_name": "Ada", "role": "CEO", "department_slugs": ["ops"],
                    "authority_scope": ["finance"], "availability": [{"weekdays": [0, 1]}]}],
        "departments": [{"slug": "ops", "title": "Ops", "head_person_name": "Ada",
                         "cadences": {"weekly": "mon"}, "headcount": 3,
                         "goals": [{"key_result": "Ship"}]}],
        "memory": {"decisions": [{"summary": "d"}], "initiatives": [{"title": "i"}],
                   "advice_given": [{"timestamp": "-1d"}],
                   "alerts": [{"headline": "h", "routed_to_person": "Ada"}]},
    })
    generated = tmp_path / "generated"
    generated.mkdir()
    generator.materialize_to_dir(generator.bundle_to_serialized(bundle, "test"), generated)
    fixture_loader.check_seed_files(generated)

    people_store.upsert_person(full_name="Ada Mine", email="ada@mine.example", db_path=env.db)
    _note(env.db, "mine")
    fixture_loader.snapshot_user_state(env.settings)
    fixture_loader.check_seed_files(env.settings.company_profile_path.parent / "_user_backup")


def test_every_curated_fixture_passes_the_seed_check() -> None:
    """Positive control for the check above: it must not refuse a real fixture."""
    root = Path(fixture_loader.__file__).resolve().parents[4] / "fixtures" / "companies"
    fixtures = sorted(p for p in root.iterdir() if (p / "profile.yaml").exists())
    assert fixtures
    for fixture in fixtures:
        fixture_loader.check_seed_files(fixture)


def test_a_failed_save_back_keeps_the_clients_save_point(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switching clients saves the outgoing one, then restores the next over
    the live database. A save that fails must abort the switch and leave the
    slot's previous state.db: without one, its next activation is a blank wipe."""
    from types import SimpleNamespace

    from openexecutive.clients import slots
    from openexecutive.memory import company_tables

    for slug in ("one", "two"):
        asyncio.run(slots.create_client_slot(env.settings, display_name=slug, slug=slug, source="blank"))
    asyncio.run(slots.activate_client_slot(env.settings, "one"))
    _note(env.db, "saved")
    asyncio.run(slots.save_active_client(env.settings))
    _note(env.db, "since the save")

    def _connect(*a: Any, **k: Any) -> Any:
        def _on(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> Any:
            if sql.startswith("VACUUM INTO"):
                raise sqlite3.OperationalError("database or disk is full")
            return conn.execute(sql, *params)

        return _Intercepted(sqlite3.connect(*a, **k), _on)

    full_disk = SimpleNamespace(connect=_connect)
    monkeypatch.setattr(company_tables, "sqlite3", full_disk)
    monkeypatch.setattr(slots, "sqlite3", full_disk)
    with pytest.raises(sqlite3.OperationalError):
        asyncio.run(slots.activate_client_slot(env.settings, "two"))
    assert slots.get_active_client(env.settings) == "one"
    assert _notes(env.db) == {"saved", "since the save"}
    slot_db = env.settings.company_profile_path.parent / "_client_slots" / "one" / "state.db"
    assert slot_db.exists()
    assert _notes(slot_db) == {"saved"}


def test_a_forced_snapshot_ends_an_unfinished_swap(env: Any) -> None:
    """It declares the live state the user's, as it does past a fixture
    sentinel; left set, the marker would refuse every unforced snapshot."""
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    asyncio.run(fixture_loader.unload_fixture(env.settings))
    fixture_loader.begin_swap(env.settings)
    with pytest.raises(fixture_loader.FixtureActiveError):
        fixture_loader.snapshot_user_state(env.settings)

    fixture_loader.snapshot_user_state(env.settings, force=True)
    assert not fixture_loader.swap_in_progress(env.settings)
    fixture_loader.snapshot_user_state(env.settings)


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
