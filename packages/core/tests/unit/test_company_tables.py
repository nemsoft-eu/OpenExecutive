"""Every table the app creates is classified, and every company swap empties
every per-company table it does not explicitly keep.

``memory.company_tables`` is the one list the three company-swapping paths
wipe from: a blank client-slot switch, the factory reset, and fixture
load/unload. Each used to keep its own hand-written list, and a table added to
one and not the others leaked one company's rows into the next (fixture load
kept ``department_notes``). Two guards keep that shut:

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
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import openexecutive
from openexecutive.cli import fixture_loader
from openexecutive.memory.company_tables import (
    COMPANY_INDEPENDENT_TABLES,
    KEPT_ON_EVERY_SWAP,
    KEPT_ON_FIXTURE_SWAP,
    PER_COMPANY_TABLES,
    wipe_per_company_tables,
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
    # A path may only keep a table the registry calls per-company, and must
    # say why.
    assert set(KEPT_ON_EVERY_SWAP) <= set(PER_COMPANY_TABLES)
    assert set(KEPT_ON_FIXTURE_SWAP) <= set(PER_COMPANY_TABLES)
    for reasons in (COMPANY_INDEPENDENT_TABLES, KEPT_ON_EVERY_SWAP, KEPT_ON_FIXTURE_SWAP):
        assert all(reason.strip() for reason in reasons.values())
    created = set(_created_tables())
    # A name nothing creates is a typo or a removed table — except the legacy
    # tables kept on purpose.
    assert set(PER_COMPANY_TABLES) - created == _LEGACY_TABLES
    assert set(COMPANY_INDEPENDENT_TABLES) <= created


def test_audit_log_is_kept_by_the_fixture_path_only() -> None:
    """Each path's treatment of the audit history, pinned: the blank switch and
    the reset wipe it, fixture load and unload keep it."""
    assert "audit_log" in PER_COMPANY_TABLES
    assert "audit_log" not in KEPT_ON_EVERY_SWAP
    assert "audit_log" in KEPT_ON_FIXTURE_SWAP


# --------------------------------------------------------------------------- #
# wipe_per_company_tables
# --------------------------------------------------------------------------- #


def test_the_wipe_deletes_children_before_parents(tmp_path: Path) -> None:
    """``people`` comes after ``person_emails`` in the registry, so reverse the
    real relationship: a parent listed after its child must still go last."""
    db = tmp_path / "fk.db"
    with sqlite3.connect(db) as conn:
        # departments is listed after department_goals; make goals the parent.
        conn.execute("CREATE TABLE department_goals (id INTEGER PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE departments (id INTEGER PRIMARY KEY, g INTEGER REFERENCES department_goals(id))"
        )
        conn.execute("CREATE TABLE generated_fixtures (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO department_goals VALUES (1)")
        conn.execute("INSERT INTO departments VALUES (1, 1)")
        conn.execute("INSERT INTO generated_fixtures VALUES (1)")
    # A plain DELETE in registry order raises under foreign_keys=ON.
    assert wipe_per_company_tables(db) == {"departments": 1, "department_goals": 1}
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM generated_fixtures").fetchone()[0] == 1


def test_the_wipe_honours_keep(tmp_path: Path) -> None:
    db = tmp_path / "keep.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE audit_log (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE watchlist (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO audit_log VALUES (1)")
        conn.execute("INSERT INTO watchlist VALUES (1)")
    assert wipe_per_company_tables(db, keep=KEPT_ON_FIXTURE_SWAP) == {"watchlist": 1}
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 1


def test_the_wipe_never_creates_a_database(tmp_path: Path) -> None:
    db = tmp_path / "absent.db"
    assert wipe_per_company_tables(db) == {}
    assert not db.exists()


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
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Every store on one temp DB, every registered table created, and the
    vector, Honcho and skills layers stubbed (side effects, not under test)."""
    from openexecutive.agents import overrides
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
    # Only a profile: the wipe must not depend on memory.json or people.yaml.
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
            if table not in existing and table not in _LEGACY_TABLES:
                conn.execute(f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY, body TEXT)')
    dept_registry.invalidate()
    people_registry.invalidate()
    overrides.invalidate_cache()
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
    overrides.invalidate_cache()


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


def _expected_survivors(seeded: dict[str, tuple[Any, ...]], *kept: dict[str, str]) -> set[str]:
    keep = set(COMPANY_INDEPENDENT_TABLES).union(*kept)
    return keep & set(seeded)


def test_every_registered_table_got_a_seeded_row(env: Any) -> None:
    """Control for the path tests: they prove nothing about a table with no row."""
    seeded = _seed_every_table(env.db)
    assert set(seeded) == (set(PER_COMPANY_TABLES) - _LEGACY_TABLES) | set(COMPANY_INDEPENDENT_TABLES)
    assert _surviving(env.db, seeded) == set(seeded)


def test_blank_slot_switch_empties_every_per_company_table(env: Any) -> None:
    from openexecutive.clients import slots

    seeded = _seed_every_table(env.db)
    slot = env.settings.company_profile_path.parent / "_client_slots" / "fresh"
    slot.mkdir(parents=True)
    asyncio.run(slots._restore_slot_state(env.settings, slot))

    assert _surviving(env.db, seeded) == _expected_survivors(seeded, KEPT_ON_EVERY_SWAP)


def test_factory_reset_empties_every_per_company_table(env: Any) -> None:
    seeded = _seed_every_table(env.db)
    asyncio.run(fixture_loader.reset_all_state(env.settings))

    assert _surviving(env.db, seeded) == _expected_survivors(seeded, KEPT_ON_EVERY_SWAP)


def test_fixture_load_empties_every_per_company_table(env: Any) -> None:
    """A fixture with only profile.yaml: the wipe must not hang off a seeder."""
    seeded = _seed_every_table(env.db)
    summary = asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert summary["auto_snapshot_taken"] is True

    # workspace_settings is kept from the wipe only because the path sets it
    # itself, from the defaults: the seeded row must not survive either.
    assert _surviving(env.db, seeded) == _expected_survivors(
        seeded, KEPT_ON_EVERY_SWAP, KEPT_ON_FIXTURE_SWAP
    ) - {"workspace_settings"}


def test_fixture_unload_empties_every_per_company_table(env: Any) -> None:
    """The demo's rows do not outlive the demo: unload restores only what the
    backup's files carry, so everything the demo wrote elsewhere goes."""
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    seeded = _seed_every_table(env.db)
    asyncio.run(fixture_loader.unload_fixture(env.settings))

    # workspace_settings comes back from the backup's workspace.yaml.
    assert _surviving(env.db, seeded) == _expected_survivors(
        seeded, KEPT_ON_EVERY_SWAP, KEPT_ON_FIXTURE_SWAP
    ) - {"workspace_settings"}


def test_fixture_load_drops_the_cached_council_overrides(env: Any) -> None:
    """The overrides are cached in-process; a wiped row must stop applying."""
    from openexecutive.agents import overrides

    overrides.set_override("finance", model="the-users-model", model_set=True, db_path=env.db)
    assert overrides.get_override("finance", db_path=env.db) is not None
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert overrides.get_override("finance", db_path=env.db) is None
