"""A fixture load then unload gives the user's people back under NEW ids.

Backup and restore are YAML-only: ``_dump_people`` writes no ids,
``_seed_people`` deletes every row and re-inserts, and ``people.id`` is
``AUTOINCREMENT`` with nothing resetting ``sqlite_sequence``. So the restored
principal is the same person by name and role, but under a fresh id, and any
state still keyed by the old id would point at nobody. The swap paths wipe
that person-keyed state (``memory/company_tables.py``), which is why the
demo and clients pages say it comes back empty.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from openexecutive.cli import fixture_loader
from openexecutive.departments import registry as dept_registry
from openexecutive.departments import store as dept_store
from openexecutive.memory import episodic, honcho_client
from openexecutive.people import store as people_store


class _FakeStore:
    RESEARCH_COLLECTION = "recent_research"
    COMPANY_COLLECTION = "company_docs"

    def __init__(self, **_kw: object) -> None: ...

    def delete_company_docs(self) -> None: ...

    def delete_documents(self, **_kw: object) -> None: ...

    def delete_notion_docs(self) -> None: ...
    def delete_drive_docs(self) -> None: ...
    def delete_onedrive_docs(self) -> None: ...
    def delete_confluence_docs(self) -> None: ...

    def delete_attachment_docs(self) -> None: ...


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    db = tmp_path / "episodic.db"
    for mod in (episodic, people_store, dept_store):
        monkeypatch.setattr(mod, "DB_PATH", db)
    episodic.initialize_db(db)
    people_store.initialize_db(db)
    dept_store.initialize_db(db)
    dept_registry.invalidate()
    monkeypatch.setattr("openexecutive.audit.log_event", lambda *a, **k: None)
    monkeypatch.setattr("openexecutive.knowledge.store.ChromaDBStore", _FakeStore)
    monkeypatch.setattr(
        "openexecutive.knowledge.notion_sync.reset_local_state", lambda **_kw: None
    )

    async def _noop(*_a: object, **_k: object) -> None:
        return None

    monkeypatch.setattr(honcho_client, "delete_workspace_and_reset_client", _noop)
    monkeypatch.setattr(honcho_client, "set_active_workspace_id", lambda *a, **k: None)
    monkeypatch.setattr(honcho_client, "clear_active_workspace_id", lambda *a, **k: None)
    monkeypatch.setattr(honcho_client, "get_active_workspace_id", lambda: "openexec")

    profile = tmp_path / "company" / "profile.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text("name: My Co\n")
    fixture = tmp_path / "fixtures" / "demo"
    fixture.mkdir(parents=True)
    (fixture / "profile.yaml").write_text("name: Demo Co\n")
    (fixture / "people.yaml").write_text(
        "people:\n  - full_name: Demo Boss\n    role: CEO\n    is_principal: true\n"
    )
    monkeypatch.setattr(fixture_loader, "FIXTURES_ROOT", tmp_path / "fixtures")
    return type("S", (), {
        "vector_store_path": tmp_path / "chroma",
        "company_profile_path": profile,
        "honcho_workspace_id": "openexec",
    })()


def test_unload_restores_the_principal_under_a_new_id(settings: Any) -> None:
    before = people_store.upsert_person(full_name="Me Myself", role="Owner", is_principal=True)

    asyncio.run(fixture_loader.load_fixture("demo", settings))
    asyncio.run(fixture_loader.unload_fixture(settings))

    principal = people_store.find_principal_person()
    assert principal is not None
    assert (principal.full_name, principal.role) == ("Me Myself", "Owner")
    # One row each for the user, the demo principal, then the user again.
    assert principal.id == before + 2
