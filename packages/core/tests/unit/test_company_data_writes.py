"""The company simulator and client-slot routes replace or wipe the whole
company, so their writes are the principal's, as the workspace settings and
the People list are. Reads stay open to the team."""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openexecutive.api.routes import clients as clients_route
from openexecutive.api.routes import fixtures as fixtures_route
from openexecutive.memory import episodic
from openexecutive.people import install_owner
from openexecutive.people import registry as people_registry
from openexecutive.people import store as people_store

OWNER = {"x-caller-email": "pat@example.com"}
TEAMMATE = {"x-caller-email": "sam@example.com"}

WRITES = [
    ("post", "/fixtures/reset"),
    ("post", "/fixtures/snapshot"),
    ("post", "/fixtures/unload"),
    ("post", "/fixtures/acme/load"),
    ("post", "/fixtures/generate"),
    ("post", "/fixtures"),
    ("delete", "/fixtures/acme"),
    ("post", "/clients"),
    ("post", "/clients/save"),
    ("post", "/clients/acme/activate"),
    ("patch", "/clients/acme"),
    ("delete", "/clients/acme"),
]


@pytest.fixture()
def people_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "people.db"
    monkeypatch.setattr(people_store, "DB_PATH", path)
    monkeypatch.setattr(episodic, "DB_PATH", path)
    monkeypatch.setattr(install_owner, "away_from_own_company", lambda: False)
    people_store.initialize_db(path)
    people_registry.invalidate()
    yield path
    people_registry.invalidate()


@pytest.fixture()
def client(people_db: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    calls: list[str] = []

    async def _reset(_settings: Any, **_kw: Any) -> dict[str, Any]:
        calls.append("reset")
        return {"reset": True}

    monkeypatch.setattr("openexecutive.cli.fixture_loader.reset_all_state", _reset)
    app = FastAPI()
    app.include_router(fixtures_route.router)
    app.include_router(clients_route.router)
    test_client = TestClient(app, raise_server_exceptions=False)
    test_client.calls = calls  # type: ignore[attr-defined]
    return test_client


def _team() -> None:
    people_store.upsert_person(full_name="Pat Owner", is_principal=True, email="pat@example.com")
    people_store.upsert_person(full_name="Sam Teammate", email="sam@example.com")
    people_registry.invalidate()


@pytest.mark.parametrize(("method", "path"), WRITES)
def test_a_teammate_cannot_reset_load_or_switch_the_company(
    client: TestClient, method: str, path: str
) -> None:
    _team()
    response = client.request(method.upper(), path, headers=TEAMMATE, json={})
    assert response.status_code == 403
    assert client.calls == []  # type: ignore[attr-defined]


def test_the_principal_can_reset(client: TestClient) -> None:
    _team()
    response = client.post("/fixtures/reset", headers=OWNER)
    assert response.status_code == 200
    assert client.calls == ["reset"]  # type: ignore[attr-defined]


def test_a_new_install_with_no_principal_can_reset(client: TestClient) -> None:
    response = client.post("/fixtures/reset", headers=TEAMMATE)
    assert response.status_code == 200


def test_reads_stay_open_to_the_team(client: TestClient) -> None:
    _team()
    assert client.get("/fixtures/status", headers=TEAMMATE).status_code != 403
    assert client.get("/clients", headers=TEAMMATE).status_code != 403


def test_the_principal_is_kept_as_the_install_owner(client: TestClient, people_db: Path) -> None:
    _team()
    client.post("/fixtures/reset", headers=OWNER)
    assert install_owner.get(people_db) == "pat@example.com"


def test_the_owner_can_switch_back_from_a_loaded_company(
    client: TestClient, people_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_owner.remember("pat@example.com", people_db)
    # A simulator company is live, with its own principal on the roster.
    people_store.upsert_person(full_name="Fixture Boss", is_principal=True, email="boss@fixture.test")
    people_registry.invalidate()
    monkeypatch.setattr(install_owner, "away_from_own_company", lambda: True)
    assert client.post("/fixtures/reset", headers=OWNER).status_code == 200
    assert client.post("/fixtures/reset", headers=TEAMMATE).status_code == 403


def test_after_a_reset_only_the_kept_owner_can_set_up_again(client: TestClient, people_db: Path) -> None:
    install_owner.remember("pat@example.com", people_db)
    assert client.post("/fixtures/reset", headers=TEAMMATE).status_code == 403
    assert client.post("/fixtures/reset", headers=OWNER).status_code == 200


def test_a_caller_naming_no_one_still_runs_an_owned_install(client: TestClient, people_db: Path) -> None:
    """With no sign-in (open mode, no caller header) the caller is the
    principal, so an empty roster with an owner kept doesn't lock them out."""
    install_owner.remember("pat@example.com", people_db)
    assert client.post("/fixtures/reset").status_code == 200


def test_whoever_sets_up_an_unclaimed_install_can_switch_back(
    client: TestClient, people_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No principal yet: the person who changes the company's data is kept
    as the owner, so a client's own principal can't lock them out later."""
    client.post("/fixtures/reset", headers=OWNER)
    assert install_owner.get(people_db) == "pat@example.com"
    people_store.upsert_person(full_name="Client CEO", is_principal=True, email="ceo@acme.test")
    people_registry.invalidate()
    monkeypatch.setattr(install_owner, "away_from_own_company", lambda: True)
    assert client.post("/fixtures/reset", headers=OWNER).status_code == 200


def test_an_incidental_write_on_an_unclaimed_install_keeps_no_owner(client: TestClient, people_db: Path) -> None:
    client.post("/fixtures/snapshot", headers=TEAMMATE)
    assert install_owner.get(people_db) is None


def test_a_failed_write_keeps_no_owner(client: TestClient, people_db: Path) -> None:
    assert client.post("/fixtures/no-such-company/load", headers=TEAMMATE).status_code >= 400
    assert install_owner.get(people_db) is None


def test_with_no_owner_kept_the_saved_copys_principal_can_switch_back(
    client: TestClient, people_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A company loaded before an owner was ever kept: the principal of the
    saved copy of the owner's own company can still switch back."""
    _team()
    copy = tmp_path / "backup" / "state.db"
    copy.parent.mkdir()
    shutil.copy(people_db, copy)
    monkeypatch.setattr("openexecutive.cli.fixture_loader._user_backup_dir", lambda _s: copy.parent)
    with sqlite3.connect(str(people_db)) as conn:
        conn.execute("DELETE FROM people")
    people_store.upsert_person(full_name="Fixture Boss", is_principal=True, email="boss@fixture.test")
    people_registry.invalidate()
    monkeypatch.setattr(install_owner, "away_from_own_company", lambda: True)
    assert client.post("/fixtures/reset", headers=TEAMMATE).status_code == 403
    assert client.post("/fixtures/reset", headers=OWNER).status_code == 200
