"""POST /progress-updates through the app ``create_app`` builds: the
shared-secret gate, signed callers and the local-login gate around it, and
the route's own order (feature off, token, body) in front of ``submit``."""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from openexecutive.audit import logger as audit_logger
from openexecutive.integrations import progress_sources, progress_store
from openexecutive.memory import episodic
from openexecutive.people import registry as people_registry
from openexecutive.people import store as people_store
from openexecutive.people.models import Person
from openexecutive.scheduler import runner
from openexecutive.scheduler.runner import PrincipalDelivery

SOURCE = "marketing-planner"
PATH = "/progress-updates"
VECTORS = json.loads(
    (Path(__file__).parents[1] / "unit" / "caller_assertion_vectors.json").read_text()
)


def body(n: int = 1, **changes: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "topic": "GTM plan",
        "status": "on_track",
        "summary": f"Positioning draft {n} is done.\nChannel plan next.",
    }
    out.update(changes)
    return out


def isolate_company_sentinels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The slot and fixture sentinels on temp paths, so a developer's own
    live slot does not turn the feature off."""
    from openexecutive.cli import fixture_loader
    from openexecutive.clients import slots

    monkeypatch.setattr(slots, "_active_client_sentinel", lambda _s: tmp_path / ".active_client")
    monkeypatch.setattr(fixture_loader, "_fixture_active_sentinel", lambda _s: tmp_path / ".fixture_active")


@pytest.fixture(autouse=True)
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Every store the route writes on one temp DB (progress updates, held
    messages, the roster, audit), every gate off, and one founder."""
    path = tmp_path / "episodic.db"
    monkeypatch.setattr(episodic, "DB_PATH", path)
    monkeypatch.setattr(people_store, "DB_PATH", path)
    episodic.initialize_db(path)
    people_store.initialize_db(path)
    people_registry.invalidate()
    monkeypatch.setattr(audit_logger, "_default_logger", audit_logger.AuditLogger(db_path=path))
    for var in ("BACKEND_SHARED_SECRET", "OE_LOCAL_LOGIN", "OE_PUBLIC_DEPLOYMENT", "CALLER_ASSERTION_PUBLIC_KEYS"):
        monkeypatch.delenv(var, raising=False)
    isolate_company_sentinels(tmp_path, monkeypatch)
    people_store.upsert_person(full_name="Ada Founder", is_principal=True, slack_user_id="UADA")
    yield path
    people_registry.invalidate()


@pytest.fixture
def sends(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``deliver_to_person`` stubbed to deliver; the texts it was given."""
    seen: list[str] = []

    async def fake(person: Person, text: str, *, label: str = "", chat_only: bool = False) -> PrincipalDelivery:
        seen.append(text)
        return PrincipalDelivery(True, "slack_dm → U", "delivered", "slack_dm")

    monkeypatch.setattr(runner, "deliver_to_person", fake)
    return seen


@pytest.fixture
def token(monkeypatch: pytest.MonkeyPatch) -> str:
    """A configured source and its token."""
    raw, entry = progress_sources.make_token(SOURCE)
    monkeypatch.setenv(progress_sources.ENV, entry)
    return raw


def client(**kwargs: Any) -> TestClient:
    from openexecutive.api.main import create_app

    return TestClient(create_app(), **kwargs)


def post(c: TestClient, token: str | None, payload: Any = None, **kwargs: Any) -> Any:
    headers = dict(kwargs.pop("headers", {}))
    if token is not None:
        headers["x-source-token"] = token
    if payload is not None:
        kwargs["json"] = payload
    return c.post(PATH, headers=headers, **kwargs)


def stored_count(db: Path) -> int:
    import sqlite3

    conn = sqlite3.connect(db)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM progress_updates").fetchone()[0])
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Stored and repeated
# --------------------------------------------------------------------------- #
def test_a_new_update_is_stored_and_relayed(token: str, sends: list[str], db: Path) -> None:
    res = post(client(), token, body())
    assert res.status_code == 201
    assert res.json() == {"id": res.json()["id"], "stored": True, "relay": "accepted"}
    assert len(sends) == 1 and f"from {SOURCE}" in sends[0]
    assert stored_count(db) == 1


def test_a_repeat_is_200_with_the_first_id_and_sends_nothing(token: str, sends: list[str]) -> None:
    c = client()
    first = post(c, token, body())
    again = post(c, token, body())
    assert again.status_code == 200
    assert again.json() == {"id": first.json()["id"], "stored": False, "relay": "duplicate"}
    assert len(sends) == 1


def test_a_held_copy_reads_accepted_never_held(token: str, monkeypatch: pytest.MonkeyPatch) -> None:
    async def suppressed(person: Person, text: str, *, label: str = "", chat_only: bool = False) -> PrincipalDelivery:
        return PrincipalDelivery(False, "slack_dm → U", "suppressed", "slack_dm")

    monkeypatch.setattr(runner, "deliver_to_person", suppressed)
    res = post(client(), token, body())
    assert res.status_code == 201 and res.json()["relay"] == "accepted"


# --------------------------------------------------------------------------- #
# Feature off, token
# --------------------------------------------------------------------------- #
def test_feature_off_is_404_before_any_token_work(monkeypatch: pytest.MonkeyPatch, sends: list[str]) -> None:
    monkeypatch.delenv(progress_sources.ENV, raising=False)

    def no_token_work(token: str | None) -> str | None:
        raise AssertionError("the token was looked at with the feature off")

    monkeypatch.setattr(progress_sources, "source_for", no_token_work)
    res = post(client(), "anything", body())
    assert res.status_code == 404
    assert sends == []


def test_a_live_client_slot_is_404_and_stores_nothing(
    token: str, sends: list[str], db: Path, tmp_path: Path
) -> None:
    (tmp_path / ".active_client").write_text("acme")
    res = post(client(), token, body())
    assert res.status_code == 404
    assert sends == [] and stored_count(db) == 0


def test_a_slot_going_live_after_the_check_is_404(
    token: str, sends: list[str], db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.integrations import progress_updates as pu

    async def paused(source: str, payload: Any) -> pu.SubmitResult:
        raise pu.Paused("a slot went live")

    monkeypatch.setattr(pu, "submit", paused)
    assert post(client(), token, body()).status_code == 404


@pytest.mark.parametrize(
    ("presented", "payload"),
    [
        (None, body()),
        ("not-the-token", body()),
        ("", body()),
    ],
    ids=["no-token", "wrong-token", "empty-token"],
)
def test_a_missing_or_unknown_token_is_401(
    token: str, sends: list[str], db: Path, presented: str | None, payload: dict[str, Any]
) -> None:
    res = post(client(), presented, payload)
    assert res.status_code == 401
    assert sends == [] and stored_count(db) == 0


def test_a_token_in_the_body_only_is_401(token: str, sends: list[str], db: Path) -> None:
    res = post(client(), None, body(**{"x-source-token": token, "token": token}))
    assert res.status_code == 401
    assert stored_count(db) == 0


def test_two_token_headers_are_401(token: str, sends: list[str]) -> None:
    res = client().post(
        PATH, headers=[("x-source-token", token), ("x-source-token", token)], json=body()
    )
    assert res.status_code == 401


def test_the_body_is_not_read_without_a_valid_token(token: str, sends: list[str]) -> None:
    # an oversized body with a bad token is refused on the token, not the size
    res = post(client(), "not-the-token", content=b"x" * 20_000, headers={"content-type": "application/json"})
    assert res.status_code == 401


# --------------------------------------------------------------------------- #
# Body size
# --------------------------------------------------------------------------- #
def test_a_declared_body_over_8kb_is_413(token: str, sends: list[str], db: Path) -> None:
    res = post(client(), token, body(summary="x" * 9000))
    assert res.status_code == 413
    assert stored_count(db) == 0


def test_a_chunked_body_over_8kb_is_413(token: str, sends: list[str], db: Path) -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(10):
            yield b"x" * 1024

    res = post(client(), token, content=chunks(), headers={"content-type": "application/json"})
    assert "content-length" not in {k.lower() for k in res.request.headers}
    assert res.status_code == 413
    assert stored_count(db) == 0


# --------------------------------------------------------------------------- #
# Rejections, caps, store failure
# --------------------------------------------------------------------------- #
def test_a_guard_rejection_is_422_naming_the_field_without_the_text(token: str, sends: list[str], db: Path) -> None:
    res = post(client(), token, body(summary="Mail ada@example.com about the plan."))
    assert res.status_code == 422
    detail = res.json()["detail"]
    assert detail["field"] == "summary"
    assert "ada@example.com" not in res.text
    assert sends == [] and stored_count(db) == 0


def test_a_source_field_is_422(token: str, sends: list[str], db: Path) -> None:
    res = post(client(), token, body(source="ceo"))
    assert res.status_code == 422
    assert res.json()["detail"]["field"] == "source"
    assert "ceo" not in res.text
    assert stored_count(db) == 0


@pytest.mark.parametrize(
    ("content", "field"),
    [
        (b"{not json", "payload"),
        (b"[]", "payload"),
        (json.dumps({"topic": 5, "status": "done", "summary": "A number for a topic."}).encode(), "topic"),
        (json.dumps({"status": "done", "summary": "No topic at all here."}).encode(), "topic"),
    ],
    ids=["bad-json", "not-an-object", "wrong-type", "missing-field"],
)
def test_a_malformed_body_is_422(token: str, sends: list[str], content: bytes, field: str) -> None:
    res = post(client(), token, content=content, headers={"content-type": "application/json"})
    assert res.status_code == 422
    assert res.json()["detail"]["field"] == field


def test_over_the_daily_cap_is_429(token: str, sends: list[str], monkeypatch: pytest.MonkeyPatch, db: Path) -> None:
    from openexecutive import config

    real = config.get_settings()
    monkeypatch.setattr(
        config, "get_settings", lambda: real.model_copy(update={"progress_max_per_source_per_day": 1})
    )
    c = client()
    assert post(c, token, body(1)).status_code == 201
    res = post(c, token, body(2))
    assert res.status_code == 429
    assert len(sends) == 1 and stored_count(db) == 1


def test_a_store_failure_is_503_and_sends_nothing(
    token: str, sends: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise progress_store.StoreUnavailable("disk I/O error at /secret/path")

    monkeypatch.setattr(progress_store, "admit", broken)
    res = post(client(), token, body())
    assert res.status_code == 503
    assert "/secret/path" not in res.text
    assert sends == []


# --------------------------------------------------------------------------- #
# The gates around it
# --------------------------------------------------------------------------- #
def test_the_exemption_set_gained_this_path_and_nothing_else() -> None:
    from openexecutive.api import main

    exempt = main._UNAUTHENTICATED_PATHS
    assert exempt == {"/health", "/webhook/telegram", "/webhook/google-chat", "/webhook/slack", PATH}
    # under local login, a loopback Host is still required
    assert not main._webhook_verifies_its_caller(PATH)


def test_with_a_shared_secret_the_token_alone_is_enough(
    token: str, sends: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BACKEND_SHARED_SECRET", "s3cret")
    c = client()
    assert post(c, token, body()).status_code == 201
    assert post(c, None, body(2)).status_code == 401
    # every other path still needs x-api-key
    assert c.get("/people").status_code == 401
    assert c.post("/chat", json={"message": "hi"}).status_code == 401
    assert c.get("/people", headers={"x-api-key": "s3cret"}).status_code != 401


def test_with_signed_callers_on_the_token_alone_is_enough(
    token: str, sends: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CALLER_ASSERTION_PUBLIC_KEYS", VECTORS["public_keys"])
    assert post(client(), token, body()).status_code == 201


def test_under_local_login_only_a_loopback_host_gets_through(
    token: str, sends: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OE_LOCAL_LOGIN", "1")
    local = client(base_url="http://127.0.0.1:8000")
    assert post(local, token, body(1)).status_code == 201
    assert post(client(base_url="http://exec.example.com"), token, body(2)).status_code == 403
    res = post(local, token, body(3), headers={"sec-fetch-site": "cross-site"})
    assert res.status_code == 403
    assert len(sends) == 1
