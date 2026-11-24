from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openexecutive.api.routes import evals as evals_route
from openexecutive.utils.public_errors import GENERIC_FAILURE, PublicError, public_error


def test_public_error_never_carries_the_exception_text() -> None:
    exc = RuntimeError("token=abc123 at /srv/data/secret.db")
    assert public_error(exc) == GENERIC_FAILURE
    assert "abc123" not in public_error(exc)


def test_a_public_error_keeps_its_own_text() -> None:
    assert public_error(PublicError("Add a topic first.")) == "Add a topic first."
    assert public_error(PublicError("  ")) == GENERIC_FAILURE


def test_a_failed_eval_run_shows_and_stores_plain_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed: dict[str, str] = {}

    async def _crash(**_kw: Any) -> AsyncIterator[dict]:
        raise RuntimeError("cannot open /srv/data/secret.db")
        yield {}  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(evals_route, "load_scenarios", lambda **_kw: [{"id": "x_001"}])
    monkeypatch.setattr(evals_route, "create_eval_run", lambda **_kw: None)
    monkeypatch.setattr(
        evals_route, "fail_eval_run", lambda *, run_id, error: failed.update(error=error)
    )
    monkeypatch.setattr(evals_route, "run_scenarios", _crash)

    app = FastAPI()
    app.include_router(evals_route.router)
    r = TestClient(app).post("/evals/runs", json={"kind": "chat"})

    assert "secret.db" not in r.text
    assert f'"message": "{GENERIC_FAILURE}"' in r.text
    assert failed["error"] == GENERIC_FAILURE
