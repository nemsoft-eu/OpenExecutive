"""The web chat route must record which alert ids it showed the model.

`ack_alert` accepts only ids in `session.trusted_alert_ids`. The route is the
only thing that populates them for a browser turn, so if this wiring breaks,
the failure is silent in both directions: the Executive can no longer clear a
card the principal just approved, and (before the guard covered web at all)
it could clear any id an inbound email had written into an alert body.

Unit-level tests of `render_and_trust` cannot catch that — they pass a session
in directly. This drives the real route.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openexecutive.alerts import store as alert_store
from openexecutive.api.routes import chat as chat_route
from openexecutive.memory import episodic, session_store
from openexecutive.memory.company_profile import CompanyProfile
from openexecutive.people import store as people_store


@pytest.fixture(autouse=True)
def _reset_route_state() -> None:
    chat_route._sessions.clear()
    chat_route._last_turn_events.clear()
    chat_route._last_turn_meta.clear()


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    db_path = Path("./episodic_memory.db").resolve()
    monkeypatch.setattr(episodic, "DB_PATH", db_path)
    monkeypatch.setattr(session_store, "DB_PATH", db_path)
    monkeypatch.setattr(people_store, "DB_PATH", tmp_path / "people.db")
    monkeypatch.setattr(alert_store, "DB_PATH", tmp_path / "alerts.db")
    episodic.initialize_db(db_path)
    people_store.initialize_db()
    alert_store.initialize_db(tmp_path / "alerts.db")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from openexecutive.onboarding import profile_builder

    monkeypatch.setattr(profile_builder, "load_or_create_profile", lambda: CompanyProfile())

    from openexecutive.knowledge import retriever as retriever_mod

    monkeypatch.setattr(retriever_mod, "retrieve", lambda **_: "")

    from openexecutive.orchestrator import executive as exec_mod

    class _StubExecutive:
        _THINKING = exec_mod.Executive._THINKING

        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def stream_chat(self, **_kwargs: Any) -> AsyncIterator[str]:
            yield "ok"

        async def stream_chat_with_committee(self, **_kwargs: Any) -> AsyncIterator[str]:
            yield "ok"

    monkeypatch.setattr(exec_mod, "Executive", _StubExecutive)
    return tmp_path


def _post(message: str = "what's on my plate?") -> None:
    app = FastAPI()
    app.include_router(chat_route.router)
    client = TestClient(app)
    resp = client.post("/chat", json={"message": message})
    assert resp.status_code == 200
    _ = resp.text  # drain the SSE stream so the turn completes


def _only_session() -> Any:
    assert len(chat_route._sessions) == 1
    return next(iter(chat_route._sessions.values()))


def test_web_turn_trusts_the_live_alert_it_was_shown(env: Path) -> None:
    people_store.upsert_person(full_name="Alex", is_principal=True)
    live = alert_store.insert_alert(
        source="email", external_id="live-1", severity="high",
        headline="Approve the Q3 budget", body="Needs a decision.",
        db_path=env / "alerts.db",
    )
    assert live is not None

    _post()

    assert _only_session().trusted_alert_ids == {live}


def test_web_turn_does_not_trust_a_closed_alert(env: Path) -> None:
    """A dismissed alert is named in the digest's handled tail so the Executive
    knows it is settled — but it must not become ackable."""
    people_store.upsert_person(full_name="Alex", is_principal=True)
    closed = alert_store.insert_alert(
        source="email", external_id="closed-1", severity="high",
        headline="St. Albans reconciliation gap", body="b",
        db_path=env / "alerts.db",
    )
    assert closed is not None
    alert_store.set_status(closed, "dismissed", db_path=env / "alerts.db")

    _post()

    assert _only_session().trusted_alert_ids == set()


def test_web_turn_with_no_alerts_trusts_nothing(env: Path) -> None:
    people_store.upsert_person(full_name="Alex", is_principal=True)

    _post()

    assert _only_session().trusted_alert_ids == set()


def test_turn_start_clears_a_stale_trusted_set(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A turn that never reaches the recorder must not inherit the last one's set.

    Web sessions are long-lived, so if `render_and_trust` is skipped — the
    to_thread wrapper fails to schedule, the gather is cancelled — the previous
    turn's ids would still be sitting on the session and `ack_alert` would
    accept them. The clear happens before the gather, so it holds regardless.
    """
    people_store.upsert_person(full_name="Alex", is_principal=True)
    live = alert_store.insert_alert(
        source="email", external_id="live-1", severity="high",
        headline="Approve the Q3 budget", body="b", db_path=env / "alerts.db",
    )
    assert live is not None

    _post()
    assert _only_session().trusted_alert_ids == {live}

    # Same session, next turn, digest unavailable.
    from openexecutive.briefing import context as ctx

    def _boom(*_a: object, **_kw: object) -> str:
        raise RuntimeError("store down")

    monkeypatch.setattr(ctx, "render_and_trust", _boom)
    session_id = _only_session().session_id

    app = FastAPI()
    app.include_router(chat_route.router)
    client = TestClient(app)
    resp = client.post("/chat", json={"message": "and now?", "session_id": session_id})
    assert resp.status_code == 200
    _ = resp.text

    assert _only_session().trusted_alert_ids == set()
