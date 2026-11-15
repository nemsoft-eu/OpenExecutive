"""post_progress_update over the real mounted /mcp app.

Drives MCP's streamable HTTP JSON-RPC through the FastAPI app ``create_app``
builds (shared-secret gate, ``caller_gate``, the ``/mcp`` mount), so the
token is read from the header the client actually sent. The relay takes the
real send path to Slack with the Slack client stubbed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings

from openexecutive.audit import logger as audit_logger
from openexecutive.integrations import progress_sources
from openexecutive.memory import episodic
from openexecutive.people import registry as people_registry
from openexecutive.people import store as people_store
from openexecutive.scheduler import runner

VECTORS = json.loads(
    (Path(__file__).parents[1] / "unit" / "caller_assertion_vectors.json").read_text()
)
SOURCE = "marketing-planner"
# the wire name, stated here rather than read from the server
HEADER = "x-source-token"
# a fixed test token; only its hash is configured
TOKEN = "progress-test-token-0123456789abcdefghijklmnop"
ARGS = {"topic": "GTM plan", "status": "on_track", "summary": "Positioning draft is done.\nChannel plan next."}


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Every store this path writes on one temp DB: episodic (progress
    updates, held messages, sends), the roster, and the audit logger, which
    keeps its own path. One founder, on Slack."""
    path = tmp_path / "progress_mcp.db"
    monkeypatch.setattr(episodic, "DB_PATH", path)
    monkeypatch.setattr(people_store, "DB_PATH", path)
    episodic.initialize_db(path)
    people_store.initialize_db(path)
    monkeypatch.setattr(audit_logger, "_default_logger", audit_logger.AuditLogger(db_path=path))
    # the slot and fixture sentinels on temp paths: a developer's own live
    # slot must not turn the feature off
    from openexecutive.cli import fixture_loader
    from openexecutive.clients import slots

    monkeypatch.setattr(slots, "_active_client_sentinel", lambda _s: tmp_path / ".active_client")
    monkeypatch.setattr(fixture_loader, "_fixture_active_sentinel", lambda _s: tmp_path / ".fixture_active")
    people_registry.invalidate()
    people_store.upsert_person(
        full_name="Ada Founder", is_principal=True, slack_user_id="UADA", email="ada@co.example"
    )
    people_registry.invalidate()
    yield path
    people_registry.invalidate()


@pytest.fixture
def slack(db: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AsyncMock]:
    """The source configured and Slack on, with its client stubbed; the
    email leg fails the test if taken. Returns the ``chat_postMessage`` stub."""
    from openexecutive import config

    monkeypatch.setenv(progress_sources.ENV, f"{SOURCE}:{progress_sources.token_hash(TOKEN)}")
    real = config.get_settings()
    monkeypatch.setattr(
        config, "get_settings", lambda: real.model_copy(update={"slack_bot_token": "xoxb-test"})
    )
    monkeypatch.setattr(
        "openexecutive.integrations.conversations.record_delivery", lambda *a, **k: "c1"
    )

    async def no_email(*_a: Any, **_k: Any) -> str | None:
        raise AssertionError("a relay must never take the email leg")

    monkeypatch.setattr(runner, "_email_principal", no_email)
    post = AsyncMock(return_value={"ok": True, "ts": "1.0"})

    class _Client:
        def __init__(self, token: str) -> None:
            self.token = token

        chat_postMessage = post

    with patch("slack_sdk.web.async_client.AsyncWebClient", _Client):
        yield post


def _app(monkeypatch: pytest.MonkeyPatch, *, signed: bool = False, secret: str | None = None) -> Any:
    """The API app, and the /mcp endpoint whose session manager each request
    replaces (a manager runs once per instance, as in
    ``test_mcp_server_caller``)."""
    from openexecutive.api.main import create_app
    from openexecutive.mcp_server import server as mcp_server

    if secret:
        monkeypatch.setenv("BACKEND_SHARED_SECRET", secret)
    else:
        monkeypatch.delenv("BACKEND_SHARED_SECRET", raising=False)
    if signed:
        monkeypatch.setenv("CALLER_ASSERTION_PUBLIC_KEYS", VECTORS["public_keys"])
    else:
        monkeypatch.delenv("CALLER_ASSERTION_PUBLIC_KEYS", raising=False)
    app = create_app()
    endpoint = next(
        getattr(r, "endpoint", None) or getattr(r, "app", None)
        for r in mcp_server._http_app.routes
        if hasattr(getattr(r, "endpoint", None) or getattr(r, "app", None), "session_manager")
    )
    monkeypatch.setattr(endpoint, "session_manager", endpoint.session_manager)
    return app, endpoint


def _post(app: Any, endpoint: Any, headers: dict[str, str], args: dict[str, Any] = ARGS) -> dict[str, Any]:
    """One ``tools/call`` of post_progress_update; the JSON-RPC result."""
    from openexecutive.mcp_server import server as mcp_server

    manager = StreamableHTTPSessionManager(
        app=mcp_server.mcp._mcp_server,
        stateless=True,
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    endpoint.session_manager = manager

    async def _go() -> dict[str, Any]:
        async with manager.run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/mcp/",
                    json={
                        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": "post_progress_update", "arguments": args},
                    },
                    headers={
                        "accept": "application/json, text/event-stream",
                        "mcp-protocol-version": "2025-06-18",
                        **headers,
                    },
                )
        assert response.status_code == 200, response.text
        data = [line[5:].strip() for line in response.text.splitlines() if line.startswith("data:")]
        return dict(json.loads(data[-1] if data else response.text)["result"])

    return asyncio.run(_go())


def _ok(result: dict[str, Any]) -> dict[str, Any]:
    assert not result.get("isError"), result
    return dict(json.loads(result["content"][0]["text"]))


def _error(result: dict[str, Any]) -> str:
    assert result.get("isError") is True, result
    return str(result["content"][0]["text"])


def _rows(db: Path, sql: str) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _stored(db: Path) -> list[tuple[Any, ...]]:
    try:
        return _rows(db, "SELECT id, source, topic, relay_state FROM progress_updates ORDER BY id")
    except sqlite3.OperationalError:
        # the table is created on first use
        return []


def _audit_dump(db: Path) -> str:
    # every column of every row, the full drill-down payload included
    return json.dumps(_rows(db, "SELECT * FROM audit_log"), default=str)


@pytest.mark.parametrize("signed", [False, True], ids=["signed-callers-off", "signed-callers-on"])
def test_a_valid_token_stores_and_relays(
    db: Path, slack: AsyncMock, monkeypatch: pytest.MonkeyPatch, signed: bool
) -> None:
    # the same with signed callers on (a request with no assertion is a
    # service) and off: the tool never reads the caller
    app, endpoint = _app(monkeypatch, signed=signed)
    out = _ok(_post(app, endpoint, {HEADER: TOKEN}))
    assert out == {"id": 1, "stored": True, "relay": "accepted"}
    assert _stored(db) == [(1, SOURCE, "GTM plan", "sent")]
    slack.assert_awaited_once()
    text = slack.await_args.kwargs["text"]
    assert text.startswith(f"Progress update #1 from {SOURCE}")
    assert "> Positioning draft is done." in text


def test_a_repeat_is_a_duplicate_and_sends_nothing(
    db: Path, slack: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, endpoint = _app(monkeypatch)
    _ok(_post(app, endpoint, {HEADER: TOKEN}))
    again = _ok(_post(app, endpoint, {HEADER: TOKEN}))
    assert again == {"id": 1, "stored": False, "relay": "duplicate"}
    slack.assert_awaited_once()


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="no-token"),
        pytest.param({"x-source-token": "not-the-token"}, id="wrong-token"),
        pytest.param({"x-source-token": ""}, id="empty-token"),
    ],
)
def test_without_the_token_the_tool_refuses(
    db: Path, slack: AsyncMock, monkeypatch: pytest.MonkeyPatch, headers: dict[str, str]
) -> None:
    from openexecutive.mcp_server.server import PROGRESS_REFUSED

    app, endpoint = _app(monkeypatch)
    assert PROGRESS_REFUSED in _error(_post(app, endpoint, headers))
    assert _stored(db) == []
    slack.assert_not_awaited()


def test_the_token_is_not_an_argument(db: Path, slack: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    # a token in the arguments is not one in the header: the call is refused
    # (an unknown argument is ignored by the schema) and nothing is stored
    from openexecutive.mcp_server.server import PROGRESS_REFUSED

    app, endpoint = _app(monkeypatch)
    out = _post(app, endpoint, {}, {**ARGS, "x_source_token": TOKEN, "source": SOURCE})
    assert PROGRESS_REFUSED in _error(out)
    assert _stored(db) == []


def test_with_the_feature_off_the_tool_refuses(
    db: Path, slack: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive import config
    from openexecutive.mcp_server.server import PROGRESS_REFUSED

    real = config.get_settings()
    monkeypatch.setattr(config, "get_settings", lambda: real.model_copy(update={"progress_sources": ""}))
    app, endpoint = _app(monkeypatch)
    assert PROGRESS_REFUSED in _error(_post(app, endpoint, {HEADER: TOKEN}))
    assert _stored(db) == []


@pytest.mark.parametrize(
    "extra",
    [pytest.param({}, id="no-actions-header"), pytest.param({"x-mcp-client-actions": "on"}, id="actions-on")],
)
def test_behind_a_shared_secret_the_token_works_and_may_act_does_not_matter(
    db: Path, slack: AsyncMock, monkeypatch: pytest.MonkeyPatch, extra: dict[str, str]
) -> None:
    app, endpoint = _app(monkeypatch, secret="s3cret")
    headers = {"x-api-key": "s3cret", HEADER: TOKEN, **extra}
    assert _ok(_post(app, endpoint, headers))["relay"] == "accepted"
    slack.assert_awaited_once()


def test_the_actions_header_does_not_stand_in_for_the_token(
    db: Path, slack: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.mcp_server.server import PROGRESS_REFUSED

    app, endpoint = _app(monkeypatch, secret="s3cret")
    out = _post(app, endpoint, {"x-api-key": "s3cret", "x-mcp-client-actions": "on"})
    assert PROGRESS_REFUSED in _error(out)
    slack.assert_not_awaited()


def test_a_guard_rejection_names_the_field(db: Path, slack: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    app, endpoint = _app(monkeypatch)
    out = _post(app, endpoint, {HEADER: TOKEN}, {**ARGS, "summary": "Mail me at a@b.example now."})
    message = _error(out)
    assert "summary:" in message
    assert "a@b.example" not in message
    assert _stored(db) == []
    slack.assert_not_awaited()


def test_a_store_failure_returns_the_fixed_error(
    db: Path, slack: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.integrations import progress_store
    from openexecutive.mcp_server.server import PROGRESS_UNAVAILABLE

    def broken(*_a: Any, **_k: Any) -> Any:
        raise progress_store.StoreUnavailable("database is locked at /srv/oe/episodic_memory.db")

    monkeypatch.setattr(progress_store, "admit", broken)
    app, endpoint = _app(monkeypatch)
    message = _error(_post(app, endpoint, {HEADER: TOKEN}))
    assert PROGRESS_UNAVAILABLE in message
    assert "/srv/oe" not in message
    slack.assert_not_awaited()


def test_the_token_never_reaches_a_log_line_or_audit_row(
    db: Path, slack: AsyncMock, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from openexecutive.integrations import progress_store

    app, endpoint = _app(monkeypatch)
    # the openexecutive logger does not propagate to the root (api.main), and
    # httpx and httpcore are held at WARNING: capture all three at DEBUG
    oe_logger = logging.getLogger("openexecutive")
    oe_logger.addHandler(caplog.handler)
    with ExitStack() as stack:
        stack.callback(oe_logger.removeHandler, caplog.handler)
        for name in (None, "openexecutive", "httpx", "httpcore"):
            stack.enter_context(caplog.at_level(logging.DEBUG, logger=name))
        _ok(_post(app, endpoint, {HEADER: TOKEN}))
        _error(_post(app, endpoint, {HEADER: TOKEN + "x"}))
        monkeypatch.setattr(
            progress_store, "admit",
            lambda *a, **k: (_ for _ in ()).throw(progress_store.StoreUnavailable("down")),
        )
        _error(_post(app, endpoint, {HEADER: TOKEN}, {**ARGS, "topic": "Another topic"}))
    # positive control: the run's own logs were captured, and it wrote the
    # ingest audit row
    assert any(r.name == "openexecutive.mcp_server.server" for r in caplog.records)
    audit = _audit_dump(db)
    assert f"source:{SOURCE}" in audit
    assert TOKEN not in caplog.text
    assert TOKEN not in audit
    assert progress_sources.token_hash(TOKEN) not in caplog.text
    assert progress_sources.token_hash(TOKEN) not in audit
