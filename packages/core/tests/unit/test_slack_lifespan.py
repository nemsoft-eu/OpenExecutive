"""The Slack Socket Mode listener starts and stops with the FastAPI app.

Regression cover for #131: `run_slack_bot()` existed but nothing in the
application lifecycle ever called it, so `make dev` brought up uvicorn and the
UI while Slack silently never listened.

These tests drive the real `create_app()` lifespan through `TestClient` as a
context manager (entering it is what runs startup/shutdown) and assert on a
faked `create_slack_app`, so no Slack credentials or network are involved.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import suppress
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from openexecutive.api.main import create_app


@pytest.fixture
def fake_handler() -> MagicMock:
    """Stand-in for `AsyncSocketModeHandler`."""
    handler = MagicMock()
    handler.connect_async = AsyncMock()
    handler.close_async = AsyncMock()
    return handler


def _settings_with_slack(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    """Make `get_settings()` inside the lifespan report the given Slack tokens.

    A real `Settings` copy, not a mock — the lifespan reads dozens of other
    fields off it, and every one of them must keep its true value.
    """
    from openexecutive.config import get_settings

    stub = get_settings().model_copy(
        update={
            "slack_bot_token": overrides.get("slack_bot_token", "xoxb-test"),
            "slack_app_token": overrides.get("slack_app_token", "xapp-test"),
        }
    )
    monkeypatch.setattr("openexecutive.config.get_settings", lambda: stub)


def test_lifespan_connects_and_closes_slack_socket(
    monkeypatch: pytest.MonkeyPatch, fake_handler: MagicMock
) -> None:
    """With both tokens set, startup connects the socket and shutdown closes it."""
    _settings_with_slack(monkeypatch)
    create = AsyncMock(return_value=(MagicMock(), fake_handler))
    monkeypatch.setattr(
        "openexecutive.integrations.slack_bot.create_slack_app", create
    )

    with TestClient(create_app()) as client:
        # Startup has run by the time the context manager yields.
        create.assert_awaited_once()
        # Called, not yet necessarily awaited: create_task() records the call
        # synchronously when it evaluates its argument.
        fake_handler.connect_async.assert_called_once()
        fake_handler.close_async.assert_not_awaited()
        assert client.get("/health").status_code == 200

    # Exiting the context manager runs shutdown. By now the background connect
    # task has actually been driven, so assert it was *awaited*, not just called.
    fake_handler.connect_async.assert_awaited_once()
    fake_handler.close_async.assert_awaited_once()


@pytest.mark.parametrize(
    ("bot_token", "app_token"),
    [(None, "xapp-test"), ("xoxb-test", None), (None, None)],
)
def test_lifespan_skips_slack_without_both_tokens(
    monkeypatch: pytest.MonkeyPatch,
    fake_handler: MagicMock,
    bot_token: str | None,
    app_token: str | None,
) -> None:
    """Socket Mode needs both tokens — a partial config must not half-start it.

    This is also what keeps every other full-app test isolated: conftest never
    sets the Slack tokens, so the block stays inert.
    """
    _settings_with_slack(monkeypatch, slack_bot_token=bot_token, slack_app_token=app_token)
    create = AsyncMock(return_value=(MagicMock(), fake_handler))
    monkeypatch.setattr(
        "openexecutive.integrations.slack_bot.create_slack_app", create
    )

    with TestClient(create_app()):
        pass

    create.assert_not_awaited()
    fake_handler.connect_async.assert_not_awaited()
    fake_handler.close_async.assert_not_awaited()


def test_slack_connect_failure_does_not_block_startup(
    monkeypatch: pytest.MonkeyPatch, fake_handler: MagicMock
) -> None:
    """A bad token must degrade to "no Slack", never take the whole app down."""
    _settings_with_slack(monkeypatch)
    fake_handler.connect_async = AsyncMock(side_effect=RuntimeError("invalid auth"))
    monkeypatch.setattr(
        "openexecutive.integrations.slack_bot.create_slack_app",
        AsyncMock(return_value=(MagicMock(), fake_handler)),
    )

    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200

    # The handler object still exists, so shutdown must still release it —
    # a half-open socket-mode client left unclosed keeps aiohttp resources
    # alive and hangs the process.
    fake_handler.close_async.assert_awaited_once()


def test_slack_connect_that_never_returns_does_not_block_startup(
    monkeypatch: pytest.MonkeyPatch, fake_handler: MagicMock
) -> None:
    """The real hazard: `connect_async()` does not fail fast.

    On a bad app token or an unreachable Slack it retries internally and never
    returns. Awaiting it in the lifespan hung boot forever (observed: startup
    never completed, process had to be killed). It must run as a background
    task instead, so startup finishes and shutdown cancels it.
    """
    _settings_with_slack(monkeypatch)
    never_returns = asyncio.Event()  # never set

    async def _hang() -> None:
        await never_returns.wait()

    fake_handler.connect_async = AsyncMock(side_effect=_hang)
    monkeypatch.setattr(
        "openexecutive.integrations.slack_bot.create_slack_app",
        AsyncMock(return_value=(MagicMock(), fake_handler)),
    )

    # Run the whole lifespan on a worker thread so a regression FAILS this
    # test instead of hanging the suite forever (there is no pytest-timeout
    # plugin here, and a blocked startup never reaches an assert).
    done = threading.Event()
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            with TestClient(create_app()) as client:
                box["status"] = client.get("/health").status_code
        except BaseException as exc:  # noqa: BLE001 - reported below
            box["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    finished = done.wait(timeout=60)

    assert finished, "startup blocked on Slack connect — lifespan awaited it directly"
    assert "error" not in box, f"lifespan raised: {box.get('error')!r}"
    assert box["status"] == 200

    # Shutdown cancels the still-pending connect and closes the handler.
    fake_handler.close_async.assert_awaited_once()


def test_slack_close_failure_does_not_break_shutdown(
    monkeypatch: pytest.MonkeyPatch, fake_handler: MagicMock
) -> None:
    """A stalled disconnect is logged, not propagated out of the lifespan."""
    _settings_with_slack(monkeypatch)
    fake_handler.close_async = AsyncMock(side_effect=RuntimeError("socket gone"))
    monkeypatch.setattr(
        "openexecutive.integrations.slack_bot.create_slack_app",
        AsyncMock(return_value=(MagicMock(), fake_handler)),
    )

    with TestClient(create_app()):
        pass  # exiting must not raise

    fake_handler.close_async.assert_awaited_once()


def test_shutdown_is_bounded_even_if_the_socket_refuses_to_die(
    monkeypatch: pytest.MonkeyPatch, fake_handler: MagicMock
) -> None:
    """Shutdown must honour its deadline, not hang, on a wedged socket.

    Regression guard. The first version of this teardown did
    ``with suppress(CancelledError, Exception): await slack_connect_task`` —
    but `asyncio.wait_for` enforces its deadline BY cancelling the coroutine,
    so that suppress() swallowed the deadline and the lifespan hung forever.
    Here the connect task ignores cancellation and close_async() never
    returns; shutdown must still finish, on the timeout path.
    """
    _settings_with_slack(monkeypatch)
    monkeypatch.setattr("openexecutive.api.main.SLACK_SHUTDOWN_TIMEOUT_S", 0.5)

    async def _ignores_cancellation() -> None:
        # Swallows cancellation for well past the deadline, then exits. Short
        # sleeps, not one long one: a swallowed cancel would otherwise leave
        # the task parked in the next sleep forever, and an immortal task
        # wedges the event loop's own teardown rather than testing shutdown.
        for _ in range(60):
            with suppress(asyncio.CancelledError):
                await asyncio.sleep(0.05)

    async def _never_returns() -> None:
        await asyncio.sleep(3600)

    fake_handler.connect_async = AsyncMock(side_effect=_ignores_cancellation)
    fake_handler.close_async = AsyncMock(side_effect=_never_returns)
    monkeypatch.setattr(
        "openexecutive.integrations.slack_bot.create_slack_app",
        AsyncMock(return_value=(MagicMock(), fake_handler)),
    )

    done = threading.Event()
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            with TestClient(create_app()) as client:
                box["status"] = client.get("/health").status_code
        except BaseException as exc:  # noqa: BLE001 - reported below
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True).start()

    assert done.wait(timeout=30), (
        "lifespan hung on shutdown — the wait_for deadline was swallowed"
    )
    assert "error" not in box, f"lifespan raised: {box.get('error')!r}"
    assert box["status"] == 200
