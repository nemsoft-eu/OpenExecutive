"""Slack bot embedded in the API lifespan: async handler, bot lifecycle, wiring.

The bot runs on async Bolt, so listeners are asyncio tasks on the API loop
(where the MCP gateway lives) and nothing runs on threads of its own. These
tests pin the token checks, the connect/retry/stop lifecycle, the lifespan
wiring, and the async message path, without a real Slack connection.
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-not-used")

from fastapi.testclient import TestClient  # noqa: E402

from openexecutive.integrations import slack_bot  # noqa: E402

# --------------------------------------------------------------------------
# token checks
# --------------------------------------------------------------------------


@pytest.fixture
def slack_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")


@pytest.mark.asyncio
async def test_create_slack_app_verifies_both_tokens(
    slack_tokens: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # restored afterwards so the resolved id does not leak into other tests
    monkeypatch.setattr(slack_bot, "_bot_user_id", None)
    with (
        patch(
            "slack_sdk.web.async_client.AsyncWebClient.auth_test",
            new=AsyncMock(return_value={"user_id": "UBOT"}),
        ) as auth_test,
        patch(
            "slack_sdk.web.async_client.AsyncWebClient.apps_connections_open",
            new=AsyncMock(return_value={"url": "wss://example"}),
        ) as connections_open,
    ):
        await slack_bot.create_slack_app()

    auth_test.assert_awaited_once()
    connections_open.assert_awaited_once_with(app_token="xapp-test")
    assert slack_bot._bot_user_id == "UBOT"


@pytest.mark.asyncio
async def test_create_slack_app_propagates_rejected_app_token(
    slack_tokens: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Socket Mode client would retry this forever without raising, so the
    probe must surface it."""
    # auth_test succeeds before the app token is rejected and sets the cache
    monkeypatch.setattr(slack_bot, "_bot_user_id", None)
    with (
        patch(
            "slack_sdk.web.async_client.AsyncWebClient.auth_test",
            new=AsyncMock(return_value={"user_id": "UBOT"}),
        ),
        patch(
            "slack_sdk.web.async_client.AsyncWebClient.apps_connections_open",
            new=AsyncMock(side_effect=RuntimeError("not_allowed_token_type")),
        ),
        pytest.raises(RuntimeError, match="not_allowed_token_type"),
    ):
        await slack_bot.create_slack_app()


# --------------------------------------------------------------------------
# message event routing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ({"bot_id": "B1", "channel_type": "im", "text": "hi"}, None),
        ({"subtype": "message_changed", "channel_type": "im", "text": "hi"}, None),
        ({"channel_type": "im", "text": "hi", "ts": "1.0"}, "dm"),
        ({"text": "<@UBOT> hi", "ts": "2.0", "thread_ts": "1.0"}, None),
        ({"text": "hi", "ts": "1.0"}, None),
        ({"text": "hi", "ts": "1.0", "thread_ts": "1.0"}, None),
        ({"text": "hi", "ts": "2.0", "thread_ts": "1.0"}, "thread_continuation"),
    ],
    ids=[
        "bot-message",
        "edit",
        "dm",
        "mention-left-to-app_mention",
        "top-level-channel-message",
        "thread-starter",
        "threaded-reply",
    ],
)
def test_message_event_mode(
    monkeypatch: pytest.MonkeyPatch, event: dict[str, Any], expected: str | None
) -> None:
    monkeypatch.setattr(slack_bot, "_bot_user_id", "UBOT")
    assert slack_bot._message_event_mode(event) == expected


# --------------------------------------------------------------------------
# EmbeddedSlackBot lifecycle
# --------------------------------------------------------------------------


class _FakeHandler:
    def __init__(self, connect_error: BaseException | None = None) -> None:
        self.connect_error = connect_error
        self.connected = False
        self.closed = False

    async def connect_async(self) -> None:
        if self.connect_error is not None:
            raise self.connect_error
        self.connected = True

    async def close_async(self) -> None:
        self.closed = True


@pytest.fixture
def fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(slack_bot, "_CONNECT_RETRY_INITIAL_S", 0.0)


def _script_app(monkeypatch: pytest.MonkeyPatch, outcomes: list[BaseException | None]) -> list[int]:
    """create_slack_app raises each exception in turn, then succeeds."""
    calls: list[int] = []

    async def _fake_create() -> str:
        calls.append(1)
        outcome = outcomes.pop(0) if outcomes else None
        if outcome is not None:
            raise outcome
        return "app"

    monkeypatch.setattr(slack_bot, "create_slack_app", _fake_create)
    return calls


def _script_handler(monkeypatch: pytest.MonkeyPatch, handler: _FakeHandler) -> list[tuple[Any, Any]]:
    built: list[tuple[Any, Any]] = []

    def _factory(app: Any, token: Any) -> _FakeHandler:
        built.append((app, token))
        return handler

    monkeypatch.setattr(slack_bot, "_socket_mode_handler", _factory)
    return built


@pytest.mark.asyncio
async def test_run_connects_and_stop_closes(
    monkeypatch: pytest.MonkeyPatch, slack_tokens: None
) -> None:
    _script_app(monkeypatch, [])
    handler = _FakeHandler()
    built = _script_handler(monkeypatch, handler)
    bot = slack_bot.EmbeddedSlackBot()

    await bot.run()
    assert built == [("app", "xapp-test")]
    assert handler.connected

    await bot.stop()
    assert handler.closed
    await bot.stop()  # idempotent


@pytest.mark.asyncio
async def test_run_retries_transient_token_check_failure(
    monkeypatch: pytest.MonkeyPatch, slack_tokens: None, fast_retry: None
) -> None:
    calls = _script_app(monkeypatch, [ConnectionError("slack.com unreachable")])
    handler = _FakeHandler()
    built = _script_handler(monkeypatch, handler)

    await slack_bot.EmbeddedSlackBot().run()

    assert len(calls) == 2
    assert len(built) == 1  # no socket session until the tokens are verified
    assert handler.connected


@pytest.mark.parametrize("code", slack_bot._FATAL_AUTH_ERRORS)
@pytest.mark.asyncio
async def test_run_gives_up_on_permanent_errors(
    monkeypatch: pytest.MonkeyPatch, slack_tokens: None, fast_retry: None, code: str
) -> None:
    calls = _script_app(
        monkeypatch, [RuntimeError(f"The server responded with: {{'ok': False, 'error': '{code}'}}")]
    )
    built = _script_handler(monkeypatch, _FakeHandler())

    await slack_bot.EmbeddedSlackBot().run()

    assert len(calls) == 1  # no retry
    assert built == []  # never opened a socket session


@pytest.mark.asyncio
async def test_cancelled_connect_closes_handler(
    monkeypatch: pytest.MonkeyPatch, slack_tokens: None
) -> None:
    _script_app(monkeypatch, [])
    connecting = asyncio.Event()

    class _Hanging(_FakeHandler):
        async def connect_async(self) -> None:
            connecting.set()
            await asyncio.Event().wait()

    handler = _Hanging()
    _script_handler(monkeypatch, handler)
    bot = slack_bot.EmbeddedSlackBot()

    task = asyncio.create_task(bot.run())
    await asyncio.wait_for(connecting.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert handler.closed


# --------------------------------------------------------------------------
# async message path
# --------------------------------------------------------------------------


@pytest.fixture
def chat_path() -> Iterator[dict[str, Any]]:
    """Patch everything _handle_message calls beyond Slack itself."""
    executive = MagicMock()
    executive.chat = AsyncMock(return_value="Here is your answer.")
    with (
        patch("openexecutive.people.store.find_person_by_slack_id") as find_person,
        patch("openexecutive.audit.log_event") as audit,
        patch(
            "openexecutive.workflows.inbound_resolver.resolve_inbound_message",
            new=AsyncMock(return_value=None),
        ),
        patch("openexecutive.alerts.pipeline.schedule_evaluation") as schedule,
        patch("openexecutive.knowledge.retriever.retrieve", return_value="ctx") as retrieve,
        patch("openexecutive.memory.episodic.format_for_prompt", return_value=""),
        patch("openexecutive.onboarding.profile_builder.load_or_create_profile") as profile,
        patch("openexecutive.orchestrator.executive.Executive", return_value=executive),
        patch("openexecutive.orchestrator.mcp_gateway.get_active_gateway", return_value=None),
        patch(
            "openexecutive.integrations.inbound_hydration.hydrate_user_message",
            side_effect=lambda **kw: kw["user_message"],
        ),
    ):
        profile.return_value.is_empty.return_value = True
        yield {
            "find_person": find_person,
            "audit": audit,
            "schedule": schedule,
            "retrieve": retrieve,
            "executive": executive,
        }


@pytest.mark.asyncio
async def test_rostered_dm_is_answered(chat_path: dict[str, Any]) -> None:
    chat_path["find_person"].return_value = MagicMock(id=7)
    say = AsyncMock()
    event = {"text": "hello", "user": "U1", "channel": "D1", "ts": "1.0", "channel_type": "im"}

    await slack_bot._handle_message(event, say, client=None, mode="dm")

    assert chat_path["executive"].chat.await_args.kwargs["person_id"] == 7
    chat_path["retrieve"].assert_called_once_with(query="hello")
    chat_path["schedule"].assert_called_once()
    say.assert_awaited_once_with(text="Here is your answer.", thread_ts="1.0")


@pytest.mark.asyncio
async def test_unrostered_sender_is_rejected_before_any_work(chat_path: dict[str, Any]) -> None:
    chat_path["find_person"].return_value = None
    say = AsyncMock()
    event = {"text": "hello", "user": "UX", "channel": "D1", "ts": "1.0", "channel_type": "im"}

    await slack_bot._handle_message(event, say, client=None, mode="dm")

    chat_path["executive"].chat.assert_not_awaited()
    chat_path["schedule"].assert_not_called()
    say.assert_not_awaited()
    outcomes = [
        c.kwargs.get("details", {}).get("outcome") for c in chat_path["audit"].call_args_list
    ]
    assert "rejected_unknown_sender" in outcomes


@pytest.mark.asyncio
async def test_workflow_reply_is_recorded_without_a_chat_turn(chat_path: dict[str, Any]) -> None:
    chat_path["find_person"].return_value = MagicMock(id=7)
    say = AsyncMock()
    event = {"text": "approved", "user": "U1", "channel": "D1", "ts": "1.0", "channel_type": "im"}

    with (
        patch(
            "openexecutive.workflows.inbound_resolver.resolve_inbound_message",
            new=AsyncMock(return_value=MagicMock(run_id="run-1")),
        ),
        patch(
            "openexecutive.workflows.resumer.apply_resolution", new=AsyncMock(return_value=True)
        ) as apply_resolution,
    ):
        await slack_bot._handle_message(event, say, client=None, mode="dm")

    apply_resolution.assert_awaited_once()
    say.assert_awaited_once_with(text="Got it — your response has been recorded.", thread_ts="1.0")
    chat_path["executive"].chat.assert_not_awaited()
    chat_path["schedule"].assert_not_called()


@pytest.mark.asyncio
async def test_thread_continuation_ignored_where_bot_never_replied(
    chat_path: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(slack_bot, "_bot_user_id", "UBOT")
    client = MagicMock()
    client.conversations_replies = AsyncMock(
        return_value={"messages": [{"user": "U1", "text": "start"}, {"user": "U2", "text": "hm"}]}
    )
    say = AsyncMock()
    event = {"text": "anyone?", "user": "U2", "channel": "C1", "ts": "2.0", "thread_ts": "1.0"}

    await slack_bot._handle_message(event, say, client=client, mode="thread_continuation")

    chat_path["audit"].assert_not_called()
    chat_path["executive"].chat.assert_not_awaited()
    say.assert_not_awaited()


@pytest.mark.asyncio
async def test_response_gate_can_skip_a_multi_person_thread(
    chat_path: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(slack_bot, "_bot_user_id", "UBOT")
    chat_path["find_person"].return_value = MagicMock(id=7, display_name="Nick")
    client = MagicMock()
    client.conversations_replies = AsyncMock(
        return_value={
            "messages": [
                {"user": "U1", "text": "question"},
                {"user": "UBOT", "text": "answer"},
                {"user": "U2", "text": "side remark"},
            ]
        }
    )
    say = AsyncMock()
    event = {"text": "thanks U1", "user": "U2", "channel": "C1", "ts": "3.0", "thread_ts": "1.0"}

    with patch(
        "openexecutive.integrations.response_gate.should_respond",
        new=AsyncMock(return_value=MagicMock(allow=False, reason="not_addressed")),
    ) as should_respond:
        await slack_bot._handle_message(event, say, client=client, mode="thread_continuation")

    should_respond.assert_awaited_once()
    chat_path["executive"].chat.assert_not_awaited()
    say.assert_not_awaited()
    outcomes = [
        c.kwargs.get("details", {}).get("outcome") for c in chat_path["audit"].call_args_list
    ]
    assert "skipped_gate" in outcomes


@pytest.mark.asyncio
async def test_thread_history_fetch_is_bounded(
    chat_path: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hung conversations.replies degrades to no history instead of stalling
    the reply; the timeout is local, not a parameter sent to Slack."""
    monkeypatch.setattr(slack_bot, "_REPLIES_TIMEOUT_S", 0.05)
    chat_path["find_person"].return_value = MagicMock(id=7)
    client = MagicMock()

    async def _hang(**kwargs: Any) -> dict[str, Any]:
        assert "timeout" not in kwargs
        await asyncio.Event().wait()
        return {}

    client.conversations_replies = _hang
    say = AsyncMock()
    event = {"text": "<@UBOT> hi", "user": "U1", "channel": "C1", "ts": "2.0", "thread_ts": "1.0"}

    await asyncio.wait_for(
        slack_bot._handle_message(event, say, client=client, mode="mention"), timeout=5
    )

    assert chat_path["executive"].chat.await_args.kwargs["co_present_person_ids"] is None
    say.assert_awaited_once()


# --------------------------------------------------------------------------
# lifespan wiring
# --------------------------------------------------------------------------


def _lifespan_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from openexecutive.api.main import create_app

    monkeypatch.delenv("BACKEND_SHARED_SECRET", raising=False)
    monkeypatch.delenv("OE_PUBLIC_DEPLOYMENT", raising=False)
    return TestClient(create_app())


@pytest.mark.asyncio
async def test_log_bot_crash_reports_only_real_crashes(caplog: pytest.LogCaptureFixture) -> None:
    from openexecutive.api.main import _log_bot_crash

    async def _crash() -> None:
        raise RuntimeError("gateway 4004")

    async def _finish() -> None:
        return None

    async def _block() -> None:
        await asyncio.Event().wait()

    crashed = asyncio.create_task(_crash())
    finished = asyncio.create_task(_finish())
    cancelled = asyncio.create_task(_block())
    await asyncio.sleep(0)
    cancelled.cancel()
    await asyncio.wait({crashed, finished, cancelled})

    # the API's logging setup turns off propagation on this logger, so
    # capture on it directly rather than through the root handler
    app_logger = logging.getLogger("openexecutive")
    app_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level("ERROR", logger="openexecutive"):
            for task, name in ((crashed, "Crashy"), (finished, "Clean"), (cancelled, "Stopped")):
                _log_bot_crash(name)(task)
    finally:
        app_logger.removeHandler(caplog.handler)

    messages = [r.getMessage() for r in caplog.records if r.name == "openexecutive"]
    assert messages == ["Crashy bot exited unexpectedly"]


# cold lifespan boot seeds the knowledge store, so the overall test deadline
# is generous; shutdown itself is measured against the real bound
_LIFESPAN_TEST_DEADLINE_S = 120.0


class _RecordingBot:
    instances: list[_RecordingBot] = []
    run_error: Exception | None = None
    run_blocks = False
    cancel_cleanup_stalls = False

    def __init__(self) -> None:
        self.ran = False
        self.run_cancelled = False
        self.stopped = False
        _RecordingBot.instances.append(self)

    async def run(self) -> None:
        self.ran = True
        if _RecordingBot.run_error is not None:
            raise _RecordingBot.run_error
        if _RecordingBot.run_blocks:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.run_cancelled = True
                if _RecordingBot.cancel_cleanup_stalls:
                    # a Socket Mode close that never returns
                    await asyncio.shield(asyncio.Event().wait())
                raise

    async def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def recording_bot(
    monkeypatch: pytest.MonkeyPatch, slack_tokens: None
) -> type[_RecordingBot]:
    _RecordingBot.instances = []
    _RecordingBot.run_error = None
    _RecordingBot.run_blocks = False
    _RecordingBot.cancel_cleanup_stalls = False
    monkeypatch.setattr(slack_bot, "EmbeddedSlackBot", _RecordingBot)
    return _RecordingBot


async def _measure_lifespan_shutdown(
    monkeypatch: pytest.MonkeyPatch, recording_bot: type[_RecordingBot]
) -> float:
    """Boot the real lifespan until the Slack run has started, then return
    how long its shutdown takes."""
    from openexecutive.api.main import create_app

    monkeypatch.delenv("BACKEND_SHARED_SECRET", raising=False)
    monkeypatch.delenv("OE_PUBLIC_DEPLOYMENT", raising=False)
    app = create_app()
    loop = asyncio.get_running_loop()
    shutdown_started: list[float] = []

    async def _boot_and_shut_down() -> None:
        async with app.router.lifespan_context(app):
            while not recording_bot.instances or not recording_bot.instances[0].ran:
                await asyncio.sleep(0.01)
            shutdown_started.append(loop.time())

    # driven directly with a deadline: a teardown that never finishes would
    # otherwise hang the test client instead of failing
    await asyncio.wait_for(_boot_and_shut_down(), timeout=_LIFESPAN_TEST_DEADLINE_S)
    return loop.time() - shutdown_started[0]


def test_lifespan_runs_and_stops_slack_when_both_tokens_set(
    monkeypatch: pytest.MonkeyPatch, recording_bot: type[_RecordingBot]
) -> None:
    with _lifespan_client(monkeypatch) as client:
        assert client.get("/health").status_code == 200
        (bot,) = recording_bot.instances
        assert bot.ran
        assert not bot.stopped
    assert bot.stopped


@pytest.mark.asyncio
async def test_lifespan_cancels_a_run_still_retrying_at_shutdown(
    monkeypatch: pytest.MonkeyPatch, recording_bot: type[_RecordingBot]
) -> None:
    from openexecutive.api import main

    recording_bot.run_blocks = True

    shutdown_s = await _measure_lifespan_shutdown(monkeypatch, recording_bot)

    (bot,) = recording_bot.instances
    # an uncancelled run would hold the Slack shutdown until the bound fires
    assert shutdown_s < main._BOT_SHUTDOWN_TIMEOUT_S
    assert bot.run_cancelled
    assert bot.stopped


@pytest.mark.asyncio
async def test_lifespan_bounds_a_stalled_slack_teardown(
    monkeypatch: pytest.MonkeyPatch, recording_bot: type[_RecordingBot]
) -> None:
    from openexecutive.api import main

    bound_s = 0.5
    monkeypatch.setattr(main, "_BOT_SHUTDOWN_TIMEOUT_S", bound_s)
    recording_bot.run_blocks = True
    recording_bot.cancel_cleanup_stalls = True

    shutdown_s = await _measure_lifespan_shutdown(monkeypatch, recording_bot)

    (bot,) = recording_bot.instances
    assert bot.run_cancelled
    # the rest of the lifespan teardown still ran, within a few bounds
    assert shutdown_s < bound_s * 10


@pytest.mark.parametrize("missing", ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"])
def test_lifespan_skips_slack_without_both_tokens(
    monkeypatch: pytest.MonkeyPatch, recording_bot: type[_RecordingBot], missing: str
) -> None:
    monkeypatch.delenv(missing)
    with _lifespan_client(monkeypatch) as client:
        assert client.get("/health").status_code == 200
    assert recording_bot.instances == []


def test_lifespan_boots_and_stops_cleanly_when_slack_run_fails(
    monkeypatch: pytest.MonkeyPatch, recording_bot: type[_RecordingBot]
) -> None:
    recording_bot.run_error = RuntimeError("boom")
    with _lifespan_client(monkeypatch) as client:
        assert client.get("/health").status_code == 200
    (bot,) = recording_bot.instances
    assert bot.ran
    assert bot.stopped
