"""A chat turn shows what a long tool is doing while it still runs.

A workflow started from chat runs all its steps inside one ``run_workflow``
call; what it posts to the turn's live feed (``orchestrator.turn_live``)
streams before the call ends, and nothing is posted outside a chat turn.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

import pytest

from openexecutive.orchestrator import turn_live
from openexecutive.orchestrator.executive import Executive

_RUN = "zz_test_run"


class _TextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _ToolUseBlock:
    type = "tool_use"

    def __init__(self, id_: str, name: str, input_: dict[str, Any]) -> None:
        self.id = id_
        self.name = name
        self.input = input_


class _FinalMsg:
    usage = None

    def __init__(self, content: list[Any], stop_reason: str) -> None:
        self.content = content
        self.stop_reason = stop_reason


class _FakeStream:
    def __init__(self, final_msg: _FinalMsg) -> None:
        self._final_msg = final_msg

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *_a: Any) -> None:
        return None

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration

    async def get_final_message(self) -> _FinalMsg:
        return self._final_msg


class _ScriptedProvider:
    def __init__(self, final_msgs: list[_FinalMsg]) -> None:
        self._final_msgs = list(final_msgs)
        self.calls: list[dict[str, Any]] = []

    def messages_stream(self, **kwargs: Any) -> _FakeStream:
        self.calls.append(kwargs)
        return _FakeStream(self._final_msgs.pop(0))


@pytest.fixture(autouse=True)
def _no_audit_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("openexecutive.orchestrator.executive.audit_log", lambda *_a, **_k: None)


# Whether the tool below had finished when each item reached the chat.
_running: dict[str, bool] = {}


async def _workflow(*_a: Any, **_k: Any) -> str:
    _running["now"] = True
    turn_live.say("Finding sailings for two")
    await asyncio.sleep(0.05)
    turn_live.post({"type": "note", "text": "step two"})
    await asyncio.sleep(0.05)
    _running["now"] = False
    return json.dumps({"ok": True, "artifact": "Ship 7: $1,240"})


_provider: dict[str, _ScriptedProvider] = {}


def _turn() -> list[Any]:
    provider = _provider["last"] = _ScriptedProvider([
        _FinalMsg([_ToolUseBlock("tu_1", _RUN, {})], "tool_use"),
        _FinalMsg([_TextBlock("It is $1,240 for two.")], "end_turn"),
    ])

    async def go() -> list[Any]:
        items: list[Any] = []
        with patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider,
        ), patch.dict(
            "openexecutive.orchestrator.executive._ALL_SKILL_HANDLERS", {_RUN: _workflow},
        ):
            async for item in Executive()._stream_agent_loop(
                system_blocks=[], messages=[{"role": "user", "content": "price the cruise"}], model="claude-test",
            ):
                if isinstance(item, dict) and item.get("tool", "") == "" and item.get("type") in ("activity", "note"):
                    item = {**item, "while_running": _running.get("now", False)}
                items.append(item)
        return items

    return asyncio.run(go())


def test_a_long_tool_shows_what_it_is_doing_while_it_runs() -> None:
    live = [it for it in _turn() if isinstance(it, dict) and it.get("while_running")]
    assert [it["type"] for it in live] == ["activity", "note"]
    assert live[0]["label"] == "Finding sailings for two…"


def test_the_tool_result_still_follows_its_call() -> None:
    """Streaming while the tools run leaves the round's results paired with
    its calls: the next model call gets the tool's own answer."""
    _turn()
    sent = json.dumps(_provider["last"].calls[1]["messages"], default=str)
    assert "tu_1" in sent and "Ship 7: $1,240" in sent


def test_nothing_is_posted_outside_a_chat_turn() -> None:
    assert turn_live.watching() is False
    assert turn_live.say("Finding sailings") is False
    assert turn_live.post({"type": "note"}) is False


def test_a_stopped_turn_cancels_the_running_tools() -> None:
    """The reader of the feed stops early (Stop, a dropped client): the
    round's tools are cancelled, not left running against no one."""

    async def go() -> bool:
        feed = turn_live.Feed()
        started = asyncio.Event()

        async def slow() -> None:
            turn_live.say("Working on step one")
            started.set()
            await asyncio.sleep(10)

        with feed.bound():
            task = asyncio.ensure_future(slow())
        # Unbound again: only the task sees the feed.
        assert turn_live.watching() is False
        reader = feed.until(task)
        first = await anext(reader)
        assert first["label"] == "Working on step one…"
        await reader.aclose()
        return task.cancelled()

    assert asyncio.run(go()) is True


def test_a_long_line_is_cut_to_one_line() -> None:
    async def go() -> list[dict[str, Any]]:
        feed = turn_live.Feed()

        async def talk() -> None:
            turn_live.say("  Finding\nsailings  " + "x" * 200)
            turn_live.say("Do\u202ene\u2060.\u00ad\x07")
            turn_live.say("   ")

        with feed.bound():
            task = asyncio.ensure_future(talk())
        return [event async for event in feed.until(task)]

    events = asyncio.run(go())
    assert len(events) == 2
    assert len(events[0]["label"]) == turn_live.MAX_LINE_CHARS and events[0]["label"].endswith("…")
    assert "\n" not in events[0]["label"]
    assert events[1]["label"] == "Done…"


def test_a_turn_stopped_mid_round_cancels_its_tools() -> None:
    """The chat stops reading the agent loop while a tool runs (Stop, a
    dropped client): closing the loop cancels the tool at once."""
    provider = _ScriptedProvider([
        _FinalMsg([_ToolUseBlock("tu_1", _RUN, {})], "tool_use"),
        _FinalMsg([_TextBlock("unused")], "end_turn"),
    ])
    seen: dict[str, bool] = {}

    async def slow(*_a: Any, **_k: Any) -> str:
        turn_live.say("Working on step one")
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            seen["cancelled"] = True
            raise
        return "{}"

    async def go() -> None:
        with patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider,
        ), patch.dict(
            "openexecutive.orchestrator.executive._ALL_SKILL_HANDLERS", {_RUN: slow},
        ):
            loop = Executive()._stream_agent_loop(
                system_blocks=[], messages=[{"role": "user", "content": "go"}], model="claude-test",
            )
            async for item in loop:
                if isinstance(item, dict) and item.get("label") == "Working on step one…":
                    break
            await loop.aclose()
            # At once, not when the loop's leftover generators are collected.
            assert seen.get("cancelled") is True

    asyncio.run(asyncio.wait_for(go(), 5))


def test_executive_loop_a_failing_tool_still_answers_with_its_error() -> None:
    """A handler that raises after posting a live line: the line is streamed
    and the next model call still gets an error result for that call."""
    provider = _ScriptedProvider([
        _FinalMsg([_ToolUseBlock("tu_1", _RUN, {})], "tool_use"),
        _FinalMsg([_TextBlock("That didn't work.")], "end_turn"),
    ])

    async def broken(*_a: Any, **_k: Any) -> str:
        turn_live.say("Trying the booking site")
        raise RuntimeError("site down")

    async def go() -> list[Any]:
        with patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider,
        ), patch.dict(
            "openexecutive.orchestrator.executive._ALL_SKILL_HANDLERS", {_RUN: broken},
        ):
            return [
                item async for item in Executive()._stream_agent_loop(
                    system_blocks=[], messages=[{"role": "user", "content": "book it"}], model="claude-test",
                )
            ]

    items = asyncio.run(go())
    assert any(isinstance(it, dict) and it.get("label") == "Trying the booking site…" for it in items)
    sent = json.dumps(provider.calls[1]["messages"], default=str)
    assert "tu_1" in sent and "The failure is recorded" in sent
