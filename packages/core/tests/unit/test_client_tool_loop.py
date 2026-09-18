"""Unit tests for BaseAgent.analyze_with_tools' bounded client-tool loop.

The loop exists so a model with no server-side search can still search
before emitting its findings. Its contract is awkward on purpose: the
callers advertise an output tool that has NO handler and then read it off
the returned message, so "handled tool" and "terminal tool" have to be
separate concepts. These tests pin the sequences that get that wrong.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-not-used")
os.environ.setdefault("EXEC_EMAIL_ADDRESS", "exec@example.com")

import pytest

from openexecutive.agents.base import BaseAgent
from openexecutive.agents.tool_outcome import ToolOutcome

from ._search_helpers import msg as _message
from ._search_helpers import tool_use_block as _tool_use

TOOLS: list[dict[str, Any]] = [
    {"name": "emit_findings", "input_schema": {"type": "object", "properties": {}}},
    {"name": "web_search", "input_schema": {"type": "object", "properties": {}}},
]


class _Agent(BaseAgent):
    name = "test_agent"
    domain = "test"
    model = "claude-sonnet-5"

    def get_system_prompt(self) -> str:
        return "SYSTEM"


class _FakeProvider:
    """Returns a scripted message per call and records the kwargs it got."""

    def __init__(self, messages: list[Any]) -> None:
        self._messages = list(messages)
        self.calls: list[dict[str, Any]] = []

    async def messages_create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._messages[min(len(self.calls) - 1, len(self._messages) - 1)]


def _run(
    provider: _FakeProvider,
    *,
    handlers: dict[str, Any] | None = None,
    terminal: set[str] | None = None,
    rounds: int = 3,
) -> Any:
    agent = _Agent()
    with (
        patch("openexecutive.agents.base.get_provider", return_value=provider),
        patch("openexecutive.agents.base.log_model_usage") as usage,
    ):
        message = asyncio.run(
            agent.analyze_with_tools(
                "CONTENT",
                tools=TOOLS,
                client_tool_handlers=handlers,
                terminal_tool_names=terminal,
                max_client_tool_rounds=rounds,
            )
        )
    return message, usage


def test_no_handlers_keeps_the_historical_single_shot_call() -> None:
    """Callers that pass no handlers must behave exactly as before."""
    provider = _FakeProvider([_message(_tool_use("emit_findings"))])
    message, usage = _run(provider)
    assert len(provider.calls) == 1
    assert usage.call_count == 1
    assert message.content[0].name == "emit_findings"


def test_search_then_emit_feeds_the_result_back_and_returns_the_emit() -> None:
    search_msg = _message(_tool_use("web_search", "tu_s", query="prices"))
    emit_msg = _message(_tool_use("emit_findings", "tu_e"))
    provider = _FakeProvider([search_msg, emit_msg])

    async def handler(_inp: dict) -> ToolOutcome:
        return ToolOutcome('{"results": []}')

    message, usage = _run(
        provider, handlers={"web_search": handler}, terminal={"emit_findings"}
    )

    assert message is emit_msg
    assert len(provider.calls) == 2
    # Every billed generation is accounted, not just the first.
    assert usage.call_count == 2
    # The second call carries the assistant turn and a matching tool_result.
    second = provider.calls[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["content"][0]["id"] == "tu_s"
    assert second[-1]["content"][0]["tool_use_id"] == "tu_s"


def test_assistant_turn_replays_text_and_reasoning_not_just_tool_use() -> None:
    """Dropping the preamble or an OpenRouter reasoning block breaks continuity.

    Every other tool loop replays the whole turn; this one must too.
    """
    reasoning = SimpleNamespace(
        type="openrouter_reasoning",
        reasoning_details=[{"type": "reasoning.text", "text": "think"}],
    )
    search_msg = _message(
        reasoning,
        SimpleNamespace(type="text", text="Let me look that up."),
        _tool_use("web_search", "tu_s", query="prices"),
    )
    emit_msg = _message(_tool_use("emit_findings", "tu_e"))
    provider = _FakeProvider([search_msg, emit_msg])

    async def handler(_inp: dict) -> ToolOutcome:
        return ToolOutcome('{"results": []}')

    _run(provider, handlers={"web_search": handler}, terminal={"emit_findings"})

    replayed = provider.calls[1]["messages"][-2]
    assert replayed["role"] == "assistant"
    assert [b["type"] for b in replayed["content"]] == [
        "openrouter_reasoning", "text", "tool_use",
    ]
    assert replayed["content"][0]["reasoning_details"] == [
        {"type": "reasoning.text", "text": "think"}
    ]
    assert replayed["content"][1]["text"] == "Let me look that up."


def test_terminal_tool_wins_when_it_arrives_alongside_a_search() -> None:
    """Running a search whose result cannot reach the emitted output is waste."""
    both = _message(
        _tool_use("web_search", "tu_s", query="x"),
        _tool_use("emit_findings", "tu_e"),
    )
    provider = _FakeProvider([both])
    calls: list[dict] = []

    async def handler(inp: dict) -> ToolOutcome:
        calls.append(inp)
        return ToolOutcome("{}")

    message, _ = _run(
        provider, handlers={"web_search": handler}, terminal={"emit_findings"}
    )
    assert message is both
    assert calls == [], "search ran even though the model had already emitted"


def test_prose_only_response_returns_immediately() -> None:
    prose = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="I think...")],
        stop_reason="end_turn",
    )
    provider = _FakeProvider([prose])

    async def handler(_inp: dict) -> ToolOutcome:  # pragma: no cover
        raise AssertionError("handler must not run")

    message, _ = _run(
        provider, handlers={"web_search": handler}, terminal={"emit_findings"}
    )
    assert message is prose
    assert len(provider.calls) == 1


def test_unknown_tool_is_answered_with_an_error_result_not_dropped() -> None:
    """Every tool_use must get a tool_result or the transcript is invalid."""
    bogus = _message(_tool_use("make_coffee", "tu_b"))
    emit = _message(_tool_use("emit_findings", "tu_e"))
    provider = _FakeProvider([bogus, emit])

    async def handler(_inp: dict) -> ToolOutcome:  # pragma: no cover
        raise AssertionError("handler must not run")

    message, _ = _run(
        provider, handlers={"web_search": handler}, terminal={"emit_findings"}
    )
    assert message is emit
    results = provider.calls[1]["messages"][-1]["content"]
    assert len(results) == 1
    assert results[0]["tool_use_id"] == "tu_b"
    assert results[0]["is_error"] is True
    assert "Unknown tool" in results[0]["content"]


def test_error_outcome_sets_is_error_on_the_tool_result() -> None:
    search = _message(_tool_use("web_search", "tu_s", query="x"))
    emit = _message(_tool_use("emit_findings", "tu_e"))
    provider = _FakeProvider([search, emit])

    async def handler(_inp: dict) -> ToolOutcome:
        return ToolOutcome("The search timed out.", is_error=True)

    _run(provider, handlers={"web_search": handler}, terminal={"emit_findings"})
    result = provider.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert isinstance(result["content"], str)


def test_plain_string_handler_result_is_accepted() -> None:
    """Skill handlers return bare strings; the loop must not require ToolOutcome."""
    search = _message(_tool_use("web_search", "tu_s", query="x"))
    emit = _message(_tool_use("emit_findings", "tu_e"))
    provider = _FakeProvider([search, emit])

    async def handler(_inp: dict) -> str:
        return "plain string"

    _run(provider, handlers={"web_search": handler}, terminal={"emit_findings"})
    result = provider.calls[1]["messages"][-1]["content"][0]
    assert result["content"] == "plain string"
    assert "is_error" not in result


def test_final_round_withdraws_the_search_tool_to_force_an_emit() -> None:
    """A model that searches forever must still be given a chance to emit.

    Without withdrawing the handled tools on the last round, the loop ends
    on a search-only message and the caller's extractor finds nothing.
    """
    search = _message(_tool_use("web_search", "tu_s", query="x"))
    provider = _FakeProvider([search])

    async def handler(_inp: dict) -> ToolOutcome:
        return ToolOutcome("{}")

    _run(
        provider,
        handlers={"web_search": handler},
        terminal={"emit_findings"},
        rounds=2,
    )
    assert len(provider.calls) == 2
    offered_first = {t["name"] for t in provider.calls[0]["tools"]}
    offered_last = {t["name"] for t in provider.calls[1]["tools"]}
    assert offered_first == {"emit_findings", "web_search"}
    assert offered_last == {"emit_findings"}


def test_search_on_the_final_round_is_not_executed() -> None:
    """The tool was withdrawn; a model that searches anyway gets no search.

    Its result could never be read — there is no later generation — so
    running it would only spend budget and egress.
    """
    search = _message(_tool_use("web_search", "tu_s", query="x"))
    provider = _FakeProvider([search])
    calls: list[dict] = []

    async def handler(inp: dict) -> ToolOutcome:
        calls.append(inp)
        return ToolOutcome("{}")

    message, _ = _run(
        provider,
        handlers={"web_search": handler},
        terminal={"emit_findings"},
        rounds=1,
    )
    assert calls == []
    assert message is search
    assert len(provider.calls) == 1


def test_loop_is_bounded_by_max_rounds() -> None:
    search = _message(_tool_use("web_search", "tu_s", query="x"))
    provider = _FakeProvider([search])

    async def handler(_inp: dict) -> ToolOutcome:
        return ToolOutcome("{}")

    _run(
        provider,
        handlers={"web_search": handler},
        terminal={"emit_findings"},
        rounds=4,
    )
    assert len(provider.calls) == 4


@pytest.mark.parametrize("rounds", [0, -1])
def test_non_positive_round_count_still_makes_one_call(rounds: int) -> None:
    """Zero rounds would skip the loop body and leave nothing to return."""
    emit = _message(_tool_use("emit_findings"))
    provider = _FakeProvider([emit])

    async def handler(_inp: dict) -> ToolOutcome:  # pragma: no cover
        raise AssertionError("handler must not run")

    message, _ = _run(
        provider,
        handlers={"web_search": handler},
        terminal={"emit_findings"},
        rounds=rounds,
    )
    assert message is emit
    assert len(provider.calls) == 1


@pytest.mark.parametrize("rounds", [1, 2, 5])
def test_every_generation_is_billed_to_usage(rounds: int) -> None:
    search = _message(_tool_use("web_search", "tu_s", query="x"))
    provider = _FakeProvider([search])

    async def handler(_inp: dict) -> ToolOutcome:
        return ToolOutcome("{}")

    _, usage = _run(
        provider,
        handlers={"web_search": handler},
        terminal={"emit_findings"},
        rounds=rounds,
    )
    assert usage.call_count == rounds


def test_timeout_bounds_the_whole_loop_not_each_round() -> None:
    """Rounds that each fit the timeout must not add up past it.

    Each generation takes 0.3s against a 0.5s timeout: a per-round bound
    would let all three rounds (0.9s) through; the loop-wide one must not.
    """
    search = _message(_tool_use("web_search", "tu_s", query="x"))

    class _SlowProvider(_FakeProvider):
        async def messages_create(self, **kwargs: Any) -> Any:
            message = await super().messages_create(**kwargs)
            await asyncio.sleep(0.3)
            return message

    provider = _SlowProvider([search])

    async def handler(_inp: dict) -> ToolOutcome:
        return ToolOutcome("{}")

    agent = _Agent()
    with (
        patch("openexecutive.agents.base.get_provider", return_value=provider),
        patch("openexecutive.agents.base.log_model_usage"),
        pytest.raises(TimeoutError),
    ):
        asyncio.run(
            agent.analyze_with_tools(
                "CONTENT",
                tools=TOOLS,
                timeout_seconds=0.5,
                client_tool_handlers={"web_search": handler},
                terminal_tool_names={"emit_findings"},
                max_client_tool_rounds=3,
            )
        )
    # The second generation was cancelled mid-flight; a third never started.
    assert len(provider.calls) == 2
