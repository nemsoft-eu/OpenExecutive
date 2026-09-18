"""Executive-level wiring for the client-side SearXNG web_search tool.

Two things here are easy to get wrong and expensive when wrong:

  * Tool placement. The SearXNG tool is an ordinary function tool and must
    sort into the cached client list; the Anthropic server tool must stay
    appended after it with no cache_control. Getting that backwards either
    drops the cache marker or ships a server tool the API rejects.
  * Budget scope. The budget has to be per TURN. Built inside the iteration
    loop it silently becomes per-iteration, allowing WEB_SEARCH_MAX_USES
    searches on every round trip.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-not-used")

import pytest

from openexecutive.orchestrator.tool_outcome import ToolOutcome

from ._search_helpers import (
    clear_search_env,
    msg,
    stream_cm,
    tool_use_block,
    use_local_model,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_search_env(monkeypatch)


def _use_local_model(monkeypatch: pytest.MonkeyPatch, *, searxng: bool = True) -> None:
    """A local default model — the Executive's model comes from DEFAULT_MODEL."""
    use_local_model(monkeypatch, searxng=searxng)
    monkeypatch.setenv("DEFAULT_MODEL", "qwen-local")

def _drive(monkeypatch: pytest.MonkeyPatch, messages: list[Any]) -> dict[str, Any]:
    """Run one Executive turn against a scripted provider. Returns captures."""
    from openexecutive.orchestrator.executive import Executive
    from openexecutive.orchestrator.session import Session

    captured: dict[str, Any] = {"tools": [], "messages": []}
    remaining = list(messages)

    def fake_stream(**kwargs: Any) -> MagicMock:
        captured["tools"].append(kwargs["tools"])
        captured["messages"].append(kwargs["messages"])
        return stream_cm(remaining[min(len(captured["tools"]) - 1,
                                        len(remaining) - 1)])

    exec_ = Executive()
    provider = SimpleNamespace(messages_stream=fake_stream)

    async def _run() -> None:
        session = Session(session_id="t-client-search")
        async for _ in exec_.stream_chat("hi", session):
            pass

    with patch(
        "openexecutive.orchestrator.executive.get_provider", return_value=provider
    ):
        asyncio.run(_run())
    return captured


def test_searxng_tool_sorts_into_the_cached_client_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_local_model(monkeypatch)
    captured = _drive(monkeypatch, [msg(
        SimpleNamespace(type="text", text="done"), stop="end_turn"
    )])
    tools = captured["tools"][0]
    names = [t["name"] for t in tools]

    assert "web_search" in names
    # Exactly one definition of the tool, never both variants.
    assert names.count("web_search") == 1
    # It is an ordinary function tool...
    web = next(t for t in tools if t["name"] == "web_search")
    assert "input_schema" in web
    assert "type" not in web
    # ...so it participates in the alphabetical sort of the client list.
    assert names == sorted(names)
    # The cache marker is still on the final client tool.
    assert tools[-1].get("cache_control") == {"type": "ephemeral", "ttl": "1h"}


def test_claude_keeps_only_the_server_tool_even_with_searxng_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The collision the per-turn wiring exists to prevent.

    Registering the SearXNG tool module-wide would offer Claude a client
    ``web_search`` next to the server tool of the same name. SEARXNG_URL
    being set must change nothing on a model with server search.
    """
    monkeypatch.setenv("SEARXNG_URL", "http://searxng:8080")
    monkeypatch.setenv("DEFAULT_MODEL", "claude-sonnet-5")
    captured = _drive(monkeypatch, [msg(
        SimpleNamespace(type="text", text="done"), stop="end_turn"
    )])
    tools = captured["tools"][0]
    web = [t for t in tools if t.get("name") == "web_search"]
    assert len(web) == 1
    assert web[0]["type"] == "web_search_20250305"
    assert tools[-1] == web[0], "server tool must stay appended last"
    assert "cache_control" not in tools[-1]
    assert tools[-2].get("cache_control") == {"type": "ephemeral", "ttl": "1h"}


def test_client_search_is_audited_as_a_client_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same tool name as the server variant, different execution site.

    The server tool audits as kind="server_tool"; the client one must be
    distinguishable, not lumped in with ordinary skills.
    """
    _use_local_model(monkeypatch)
    messages = [
        msg(tool_use_block("web_search", "tu_1", query="ciao pricing")),
        msg(SimpleNamespace(type="text", text="done"), stop="end_turn"),
    ]
    audit_calls: list[dict[str, Any]] = []

    def _capture(event_type: str, summary: str, **kwargs: Any) -> None:
        audit_calls.append({"event_type": event_type, **kwargs})

    with (
        patch(
            "openexecutive.orchestrator.searxng_search._run_search",
            new=AsyncMock(return_value=ToolOutcome('{"results": []}')),
        ),
        patch(
            "openexecutive.orchestrator.executive.audit_log", side_effect=_capture
        ),
    ):
        _drive(monkeypatch, messages)

    search_rows = [
        c for c in audit_calls
        if c["event_type"] == "tool_invocation"
        and (c.get("details") or {}).get("tool") == "web_search"
    ]
    assert len(search_rows) == 1
    assert search_rows[0]["details"]["kind"] == "client_tool"


def test_local_model_without_searxng_gets_no_search_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_local_model(monkeypatch, searxng=False)
    captured = _drive(monkeypatch, [msg(
        SimpleNamespace(type="text", text="done"), stop="end_turn"
    )])
    assert "web_search" not in {t.get("name") for t in captured["tools"][0]}


def test_client_search_is_dispatched_and_its_result_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers the classification site as well as the dispatch site.

    Classification reads the handler map separately from dispatch; overlay
    only the dispatch lookup and the tool falls through to "Unknown tool".
    """
    _use_local_model(monkeypatch)
    messages = [
        msg(tool_use_block("web_search", "tu_1", query="ciao pricing")),
        msg(SimpleNamespace(type="text", text="done"), stop="end_turn"),
    ]
    with patch(
        "openexecutive.orchestrator.searxng_search._run_search",
        new=AsyncMock(
            return_value=ToolOutcome('{"results": [{"url": "https://x.example"}]}')
        ),
    ):
        captured = _drive(monkeypatch, messages)

    # Second generation carries the tool_result for the search.
    follow_up = captured["messages"][1]
    results = [
        block
        for turn in follow_up
        if isinstance(turn, dict) and isinstance(turn.get("content"), list)
        for block in turn["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(results) == 1
    assert results[0]["tool_use_id"] == "tu_1"
    assert "Unknown tool" not in results[0]["content"]
    assert "x.example" in results[0]["content"]


def test_budget_is_per_turn_not_per_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two iterations, one search each, budget of 1 → one real search.

    A budget rebuilt inside the iteration loop would allow both.
    """
    _use_local_model(monkeypatch)
    monkeypatch.setenv("WEB_SEARCH_MAX_USES", "1")
    messages = [
        msg(tool_use_block("web_search", "tu_1", query="first")),
        msg(tool_use_block("web_search", "tu_2", query="second")),
        msg(SimpleNamespace(type="text", text="done"), stop="end_turn"),
    ]
    run_search = AsyncMock(return_value=ToolOutcome('{"results": []}'))
    with patch(
        "openexecutive.orchestrator.searxng_search._run_search", new=run_search
    ):
        captured = _drive(monkeypatch, messages)

    assert len(captured["tools"]) >= 3, "expected multiple loop iterations"
    assert run_search.await_count == 1, (
        "budget reset between iterations — it must be created before the loop"
    )
    # The over-budget call still gets an answered, flagged tool_result.
    third = captured["messages"][2]
    results = [
        block
        for turn in third
        if isinstance(turn, dict) and isinstance(turn.get("content"), list)
        for block in turn["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    over_budget = [r for r in results if r.get("is_error")]
    assert over_budget, "over-budget search was not flagged with is_error"
