"""RegRadar is connected read-only, and the gateway is where that is enforced.

RegRadar serves its findings over MCP with three reads, the contract pinned in
nemsoft-eu/regradar#44. `_REGRADAR_READ_TOOLS` is the code-side half of the
read-only guarantee — a fail-closed allow-list of those three names. These
tests pin it: `call_tool` refuses any other `regradar__` name, `search_tools`
never offers one, and a runtime `load_mcp_server` cannot borrow a configured
prefix-gated server's name (and with it, that server's bearer token).
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
    _REGRADAR_PREFIX,
    _REGRADAR_READ_TOOLS,
    _RESERVED_SERVER_NAMES,
    MCPGateway,
    _is_blocked_regradar_tool,
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


def _load(gateway: MCPGateway, name: str) -> str:
    with patch.object(gw_module, "get_settings", return_value=_settings()):
        return asyncio.run(gateway.load_mcp_server({"name": name, "url": "https://example.test/mcp"}))


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


def test_allow_list_is_exactly_the_contracted_reads() -> None:
    """The three names nemsoft-eu/regradar#44 pins. A change here is a
    cross-repo contract change, made on purpose after reading the schema."""
    assert frozenset({"search_findings", "get_finding", "latest_digest"}) == _REGRADAR_READ_TOOLS


# --- call_tool -------------------------------------------------------------


@pytest.mark.parametrize("bare", sorted(_REGRADAR_READ_TOOLS))
def test_every_allow_listed_read_passes_the_gate(bare: str) -> None:
    gateway, session_call = _make_gateway()
    result = _call(gateway, _REGRADAR_PREFIX + bare, {"query": "cleaning sector"})
    assert result == '{"ok": true}'
    assert session_call.await_count == 1


def test_a_respelled_allowed_read_passes_the_predicate() -> None:
    """Case and hyphen drift must not turn a read into a refusal. Whether the
    server then resolves that spelling is not this gate's concern."""
    assert not _is_blocked_regradar_tool("RegRadar__Search-Findings")


@pytest.mark.parametrize("bare", ["delete_finding", "some_tool_nobody_has_vetted"])
def test_a_write_or_unknown_tool_is_refused_and_told_not_to_retry(bare: str) -> None:
    """The list fails closed: a plausible write and a name nobody thought of
    are refused alike, so a tool RegRadar ships later cannot run until someone
    adds it to `_REGRADAR_READ_TOOLS` on purpose."""
    gateway, session_call = _make_gateway()
    result = _call(gateway, _REGRADAR_PREFIX + bare, {"id": "F-1"})
    assert _blocked(result)
    assert session_call.await_count == 0
    error = _error(result)
    assert "read-only" in error
    assert "Do not retry" in error


def test_a_respelled_blocked_tool_is_refused_too() -> None:
    assert _is_blocked_regradar_tool("RegRadar__Delete-Finding")
    assert _is_blocked_regradar_tool("REGRADAR__delete_finding")
    gateway, session_call = _make_gateway()
    assert _blocked(_call(gateway, "RegRadar__Delete-Finding", {}))
    assert session_call.await_count == 0


def test_the_refusal_is_audited_as_a_blocked_outbound_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "openexecutive.audit.log_event",
        lambda kind, _summary, **kw: rows.append((kind, kw.get("details", {}))),
    )
    gateway, _ = _make_gateway()
    _call(gateway, "regradar__delete_finding", {})
    assert rows == [(
        "integration_outbound_blocked",
        {"tool": "regradar__delete_finding", "field": "tool", "address": "<regradar-write>"},
    )]


def test_the_refusal_names_regradar_in_the_audit_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A RegRadar refusal must not read as a Google Workspace egress block —
    /audit is searched by that text."""
    summaries: list[str] = []
    monkeypatch.setattr(
        "openexecutive.audit.log_event",
        lambda _kind, summary, **_kw: summaries.append(summary),
    )
    gateway, _ = _make_gateway()
    _call(gateway, "regradar__delete_finding", {})
    assert summaries == ["Blocked a RegRadar call (tool=regradar__delete_finding field=tool)"]


def test_a_crafted_tool_name_cannot_forge_a_second_audit_line() -> None:
    """Everything after `regradar__` is the caller's, so `_refuse` flattens
    it: a newline must not forge a line in the audit summary."""
    summaries: list[str] = []
    with patch(
        "openexecutive.audit.log_event",
        lambda _kind, summary, **_kw: summaries.append(summary),
    ):
        gateway, session_call = _make_gateway()
        forged = "regradar__x\nBlocked a Google Workspace call (tool=send_gmail_message field=to)"
        result = _call(gateway, forged, {})
    assert session_call.await_count == 0
    assert len(summaries) == 1 and "\n" not in summaries[0]
    assert "\n" not in _error(result)


def test_a_megabyte_long_crafted_name_is_truncated_in_the_audit_row_and_the_reason() -> None:
    summaries: list[str] = []
    rows: list[dict[str, Any]] = []
    with patch(
        "openexecutive.audit.log_event",
        lambda _kind, summary, **kw: (summaries.append(summary), rows.append(kw.get("details", {}))),
    ):
        gateway, session_call = _make_gateway()
        result = _call(gateway, "regradar__" + "A" * 1_000_000, {})
    assert session_call.await_count == 0
    assert len(rows[0]["tool"]) <= 200
    assert len(summaries[0]) < 400
    assert len(_error(result)) < 400


# --- other servers are unaffected ------------------------------------------


def test_a_non_regradar_tool_name_is_unaffected() -> None:
    gateway, session_call = _make_gateway()
    assert not _is_blocked_regradar_tool("notion__create_page")
    result = _call(gateway, "notion__create_page", {"title": "x"})
    assert result == '{"ok": true}'
    assert session_call.await_count == 1


def test_a_server_whose_name_merely_starts_with_regradar_is_unaffected() -> None:
    """The prefix is the whole namespace separator, so another server is not
    swept into RegRadar's allow-list by sharing its first letters."""
    assert not _is_blocked_regradar_tool("regradarx__delete_finding")


@pytest.mark.parametrize("tool_name", [None, 42, ["regradar__delete_finding"], {"a": 1}])
def test_a_non_str_tool_name_does_not_crash_the_check(tool_name: object) -> None:
    assert _is_blocked_regradar_tool(tool_name) is False


# --- search_tools ----------------------------------------------------------


def test_search_results_never_offer_a_blocked_regradar_tool() -> None:
    gateway, session_call = _make_gateway()
    names = (
        "regradar__search_findings",
        "regradar__delete_finding",
        "regradar__get_finding",
        "regradar__some_tool_nobody_has_vetted",
        "linear__create_issue",
        "regradar__latest_digest",
    )
    text = f"Found {len(names)} tools:\n\n" + "".join(_search_block(n) for n in names)
    session_call.return_value.content = [MagicMock(text=text)]
    out = asyncio.run(gateway.search_tools({"query": "regulatory findings"}))
    for kept in ("regradar__search_findings", "regradar__get_finding", "regradar__latest_digest"):
        assert kept in out
    for dropped in (
        "regradar__delete_finding",
        "regradar__some_tool_nobody_has_vetted",
        "linear__create_issue",
    ):
        assert dropped not in out


def test_search_results_without_a_blocked_regradar_tool_come_back_byte_identical() -> None:
    gateway, session_call = _make_gateway()
    text = "Found 2 tools:\n\n" + "".join(
        _search_block(n) for n in ("regradar__search_findings", "regradar__latest_digest")
    )
    session_call.return_value.content = [MagicMock(text=text)]
    assert asyncio.run(gateway.search_tools({"query": "regulatory findings"})) == text


# --- load_mcp_server --------------------------------------------------------


def test_every_prefix_gated_server_is_reserved() -> None:
    """A tripwire, not a proof: it finds gates named `_X_PREFIX` +
    `_X_READ_TOOLS`, so a new read-gated server following that convention
    fails the equality until someone reviews whether its name is reserved."""
    names = vars(gw_module)
    gated = {
        value.removesuffix("__")
        for key, value in names.items()
        if key.endswith("_PREFIX") and key.removesuffix("_PREFIX") + "_READ_TOOLS" in names
    }
    assert gated == {"linear", "regradar"}
    assert gated <= _RESERVED_SERVER_NAMES


@pytest.mark.parametrize("name", ["atlassian", "Atlassian", "google-workspace"])
def test_a_runtime_load_under_any_configured_server_name_is_refused(name: str) -> None:
    """Any url server with a tokens line has the same token-by-name hazard,
    so every name the config file defines is refused, not only the pair."""
    gateway, session_call = _make_gateway()
    gateway.server_names = ("atlassian", "google_workspace")
    result = _load(gateway, name)
    assert _blocked(result)
    assert session_call.await_count == 0


def test_an_unconfigured_name_still_loads_when_servers_are_configured() -> None:
    """Negative control: a non-empty configured list refuses only its own names."""
    gateway, session_call = _make_gateway()
    gateway.server_names = ("atlassian", "google_workspace")
    assert _load(gateway, "vendor_docs") == '{"ok": true}'
    session_call.assert_awaited_once()


@pytest.mark.parametrize("name", ["regradar", "RegRadar", "linear", "Linear", " linear "])
def test_a_runtime_load_under_a_reserved_name_is_refused(
    name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """extensible-mcp picks the bearer token by server name and refuses a
    runtime name only while that server is connected, so after a failed boot
    connect this load would send the static entry's token to the model's URL.
    The gateway refuses it before the client is ever asked."""
    rows: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "openexecutive.audit.log_event",
        lambda _kind, _summary, **kw: rows.append(kw.get("details", {})),
    )
    gateway, session_call = _make_gateway()
    result = _load(gateway, name)
    assert _blocked(result)
    assert "Do not retry" in _error(result)
    assert session_call.await_count == 0
    assert [r["address"] for r in rows] == ["<reserved-server>"]


def test_a_runtime_load_under_an_unreserved_name_reaches_the_client() -> None:
    gateway, session_call = _make_gateway()
    result = _load(gateway, "regradar_two")
    assert result == '{"ok": true}'
    session_call.assert_awaited_once()
    assert session_call.await_args is not None
    assert session_call.await_args.args[1] == {
        "name": "regradar_two", "url": "https://example.test/mcp",
    }
