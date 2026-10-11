"""Tests for the client-side ``web_fetch`` tool (orchestrator/web_fetch.py)."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from openexecutive.config import get_settings
from openexecutive.orchestrator import web_fetch
from openexecutive.orchestrator.web_fetch import (
    WEB_FETCH_TOOL,
    BlockedAddressError,
    PinnedTransport,
    is_public_address,
)
from openexecutive.orchestrator.web_search_tool import select_web_search_tool
from openexecutive.providers.translator import to_openai_request

from ._search_helpers import clear_search_env, use_local_model


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test states the search config it depends on."""
    clear_search_env(monkeypatch)


# --- selection --------------------------------------------------------------


def test_direct_claude_gets_anthropics_server_fetch() -> None:
    selection = select_web_search_tool("claude-sonnet-5")
    assert selection is not None and selection.kind == "server"
    assert selection.fetch_tool == {
        "type": "web_fetch_20260309",
        "name": "web_fetch",
        "max_uses": 3,
        "max_content_tokens": 4_000,
    }
    assert selection.fetch_max_uses == 3


def test_server_fetch_carries_the_domain_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_BLOCKED_DOMAINS", "example.com")
    selection = select_web_search_tool("claude-sonnet-5")
    assert selection is not None and selection.fetch_tool is not None
    assert selection.fetch_tool["blocked_domains"] == ["example.com"]


@pytest.mark.parametrize("model", ["claude-sonnet-5", "openai/gpt-6-astra"])
def test_openrouter_gets_search_but_no_fetch(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OpenRouter has no fetch to swap in, Claude id or not."""
    monkeypatch.setenv("OPENROUTER_ENABLED", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-not-used")
    selection = select_web_search_tool(model)
    assert selection is not None and selection.kind == "server"
    assert selection.fetch_tool is None
    assert selection.fetch_max_uses == 0


def test_local_model_with_searxng_gets_the_client_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_local_model(monkeypatch, "qwen-local")
    selection = select_web_search_tool("qwen-local")
    assert selection is not None and selection.kind == "client"
    assert selection.fetch_tool is WEB_FETCH_TOOL
    assert "input_schema" in WEB_FETCH_TOOL
    assert selection.fetch_max_uses == 3


def test_local_claude_lookalike_gets_the_client_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LOCAL_MODELS membership beats the Claude regex, same as get_provider."""
    use_local_model(monkeypatch, "claude-proxy-1")
    selection = select_web_search_tool("claude-proxy-1")
    assert selection is not None and selection.fetch_tool is WEB_FETCH_TOOL


def test_local_model_without_searxng_gets_no_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_local_model(monkeypatch, "qwen-local", searxng=False)
    assert select_web_search_tool("qwen-local") is None


@pytest.mark.parametrize("local", [False, True])
def test_zero_max_uses_turns_fetch_off_on_both_paths(
    local: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WEB_FETCH_MAX_USES", "0")
    model = "claude-sonnet-5"
    if local:
        use_local_model(monkeypatch, "qwen-local")
        model = "qwen-local"
    selection = select_web_search_tool(model)
    assert selection is not None  # search itself is unaffected
    assert selection.fetch_tool is None
    assert selection.fetch_max_uses == 0


def test_disabling_search_disables_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "false")
    assert select_web_search_tool("claude-sonnet-5") is None


@pytest.mark.parametrize(
    ("key", "value"),
    [("WEB_FETCH_MAX_USES", "-1"), ("WEB_FETCH_TIMEOUT_S", "0"), ("WEB_FETCH_TIMEOUT_S", "-1")],
)
def test_config_rejects_bad_fetch_bounds(
    key: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        get_settings()


def test_translator_drops_a_stray_server_fetch_tool() -> None:
    """A web_fetch server tool reaching OpenRouter must not become a
    schemaless function tool the model could call."""
    body = to_openai_request(
        "anthropic/claude-opus-4.7",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "web_fetch_20260309", "name": "web_fetch", "max_uses": 3}],
        },
    )
    assert "tools" not in body or all(
        t.get("function", {}).get("name") != "web_fetch" for t in body["tools"]
    )

# --- address policy -------------------------------------------------------


@pytest.mark.parametrize(
    "addr",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",  # cloud metadata
        "100.100.1.1",  # CGNAT / Tailscale
        "0.0.0.0",
        "224.0.0.1",  # multicast
        "::1",
        "fe80::1%eth0",
        "fd00::1",
        "::ffff:127.0.0.1",  # IPv4-mapped
        "2002:7f00:1::",  # 6to4 around 127.0.0.1
        "64:ff9b::7f00:1",  # NAT64 around 127.0.0.1
        "64:ff9b::a00:1",  # NAT64 around 10.0.0.1
        "not-an-ip",
    ],
)
def test_non_public_addresses_are_refused(addr: str) -> None:
    assert not is_public_address(addr)


@pytest.mark.parametrize(
    "addr", ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946", "64:ff9b::5db8:d822"]
)
def test_public_addresses_are_allowed(addr: str) -> None:
    assert is_public_address(addr)


# --- pinned transport ------------------------------------------------------


@pytest.fixture
async def local_server() -> AsyncIterator[tuple[int, list[bytes]]]:
    """A one-route HTTP server on 127.0.0.1 that records each request head."""
    heads: list[bytes] = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        heads.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
            b"Content-Length: 5\r\nConnection: close\r\n\r\nhello"
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield port, heads
    finally:
        server.close()
        await server.wait_closed()


async def test_pinned_transport_refuses_a_private_resolution(
    local_server: tuple[int, list[bytes]],
) -> None:
    port, heads = local_server

    async def resolver(host: str, _port: int) -> list[str]:
        return ["127.0.0.1"]

    async with httpx.AsyncClient(transport=PinnedTransport(resolver)) as client:
        with pytest.raises(httpx.ConnectError) as info:
            await client.get(f"http://looks-public.example:{port}/")
    assert isinstance(info.value.__cause__, BlockedAddressError)
    assert heads == []  # nothing reached the server


async def test_pinned_transport_refuses_mixed_public_and_private(
    local_server: tuple[int, list[bytes]],
) -> None:
    port, heads = local_server

    async def resolver(host: str, _port: int) -> list[str]:
        return ["93.184.216.34", "127.0.0.1"]

    async with httpx.AsyncClient(transport=PinnedTransport(resolver)) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get(f"http://rebind.example:{port}/")
    assert heads == []


async def test_pinned_transport_connects_to_the_checked_address(
    local_server: tuple[int, list[bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The connection goes to the resolver's address, with the URL's Host.

    ``pinned.invalid`` cannot resolve through real DNS (RFC 6761), so a
    successful response proves the backend connected to the address it
    checked rather than resolving the name again.
    """
    port, heads = local_server
    monkeypatch.setattr(web_fetch, "is_public_address", lambda _a: True)
    calls: list[str] = []

    async def resolver(host: str, _port: int) -> list[str]:
        calls.append(host)
        return ["127.0.0.1"]

    async with httpx.AsyncClient(transport=PinnedTransport(resolver)) as client:
        response = await client.get(f"http://pinned.invalid:{port}/page")
    assert response.text == "hello"
    assert calls == ["pinned.invalid"]
    assert b"Host: pinned.invalid" in heads[0]


def test_pinned_transport_still_owns_the_httpx_pool() -> None:
    """``PinnedTransport`` replaces httpx's private ``_pool``. If an httpx
    upgrade renames it, the replacement would silently stop applying."""
    transport = PinnedTransport()
    assert isinstance(transport._pool, web_fetch.httpcore.AsyncConnectionPool)
    assert isinstance(transport._pool._network_backend, web_fetch._PinnedBackend)
