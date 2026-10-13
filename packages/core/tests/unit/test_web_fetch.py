"""Tests for the client-side ``web_fetch`` tool (orchestrator/web_fetch.py)."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from openexecutive.config import get_settings
from openexecutive.orchestrator import web_fetch
from openexecutive.orchestrator.searxng_search import SearchBudget
from openexecutive.orchestrator.web_fetch import (
    WEB_FETCH_TOOL,
    BlockedAddressError,
    PinnedTransport,
)
from openexecutive.orchestrator.web_search_tool import select_web_search_tool
from openexecutive.utils import public_address

from ._search_helpers import clear_search_env, use_local_model


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test states the search config it depends on."""
    clear_search_env(monkeypatch)


# --- selection --------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "openrouter"),
    [("claude-sonnet-5", False), ("claude-sonnet-5", True), ("openai/gpt-6-astra", True)],
)
def test_a_server_search_model_gets_the_client_fetch(
    model: str, openrouter: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never Anthropic's server web_fetch: it opens any URL a client tool
    result echoes, so the turn's data could leave in a query string."""
    if openrouter:
        monkeypatch.setenv("OPENROUTER_ENABLED", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-not-used")
    selection = select_web_search_tool(model)
    assert selection is not None and selection.kind == "server"
    assert selection.fetch_tool is WEB_FETCH_TOOL
    assert selection.fetch_max_uses == 3


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


# --- address policy -------------------------------------------------------
# The classifier itself is tested in test_public_address.py; this checks only
# that the fetch uses it.


def test_the_fetch_uses_the_shared_classifier() -> None:
    assert web_fetch.is_public_address is public_address.is_public_address


# --- pinned transport ------------------------------------------------------


@pytest.fixture
def local_server_reply() -> list[bytes]:
    """What ``local_server`` answers; a test may replace the one entry."""
    return [
        b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
        b"Content-Length: 5\r\nConnection: close\r\n\r\nhello"
    ]


@pytest.fixture
async def local_server(
    local_server_reply: list[bytes],
) -> AsyncIterator[tuple[int, list[bytes]]]:
    """A one-route HTTP server on 127.0.0.1 that records each request head."""
    heads: list[bytes] = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        heads.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(local_server_reply[0])
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


async def test_pinned_transport_falls_back_to_the_next_checked_address(
    local_server: tuple[int, list[bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first address that refuses the connection is not the end of the
    fetch: the next checked address is tried, as a browser would."""
    port, heads = local_server
    monkeypatch.setattr(web_fetch, "is_public_address", lambda _a: True)

    async def resolver(host: str, _port: int) -> list[str]:
        # Nothing listens on 127.0.0.2 at this port, so it refuses.
        return ["127.0.0.2", "127.0.0.1"]

    async with httpx.AsyncClient(transport=PinnedTransport(resolver)) as client:
        response = await client.get(f"http://pinned.invalid:{port}/page")
    assert response.text == "hello"
    assert len(heads) == 1


def _scripted_transport(
    monkeypatch: pytest.MonkeyPatch, addresses: list[str]
) -> tuple[PinnedTransport, list[tuple[str, float | None]]]:
    """A pinned transport over ``addresses`` that records each attempt's
    timeout. 192.0.2.x addresses drop packets (wait out their timeout),
    127.0.0.2 refuses at once, and anything else really connects."""
    monkeypatch.setattr(web_fetch, "is_public_address", lambda _a: True)
    transport = PinnedTransport()
    backend = transport._pool._network_backend
    real_connect = backend._inner.connect_tcp
    attempts: list[tuple[str, float | None]] = []

    async def connect_tcp(
        host: str, port: int, timeout: float | None = None, **kwargs: Any
    ) -> Any:
        attempts.append((host, timeout))
        if host.startswith("192.0.2."):
            await asyncio.sleep(timeout or 0)
            raise web_fetch.httpcore.ConnectTimeout("no answer")
        if host == "127.0.0.2":
            raise web_fetch.httpcore.ConnectError("refused")
        return await real_connect(host, port, timeout=timeout, **kwargs)

    async def resolver(host: str, _port: int) -> list[str]:
        return addresses

    monkeypatch.setattr(backend, "_resolver", resolver)
    monkeypatch.setattr(backend._inner, "connect_tcp", connect_tcp)
    return transport, attempts


async def test_pinned_transport_leaves_time_for_the_next_address(
    local_server: tuple[int, list[bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first address that drops packets silently runs out its own share of
    the connect timeout, not the whole deadline, so the next one still
    loads inside it."""
    port, heads = local_server
    monkeypatch.setattr(web_fetch, "_MIN_CONNECT_ATTEMPT_S", 0.0)
    transport, attempts = _scripted_transport(monkeypatch, ["192.0.2.1", "127.0.0.1"])
    async with httpx.AsyncClient(transport=transport, timeout=0.4) as client:
        async with asyncio.timeout(0.4):
            response = await client.get(f"http://pinned.invalid:{port}/page")
    assert response.text == "hello"
    assert [host for host, _ in attempts] == ["192.0.2.1", "127.0.0.1"]
    assert attempts[0][1] == pytest.approx(0.2, abs=0.02)
    assert len(heads) == 1


async def test_pinned_transport_passes_on_time_an_instant_refusal_left(
    local_server: tuple[int, list[bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An address that refuses at once uses none of its share, so the next
    gets half of all that is left rather than its own third."""
    port, _heads = local_server
    monkeypatch.setattr(web_fetch, "_MIN_CONNECT_ATTEMPT_S", 0.0)
    transport, attempts = _scripted_transport(
        monkeypatch, ["127.0.0.2", "192.0.2.1", "127.0.0.1"]
    )
    async with httpx.AsyncClient(transport=transport, timeout=0.6) as client:
        response = await client.get(f"http://pinned.invalid:{port}/page")
    assert response.text == "hello"
    assert attempts[0][1] == pytest.approx(0.2, abs=0.02)
    assert attempts[1][1] == pytest.approx(0.3, abs=0.02)


async def test_pinned_transport_gives_each_attempt_at_least_the_floor(
    local_server: tuple[int, list[bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Many records do not cut an attempt below one SYN retry: with 16
    addresses and 15 s the first still gets 3 s, not 0.94 s, and when less
    than the floor is left, an attempt gets all of it."""
    port, _heads = local_server
    addresses = [f"127.0.0.{n}" for n in range(2, 17)] + ["127.0.0.1"]
    transport, attempts = _scripted_transport(monkeypatch, addresses)
    async with httpx.AsyncClient(transport=transport, timeout=15.0) as client:
        response = await client.get(f"http://pinned.invalid:{port}/page")
    assert response.text == "hello"
    assert attempts[0][1] == pytest.approx(3.0, abs=0.02)

    transport, attempts = _scripted_transport(monkeypatch, ["127.0.0.2", "127.0.0.1"])
    async with httpx.AsyncClient(transport=transport, timeout=1.0) as client:
        await client.get(f"http://pinned.invalid:{port}/page")
    assert attempts[0][1] == pytest.approx(1.0, abs=0.02)


def test_pinned_transport_still_owns_the_httpx_pool() -> None:
    """``PinnedTransport`` replaces httpx's private ``_pool``. If an httpx
    upgrade renames it, the replacement would silently stop applying."""
    transport = PinnedTransport()
    assert isinstance(transport._pool, web_fetch.httpcore.AsyncConnectionPool)
    assert isinstance(transport._pool._network_backend, web_fetch._PinnedBackend)


def test_every_fetch_goes_through_the_pinned_transport() -> None:
    """The handler tests below swap ``_new_transport`` out; production must
    not be able to drift away from the pinned one unnoticed."""
    assert isinstance(web_fetch._new_transport(), PinnedTransport)


# --- the seen-URL rule ------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTPS://Example.COM", "https://example.com/"),
        ("https://example.com:443/a#frag", "https://example.com/a"),
        ("http://example.com:80/", "http://example.com/"),
        ("http://example.com:8080/a?b=1&c=2", "http://example.com:8080/a?b=1&c=2"),
        ("https://example.com./a", "https://example.com/a"),
        ("https://bücher.example/x", "https://xn--bcher-kva.example/x"),
        ("https://[2606:2800:220:1::1]:8443/", "https://[2606:2800:220:1::1]:8443/"),
        ("  https://example.com/a  ", "https://example.com/a"),
        ("https://example.com/Path", "https://example.com/Path"),
        ("ftp://example.com/", None),
        ("https://user:pw@example.com/", None),
        ("https://@example.com/", None),
        ("https://example.com:99999/", None),
        # IDNA 2003 would compare these as strasse.de / σ, but httpx (IDNA
        # 2008) connects to a different, registrable domain.
        ("https://straße.de/x", None),
        ("https://ς.gr/", None),
        ("https://xn--strae-oqa.de/x", "https://xn--strae-oqa.de/x"),
        ("https:///nohost", None),
        ("not a url", None),
    ],
)
def test_normalise_url(raw: str, expected: str | None) -> None:
    assert web_fetch.normalise_url(raw) == expected


@pytest.mark.parametrize(
    ("seen_as", "fetched_as"),
    [
        ("https://example.com/café", "https://example.com/caf%C3%A9"),
        ("https://example.com/a b", "https://example.com/a%20b"),
        ("https://example.com/a/../b", "https://example.com/b"),
    ],
)
def test_two_spellings_of_one_request_compare_equal(seen_as: str, fetched_as: str) -> None:
    seen = web_fetch.SeenUrls()
    seen.add(seen_as)
    assert seen.allows(fetched_as)


def test_trailing_slash_and_query_are_part_of_the_url() -> None:
    seen = web_fetch.SeenUrls()
    seen.add("https://example.com/page?id=1")
    assert seen.allows("https://EXAMPLE.com/page?id=1#top")
    assert not seen.allows("https://example.com/page/?id=1")
    assert not seen.allows("https://example.com/page?id=2")
    assert not seen.allows("https://example.com/page")


@pytest.mark.parametrize(
    ("text", "url"),
    [
        ("<https://ciao-construct.be|ciao-construct.be> what date?", "https://ciao-construct.be/"),
        ("look at <https://example.com/a>", "https://example.com/a"),
        ("see https://example.com/x.", "https://example.com/x"),
        ("is it https://example.com/x?", "https://example.com/x"),
        ("(see https://example.com/x), then", "https://example.com/x"),
        ("[docs](https://example.com/docs)", "https://example.com/docs"),
        ("`https://example.com/code`", "https://example.com/code"),
        ('"https://example.com/q"', "https://example.com/q"),
        ("https://en.wikipedia.org/wiki/Foo_(bar)", "https://en.wikipedia.org/wiki/Foo_(bar)"),
        ("<https://example.com/?a=1&amp;b=2>", "https://example.com/?a=1&b=2"),
        ("HTTP://Example.com/caps", "http://example.com/caps"),
        ("https://en.wikipedia.org/wiki/Schindler's_List", "https://en.wikipedia.org/wiki/Schindler's_List"),
        ("'https://example.com/single'", "https://example.com/single"),
        ("**https://example.com/bold**", "https://example.com/bold"),
        ("见 https://example.com/zh。", "https://example.com/zh"),
        ("see https://example.com's homepage", "https://example.com/"),
        ("see https://example.com:8443's admin", "https://example.com:8443/"),
        # A quote in a query with no path is the query's, not a possessive.
        ("https://example.com?q='x'y", "https://example.com/?q='x'y"),
    ],
)
def test_urls_in_the_forms_people_paste_are_seen(text: str, url: str) -> None:
    seen = web_fetch.SeenUrls()
    seen.add_from_text(text)
    assert seen.allows(url)


@pytest.mark.parametrize(
    ("text", "url"),
    [
        ("what does example.com/page say?", "https://example.com/page"),
        ("see www.example.com today", "https://www.example.com/"),
        ("what does ciao-construct.be? say", "https://ciao-construct.be/"),
        ("(see example.com/x), then", "https://example.com/x"),
        ("admin at example.com:8443/a", "https://example.com:8443/a"),
        # The host is case-blind, the path is not.
        ("Example.COM/Caps", "https://example.com/Caps"),
        ("read example.com.", "https://example.com/"),
        ("example.com's homepage", "https://example.com/"),
        ("example.com?ref=x", "https://example.com/?ref=x"),
        ("medium.com/@user", "https://medium.com/@user"),
        ("example.com/?a=1&amp;b=2", "https://example.com/?a=1&b=2"),
        # Slack's <url|label>: the link keeps its own scheme.
        ("<http://a.example|a.example>", "http://a.example/"),
    ],
)
def test_a_link_written_without_a_scheme_is_seen_as_https(text: str, url: str) -> None:
    seen = web_fetch.SeenUrls()
    seen.add_from_text(text)
    assert seen.allows(url)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("version 1.2.3", set()),
        ("e.g. this, i.e. that", set()),
        ("upgrade to v2.0", set()),
        ("an IPv4 address: 10.0.0.1/x", set()),
        ("localhost/admin", set()),
        ("see docs/example.com", set()),
        # A common email address seeds neither its domain nor its local part.
        ("mail name@example.com", set()),
        ("mail jane.doe.smith@example.com", set()),
        ("mail first.last+tag@example.com", set()),
        ("mail jane.doe=x@example.com", set()),
        # A local part holding a character that ends a link seeds its dotted
        # prefix, never the domain: harmless, see add_from_text.
        ("mail jane.doe!x@example.com", {"https://jane.doe/"}),
        # IDNA 2003 and 2008 disagree on "ß" (see test_normalise_url), and a
        # scheme-less host is ASCII only, so no part of it is seen.
        ("straße.de/x", set()),
        ("bücher.example/x", set()),
        # The host-like text inside a link with a scheme is not a link.
        ("https://example.com/foo.bar/x", {"https://example.com/foo.bar/x"}),
        (
            "https://example.com/?next=other.example/x",
            {"https://example.com/?next=other.example/x"},
        ),
        # A link with a scheme inside a scheme-less one's path leaves the
        # scheme-less link whole, and is seen as written too.
        (
            "web.archive.org/web/2024/https://example.com/page",
            {
                "https://web.archive.org/web/2024/https://example.com/page",
                "https://example.com/page",
            },
        ),
        # The forms people paste: Markdown, Slack's <url>, a Slack auto-link
        # (its label seeds the https form), Slack's mailto link, a list.
        ("[x](example.com/a)", {"https://example.com/a"}),
        ("<example.com/a>", {"https://example.com/a"}),
        ("<http://a.example|a.example>", {"http://a.example/", "https://a.example/"}),
        ("<mailto:jane.doe@example.com|jane.doe@example.com>", set()),
        ("a.example,b.example", {"https://a.example/", "https://b.example/"}),
        ("**example.com**", {"https://example.com/"}),
        ("'example.com'", {"https://example.com/"}),
        ("(see example.com/x), then", {"https://example.com/x"}),
        ("read example.com.", {"https://example.com/"}),
        ("example.com: nice", {"https://example.com/"}),
        ("\u89c1 example.com\u3002", {"https://example.com/"}),
        # Curly quotes, an em dash, an ellipsis, CJK brackets and commas.
        ("\u201cexample.com\u201d", {"https://example.com/"}),
        ("\u2018example.com\u2019", {"https://example.com/"}),
        ("\u201eexample.com\u201c", {"https://example.com/"}),
        ("\u00abexample.com\u00bb", {"https://example.com/"}),
        ("example.com\u2019s homepage", {"https://example.com/"}),
        ("see\u2014example.com\u2014really", {"https://example.com/"}),
        ("example.com\u2026", {"https://example.com/"}),
        ("\u300cexample.com\u300d", {"https://example.com/"}),
        ("\uff08example.com\uff09", {"https://example.com/"}),
        ("a.example\u3001b.example", {"https://a.example/", "https://b.example/"}),
        # Those marks end a path, query or fragment too, never join it.
        ("\u201cexample.com/page\u201d", {"https://example.com/page"}),
        ("see\u2014example.com/page\u2014please", {"https://example.com/page"}),
        ("\u300cexample.com/page\u300d", {"https://example.com/page"}),
        ("\uff08example.com/page\uff09", {"https://example.com/page"}),
        ("example.com/page\u2026", {"https://example.com/page"}),
        ("example.com?x=1\u3001", {"https://example.com/?x=1"}),
        ("example.com/a?q=b\u00bb", {"https://example.com/a?q=b"}),
        # Percent-encoded, the same mark is the path's.
        ("example.com/a%E2%80%9Db", {"https://example.com/a%E2%80%9Db"}),
        # Otherwise a path runs to _URL_RE's delimiters, so text joined to it,
        # a CJK letter included, is path.
        ("example.com/x\u89c1", {"https://example.com/x%E8%A7%81"}),
        ("example.com/\u9875", {"https://example.com/%E9%A1%B5"}),
        # A port past 65535 makes no URL, so nothing is seen.
        ("example.com:99999/x", set()),
        ("example.com:8443/a", {"https://example.com:8443/a"}),
        # A ":" that starts no port: the person wrote no root URL, and the
        # host cannot back off to a shorter one ("a.example").
        ("example.com:abc/path", set()),
        ("example.com:100000/path", set()),
        ("example.com:abc", set()),
        ("a.example.com:abc", set()),
        ("example.com:/x", set()),
        ("example.com./x", set()),
        # Nor does a ":" start a link: this is not "abc.example/x".
        ("example.com:abc.example/x", set()),
        ("site:example.com", set()),
        # A letter, a zero-width space, a soft hyphen, "+", "=" or "/" right
        # before the host does not start a match, so the token's suffix is
        # not seeded.
        ("foo\u200bexample.com/x", set()),
        ("foo\u00adexample.com/x", set()),
        ("foo+example.com/x", set()),
        ("foo=example.com", set()),
        ("x/example.com", set()),
        ("\u89c1example.com", set()),
        # Nor ends mid-word.
        ("example.com\u89c1", set()),
        ("example.com-x", set()),
        # A letter from another script in the host (U+0430, the Cyrillic "a")
        # leaves no ASCII part of it seen.
        ("ex\u0430mple.com", set()),
        ("example.com\u0430", set()),
        # Only the https form of a scheme-less link.
        ("example.com/page", {"https://example.com/page"}),
        # Seeded, and pointless: the speaker's own text, under every check.
        ("open report.pdf", {"https://report.pdf/"}),
        # A brace is no URL character, so it ends a scheme-less link ({ only
        # opens one, } only closes one) and never reaches the path as %7D.
        ("{example.com/page}", {"https://example.com/page"}),
        ("{{example.com/a}}", {"https://example.com/a"}),
        ("example.com?x=1}", {"https://example.com/?x=1"}),
        ("example.com/p{q", {"https://example.com/p"}),
        # A link with a scheme keeps its braces, percent-encoded, as on main.
        ("https://e.com/x{y}", {"https://e.com/x%7By%7D"}),
    ],
)
def test_what_a_scheme_less_pass_seeds(text: str, expected: set[str]) -> None:
    seen = web_fetch.SeenUrls()
    seen.add_from_text(text)
    assert seen._urls == expected


# CJK full stop and commas, curly quotes and guillemets, CJK corner brackets
# and fullwidth parentheses, the em dash and the ellipsis.
_WIDE_MARKS = [
    "\u3002", "\uff0c", "\u3001",
    "\u201c", "\u201d", "\u2018", "\u2019", "\u201e", "\u00ab", "\u00bb",
    "\u300c", "\u300d", "\uff08", "\uff09",
    "\u2014", "\u2026",
]


def test_the_wide_marks_are_the_ones_the_pattern_uses() -> None:
    assert set(web_fetch._WIDE_DELIMITERS) == set(_WIDE_MARKS)


@pytest.mark.parametrize("mark", _WIDE_MARKS, ids=[f"U+{ord(m):04X}" for m in _WIDE_MARKS])
def test_a_wide_mark_starts_ends_and_stops_a_scheme_less_link(mark: str) -> None:
    """Each mark is a boundary on either side and is never path: a link
    stopped at a mark that could not end one would seed nothing."""
    cases = {
        f"example.com/p{mark}q": {"https://example.com/p"},
        f"{mark}example.com/p{mark}": {"https://example.com/p"},
        f"{mark}example.com{mark}": {"https://example.com/"},
        f"example.com?x=1{mark}": {"https://example.com/?x=1"},
        f"a.example{mark}b.example": {"https://a.example/", "https://b.example/"},
    }
    for text, expected in cases.items():
        seen = web_fetch.SeenUrls()
        seen.add_from_text(text)
        assert seen._urls == expected, ascii(text)


def test_a_dotted_file_name_seeds_its_host_and_nothing_the_model_can_extend() -> None:
    """``memory_text`` carries the attached files' names, and a third party may
    have chosen one. It seeds the host alone: a path, query, port or scheme
    the model adds is a different URL and stays refused."""
    seen = web_fetch.SeenUrls()
    seen.add_from_text("(Attached files: x.attacker.example)")
    assert seen.allows("https://x.attacker.example/")
    assert not seen.allows("https://x.attacker.example/exfil")
    assert not seen.allows("https://x.attacker.example/?q=1")
    assert not seen.allows("https://x.attacker.example:8443/")
    assert not seen.allows("http://x.attacker.example/")


def test_a_huge_character_reference_in_a_pasted_url_does_not_raise() -> None:
    """html.unescape raises past 4,300 digits; seeding runs before the
    turn's first model call, so a raise there would end the turn."""
    seen = web_fetch.SeenUrls()
    seen.add_from_text("see https://a.example/?x=&#" + "1" * 5000 + ";")
    assert seen.allows("https://a.example/?x=&#" + "1" * 5000 + ";")


def test_a_lookalike_of_a_seen_url_is_not_allowed() -> None:
    """Pasting business.example must not make busineß.example fetchable."""
    seen = web_fetch.SeenUrls()
    seen.add_from_text("open https://docs.business.example/share?token=s3cret")
    assert seen.allows("https://docs.business.example/share?token=s3cret")
    assert not seen.allows("https://docs.busineß.example/share?token=s3cret")


def test_a_url_outside_the_speakers_own_words_stays_unseen() -> None:
    """The web chat appends attachment text to the prompt and passes what the
    person typed as ``memory_text``; only the latter may seed the set."""
    from openexecutive.orchestrator.executive import _speaker_text

    typed = "What does https://ciao-construct.be say about the date?"
    prompt = f"{typed}\n\n<attachment>see https://attacker.example/x</attachment>"
    seen = web_fetch.SeenUrls()
    seen.add_from_text(_speaker_text(typed, prompt))
    assert seen.allows("https://ciao-construct.be/")
    assert not seen.allows("https://attacker.example/x")


def test_a_scheme_less_link_outside_the_speakers_own_words_stays_unseen() -> None:
    from openexecutive.orchestrator.executive import _speaker_text

    typed = "What does ciao-construct.be say about the date?"
    prompt = f"{typed}\n\n<attachment>see attacker.example/x</attachment>"
    seen = web_fetch.SeenUrls()
    seen.add_from_text(_speaker_text(typed, prompt))
    assert seen._urls == {"https://ciao-construct.be/"}


def _seen_from_openrouter(message: dict[str, Any]) -> web_fetch.SeenUrls:
    """The seen set an OpenRouter response leaves: every block the translator
    returns goes through ``server_search_result_urls``, as the Executive's
    block parser does for the search blocks."""
    from openexecutive.providers.translator import from_openai_response

    seen = web_fetch.SeenUrls()
    msg = from_openai_response({"choices": [{"message": message, "finish_reason": "stop"}]})
    for block in msg.content:
        for url in web_fetch.server_search_result_urls(block):
            seen.add(url)
    return seen


def _url_citation(url: str, **fields: str) -> dict[str, Any]:
    return {"type": "url_citation", "url_citation": {"url": url, **fields}}


def test_an_openrouter_citation_is_seen_and_text_the_model_wrote_is_not() -> None:
    """Only the url of a citation counts. The same response's text, a tool
    call's arguments, and the citation's own title and excerpt are written
    by the model or the page, so a URL there stays unseen."""
    seen = _seen_from_openrouter({
        "content": "Also see https://text.example/x",
        "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "web_fetch", "arguments": json.dumps({"url": "https://arg.example/y"})}}],
        "annotations": [_url_citation(
            "https://cited.example/a",
            title="https://title.example/t",
            content="go to https://excerpt.example/e",
        )],
    })
    assert seen.allows("https://cited.example/a")
    for url in ("https://text.example/x", "https://arg.example/y",
                "https://title.example/t", "https://excerpt.example/e"):
        assert not seen.allows(url), url


@pytest.mark.parametrize(
    "url",
    ["javascript:alert(1)", "ftp://files.example/a", "https://user:pw@host.example/",
     "file:///etc/passwd"],
)
def test_a_citation_that_is_not_a_plain_https_address_is_not_seen(url: str) -> None:
    """The translator carries the string as given; SeenUrls.add refuses it."""
    seen = _seen_from_openrouter({"content": "x", "annotations": [_url_citation(url)]})
    assert not seen.allows(url)
    assert seen._urls == set()


def test_a_citation_is_compared_in_its_normal_form() -> None:
    seen = _seen_from_openrouter(
        {"content": "x", "annotations": [_url_citation("HTTPS://Example.COM/page#frag")]}
    )
    assert seen.allows("https://example.com/page")
    assert not seen.allows("https://example.com/page/")


@pytest.mark.parametrize(
    "text",
    [
        "aa." * 40_000 + "@",
        " a.bc" + "." * 100_000 + "x",
        "+a.bc" * 20_000,
        "example.com/" + "." * 100_000 + "\u300c",
        "example.com/" + "a\u2014" * 50_000,
        # Every match is normalised, which costs far more than the scan, so
        # this one is sized to stay inside the bound on a loaded CI runner.
        "\u201ca.bc\u201d" * 4_000,
    ],
    ids=[
        "label-run-to-at",
        "dot-run-after-host",
        "joined-hosts",
        "dot-run-path-to-wide-mark",
        "path-of-wide-marks",
        "curly-quoted-hosts",
    ],
)
def test_the_scheme_less_pass_stays_linear(text: str) -> None:
    """A look-ahead that rescans the rest of a run of labels from every place
    a host could end (an earlier check for an email "@") took several
    seconds on the first text. The end check scans only the "." and ":" run
    after a host end, and a match is tried only where a link may start."""
    seen = web_fetch.SeenUrls()
    started = time.perf_counter()
    seen.add_from_text(text)
    assert time.perf_counter() - started < 1.0


# --- the handler ------------------------------------------------------------


Route = tuple[int, dict[str, str], bytes]


def _serve(
    monkeypatch: pytest.MonkeyPatch, routes: dict[str, Route]
) -> list[str]:
    """Route each fetch through a MockTransport serving ``routes`` by URL.

    Returns the list of URLs actually requested. An unknown URL is a 599, so
    a test that reaches one fails on its assertion rather than on DNS.
    """
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        status, headers, body = routes.get(str(request.url), (599, {}, b""))
        return httpx.Response(status, headers=headers, content=body)

    monkeypatch.setattr(web_fetch, "_new_transport", lambda: httpx.MockTransport(handler))
    return requested


def _html(charset: str = "utf-8") -> dict[str, str]:
    return {"content-type": f"text/html; charset={charset}"}


def _fetcher(*seen_urls: str, max_uses: int = 3) -> tuple[Any, SearchBudget]:
    seen = web_fetch.SeenUrls()
    for url in seen_urls:
        seen.add(url)
    budget = SearchBudget(max_uses=max_uses)
    return web_fetch.make_fetch_handler(budget, seen), budget


async def test_a_seen_html_page_comes_back_as_text(monkeypatch: pytest.MonkeyPatch) -> None:
    page = (
        b"<html><head><style>p{}</style><script>evil()</script></head>"
        b"<body><h1>Ciao</h1><p>Opening &amp; date: 1 March</p></body></html>"
    )
    # Seen without a path, opened as typed: httpx sends "/" either way.
    requested = _serve(monkeypatch, {"https://ciao.example": (200, _html(), page)})
    handle, budget = _fetcher("https://ciao.example/")
    outcome = await handle({"url": "https://ciao.example"})
    assert not outcome.is_error
    result = json.loads(outcome.content)
    assert result == {
        "notice": web_fetch.UNTRUSTED_PAGE_NOTICE,
        "url": "https://ciao.example",
        "final_url": "https://ciao.example",
        "status": 200,
        "content": "Ciao Opening & date: 1 March",
        "truncated": False,
    }
    assert requested == ["https://ciao.example"]
    assert budget.used == 1


async def test_the_fetch_asks_for_an_uncompressed_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """httpx inflates a gzip chunk with no output limit, ahead of the cap."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"ok")

    monkeypatch.setattr(web_fetch, "_new_transport", lambda: httpx.MockTransport(handler))
    handle, _ = _fetcher("https://a.example/")
    await handle({"url": "https://a.example/"})
    assert sent[0].headers["accept-encoding"] == "identity"


@pytest.mark.parametrize("encoding", ["gzip", "gzip, gzip", "deflate", "br"])
async def test_a_compressed_page_is_refused_unread(
    encoding: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stacked gzip turns a few hundred bytes into hundreds of MB inside
    httpx, before any cap of ours could look at it."""
    reads = 0

    async def body() -> AsyncIterator[bytes]:
        nonlocal reads
        reads += 1
        yield b"\x1f\x8b"

    monkeypatch.setattr(web_fetch, "_new_transport", lambda: httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/html", "content-encoding": encoding},
            content=body(),
        )
    ))
    handle, _ = _fetcher("https://a.example/")
    outcome = await handle({"url": "https://a.example/"})
    assert outcome.is_error and "compressed page" in outcome.content
    assert reads == 0


def _propagate_app_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """api.main's logging setup turns propagation off on "openexecutive" when
    any earlier test on this worker imported the app; caplog listens on the
    root logger, so turn it back on for the test (as test_top_three does)."""
    monkeypatch.setattr(logging.getLogger("openexecutive"), "propagate", True)


async def test_an_unencodable_host_label_is_a_failed_lookup(
    local_server: tuple[int, list[bytes]], monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """getaddrinfo raises UnicodeError, not OSError, for a label over 63
    characters. The backend maps it to a connect failure, as for any failed
    lookup, rather than letting it escape httpx unmapped."""
    port, heads = local_server

    async def resolver(host: str, _port: int) -> list[str]:
        raise UnicodeError("label empty or too long")

    monkeypatch.setattr(web_fetch, "_new_transport", lambda: PinnedTransport(resolver))
    _propagate_app_logs(monkeypatch)
    url = f"http://site.example:{port}/"
    handle, _ = _fetcher(url)
    with caplog.at_level("WARNING", logger=web_fetch.logger.name):
        outcome = await handle({"url": url})
    assert outcome.is_error and "could not be reached" in outcome.content
    assert "connect failed" in caplog.text
    assert heads == []


async def test_plain_text_is_returned_as_is(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, {"https://a.example/robots.txt": (
        200, {"content-type": "text/plain"}, b"User-agent: *\n<b>not markup</b>",
    )})
    handle, _ = _fetcher("https://a.example/robots.txt")
    result = json.loads((await handle({"url": "https://a.example/robots.txt"})).content)
    assert result["content"] == "User-agent: *\n<b>not markup</b>"


async def test_a_long_page_is_cut_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, {"https://a.example/": (
        200, {"content-type": "text/plain"}, b"x" * 20_000,
    )})
    handle, _ = _fetcher("https://a.example/")
    result = json.loads((await handle({"url": "https://a.example/"})).content)
    assert len(result["content"]) == 12_000 and result["truncated"] is True


@pytest.mark.parametrize(
    ("charset", "body", "expected"),
    [
        ("iso-8859-1", "Café à côté".encode("latin-1"), "Café à côté"),
        ("windows-1252", "“quoted” – €5".encode("cp1252"), "“quoted” – €5"),
        ("no-such-codec", "naïve".encode(), "naïve"),
        ("rot13", b"plain", "plain"),
        ("idna", "naïve".encode(), "naïve"),
        ("punycode", "naïve".encode(), "naïve"),
        ("undefined", b"plain", "plain"),
        # Escape codecs can yield a lone surrogate, which no JSON body carries.
        ("unicode_escape", b"hi \\ud800 there", "hi ? there"),
        ("utf-7", b"hi +2AA- there", "hi ? there"),
    ],
)
async def test_the_declared_charset_decodes_the_page(
    charset: str, body: bytes, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, {"https://a.example/": (200, _html(charset), body)})
    handle, _ = _fetcher("https://a.example/")
    outcome = await handle({"url": "https://a.example/"})
    assert json.loads(outcome.content)["content"] == expected
    assert expected in outcome.content  # not \u-escaped: it counts once against the cap
    outcome.content.encode("utf-8")  # what a provider's JSON request body needs


@pytest.mark.parametrize(
    "tool_input", [None, [], "https://a.example/", {}, {"url": None}, {"url": 3}, {"url": "  "}]
)
async def test_a_malformed_input_is_an_error_not_a_crash(
    tool_input: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = _serve(monkeypatch, {})
    handle, budget = _fetcher("https://a.example/")
    outcome = await handle(tool_input)
    assert outcome.is_error and "'url'" in outcome.content
    assert requested == [] and budget.used == 0


async def test_an_unseen_url_is_refused_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = _serve(monkeypatch, {})
    handle, budget = _fetcher("https://example.com/page")
    outcome = await handle({"url": "https://example.com/page/"})
    assert outcome.is_error
    assert "has not appeared in this conversation" in outcome.content
    assert "paste it again" in outcome.content
    assert requested == [] and budget.used == 0


def _fetcher_from_text(text: str) -> tuple[Any, SearchBudget]:
    seen = web_fetch.SeenUrls()
    seen.add_from_text(text)
    budget = SearchBudget(max_uses=3)
    return web_fetch.make_fetch_handler(budget, seen), budget


@pytest.mark.parametrize(
    "url",
    [
        # Only the model wrote these: the speaker wrote example.com/page.
        "https://other.example/",
        "https://example.com/page?q=the+turn's+data",
        "http://example.com/page",
        # Never seeded: no dot, or a numeric final label.
        "https://localhost/admin",
        "https://10.0.0.1/x",
    ],
)
async def test_a_scheme_less_link_does_not_let_the_model_widen_the_set(
    url: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = _serve(monkeypatch, {})
    handle, budget = _fetcher_from_text("summarise example.com/page, localhost/admin, 10.0.0.1/x")
    outcome = await handle({"url": url})
    assert outcome.is_error and "has not appeared in this conversation" in outcome.content
    assert requested == [] and budget.used == 0


async def test_a_scheme_less_link_opens_as_https(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://example.com/page"
    requested = _serve(monkeypatch, {url: (200, _html(), b"<p>ok</p>")})
    handle, _ = _fetcher_from_text("summarise example.com/page")
    outcome = await handle({"url": url})
    assert not outcome.is_error and requested == [url]


async def test_a_seen_url_that_is_not_plain_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_url = "https://example.com/" + "a" * 600
    requested = _serve(monkeypatch, {})
    handle, budget = _fetcher(long_url)
    outcome = await handle({"url": long_url})
    assert outcome.is_error and "not a plain http(s) address" in outcome.content
    assert requested == [] and budget.used == 0


@pytest.mark.parametrize(
    ("key", "value", "url", "refusal"),
    [
        ("WEB_SEARCH_BLOCKED_DOMAINS", "example.com", "https://www.example.com/",
         "list of blocked domains"),
        ("WEB_SEARCH_BLOCKED_DOMAINS", "example.com", "https://evil-example.com/", None),
        ("WEB_SEARCH_ALLOWED_DOMAINS", "https://example.com/blog", "https://other.org/",
         "list of allowed domains"),
        ("WEB_SEARCH_ALLOWED_DOMAINS", "example.com", "https://docs.example.com/", None),
    ],
)
async def test_the_domain_lists_apply(
    key: str, value: str, url: str, refusal: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(key, value)
    requested = _serve(monkeypatch, {url: (200, _html(), b"<p>ok</p>")})
    handle, budget = _fetcher(url)
    outcome = await handle({"url": url})
    if refusal is None:
        assert not outcome.is_error and requested == [url]
    else:
        assert outcome.is_error and refusal in outcome.content
        assert requested == [] and budget.used == 0


async def test_the_domain_lists_apply_to_a_scheme_less_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEB_SEARCH_BLOCKED_DOMAINS", "example.com")
    requested = _serve(monkeypatch, {})
    handle, budget = _fetcher_from_text("summarise example.com/page")
    outcome = await handle({"url": "https://example.com/page"})
    assert outcome.is_error and "list of blocked domains" in outcome.content
    assert requested == [] and budget.used == 0


async def test_the_handler_refuses_on_a_turn_that_read_mail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.delegation.settings import TurnDelegation
    from openexecutive.orchestrator.schedule_tools import current_session
    from openexecutive.orchestrator.session import Session

    requested = _serve(monkeypatch, {})
    handle, budget = _fetcher("https://a.example/")
    session = Session()
    session.turn_delegation = TurnDelegation(  # type: ignore[attr-defined]
        offered=True, touched_mail=True, read_mail=True, session_id=session.session_id,
    )
    token = current_session.set(session)
    try:
        outcome = await handle({"url": "https://a.example/"})
    finally:
        current_session.reset(token)
    assert outcome.is_error
    assert "next message" in json.loads(outcome.content)["error"]
    assert requested == [] and budget.used == 0


async def test_concurrent_calls_cannot_overspend_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = _serve(monkeypatch, {"https://a.example/": (200, _html(), b"ok")})
    handle, budget = _fetcher("https://a.example/", max_uses=2)
    outcomes = await asyncio.gather(*(handle({"url": "https://a.example/"}) for _ in range(5)))
    assert sum(not o.is_error for o in outcomes) == 2
    assert sum("budget for this turn is spent" in o.content for o in outcomes) == 3
    assert len(requested) == 2 and budget.used == 2


async def test_a_failed_fetch_still_spends_its_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise a failing site would allow unlimited retries."""
    _serve(monkeypatch, {"https://a.example/": (503, {}, b"")})
    handle, budget = _fetcher("https://a.example/", max_uses=1)
    assert "HTTP 503" in (await handle({"url": "https://a.example/"})).content
    assert "spent" in (await handle({"url": "https://a.example/"})).content
    assert budget.used == 1


@pytest.mark.parametrize("content_type", ["application/pdf", "image/png", ""])
async def test_a_page_that_is_not_text_is_refused_unread(
    content_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = 0

    async def body() -> AsyncIterator[bytes]:
        nonlocal reads
        reads += 1
        yield b"%PDF-1.7"

    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"content-type": content_type} if content_type else {}
        return httpx.Response(200, headers=headers, content=body())

    monkeypatch.setattr(web_fetch, "_new_transport", lambda: httpx.MockTransport(handler))
    handle, _ = _fetcher("https://a.example/f")
    outcome = await handle({"url": "https://a.example/f"})
    assert outcome.is_error and "only HTML and plain-text" in outcome.content
    assert reads == 0


async def test_the_byte_cap_stops_the_read(monkeypatch: pytest.MonkeyPatch) -> None:
    reads = 0

    async def body() -> AsyncIterator[bytes]:
        nonlocal reads
        for _ in range(10):
            reads += 1
            yield b"x" * (1024 * 1024)

    monkeypatch.setattr(web_fetch, "_new_transport", lambda: httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-type": "text/plain"}, content=body())
    ))
    handle, _ = _fetcher("https://a.example/")
    outcome = await handle({"url": "https://a.example/"})
    assert outcome.is_error and "too large" in outcome.content
    assert reads == 3  # two 1 MiB chunks reach the 2 MiB cap; the third crosses it


async def test_a_slow_drip_hits_the_whole_exchange_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """httpx's timeout is per operation; each chunk here beats it."""
    monkeypatch.setenv("WEB_FETCH_TIMEOUT_S", "0.2")

    async def body() -> AsyncIterator[bytes]:
        for _ in range(50):
            await asyncio.sleep(0.05)
            yield b"x"

    monkeypatch.setattr(web_fetch, "_new_transport", lambda: httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-type": "text/plain"}, content=body())
    ))
    handle, _ = _fetcher("https://a.example/")
    started = asyncio.get_running_loop().time()
    outcome = await handle({"url": "https://a.example/"})
    assert outcome.is_error and "did not load in time" in outcome.content
    assert asyncio.get_running_loop().time() - started < 1.5


# --- redirects --------------------------------------------------------------


def _redirect(location: str, status: int = 302) -> Route:
    return (status, {"location": location}, b"")


async def test_a_relative_redirect_is_followed_without_being_seen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = _serve(monkeypatch, {
        "http://a.example/": _redirect("https://a.example/", 301),
        "https://a.example/": _redirect("/nl/home"),
        "https://a.example/nl/home": (200, _html(), b"<p>Welkom</p>"),
    })
    handle, _ = _fetcher("http://a.example/")
    result = json.loads((await handle({"url": "http://a.example/"})).content)
    assert result["content"] == "Welkom"
    assert result["final_url"] == "https://a.example/nl/home"
    assert requested == ["http://a.example/", "https://a.example/", "https://a.example/nl/home"]


async def test_the_redirect_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = _serve(monkeypatch, {
        f"https://a.example/{i}": _redirect(f"/{i + 1}") for i in range(10)
    })
    handle, _ = _fetcher("https://a.example/0")
    outcome = await handle({"url": "https://a.example/0"})
    assert outcome.is_error and "more than 3 times" in outcome.content
    assert len(requested) == 4  # the URL itself plus three redirects


@pytest.mark.parametrize(
    "location",
    ["https://blocked.example/x", "file:///etc/passwd", "https://user:pw@a.example/"],
)
async def test_a_redirect_is_checked_like_the_first_url(
    location: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WEB_SEARCH_BLOCKED_DOMAINS", "blocked.example")
    requested = _serve(monkeypatch, {"https://a.example/": _redirect(location)})
    handle, _ = _fetcher("https://a.example/")
    outcome = await handle({"url": "https://a.example/"})
    assert outcome.is_error and "redirected to an address" in outcome.content
    assert requested == ["https://a.example/"]


async def test_a_redirect_from_a_scheme_less_link_is_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEB_SEARCH_BLOCKED_DOMAINS", "blocked.example")
    requested = _serve(monkeypatch, {"https://a.example/": _redirect("https://blocked.example/x")})
    handle, _ = _fetcher_from_text("what is on a.example")
    outcome = await handle({"url": "https://a.example/"})
    assert outcome.is_error and "redirected to an address" in outcome.content
    assert requested == ["https://a.example/"]


# --- the pinned connect, through the handler ----------------------------------


def _pin(monkeypatch: pytest.MonkeyPatch, table: dict[str, list[str]]) -> list[str]:
    """Fetch through the real PinnedTransport with ``table`` as DNS."""
    lookups: list[str] = []

    async def resolver(host: str, _port: int) -> list[str]:
        lookups.append(host)
        return table[host]

    monkeypatch.setattr(web_fetch, "_new_transport", lambda: PinnedTransport(resolver))
    return lookups


@pytest.mark.parametrize(
    "addresses",
    [
        ["127.0.0.1"],
        ["10.0.0.1"],
        ["169.254.169.254"],
        ["fe80::1"],
        ["::ffff:127.0.0.1"],
        ["64:ff9b::7f00:1"],
        ["93.184.216.34", "127.0.0.1"],
    ],
)
async def test_a_non_public_resolution_is_refused(
    addresses: list[str], local_server: tuple[int, list[bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port, heads = local_server
    _pin(monkeypatch, {"internal.example": addresses})
    url = f"http://internal.example:{port}/"
    handle, _ = _fetcher(url)
    outcome = await handle({"url": url})
    assert outcome.is_error and "private or internal address" in outcome.content
    assert heads == []


async def test_a_scheme_less_link_to_a_private_address_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """127.0.0.1.nip.io has an alphabetic final label, so it is seeded; the
    pinned connect still refuses the loopback address it resolves to."""
    lookups = _pin(monkeypatch, {"127.0.0.1.nip.io": ["127.0.0.1"]})
    handle, budget = _fetcher_from_text("open 127.0.0.1.nip.io/x")
    outcome = await handle({"url": "https://127.0.0.1.nip.io/x"})
    assert outcome.is_error and "private or internal address" in outcome.content
    assert lookups == ["127.0.0.1.nip.io"] and budget.used == 1


async def test_a_redirect_to_a_private_address_is_refused(
    local_server: tuple[int, list[bytes]], local_server_reply: list[bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first hop is allowed and answers with a redirect; the second
    resolves to an address the policy refuses, so it is never connected to.

    127.0.0.1 stands in for a public address (it is where the test server
    listens); 127.0.0.2 is the refused one.
    """
    port, heads = local_server
    monkeypatch.setattr(web_fetch, "is_public_address", lambda a: a == "127.0.0.1")
    lookups = _pin(monkeypatch, {
        "public.example": ["127.0.0.1"], "internal.example": ["127.0.0.2"],
    })
    redirect_to = f"http://internal.example:{port}/".encode()
    local_server_reply[:] = [
        b"HTTP/1.1 302 Found\r\nLocation: " + redirect_to
        + b"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    ]
    url = f"http://public.example:{port}/"
    handle, _ = _fetcher(url)
    outcome = await handle({"url": url})
    assert outcome.is_error and "private or internal address" in outcome.content
    assert lookups == ["public.example", "internal.example"]
    assert len(heads) == 1 and b"Host: public.example" in heads[0]


@pytest.mark.parametrize(
    "reply",
    [
        (500, {}, b""),  # the HTTP-status branch
        _redirect("http://[bad"),  # httpx raises, logged with exc_info
        _redirect("https://xn--/"),  # an IDNA error httpx does not map
    ],
)
async def test_failures_never_log_the_path_or_query(
    reply: Route, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    url = "https://a.example/private/path?token=s3cret"
    _serve(monkeypatch, {url: reply})
    _propagate_app_logs(monkeypatch)
    handle, _ = _fetcher(url)
    with caplog.at_level("WARNING", logger=web_fetch.logger.name):
        outcome = await handle({"url": url})
    assert outcome.is_error and "unexpectedly" not in outcome.content  # not the backstop
    assert "a.example" in caplog.text
    assert "s3cret" not in caplog.text and "/private/path" not in caplog.text


async def test_an_unexpected_failure_is_an_error_not_a_raise(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """One raised handler would abort the Executive's whole gather round."""

    async def boom(url: str, settings: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(web_fetch, "_run_fetch", boom)
    _propagate_app_logs(monkeypatch)
    url = "https://a.example/private/path?token=s3cret"
    handle, budget = _fetcher(url)
    with caplog.at_level("WARNING", logger=web_fetch.logger.name):
        outcome = await handle({"url": url})
    assert outcome.is_error and outcome.content == "The fetch failed unexpectedly."
    assert budget.used == 1
    assert "unexpected failure for host a.example" in caplog.text
    assert "s3cret" not in caplog.text and "/private/path" not in caplog.text
