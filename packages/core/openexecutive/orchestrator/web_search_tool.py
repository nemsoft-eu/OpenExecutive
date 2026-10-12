"""Anthropic native server-side web search and fetch tools, and the choice
between them and the client-side ones (``searxng_search``, ``web_fetch``).

The `web_search_20250305` and `web_fetch_20260309` tools are executed by
the Anthropic API itself — no client-side handler is needed. The model
issues `server_tool_use` blocks during a generation and the API returns
`web_search_tool_result` / `web_fetch_tool_result` blocks inline; both must
be preserved in the assistant turn echoed back on the next iteration so the
model retains context for what it searched and read.

Because the block carries a `type` field instead of an `input_schema`, it
cannot be cached the way client-side tools are. The Executive keeps it
separate from the alphabetically-sorted client-tool list so the
`cache_control` marker stays on a real client tool.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from openexecutive.config import get_settings

if TYPE_CHECKING:
    from openexecutive.orchestrator.web_fetch import SeenUrls

WEB_SEARCH_TOOL_NAME = "web_search"
WEB_FETCH_TOOL_NAME = "web_fetch"

# Roughly what one page may put in the prompt through the server fetch tool:
# the same reading budget as the client tool's page cap (web_fetch.py), so
# one page cannot crowd out the rest of the turn on either path.
WEB_FETCH_MAX_CONTENT_TOKENS = 4_000


def build_web_search_tool(*, max_uses: int | None = None) -> dict[str, Any] | None:
    """Return the Anthropic web_search tool dict, or None when disabled.

    Read at tool-list assembly time (per turn) so toggling the env var
    without a restart takes effect on the next turn. ``max_uses`` overrides
    the shared ``WEB_SEARCH_MAX_USES`` knob for callers with their own
    (the research fan-out uses ``RESEARCH_WEB_SEARCH_MAX_USES``).
    """
    settings = get_settings()
    if not settings.enable_web_search:
        return None

    tool: dict[str, Any] = {
        "type": "web_search_20250305",
        "name": WEB_SEARCH_TOOL_NAME,
        "max_uses": max_uses if max_uses is not None else settings.web_search_max_uses,
    }
    if settings.web_search_allowed_domains:
        tool["allowed_domains"] = settings.web_search_allowed_domains
    if settings.web_search_blocked_domains:
        tool["blocked_domains"] = settings.web_search_blocked_domains
    return tool


def build_web_fetch_tool() -> dict[str, Any]:
    """Return Anthropic's web_fetch server tool dict.

    The API itself fetches only URLs already present in the conversation,
    so unlike the client tool there is no seen-URL set to keep here.
    """
    settings = get_settings()
    tool: dict[str, Any] = {
        "type": "web_fetch_20260309",
        "name": WEB_FETCH_TOOL_NAME,
        "max_uses": settings.web_fetch_max_uses,
        "max_content_tokens": WEB_FETCH_MAX_CONTENT_TOKENS,
    }
    if settings.web_search_allowed_domains:
        tool["allowed_domains"] = settings.web_search_allowed_domains
    if settings.web_search_blocked_domains:
        tool["blocked_domains"] = settings.web_search_blocked_domains
    return tool


@dataclass(frozen=True)
class WebSearchSelection:
    """Which of the two search implementations a call should use.

    ``kind`` decides where the tool dict goes in the request, and the two
    places are not interchangeable:

      * ``"server"`` — Anthropic's (or, after translation, OpenRouter's)
        server tool. It carries a ``type`` instead of an ``input_schema``,
        so it must be appended AFTER the sorted client-tool list and must
        never receive a ``cache_control`` marker.
      * ``"client"`` — the SearXNG tool. It is an ordinary function tool
        and belongs inside the sorted client list, where it may end up
        carrying the cache marker like any other.

    ``max_uses`` is carried so a caller with its own cap (the research
    fan-out's ``RESEARCH_WEB_SEARCH_MAX_USES``) can size the client-side
    budget with the same number it puts on the server tool.

    ``fetch_tool`` is the matching ``web_fetch`` tool of the same kind, or
    None where there is none (a Claude id on OpenRouter, or
    ``WEB_FETCH_MAX_USES=0``). It goes where ``tool`` goes. Only the chat
    loop offers it today: the research fan-out and standing queries read
    ``tool`` alone and build their handlers without a seen-URL set, which
    leaves the client fetch handler out.
    """

    kind: Literal["server", "client"]
    tool: dict[str, Any]
    max_uses: int
    fetch_tool: dict[str, Any] | None = None
    fetch_max_uses: int = 0


def select_web_search_tool(
    model: str, *, max_uses: int | None = None
) -> WebSearchSelection | None:
    """Pick the search implementation for ``model``, or None when there is none.

    Server-side search wherever the provider runs it inside the generation;
    otherwise the client-side SearXNG tool when ``SEARXNG_URL`` is set. A
    local model with no SearXNG configured gets nothing, which is the
    behaviour that existed before the client tool.

    ``model`` must be the model the request will actually be sent with —
    the Council override or the research model, not an agent's default.
    Selecting against a different model than the call uses is how a request
    ends up carrying a tool its provider will strip or reject.
    """
    settings = get_settings()
    if not settings.enable_web_search:
        return None

    effective_max = (
        max_uses if max_uses is not None else settings.web_search_max_uses
    )

    fetch_max = settings.web_fetch_max_uses

    # Imported here rather than at module scope: the registry pulls in every
    # provider, and this module is imported by the Executive at startup.
    from openexecutive.providers.registry import (
        supports_server_web_fetch,
        supports_server_web_search,
    )

    if supports_server_web_search(model):
        server_tool = build_web_search_tool(max_uses=effective_max)
        assert server_tool is not None  # the enable flag was checked above
        server_fetch = (
            build_web_fetch_tool()
            if fetch_max > 0 and supports_server_web_fetch(model)
            else None
        )
        return WebSearchSelection(
            "server", server_tool, effective_max, server_fetch,
            fetch_max if server_fetch is not None else 0,
        )

    if not settings.searxng_url:
        return None

    # searxng_search and web_fetch import names from this module, so a
    # module-level import here would be circular.
    from openexecutive.orchestrator.searxng_search import SEARXNG_WEB_SEARCH_TOOL
    from openexecutive.orchestrator.web_fetch import WEB_FETCH_TOOL

    client_fetch = WEB_FETCH_TOOL if fetch_max > 0 else None
    return WebSearchSelection(
        "client", SEARXNG_WEB_SEARCH_TOOL, effective_max, client_fetch,
        fetch_max if client_fetch is not None else 0,
    )


def client_search_handlers(
    selection: WebSearchSelection | None, seen: SeenUrls | None = None
) -> dict[str, Any]:
    """Handler map for ``selection``: the budgeted client handlers, or empty.

    Only the client variant needs a handler; the server variant runs inside
    the provider's generation. Each call builds a FRESH budget, so call it
    once per turn (or per research call) — never inside a tool-use loop,
    where a new budget per iteration would reset on every model round trip
    and multiply the cap.

    With ``seen`` (the chat loop's turn), a selection carrying a fetch tool
    also gets the ``web_fetch`` handler, with its own budget, and every
    search result URL joins ``seen`` so it can be opened. Without it (the
    research fan-out, standing queries) only the search handler is built.
    """
    if selection is None or selection.kind != "client":
        return {}
    from openexecutive.orchestrator.searxng_search import (
        SearchBudget,
        make_search_handler,
    )

    handlers: dict[str, Any] = {
        WEB_SEARCH_TOOL_NAME: make_search_handler(
            SearchBudget(max_uses=selection.max_uses),
            on_result_url=seen.add if seen is not None else None,
        )
    }
    if seen is not None and selection.fetch_tool is not None:
        from openexecutive.orchestrator.web_fetch import make_fetch_handler

        handlers[WEB_FETCH_TOOL_NAME] = make_fetch_handler(
            SearchBudget(max_uses=selection.fetch_max_uses), seen
        )
    return handlers


def client_tool_rounds(selection: WebSearchSelection | None) -> int:
    """Generations a client-tool loop may take for ``selection``.

    One round per search plus the round that emits the caller's output. A
    server or absent selection needs no loop: the single call is the answer.
    """
    if selection is None or selection.kind != "client":
        return 1
    return selection.max_uses + 1
