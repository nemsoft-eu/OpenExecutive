"""Anthropic native server-side web search tool.

The `web_search_20250305` tool is executed by the Anthropic API itself — no
client-side handler is needed. The model issues `server_tool_use` blocks
during a generation and the API returns `web_search_tool_result` blocks
inline; both must be preserved in the assistant turn echoed back on the
next iteration so the model retains context for what it searched.

Because the block carries a `type` field instead of an `input_schema`, it
cannot be cached the way client-side tools are. The Executive keeps it
separate from the alphabetically-sorted client-tool list so the
`cache_control` marker stays on a real client tool.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from openexecutive.config import get_settings

WEB_SEARCH_TOOL_NAME = "web_search"


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
    """

    kind: Literal["server", "client"]
    tool: dict[str, Any]
    max_uses: int


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

    # Imported here rather than at module scope: the registry pulls in every
    # provider, and this module is imported by the Executive at startup.
    from openexecutive.providers.registry import supports_server_web_search

    if supports_server_web_search(model):
        server_tool = build_web_search_tool(max_uses=effective_max)
        assert server_tool is not None  # the enable flag was checked above
        return WebSearchSelection("server", server_tool, effective_max)

    if not settings.searxng_url:
        return None

    # searxng_search imports WEB_SEARCH_TOOL_NAME from this module, so a
    # module-level import here would be circular.
    from openexecutive.orchestrator.searxng_search import SEARXNG_WEB_SEARCH_TOOL

    return WebSearchSelection("client", SEARXNG_WEB_SEARCH_TOOL, effective_max)


def client_search_handlers(selection: WebSearchSelection | None) -> dict[str, Any]:
    """Handler map for ``selection``: one budgeted handler, or empty.

    Only the client variant needs a handler; the server variant runs inside
    the provider's generation. Each call builds a FRESH budget, so call it
    once per turn (or per research call) — never inside a tool-use loop,
    where a new budget per iteration would reset on every model round trip
    and multiply the cap.
    """
    if selection is None or selection.kind != "client":
        return {}
    from openexecutive.orchestrator.searxng_search import (
        SearchBudget,
        make_search_handler,
    )

    return {
        WEB_SEARCH_TOOL_NAME: make_search_handler(
            SearchBudget(max_uses=selection.max_uses)
        )
    }


def client_tool_rounds(selection: WebSearchSelection | None) -> int:
    """Generations a client-tool loop may take for ``selection``.

    One round per search plus the round that emits the caller's output. A
    server or absent selection needs no loop: the single call is the answer.
    """
    if selection is None or selection.kind != "client":
        return 1
    return selection.max_uses + 1
