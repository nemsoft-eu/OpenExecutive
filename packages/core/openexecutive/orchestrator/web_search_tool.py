"""Anthropic native server-side web search tool, the choice between it and
the client-side SearXNG one (``searxng_search``), and the client-side
``web_fetch`` that goes with either.

The `web_search_20250305` tool is executed by the Anthropic API itself — no
client-side handler is needed. The model issues `server_tool_use` blocks
during a generation and the API returns `web_search_tool_result` blocks
inline; both must be preserved in the assistant turn echoed back on the
next iteration so the model retains context for what it searched.

Because the block carries a `type` field instead of an `input_schema`, it
cannot be cached the way client-side tools are. The Executive keeps it
separate from the alphabetically-sorted client-tool list so the
`cache_control` marker stays on a real client tool.

Page reads are always the client tool, never Anthropic's ``web_fetch``
server tool: that one opens any URL in a client tool result, including one
echoing text the model wrote (a specialist's answer quoting it), so an
injected instruction could send the turn's data out in a query string. The
client tool opens only URLs the person wrote or a search returned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from openexecutive.config import get_settings

if TYPE_CHECKING:
    from openexecutive.orchestrator.web_fetch import SeenUrls

WEB_SEARCH_TOOL_NAME = "web_search"
WEB_FETCH_TOOL_NAME = "web_fetch"


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

    ``fetch_tool`` is the client-side ``web_fetch`` tool, whatever the
    search kind, or None when its cap (``WEB_FETCH_MAX_USES``, or the
    caller's ``fetch_max_uses``) is 0. It is a function tool, so it always
    joins the sorted client list. Carrying it does not offer it: a caller
    offers it by listing it and passing a seen-URL set to
    ``client_search_handlers``. The chat loop does, whatever the search
    kind; the research fan-out does beside the client search only
    (``monitoring.research.specialist_research``); standing queries never
    do (``monitoring.sources.query``).
    """

    kind: Literal["server", "client"]
    tool: dict[str, Any]
    max_uses: int
    fetch_tool: dict[str, Any] | None = None
    fetch_max_uses: int = 0


def select_web_search_tool(
    model: str,
    *,
    max_uses: int | None = None,
    fetch_max_uses: int | None = None,
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

    ``max_uses`` and ``fetch_max_uses`` override the chat knobs
    (``WEB_SEARCH_MAX_USES``, ``WEB_FETCH_MAX_USES``) for a caller with its
    own caps: the research fan-out passes ``RESEARCH_WEB_SEARCH_MAX_USES``
    and ``RESEARCH_WEB_FETCH_MAX_USES``.
    """
    settings = get_settings()
    if not settings.enable_web_search:
        return None

    effective_max = (
        max_uses if max_uses is not None else settings.web_search_max_uses
    )

    # Imported here rather than at module scope: the registry pulls in every
    # provider, and this module is imported by the Executive at startup.
    # searxng_search and web_fetch import names from this module, so a
    # module-level import of either would be circular.
    from openexecutive.orchestrator.web_fetch import WEB_FETCH_TOOL
    from openexecutive.providers.registry import supports_server_web_search

    fetch_max = (
        fetch_max_uses if fetch_max_uses is not None else settings.web_fetch_max_uses
    )
    fetch_tool = WEB_FETCH_TOOL if fetch_max > 0 else None

    if supports_server_web_search(model):
        server_tool = build_web_search_tool(max_uses=effective_max)
        assert server_tool is not None  # the enable flag was checked above
        return WebSearchSelection(
            "server", server_tool, effective_max,
            fetch_tool=fetch_tool, fetch_max_uses=fetch_max,
        )

    if not settings.searxng_url:
        return None

    from openexecutive.orchestrator.searxng_search import SEARXNG_WEB_SEARCH_TOOL

    return WebSearchSelection(
        "client", SEARXNG_WEB_SEARCH_TOOL, effective_max,
        fetch_tool=fetch_tool, fetch_max_uses=fetch_max,
    )


def client_search_handlers(
    selection: WebSearchSelection | None, seen: SeenUrls | None = None
) -> dict[str, Any]:
    """Handler map for ``selection``: the budgeted client handlers, or empty.

    The client search needs a handler; the server search runs inside the
    provider's generation. Each call builds a FRESH budget, so call it
    once per turn (or per research call) — never inside a tool-use loop,
    where a new budget per iteration would reset on every model round trip
    and multiply the cap.

    With ``seen``, a selection carrying a fetch tool also gets the
    ``web_fetch`` handler, with its own budget, whatever the search kind,
    and every client search result URL joins ``seen`` so it can be opened.
    The chat loop passes its turn's set (and adds the server search's
    results itself); the research fan-out passes an empty one beside the
    client search, so only its own results become openable. Without it
    (standing queries, and research beside a server search) only the
    client search handler is built, if the search is the client one.
    """
    if selection is None:
        return {}
    from openexecutive.orchestrator.searxng_search import (
        SearchBudget,
        make_search_handler,
    )

    handlers: dict[str, Any] = {}
    if selection.kind == "client":
        handlers[WEB_SEARCH_TOOL_NAME] = make_search_handler(
            SearchBudget(max_uses=selection.max_uses),
            on_result_url=seen.add if seen is not None else None,
        )
    if seen is not None and selection.fetch_tool is not None:
        from openexecutive.orchestrator.web_fetch import make_fetch_handler

        handlers[WEB_FETCH_TOOL_NAME] = make_fetch_handler(
            SearchBudget(max_uses=selection.fetch_max_uses), seen
        )
    return handlers


def client_tool_rounds(
    selection: WebSearchSelection | None, *, with_fetch: bool = False
) -> int:
    """Generations a client-tool loop may take for ``selection``.

    One round per search plus the round that emits the caller's output, and
    with ``with_fetch`` (the caller built the fetch handler) one per page
    read too. A server or absent selection needs no loop: the single call
    is the answer.

    The count is a bound, not a reservation. A round that runs no search
    and reads no page still spends a round: a refused open (a URL the run
    has not seen, a site the domain lists refuse), a call over its budget,
    an unknown tool. So a specialist that wastes one with every search already used
    reaches the final round, where the handled tools are withdrawn, without
    having read a page. The rounds are not extended for that: the bound is
    what caps a call's generations, and so its cost and its share of the
    timeout, and a refusal is only an error tool_result to the loop, the
    same as a fetch that ran and failed. The research call tells the model
    which URLs it can open instead
    (``monitoring.research.prompts.RESEARCH_FETCH_ADDENDUM``).
    """
    if selection is None or selection.kind != "client":
        return 1
    fetches = (
        selection.fetch_max_uses
        if with_fetch and selection.fetch_tool is not None
        else 0
    )
    return selection.max_uses + fetches + 1


def _result_error(block: dict[str, Any]) -> str | None:
    """The error code of a ``web_search_tool_result`` block, or None when it
    holds results."""
    content = block.get("content")
    if isinstance(content, dict) and content.get("type") == "web_search_tool_result_error":
        return str(content.get("error_code") or "error")
    return None


def searches_run(blocks: list[dict[str, Any]]) -> int:
    """How many web searches in this round's blocks ran: every search but
    one the limit stopped (``max_uses_exceeded``), which never ran."""
    stopped = {
        str(b.get("tool_use_id")) for b in blocks
        if b.get("type") == "web_search_tool_result" and _result_error(b) == "max_uses_exceeded"
    }
    return sum(
        1 for b in blocks
        if b.get("type") == "server_tool_use" and b.get("name") == WEB_SEARCH_TOOL_NAME
        and str(b.get("id")) not in stopped
    )


def turn_search_tool() -> dict[str, Any] | None:
    """The web_search tool for every round of a chat turn: one budget for
    the whole turn (``WEB_SEARCH_TURN_BUDGET``), so an ask with many facts is
    looked up to the end, not cut off after a round's first few searches.
    The same tool every round, since the tools lead the cached prompt and a
    changed ``max_uses`` would miss the cache on every later round; once the
    turn has spent its budget the round's results say so instead
    (``SEARCHES_SPENT_NOTE``), and a search after that ends the turn on its
    write-up (``SEARCHES_OVER_NOTE``). None when search is off."""
    return build_web_search_tool(max_uses=get_settings().web_search_turn_budget)


# Added to a round's results once the turn has run its searches. Its other
# steps still run; a search after it makes the next round the write-up
# (SEARCHES_OVER_NOTE), as does a round a long search paused past it. The
# tool itself can't shrink without missing the cache, so the write-up round
# could still search: a turn that ignores every note runs at most about four
# budgets; one that heeds them stops at the budget.
SEARCHES_SPENT_NOTE = (
    "This turn has used its web searches. Don't search again: finish the answer from what the searches"
    " found and what the company knows, and say plainly what is still unconfirmed."
)
SEARCHES_OVER_NOTE = (
    "This turn has run past its web searches. Write the answer now from what you have, with no more"
    " searches or other steps, and say plainly what is still unconfirmed."
)
SEARCHES_EMPTY_REPLY = (
    "I've used all the web searches I get for one message. Ask me to carry on and I'll look up the rest."
)


def spent(used: int) -> bool:
    """Whether a turn that has run ``used`` searches has spent its budget."""
    return used >= get_settings().web_search_turn_budget


def searched(blocks: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Each web search in this round's blocks as (call, result) for the
    after-reply checks (``claim_check`` / ``reply_check``): a search is a
    look, like any read."""
    results = {
        str(b.get("tool_use_id")): b for b in blocks if b.get("type") == "web_search_tool_result"
    }
    calls: list[tuple[str, str]] = []
    for b in blocks:
        if b.get("type") != "server_tool_use" or b.get("name") != WEB_SEARCH_TOOL_NAME:
            continue
        query = (b.get("input") or {}).get("query", "")
        result = results.get(str(b.get("id")))
        if result is None:
            outcome = "no result"
        elif (error := _result_error(result)) is not None:
            outcome = f"error: {error}"
        else:
            found = result.get("content")
            titles = [
                str(r.get("title", "")) for r in (found if isinstance(found, list) else []) if isinstance(r, dict)
            ]
            outcome = f"{len(titles)} results: " + "; ".join(t for t in titles[:5] if t)
        calls.append((f"{WEB_SEARCH_TOOL_NAME} {query}".strip(), outcome))
    return calls
