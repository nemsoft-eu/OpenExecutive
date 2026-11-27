"""FastMCP server definition: resources + tools + the FastAPI mount.

Design notes
------------
* **Transport.** Streamable-HTTP, mounted into the existing FastAPI app at
  ``/mcp``. We set ``streamable_http_path="/"`` and ``app.mount("/mcp", ...)``
  so the external endpoint is exactly ``/mcp`` (no double ``/mcp/mcp``).
  Compliant MCP clients follow the trailing-slash 307 to ``/mcp/``.

* **Lifespan.** Mounting a sub-app does NOT run its lifespan, so
  ``mcp.session_manager.run()`` is chained into the API lifespan in
  ``api/main.py`` — without it every ``/mcp`` request 500s.

* **Auth.** The endpoint is intentionally NOT added to the API's
  ``_UNAUTHENTICATED_PATHS``; the existing shared-secret middleware gates it,
  so clients pass ``x-api-key: $BACKEND_SHARED_SECRET`` like the UI does.

* **Who is asking.** Every handler answers as the caller the API already
  verified (``api.caller``): the mounted app shares the outer request's
  ``request.state``, so ``caller_gate``'s verdict reaches the handler through
  FastMCP's request context (``_request``). A deployment that fronts ``/mcp``
  with its own sign-in and stamps the signed caller, as the UI proxy does,
  gets per-person answers by the web app's rules: a signed-in email is that
  person (no one when off the roster); open mode with no ``x-caller-email``,
  or an operator, is the principal; a service (signed callers on, no
  assertion) is no one. Episodic memory, the activity feed, the briefing and
  the roster go to a rostered caller, as the web app gives them to anyone on
  the team, the briefing narrowed as ``GET /today`` narrows it, and so do
  the watches, which name the companies and people being watched; Needs
  you, Done for you, tasks and Approvals are read through the web app's
  own routes with the same request, so they follow its rules exactly; the
  company profile, departments, specialists, knowledge and the workflow
  catalog go to anyone who reached ``/mcp``.

* **An AI app is the person signed in to it.** ``ask_executive``,
  ``run_workflow`` and ``answer_card`` each run one Executive turn marked
  ``Session.from_mcp_client`` for the verified caller, the same turn the web
  chat runs: Approvals, then Your rules, then In training decide each step
  (``orchestrator.levels``), and nothing else does. Every audit row and card
  the turn writes names the app (``audit.via_app``). A caller that is no one
  (a service) is someone not in People, so everything it asks to send or
  change waits on the owner's card.

* **post_progress_update writes and sends outside Approvals.** It is the one
  tool that does, and it does not depend on who the caller is: the update is
  filed under the source its ``x-source-token`` header belongs to
  (``integrations.progress_sources``, a hashed per-source token, so the
  source is never an argument and the token never reaches model context),
  and with a shared secret configured the request must also have passed the
  gate. ``integrations.progress_updates.submit`` holds every rule: the
  content guard rejects personal data, credentials, markup and stray URLs;
  per-source daily and relay caps bound the volume; the text goes only to
  the active principals, only to their chat apps, quoted behind a header
  the server writes. A missing or unknown token, the feature off, or an
  in-process call gets one fixed refusal.

* **DNS-rebinding protection** (FastMCP's Host-header check) is disabled: the
  endpoint authenticates with a header secret (not a cookie), so a browser
  rebinding attack cannot supply credentials, and the server runs behind the
  deployment's TLS terminator. The shared-secret gate is the real access
  control.

* **No prompt-caching impact.** This path is parallel to chat. Resources are
  pure reads; ``consult_specialist`` delegates to ``BaseAgent.analyze`` which
  reuses the existing cached system blocks unchanged.

* **State.** Handlers get no FastAPI ``Request`` argument (only the caller
  read above goes through the request). Most service functions build
  their own SQLite/ChromaDB handles; the one that benefits from the app's warm
  store (``search_knowledge``) reads it from the ``set_store`` singleton.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, Literal, get_args

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field

from openexecutive.api import caller as api_caller
from openexecutive.integrations.progress_guard import STATUSES
from openexecutive.integrations.progress_sources import TOKEN_HEADER as SOURCE_TOKEN_HEADER

logger = logging.getLogger(__name__)

# Sorted roster of specialist keys, kept in lockstep with
# ``orchestrator/router.SPECIALIST_REGISTRY`` by a unit test. This is the
# canonical closed set, used to derive the advertised enum and by
# ``specialist_keys()``.
#
# Do NOT annotate a tool parameter with it. FastMCP validates arguments against
# the handler's type hints before the body runs, so a ``Literal`` parameter is a
# hard schema-level rejection of anything outside the set — see ``SpecialistArg``
# for why that is the wrong contract for a model-emitted value.
SpecialistKey = Literal[
    "board_comms", "cfo", "chro", "cmo", "coo", "cpo", "cso", "gc", "sales", "triage"
]

# What the ``consult_specialist`` tool accepts on the wire. The roster above is
# advertised verbatim in the generated JSON Schema — clients still see the ten
# canonical values — but the annotation is ``str``, so *server-side* validation
# does not reject a value outside it and `resolve_specialist_name` gets its
# chance to recover a re-cased or truncated name like `csO` or `cs`. With a bare
# ``Literal`` the resolver was unreachable: Pydantic raised `literal_error`
# first, which both diverged from the chat path and lost the `routing_anomaly`
# audit row `route_to_specialist` writes when it corrects a name.
#
# The tolerance is server-side only, and deliberately so. The advertised schema
# still carries `enum`, so a client that validates outbound arguments against it
# will refuse to send `csO` before the request leaves — correct behaviour, since
# a client that can tell the name is wrong should say so early. What this buys
# is the case the resolver was written for: a *model* emitting a truncated name
# through a client that passes arguments straight through.
SpecialistArg = Annotated[
    str, Field(json_schema_extra={"enum": list(get_args(SpecialistKey))})
]

_INSTRUCTIONS = (
    "Open Executive: a company's Executive and its council of specialists, "
    "answering as the person signed in.\n\n"
    "Read: the company profile, today's brief, Needs you (the cards waiting "
    "on this person), Done for you, tasks, watches, departments, people, "
    "Approvals (what it may do on its own), and past decisions, initiatives "
    "and advice.\n\n"
    "Do: `ask_executive` (one answer from the Executive, who can also act), "
    "`answer_card` (Approve, Skip or reply to a Needs you card), "
    "`run_workflow` (start a workflow from `list_workflows`), "
    "`consult_specialist` (one specialist's analysis) and `search_knowledge` "
    "(the knowledge base and company documents).\n\n"
    "This app acts as the person, under their Approvals: anything that "
    "reaches someone else goes on its own or waits on a card for them, as it "
    "would in Open Executive.\n\n"
    "`post_progress_update` is for other agents working for the founders: it "
    "stores a short progress update and sends it, quoted as received, to the "
    "founders' chat apps. It needs a source token in the `x-source-token` "
    "header and is refused without one."
)

# What the API's shared-secret gate sets on a request it let through.
SHARED_SECRET_OK = "shared_secret_ok"
_CLIENT_NAME_MAX = 60

mcp = FastMCP(
    name="open-executive",
    instructions=_INSTRUCTIONS,
    stateless_http=True,
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    ),
)


# ---------------------------------------------------------------------------
# Process-wide store handoff (mirrors mcp_gateway.set_active_gateway).
# ---------------------------------------------------------------------------
_active_store: Any = None


def set_store(store: Any) -> None:
    """Hand the API's warm ChromaDB store to MCP handlers (called from lifespan)."""
    global _active_store
    _active_store = store


def get_store() -> Any:
    """The shared ChromaDB store, or ``None`` (handlers then build their own)."""
    return _active_store


def _json(payload: Any) -> str:
    """Serialize a resource payload as compact JSON, dates → ISO strings."""
    return json.dumps(payload, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Who is asking.
# ---------------------------------------------------------------------------
ROSTER_ONLY = "Only someone on the team can read this."


class CallerRefused(PermissionError):
    """This caller may not read what it asked for. FastMCP returns the
    message to the client as the error."""


@dataclass(frozen=True)
class McpCaller:
    """Who an MCP request asks as: their People entry (None for no one) and
    whether they are the principal, by the web app's rules."""

    person_id: int | None
    is_principal: bool


# No MCP request is in progress: a direct in-process call (tests, scripts).
_NO_MCP_REQUEST = object()


def _request() -> Any:
    """The HTTP request behind the MCP call in progress; None when the call
    has a request context but no HTTP request; ``_NO_MCP_REQUEST`` outside
    any MCP request. Read in the handler's own task, where FastMCP's request
    context is set."""
    try:
        context = mcp.get_context().request_context
    except (LookupError, ValueError):
        return _NO_MCP_REQUEST
    return getattr(context, "request", None)


def _who() -> api_caller.Caller:
    """The verified caller of the request in progress. Outside any MCP
    request (a direct in-process call) it is what a request naming no one
    would be: the principal, or no one once signed callers are on. An MCP
    call that carries no HTTP request names no one (fails closed)."""
    request = _request()
    if request is _NO_MCP_REQUEST:
        return api_caller.Caller("service") if api_caller.signing_on() else api_caller.Caller("open")
    if request is None:
        return api_caller.Caller("service")
    return api_caller.caller(request)


def client_name() -> str:
    """The MCP client's own name (``clientInfo.name`` from its initialize),
    as one plain line, or "an AI app" when it gave none (a stateless request
    has no initialize of its own)."""
    from openexecutive.utils.prompt_blocks import plain

    name = ""
    try:
        params = mcp.get_context().session.client_params
        info = getattr(params, "clientInfo", None)
        name = str(getattr(info, "name", "") or "")
    except Exception:
        name = ""
    cleaned = " ".join(plain(name).split())[:_CLIENT_NAME_MAX].strip()
    return cleaned or "an AI app"


def _resolve(who: api_caller.Caller) -> McpCaller:
    """``who`` on the roster, by ``api.caller.person_id_for``. Fails closed:
    a roster that can't be read makes them no principal."""
    from openexecutive.people.store import is_principal_or_self

    person_id = api_caller.person_id_for(who)
    if who.defaults_to_principal:
        return McpCaller(person_id, True)
    try:
        principal = person_id is not None and is_principal_or_self(person_id, None)
    except Exception:
        logger.warning("mcp_server: principal check failed — answering no", exc_info=True)
        principal = False
    return McpCaller(person_id, principal)


async def current_caller() -> McpCaller:
    """Who the MCP request in progress asks as."""
    who = _who()
    return await asyncio.to_thread(_resolve, who)


async def _require_person() -> McpCaller:
    """The principal, or someone on the roster; a service or an unrostered
    caller names no one, so reads nothing about the people here."""
    resolved = await current_caller()
    if resolved.person_id is None and not resolved.is_principal:
        raise CallerRefused(ROSTER_ONLY)
    return resolved


# ---------------------------------------------------------------------------
# Resources — company-grounded context. Read-only, no LLM calls. The unique
# thing OE has that a generic agent does not.
# ---------------------------------------------------------------------------
def _http_request() -> Any:
    """The HTTP request of the MCP call in progress, for reading through the
    web app's own routes with the caller it verified. None outside one."""
    request = _request()
    return None if request is _NO_MCP_REQUEST else request


async def _through_route(route: Any, *args: Any, **kwargs: Any) -> Any:
    """Call one of the web app's own GET routes with this MCP request, so the
    answer follows that route's rules for the caller. A refusal comes back as
    the route's own words."""
    from fastapi import HTTPException

    request = _http_request()
    if request is None:
        raise CallerRefused(ROSTER_ONLY)
    try:
        if inspect.iscoroutinefunction(route):
            return await route(request, *args, **kwargs)
        return await asyncio.to_thread(route, request, *args, **kwargs)
    except HTTPException as exc:
        raise CallerRefused(exc.detail if isinstance(exc.detail, str) else ROSTER_ONLY) from exc


@mcp.resource(
    "oe://company/profile",
    name="Company profile",
    description="What the company is: the profile Open Executive works from.",
    mime_type="text/markdown",
)
async def company_profile() -> str:
    from openexecutive.onboarding.profile_builder import load_or_create_profile

    profile = await asyncio.to_thread(load_or_create_profile)
    if profile.is_empty():
        return "No company profile has been configured yet."
    return profile.to_prompt_block()


async def _today_for(who: McpCaller) -> Any:
    """Home's snapshot for ``who``, as ``GET /today`` builds it: the
    principal's private rows for the principal alone, a drafted artifact's
    card for its owner."""
    from openexecutive.api.routes.today import _build_today
    from openexecutive.orchestrator.artifact_records import viewer_for_person

    viewer = await asyncio.to_thread(viewer_for_person, who.person_id)
    today = await asyncio.to_thread(_build_today, include_private=who.is_principal, viewer=viewer)
    today.caller_person_id = who.person_id
    return today


@mcp.resource(
    "oe://today/briefing",
    name="Today's brief",
    description=(
        "Home for the person signed in: departments and their Approvals "
        "levels, people, and the cards waiting on them."
    ),
    mime_type="application/json",
)
async def today_briefing() -> str:
    today = await _today_for(await _require_person())
    return _json(today.model_dump(mode="json"))


@mcp.resource(
    "oe://cards/needs-you",
    name="Needs you",
    description=(
        "The cards waiting on the person signed in, as Home shows them. Answer "
        "one with answer_card."
    ),
    mime_type="application/json",
)
async def needs_you() -> str:
    today = await _today_for(await _require_person())
    return _json([p.model_dump(mode="json") for p in today.proposals])


@mcp.resource(
    "oe://today/done-for-you",
    name="Done for you",
    description="What the Executive did for the person signed in in the last day, as Home shows it.",
    mime_type="application/json",
)
async def done_for_you() -> str:
    from openexecutive.api.routes.today import get_today_done

    rows = await _through_route(get_today_done)
    return _json([r.model_dump(mode="json") for r in rows])


@mcp.resource(
    "oe://tasks",
    name="Tasks",
    description="Open tasks, soonest due first: the owner sees everyone's, anyone else their own; a task from someone's own chat only they and its assignee see.",
    mime_type="application/json",
)
async def tasks() -> str:
    from openexecutive.api.routes.people import list_tasks_route

    found = await _through_route(list_tasks_route, limit=200)
    return _json(found.model_dump(mode="json"))


@mcp.resource(
    "oe://watches",
    name="Watches",
    description="What the Executive is watching (companies, people, pages, topics) and how each is set.",
    mime_type="application/json",
)
async def watches() -> str:
    from openexecutive.api.routes.watchlist import list_watchlist

    await _require_person()
    # Through the route, so an email watch shows to its owner alone.
    rows = await _through_route(list_watchlist)
    return _json([r.model_dump(mode="json") for r in rows])


@mcp.resource(
    "oe://approvals",
    name="Approvals",
    description=(
        "What the Executive may do on its own for the person signed in, and "
        "what waits on a card: their Approvals level, As you, Your "
        "rules and In training, as the What it may do page shows them."
    ),
    mime_type="application/json",
)
async def approvals_summary() -> str:
    from openexecutive.api.routes.may_do import get_what_it_may_do

    found = await _through_route(get_what_it_may_do)
    return _json(found.model_dump(mode="json"))


@mcp.resource(
    "oe://people/roster",
    name="People",
    description="Who is on the team. For someone on the roster.",
    mime_type="application/json",
)
async def people_roster() -> str:
    from openexecutive.people.store import list_people

    # The team, never the principal's contacts (as GET /people gives it).
    who = await _require_person()
    people = await asyncio.to_thread(list_people)
    # A teammate's phone, website and other details go only to the principal
    # and to that person (as GET /people shows them).
    shown = [
        p if who.is_principal or not p.details or p.id == who.person_id
        else p.model_copy(update={"details": []})
        for p in people
    ]
    return _json([p.model_dump(mode="json") for p in shown])


async def _reader(who: McpCaller) -> Any:
    """``who``'s view of follow-ups and advice (``api.own_rows``):
    the team's rows and their own."""
    from openexecutive.api.own_rows import Reader
    from openexecutive.api.routes.today import principal_id

    viewer = who.person_id
    if viewer is None and who.is_principal:
        # The principal with no roster id of their own (a local, unsigned
        # caller): their own rows are the roster principal's.
        viewer = await asyncio.to_thread(principal_id)
    return await asyncio.to_thread(Reader, viewer)


def _department_view(state: Any, names: dict[int, str]) -> dict[str, Any]:
    """One department as its page says it: its Approvals level by name, its
    lead and people by name, its goals. No ids or channel settings."""
    config = state.config
    return {
        "slug": config.slug,
        "title": config.title,
        "mission": config.charter.mission,
        "scope": config.charter.scope,
        "out_of_scope": config.charter.out_of_scope,
        "approvals": config.approvals,
        "lead": names.get(config.head_person_id) if config.head_person_id is not None else None,
        "people": [names[i] for i in state.member_person_ids if i in names],
        "check_in": config.cadences.get("check_in") or None,
        "watching": config.watched_entities,
        "headcount": state.headcount,
        "budget_usd": state.budget_usd,
        "goals": [
            {
                "goal": g.key_result,
                "target": g.target,
                "current": g.current,
                "status": g.status,
                "period": g.period_value,
            }
            for g in state.goals
        ],
    }


@mcp.resource(
    "oe://departments/state",
    name="Departments",
    description="Each department: what it does, its Approvals level, who leads it, its goals.",
    mime_type="application/json",
)
async def departments_state() -> str:
    from openexecutive.departments.store import list_departments
    from openexecutive.people.store import list_people

    depts = await asyncio.to_thread(list_departments)
    names: dict[int, str] = {}
    try:
        # Who leads and works in each one is the team's to know, as the
        # People list is: none for a caller who is no one.
        if await asyncio.to_thread(api_caller.person_id_for, _who()) is not None:
            people = await asyncio.to_thread(list_people)
            names = {p.id: p.full_name for p in people if p.id is not None}
    except Exception:
        logger.warning("mcp_server: couldn't read the People list — departments without names", exc_info=True)
        names = {}
    return _json([_department_view(d, names) for d in depts])


@mcp.resource(
    "oe://memory/decisions",
    name="Decisions",
    description="Decisions on record: the team's, and the caller's own. For someone on the roster.",
    mime_type="application/json",
)
async def memory_decisions() -> str:
    from openexecutive.memory.episodic import list_decisions

    # The team's, and the caller's own private ones.
    who = await _require_person()
    reader = who.person_id
    if reader is None and who.is_principal:
        from openexecutive.api.routes.today import principal_id

        reader = await asyncio.to_thread(principal_id)
    rows = await asyncio.to_thread(list_decisions, reader=reader)
    return _json([r.model_dump(mode="json") for r in rows])


@mcp.resource(
    "oe://memory/initiatives",
    name="Initiatives",
    description="Initiatives being tracked. For someone on the roster.",
    mime_type="application/json",
)
async def memory_initiatives() -> str:
    from openexecutive.memory.episodic import list_initiatives

    await _require_person()
    rows = await asyncio.to_thread(list_initiatives)
    return _json([r.model_dump(mode="json") for r in rows])


@mcp.resource(
    "oe://memory/advice",
    name="Advice",
    description="Advice the Executive gave: the team's, and the caller's own. For someone on the roster.",
    mime_type="application/json",
)
async def memory_advice() -> str:
    from openexecutive.memory.episodic import list_advice

    reader = await _reader(await _require_person())
    rows = await asyncio.to_thread(lambda: [r for r in list_advice() if reader.sees_session_row(r)])
    return _json([r.model_dump(mode="json") for r in rows])


# ---------------------------------------------------------------------------
# Tools. consult_specialist, search_knowledge and list_workflows only read.
# ask_executive, answer_card and run_workflow each run one Executive turn as
# the person signed in (``_executive_turn``), under their Approvals.
# post_progress_update (below) runs as no one: as the source its token names.
# ---------------------------------------------------------------------------
_READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
_ACTS = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True)


@mcp.tool(title="Consult a specialist", annotations=_READ_ONLY)
async def consult_specialist(
    specialist: SpecialistArg, query: str, context: str = ""
) -> str:
    """Ask one of Open Executive's specialists for analysis in their area.

    Each specialist searches the company's knowledge and documents and gives
    a grounded read. Specialists: cso (strategy, M&A, OKRs), cfo (finance,
    unit economics, fundraising), chro (people, pay, org design), gc (legal,
    contracts, compliance), coo (operations, process, metrics), cmo (go to
    market, brand, PR), cpo (product, roadmap), sales (pipeline, deals,
    pricing, forecast), board_comms (board decks, investors, governance),
    triage (chief of staff: what in what came in needs someone).

    Args:
        specialist: Which specialist to ask.
        query: The question or task. The specialist sees only this and
            ``context``, so be precise.
        context: Background from your own task that grounds the answer.
    """
    from openexecutive.orchestrator.router import (
        NAME_PREVIEW_CHARS,
        SPECIALIST_REGISTRY,
        resolve_specialist_name,
        route_to_specialist,
    )

    # Resolve rather than reject on an exact match. A strict membership test
    # here would give external MCP clients a different policy from the chat
    # path — `csO` normalised in one and refused in the other — and would
    # short-circuit `route_to_specialist` before it can audit the anomaly.
    # A genuinely unresolvable name still raises, because an MCP caller is a
    # program that wants an error, not a model that wants a recoverable
    # tool_result. The RAW name is what gets forwarded — `route_to_specialist`
    # resolves it again and writes the `routing_anomaly` row at the point it
    # makes the correction, so normalising here would erase the anomaly.
    if resolve_specialist_name(specialist) is None:
        valid = ", ".join(sorted(SPECIALIST_REGISTRY))
        # Truncated for the same reason router.py truncates it in both its own
        # messages: since the annotation widened to `str`, this value is
        # arbitrary unbounded text from an external client, and it lands in the
        # tool error and in FastMCP's ERROR log line.
        shown = specialist[:NAME_PREVIEW_CHARS]
        raise ValueError(f"Unknown specialist {shown!r}. Valid: {valid}")
    return await route_to_specialist(
        specialist, query, context=context, actor="specialist_mcp",
    )


@mcp.tool(title="Search knowledge", annotations=_READ_ONLY)
async def search_knowledge(query: str) -> str:
    """Search Open Executive's business knowledge base and the company's documents.

    Returns the most relevant passages, each starting with its ``[source]``.
    No model call: raw retrieval to cite or reason over.
    """
    from openexecutive.knowledge.retriever import retrieve

    return await asyncio.to_thread(retrieve, query=query, store=get_store())


@mcp.tool(title="List workflows", annotations=_READ_ONLY)
async def list_workflows() -> str:
    """List the workflows Open Executive can run, with the inputs each takes.

    Returns a JSON array (name, title, description, estimated minutes, input
    schema, steps). Start one with run_workflow.
    """
    from openexecutive.workflows import list_workflows as _list

    workflows = await asyncio.to_thread(_list)
    return _json([w.meta().model_dump(mode="json") for w in workflows])


async def _open_card(card_id: int) -> None:
    """Refuse a card the caller can't see on Home or that is already answered,
    by the alerts route's own visibility rule, before any model call."""
    from fastapi import HTTPException

    from openexecutive.api.routes.alerts import _visible_alert
    from openexecutive.orchestrator.schedule_tools import _FIND_ALERTS_ACKABLE_STATUSES

    request = _http_request()
    if request is None:
        raise CallerRefused(ROSTER_ONLY)
    try:
        alert = await asyncio.to_thread(_visible_alert, card_id, request)
    except HTTPException:
        alert = None
    if alert is None or alert.status not in _FIND_ALERTS_ACKABLE_STATUSES:
        raise ValueError(f"No open card {card_id}. Pick one from the Needs you resource.")


# The web chat's own cap on one message (``POST /chat``), for every turn here.
MAX_MESSAGE = 32_000


async def _executive_turn(message: str, card_id: int | None = None) -> str:
    """One Executive turn for the verified caller, as the web chat runs it:
    their live board shown, Approvals deciding each step, and every row it
    writes naming the app. With ``card_id`` the turn may answer that card
    and no other: it is the only trusted id, and ``find_alerts`` adds none."""
    from openexecutive.audit import via_app
    from openexecutive.briefing.context import render_and_trust
    from openexecutive.onboarding.profile_builder import load_or_create_profile
    from openexecutive.orchestrator.artifact_records import NOBODY
    from openexecutive.orchestrator.executive import Executive
    from openexecutive.orchestrator.mcp_gateway import get_active_gateway
    from openexecutive.orchestrator.session import Session

    if len(message) > MAX_MESSAGE:
        raise ValueError(f"That is too long for one message; keep it under {MAX_MESSAGE:,} characters.")

    # Who is asking comes from the verified request, never from the message:
    # a signed-in person, the principal for a request that names no one in
    # open mode, no one for a service.
    who = _who()
    person_id = await asyncio.to_thread(api_caller.person_id_for, who)
    profile = await asyncio.to_thread(load_or_create_profile)
    session = Session(
        company_profile=profile if not profile.is_empty() else None,
        # A client that resolves to no one reads and publishes no one's
        # documents (``artifact_records.current_viewer`` would otherwise take
        # a speakerless session for the principal's own work).
        documents_viewer=NOBODY if person_id is None else None,
        from_mcp_client=True,
        caller_person_id=person_id,
        # The web chat's rule for what needs a sign-in (Act as me): a
        # request that names someone, never one that only holds the secret.
        web_caller_signed_in=bool(who.email),
    )
    executive = Executive(mcp_gateway=get_active_gateway())
    # The live board, as the web chat shows it, and what a card answer may
    # touch: only for someone on the roster.
    briefing_context = (
        await asyncio.to_thread(render_and_trust, session) if person_id is not None else ""
    )
    if card_id is not None:
        if person_id is None:
            raise CallerRefused(ROSTER_ONLY)
        await _open_card(card_id)
        session.trusted_alert_ids = {card_id}
        session.answering_card = card_id
    # ``message`` is the caller's own words (in a fixed frame for a card or
    # workflow), nothing third-party packed around it, so a link in it may be
    # opened when the caller resolved to a known person
    # (``_who``: the verified caller, as the web chat route reads it).
    with via_app(client_name()):
        return await executive.chat(
            user_message=message, session=session, person_id=person_id,
            briefing_context=briefing_context,
            memory_text=message, fetch_own_urls=person_id is not None,
        )


@mcp.tool(title="Ask the Executive", annotations=_ACTS)
async def ask_executive(message: str) -> str:
    """Ask the Executive anything, or ask it to do something, as the person signed in.

    It answers as it would in Open Executive, drawing on the specialists,
    company knowledge and memory, and can act: tasks, follow-ups, watches,
    goals, decisions, notes, email and messages. What it does follows the
    person's Approvals: a step goes on its own or waits on a card for them,
    exactly as in Open Executive. The conversation is kept in Open
    Executive's history. Non-streaming.

    Args:
        message: What to ask or ask it to do.
    """
    return await _executive_turn(message)


CardAnswer = Literal["approve", "skip", "reply"]


@mcp.tool(title="Answer a card", annotations=_ACTS)
async def answer_card(card_id: int, answer: CardAnswer, reply: str = "") -> str:
    """Approve, skip or reply to one of the person's Needs you cards.

    Does what the card's own button or reply box does on Home: approve
    carries it out, skip drops it, reply answers a card that asks something
    (or tells it what to change). Only the person a card is for can approve
    or answer it, as on Home (``card_answers``).

    Args:
        card_id: The card's ``alert_id`` from the Needs you resource.
        answer: "approve", "skip" or "reply".
        reply: The person's words, for "reply".
    """
    if answer == "approve":
        message = f"Yes, approve card {card_id}."
    elif answer == "skip":
        message = f"No, drop card {card_id}."
    else:
        words = " ".join(reply.split())
        if not words:
            raise ValueError("Give the person's words in `reply`.")
        message = f"About card {card_id}: {words}"
    return await _executive_turn(message, card_id=card_id)


@mcp.tool(title="Run a workflow", annotations=_ACTS)
async def run_workflow(workflow: str, inputs: dict[str, Any] | None = None) -> str:
    """Run one of the workflows from list_workflows, as the person signed in.

    It runs as if they asked for it in Open Executive: its steps follow their
    Approvals, and it returns what the workflow made.

    Args:
        workflow: The workflow's ``name`` from list_workflows.
        inputs: Its input fields, matching that workflow's input schema.
    """
    from pydantic import ValidationError

    from openexecutive.workflows import list_workflows as _list

    name = " ".join(workflow.split())
    known = {w.name: w for w in await asyncio.to_thread(_list)}
    if name not in known:
        raise ValueError("No workflow by that name. Pick one from list_workflows.")
    # The inputs are the workflow's own fields, checked against its input
    # schema as its run is, and no longer than a person would type.
    try:
        checked = known[name].input_model().model_validate(inputs or {})
    except ValidationError as exc:
        wrong = ", ".join(str(e["loc"][0]) for e in exc.errors() if e.get("loc")) or "inputs"
        raise ValueError(f"These inputs don't fit the workflow: {wrong}. See its input_schema in list_workflows.") from exc
    fields = _json(checked.model_dump(mode="json"))
    return await _executive_turn(f"Run the {name} workflow with these inputs: {fields}")


# ---------------------------------------------------------------------------
# Progress updates — the one tool that stores and sends outside Approvals.
# The source is whoever the ``x-source-token`` header belongs to, never an
# argument, so the token stays out of model context and tool-call logs.
# ---------------------------------------------------------------------------
# one message for every refusal, so it says nothing about which check failed
PROGRESS_REFUSED = "This server does not accept progress updates from this caller."
PROGRESS_UNAVAILABLE = "Progress updates can't be stored right now; nothing was stored or sent. Try again later."
# how much of a guard field name goes back in an error, as consult_specialist
# truncates the name it echoes
_FIELD_PREVIEW_CHARS = 40

StatusArg = Annotated[str, Field(json_schema_extra={"enum": list(STATUSES)})]


def _progress_source() -> str:
    """The source the request's ``x-source-token`` belongs to. Raises
    :class:`CallerRefused` with :data:`PROGRESS_REFUSED` for an in-process
    call, a call with no HTTP request, the feature off (no sources, or a
    client slot or demo fixture live), no token or an unknown one, and, with
    a shared secret configured, a request the gate did not mark as
    authenticated by it (``SHARED_SECRET_OK``)."""
    import os

    from openexecutive.integrations import progress_sources, progress_updates

    request = _request()
    if request is None or request is _NO_MCP_REQUEST:
        raise CallerRefused(PROGRESS_REFUSED)
    if os.environ.get("BACKEND_SHARED_SECRET", "").strip():
        state = getattr(request, "state", None)
        if getattr(state, SHARED_SECRET_OK, False) is not True:
            raise CallerRefused(PROGRESS_REFUSED)
    if not progress_updates.accepting():
        raise CallerRefused(PROGRESS_REFUSED)
    source = progress_sources.source_for(request.headers.get(SOURCE_TOKEN_HEADER))
    if source is None:
        raise CallerRefused(PROGRESS_REFUSED)
    return source


@mcp.tool(
    title="Post a progress update",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def post_progress_update(
    topic: str,
    status: StatusArg,
    summary: str,
    ask: str = "",
    link: str = "",
    idempotency_key: str = "",
) -> dict[str, Any]:
    """Post a short progress update on work you are doing for the founders.

    The update is stored and sent, quoted as received, to each founder as a
    chat-app message; a founder it does not reach gets it in their next
    morning brief. Past your source's relay cap an update is stored but not
    sent, and each founder gets one notice that more updates are waiting.
    It is filed under the source your ``x-source-token`` header belongs to;
    there is no source argument.

    Content contract: write for the founders, in plain text. Do not post
    unpublished prices or other confidential commercial terms, personal
    data (names of private individuals, contact details, health or HR
    detail), credentials, or anything you would not put in a team chat.
    Amounts are accepted, so keeping an unpublished price out is your
    responsibility. Updates with email addresses, phone numbers, IBANs,
    credential-like strings, ``<`` or ``>``, control characters, or a URL
    outside ``link`` are rejected with the field and the reason. A bare
    domain name such as chainsafe.be counts as a URL, so name a site in
    words ("the ChainSafe site").

    Returns ``{id, stored, relay}``: ``relay`` is ``accepted`` for a new
    update and ``duplicate`` for a repeat of an update already stored
    (nothing sent). Whether a message went out is not returned.

    Args:
        topic: What the update is about, one line, 3-80 characters.
        status: on_track, at_risk, blocked, done or info.
        summary: What happened, 10-800 characters, at most 8 lines.
        ask: What you need from the founders, one line, up to 300 characters.
        link: An https link on an allowed host, for the details: up to 300
            characters, no %-encoding or #fragment, and query parameters
            only from id, resourcekey, usp, tab, gid, v and node-id.
        idempotency_key: Your own retry key (8-64 of A-Z, a-z, 0-9, _ and -):
            posting the same key again returns the first update.
    """
    from openexecutive.integrations import progress_updates

    source = _progress_source()
    payload: dict[str, Any] = {"topic": topic, "status": status, "summary": summary}
    # an empty optional is an omitted one, not a value the guard should check
    optional = {"ask": ask, "link": link, "idempotency_key": idempotency_key}
    payload.update({k: v for k, v in optional.items() if v})
    try:
        result = await progress_updates.submit(source, payload)
    except progress_updates.Rejected as exc:
        raise ValueError(f"{exc.field[:_FIELD_PREVIEW_CHARS]}: {exc.reason}") from None
    except progress_updates.Paused:
        raise CallerRefused(PROGRESS_REFUSED) from None
    except progress_updates.OverLimit as exc:
        raise ValueError(str(exc)) from None
    except progress_updates.StoreUnavailable:
        logger.warning("mcp_server: progress update from %s not stored: store unavailable", source, exc_info=True)
        raise RuntimeError(PROGRESS_UNAVAILABLE) from None
    return {"id": result.id, "stored": result.stored, "relay": result.relay}


# ---------------------------------------------------------------------------
# Mount.
# ---------------------------------------------------------------------------
_http_app: Any = None


def mount(app: Any) -> None:
    """Attach the Streamable-HTTP MCP app to a FastAPI app at ``/mcp``.

    The first call builds the Streamable-HTTP ASGI app, which also lazily
    creates ``mcp.session_manager`` (run by the API lifespan — see
    ``api/main.py``). The result is cached so repeated ``create_app()`` calls
    reuse the *same* sub-app and session manager rather than orphaning the one
    the lifespan will run.
    """
    global _http_app
    if _http_app is None:
        _http_app = mcp.streamable_http_app()
    app.mount("/mcp", _http_app)
    logger.info("mcp_server mounted at /mcp")


_session_started = False


@asynccontextmanager
async def run_session_manager() -> AsyncIterator[None]:
    """Run the Streamable-HTTP session manager for the app's serving phase.

    ``StreamableHTTPSessionManager.run()`` can be entered **only once per
    instance**, and FastMCP caches a single manager on the module-global ``mcp``.
    Production starts one app once, so this runs the manager for the app's life.
    The guard makes it a no-op on any *subsequent* lifespan entry in the same
    process — which only happens in the test suite, where apps are rebuilt via
    ``create_app()`` (a test that drives the live ``/mcp`` protocol gives the
    endpoint a session manager of its own) — instead of raising
    ``RuntimeError``.
    """
    global _session_started
    if _session_started:
        yield
        return
    _session_started = True
    async with mcp.session_manager.run():
        yield


def specialist_keys() -> tuple[str, ...]:
    """The specialist enum advertised in the ``consult_specialist`` schema.

    Exposed for the drift test that pins it to ``SPECIALIST_REGISTRY``.
    """
    return get_args(SpecialistKey)
