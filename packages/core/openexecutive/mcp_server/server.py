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

* **DNS-rebinding protection** (FastMCP's Host-header check) is disabled: the
  endpoint authenticates with a header secret (not a cookie), so a browser
  rebinding attack cannot supply credentials, and the server runs behind the
  deployment's TLS terminator. The shared-secret gate is the real access
  control.

* **No prompt-caching impact.** This path is parallel to chat. Resources are
  pure reads; ``consult_specialist`` delegates to ``BaseAgent.analyze`` which
  reuses the existing cached system blocks unchanged.

* **State.** Handlers get no FastAPI ``Request``. Most service functions build
  their own SQLite/ChromaDB handles; the one that benefits from the app's warm
  store (``search_knowledge``) reads it from the ``set_store`` singleton.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, get_args

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field

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
    "board_comms", "cfo", "chro", "cmo", "coo", "cpo", "cso", "gc", "triage"
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
    "Open Executive exposed as an MCP server. It surfaces a company's "
    "executive context and a council of specialist analysts so another agent "
    "can ground itself in this company without re-explaining it.\n\n"
    "Resources (read-only, company-internal): company profile, today's "
    "briefing and recent activity, the people roster, department state, "
    "and episodic memory (past decisions, initiatives, advice).\n\n"
    "Tools: `consult_specialist` (domain analysis from a CFO/CSO/etc., grounded "
    "in company knowledge), `search_knowledge` (curated MBA + company-doc "
    "retrieval), `list_workflows` (catalog of multi-step executive workflows), "
    "and `ask_executive` (a single synthesized answer from the Executive — a "
    "fallback; prefer the specialist tool and resources)."
)

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
# Resources — company-grounded context. Read-only, no LLM calls. The unique
# thing OE has that a generic agent does not.
# ---------------------------------------------------------------------------
@mcp.resource(
    "oe://company/profile",
    name="Company profile",
    description="Structured company profile rendered as a prompt block.",
    mime_type="text/markdown",
)
async def company_profile() -> str:
    from openexecutive.onboarding.profile_builder import load_or_create_profile

    profile = await asyncio.to_thread(load_or_create_profile)
    if profile.is_empty():
        return "No company profile has been configured yet."
    return profile.to_prompt_block()


@mcp.resource(
    "oe://today/briefing",
    name="Today briefing",
    description="The morning-brief snapshot: department health, roster, proposals.",
    mime_type="application/json",
)
async def today_briefing() -> str:
    from openexecutive.api.routes.today import _build_today

    today = await asyncio.to_thread(_build_today)
    return _json(today.model_dump(mode="json"))


@mcp.resource(
    "oe://today/activity",
    name="Recent activity",
    description="Recent self-initiated Executive activity (last 20 actions).",
    mime_type="application/json",
)
async def today_activity() -> str:
    from openexecutive.api.routes.today import _build_activity

    activity = await asyncio.to_thread(_build_activity, 20)
    return _json(activity.model_dump(mode="json"))


@mcp.resource(
    "oe://people/roster",
    name="People roster",
    description="Active people on the company roster.",
    mime_type="application/json",
)
async def people_roster() -> str:
    from openexecutive.people.store import list_people

    people = await asyncio.to_thread(list_people)
    return _json([p.model_dump(mode="json") for p in people])


@mcp.resource(
    "oe://departments/state",
    name="Department state",
    description="Departments with goals, cadences, and current state.",
    mime_type="application/json",
)
async def departments_state() -> str:
    from openexecutive.departments.store import list_departments

    depts = await asyncio.to_thread(list_departments)
    return _json([d.model_dump(mode="json") for d in depts])


@mcp.resource(
    "oe://memory/decisions",
    name="Episodic memory: decisions",
    description="Past decisions recorded in episodic memory.",
    mime_type="application/json",
)
async def memory_decisions() -> str:
    from openexecutive.memory.episodic import list_decisions

    rows = await asyncio.to_thread(list_decisions)
    return _json([r.model_dump(mode="json") for r in rows])


@mcp.resource(
    "oe://memory/initiatives",
    name="Episodic memory: initiatives",
    description="Tracked initiatives recorded in episodic memory.",
    mime_type="application/json",
)
async def memory_initiatives() -> str:
    from openexecutive.memory.episodic import list_initiatives

    rows = await asyncio.to_thread(list_initiatives)
    return _json([r.model_dump(mode="json") for r in rows])


@mcp.resource(
    "oe://memory/advice",
    name="Episodic memory: advice",
    description="Advice given, recorded in episodic memory.",
    mime_type="application/json",
)
async def memory_advice() -> str:
    from openexecutive.memory.episodic import list_advice

    rows = await asyncio.to_thread(list_advice)
    return _json([r.model_dump(mode="json") for r in rows])


# ---------------------------------------------------------------------------
# Tools — structured capabilities. The flagship is consult_specialist.
# ---------------------------------------------------------------------------
@mcp.tool()
async def consult_specialist(
    specialist: SpecialistArg, query: str, context: str = ""
) -> str:
    """Consult one of Open Executive's specialist executives for domain analysis.

    Each specialist runs its own knowledge retrieval and returns a grounded,
    domain-expert read. Specialists: cso (strategy/M&A/OKRs), cfo (finance/unit
    economics/fundraising), chro (people/comp/org design), gc (legal/contracts/
    compliance), coo (operations/process/metrics), cmo (GTM/brand/PR), cpo
    (product/roadmap), board_comms (board decks/IR/governance),
    triage (chief of staff — significance of inbound events).

    Args:
        specialist: Which specialist to consult.
        query: The specific question or task. The specialist sees only this
            query plus ``context``, so be precise.
        context: Relevant background from your own task to ground the answer.
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


@mcp.tool()
async def search_knowledge(query: str) -> str:
    """Search Open Executive's curated MBA knowledge base and indexed company docs.

    Returns the most relevant retrieved chunks, each prefixed with its
    ``[source]``. No LLM call — this is raw retrieval you can cite or feed back
    into your own reasoning.
    """
    from openexecutive.knowledge.retriever import retrieve

    return await asyncio.to_thread(retrieve, query=query, store=get_store())


@mcp.tool()
async def list_workflows() -> str:
    """List the available multi-step executive workflows and their input schemas.

    Returns a JSON array of workflow metadata (name, title, description,
    estimated minutes, input schema, steps). Execution is intentionally not
    exposed over MCP yet — workflows are long-running and would exceed a single
    tool-call's timeout.
    """
    from openexecutive.workflows import list_workflows as _list

    workflows = await asyncio.to_thread(_list)
    return _json([w.meta().model_dump(mode="json") for w in workflows])


@mcp.tool()
async def ask_executive(message: str, caller_email: str = "") -> str:
    """Ask the Executive a question and get one synthesized answer (fallback tool).

    Prefer ``consult_specialist`` and the company resources — this duplicates
    OE's existing chat channels and is non-streaming. Use it only when you want
    a single coherent, cross-domain answer.

    Args:
        message: Your question for the Executive.
        caller_email: Optional — resolve the caller to a known person for
            person-scoped memory; defaults to the company principal.
    """
    from openexecutive.orchestrator.executive import Executive
    from openexecutive.orchestrator.mcp_gateway import get_active_gateway
    from openexecutive.orchestrator.session import Session
    from openexecutive.people.store import (
        find_person_by_email,
        find_principal_person,
    )

    def _resolve_person() -> int | None:
        # Degrade to None on a DB error rather than surfacing it to the client,
        # matching the chat route's caller resolution (api/routes/chat.py).
        try:
            person = (
                find_person_by_email(caller_email.strip().lower())
                if caller_email.strip()
                else find_principal_person()
            )
        except (OSError, sqlite3.Error):
            logger.warning("ask_executive caller lookup failed", exc_info=True)
            return None
        return person.id if person is not None else None

    person_id = await asyncio.to_thread(_resolve_person)

    from openexecutive.onboarding.profile_builder import load_or_create_profile

    profile = await asyncio.to_thread(load_or_create_profile)
    session = Session(
        company_profile=profile if not profile.is_empty() else None,
    )

    executive = Executive(mcp_gateway=get_active_gateway())
    return await executive.chat(
        user_message=message, session=session, person_id=person_id
    )


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
    ``create_app()`` and never exercise the live ``/mcp`` protocol — instead of
    raising ``RuntimeError``.
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
