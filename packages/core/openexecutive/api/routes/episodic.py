from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from pydantic import BaseModel

from openexecutive.api import own_rows
from openexecutive.memory.episodic import (
    Advice,
    Decision,
    Initiative,
    _as_decision,
    current_decision,
    delete_advice,
    delete_decision,
    delete_initiative,
    get_advice,
    get_initiative,
    list_advice,
    list_initiatives,
    update_advice,
    update_decision,
    update_initiative,
)
from openexecutive.memory.facts import (
    Fact,
    approve_fact,
    decision_rows,
    decline_fact,
    get_fact,
    list_facts,
    readable,
    record_fact,
    retire_fact,
    share_decision,
)
from openexecutive.memory.honcho_client import (
    PERSON_CONCLUSIONS_MAX_PAGE,
    PeopleMemory,
    PersonConclusionsPage,
    PersonMemory,
    forget_person_conclusion,
    people_overview,
    person_conclusions,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class DecisionUpdate(BaseModel):
    domain: str | None = None
    summary: str | None = None
    rationale: str | None = None
    outcome: str | None = None
    tags: str | None = None


class InitiativeUpdate(BaseModel):
    title: str | None = None
    status: str | None = None
    summary: str | None = None


class AdviceUpdate(BaseModel):
    domain: str | None = None
    query_summary: str | None = None
    advice_summary: str | None = None


# --- Decisions ---


class DecisionOut(Decision):
    # Only the caller reads it: they made it in their own chat and have not
    # shared it with the team.
    private: bool = False
    # The caller may share it with the team (their own private decision).
    can_share: bool = False
    # The caller may make it private again (a shared one they recorded).
    can_make_private: bool = False
    # Shared by a teammate whose facts the principal approves first: not the
    # team's yet.
    waiting_for_approval: bool = False


class DecisionShare(BaseModel):
    shared: bool


def _decision_reader(request: Request) -> tuple[int | None, bool]:
    """``(reader, principal)``: whose private decisions this caller reads
    (their own; a request with no caller is the principal's), and whether
    they are the principal."""
    principal = _caller_is_principal(request)
    reader = _caller_id(request)
    if reader is None and principal:
        from openexecutive.api.routes.today import principal_id

        reader = principal_id()
    return reader, principal


def _decision_out(fact: Fact, reader: int | None, principal: bool) -> DecisionOut:
    private = fact.private_to_person_id is not None
    own = reader is not None and (
        fact.recorded_by_role == "principal" if principal
        else fact.recorded_by_role == "teammate" and fact.recorded_by_person_id == reader
        and fact.approved_at is None
    )
    return DecisionOut(
        **_as_decision(fact).model_dump(),
        private=private,
        can_share=private and fact.private_to_person_id == reader and fact.status == "active",
        can_make_private=not private and own and (
            fact.status == "active" or (fact.status == "proposed" and fact.recorded_by_role == "teammate")
        ),
        waiting_for_approval=fact.status == "proposed",
    )


def _own_private_decision(decision_id: int, reader: int | None) -> bool:
    fact = get_fact(decision_id)
    return (
        fact is not None and fact.decision and reader is not None
        and fact.private_to_person_id == reader
    )


@router.get("/memories/decisions", response_model=list[DecisionOut])
def get_decisions(request: Request) -> list[DecisionOut]:
    """The team's decisions, and the caller's own private ones (a decision
    made in someone's own chat is theirs alone until they share it)."""
    reader, principal = _decision_reader(request)
    return [_decision_out(f, reader, principal) for f in decision_rows(limit=1000, reader=reader)]


@router.patch("/memories/decisions/{decision_id}", response_model=DecisionOut)
def patch_decision(decision_id: int, body: DecisionUpdate, request: Request) -> DecisionOut:
    # A shared decision is a standing fact read on everyone's turns: only the
    # principal changes or removes one. A private one is its person's own.
    reader, principal = _decision_reader(request)
    if not _own_private_decision(decision_id, reader):
        _require_principal(request, "change a decision")
    if not update_decision(
        decision_id, **body.model_dump(exclude_unset=True), reader=reader, own_only=not principal
    ):
        raise HTTPException(status_code=404, detail="Decision not found")
    # A new summary is an edit, kept as a new fact that replaces this one.
    updated = current_decision(decision_id, reader=reader)
    fact = get_fact(updated.id) if updated is not None and updated.id is not None else None
    if fact is None:
        raise HTTPException(status_code=404, detail="Decision not found")
    return _decision_out(fact, reader, principal)


@router.delete("/memories/decisions/{decision_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_decision(decision_id: int, request: Request) -> Response:
    reader, _principal = _decision_reader(request)
    current = current_decision(decision_id, reader=reader)
    if not (current is not None and current.id is not None and _own_private_decision(current.id, reader)):
        _require_principal(request, "remove a decision")
    if not delete_decision(decision_id, reader=reader):
        raise HTTPException(status_code=404, detail="Decision not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/memories/decisions/{decision_id}/share", response_model=DecisionOut)
def share_decision_route(decision_id: int, body: DecisionShare, request: Request) -> DecisionOut:
    """Share the caller's own private decision with the team
    (``shared: true``), or make one they recorded private to them again
    (``shared: false``)."""
    reader, principal = _decision_reader(request)
    if reader is None:
        raise HTTPException(status_code=403, detail="Sign in as someone on the People list")
    fact = share_decision(decision_id, shared=body.shared, person_id=reader, is_principal=principal)
    if fact is None:
        raise HTTPException(status_code=404, detail="Decision not found")
    _audit_fact(
        "decision_shared" if body.shared else "decision_made_private",
        f"Decision {decision_id} {'shared with the team' if body.shared else 'made private'}",
        {"fact_id": decision_id, "status": fact.status, "caller_person_id": reader},
        actor="principal" if principal else "teammate",
    )
    return _decision_out(fact, reader, principal)


# --- Initiatives ---


@router.get("/memories/initiatives", response_model=list[Initiative])
def get_initiatives() -> list[Initiative]:
    return list_initiatives()


@router.patch("/memories/initiatives/{initiative_id}", response_model=Initiative)
def patch_initiative(initiative_id: int, body: InitiativeUpdate) -> Initiative:
    if not update_initiative(initiative_id, **body.model_dump(exclude_unset=True)):
        raise HTTPException(status_code=404, detail="Initiative not found")
    updated = get_initiative(initiative_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="Initiative not found")
    return updated


@router.delete("/memories/initiatives/{initiative_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_initiative(initiative_id: int) -> Response:
    if not delete_initiative(initiative_id):
        raise HTTPException(status_code=404, detail="Initiative not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- Advice ---


@router.get("/memories/advice", response_model=list[Advice])
def get_advice_list(request: Request) -> list[Advice]:
    # The team's and the caller's own (``api.own_rows``).
    reader = own_rows.reader_for(request)
    return [a for a in list_advice() if reader.sees_session_row(a)]


def _own_advice(advice_id: int, request: Request) -> Advice:
    """The advice row, or 404 when there is none or it is another person's."""
    row = get_advice(advice_id)
    if row is None or not own_rows.reader_for(request).sees_session_row(row):
        raise HTTPException(status_code=404, detail="Advice not found")
    return row


@router.patch("/memories/advice/{advice_id}", response_model=Advice)
def patch_advice(advice_id: int, body: AdviceUpdate, request: Request) -> Advice:
    _own_advice(advice_id, request)
    if not update_advice(advice_id, **body.model_dump(exclude_unset=True)):
        raise HTTPException(status_code=404, detail="Advice not found")
    updated = get_advice(advice_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="Advice not found")
    return updated


@router.delete("/memories/advice/{advice_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_advice(advice_id: int, request: Request) -> Response:
    _own_advice(advice_id, request)
    if not delete_advice(advice_id):
        raise HTTPException(status_code=404, detail="Advice not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- Standing facts (corrections that persist everywhere) ---


class FactsPage(BaseModel):
    facts: list[Fact]
    # Whether this caller may retire any fact (the principal only).
    can_retire: bool
    # The facts this caller may retire: every active one for the principal; a
    # teammate's own attributed ones for them.
    retirable_ids: list[int] = []
    # Whether this caller approves or declines a decision copied in from mail
    # that waits for their OK (the principal only).
    can_review: bool = False
    # The caller's own private decisions, which they may share with the team.
    shareable_ids: list[int] = []
    # Shared decisions the caller recorded, which they may make private again.
    unshareable_ids: list[int] = []


class FactRetire(BaseModel):
    reason: str = ""


class FactEdit(BaseModel):
    statement: str


class FactReview(BaseModel):
    reason: str = ""


def _caller_id(request: Request) -> int | None:
    from openexecutive.api.routes.chat import _resolve_caller_person_id

    try:
        return _resolve_caller_person_id(request)
    except Exception:
        return None


def _is_own_teammate_fact(fact: Fact, caller: int | None) -> bool:
    """A teammate's own fact, not yet the principal's: once the principal
    approved it (``principal_owned``), only the principal may retire it."""
    return (
        caller is not None
        and fact.recorded_by_role == "teammate"
        and fact.recorded_by_person_id == caller
        and not fact.principal_owned
    )


def _is_own_fact(fact: Fact, caller: int | None) -> bool:
    """A fact this teammate may edit or retire: their own attributed one, or
    a decision private to them."""
    return _is_own_teammate_fact(fact, caller) or (
        caller is not None and fact.private_to_person_id == caller
    )


def _readable_fact(fact_id: int, request: Request) -> Fact | None:
    """The fact, or None when it is someone else's private decision."""
    fact = get_fact(fact_id)
    if fact is None:
        return None
    reader, _ = _decision_reader(request)
    return fact if readable(fact, reader) else None


def _is_own_quote(fact: Fact, caller: int | None) -> bool:
    return (
        caller is not None
        and fact.recorded_by_role == "teammate"
        and fact.recorded_by_person_id == caller
    )


def _audit_text(fact: Fact, **text: str) -> dict[str, object]:
    """What a fact says, for its audit row: nothing for a private decision,
    which is its person's alone and the audit page is the principal's."""
    return {"private": True} if fact.private_to_person_id is not None else dict(text)


def _audit_subject(fact: Fact) -> str:
    return "" if fact.private_to_person_id is not None else f": {fact.subject[:80]}"


def _audit_fact(event: str, summary: str, details: dict[str, object], actor: str) -> None:
    try:
        from openexecutive.audit import log_event as audit_log

        # Private: the row names the fact, and a teammate must not read one the
        # principal retired or declined through /audit when the facts route
        # hides it from them.
        audit_log(event, summary, actor=actor, details=details, private=True)
    except Exception:  # noqa: BLE001 - the change already landed.
        logger.warning("%s audit row failed", event, exc_info=True)


@router.get("/memories/facts", response_model=FactsPage)
def get_facts(
    request: Request,
    include_inactive: bool = Query(True),
    limit: int = Query(200, ge=1, le=500),
) -> FactsPage:
    """The standing facts and corrections the principal and teammates asked
    to keep (``memory.facts``), newest first — with their replaced, retired,
    proposed and declined history unless ``include_inactive=false`` — and the
    company-profile fields changed from chat. Every prompt that produces
    output reads the active ones. The history is the principal's alone:
    anyone else gets the active rows.

    The facts themselves are company knowledge every conversation already
    sees, and so is who stated a teammate's (it renders "(per <name>)"); the
    rest of the provenance (the quote, a retire reason, and the session, turn
    and person id it came from) is shown to the principal only."""
    principal = _caller_is_principal(request)
    caller = None if principal else _caller_id(request)
    # A private decision is listed for its person alone.
    reader, _ = _decision_reader(request)
    # A teammate sees only what is in force: a fact the principal retired or
    # replaced (perhaps because it was wrong or too sensitive) no longer
    # renders anywhere, so its text is not theirs to read either.
    if principal:
        rows = list_facts(include_inactive=include_inactive, limit=limit, reader=reader)
        retirable = [f.id for f in rows if f.status == "active" and f.kind != "profile"]
        flags = {f.id: _decision_out(f, reader, principal) for f in rows if f.decision}
    else:
        rows = list_facts(limit=limit, reader=reader)
        retirable = [
            f.id for f in rows
            if f.status == "active"
            and (_is_own_teammate_fact(f, caller) or f.private_to_person_id is not None)
        ]
        flags = {f.id: _decision_out(f, reader, principal) for f in rows if f.decision}
        # Provenance is the principal's: their words, and which of their
        # chats and turns a fact came from (ids other routes may key on).
        rows = [
            f.model_copy(update={
                "source_quote": f.source_quote if _is_own_quote(f, caller) else "",
                "retired_reason": "",
                "session_id": None, "turn_id": None, "recorded_by_person_id": None,
            })
            for f in rows
        ]
    return FactsPage(
        facts=rows, can_retire=principal, retirable_ids=retirable, can_review=principal,
        shareable_ids=[i for i, d in flags.items() if d.can_share],
        unshareable_ids=[i for i, d in flags.items() if d.can_make_private],
    )


@router.post("/memories/facts/{fact_id}/retire", response_model=Fact)
def retire_standing_fact(fact_id: int, request: Request, body: FactRetire | None = None) -> Fact:
    """Stop an active fact rendering into any prompt. The row stays, as
    ``retired``, so the Pulse page still shows what it said. The principal
    can retire any; a teammate only one they recorded themselves."""
    principal = _caller_is_principal(request)
    existing = _readable_fact(fact_id, request)
    caller = None if principal else _caller_id(request)
    if not principal and not (existing is not None and _is_own_fact(existing, caller)):
        raise HTTPException(
            status_code=403,
            detail="Only the principal, or the teammate who recorded it, can retire a standing fact",
        )
    if existing is None or existing.kind == "profile":
        raise HTTPException(status_code=404, detail="Fact not found")
    reason = " ".join(((body.reason if body else "") or "retired from the Memory page").split())
    retired = retire_fact(
        fact_id, reason=reason[:280],
        # A private decision is its person's own, whoever outranks whom.
        teammate_id=None if principal or existing.private_to_person_id is not None else caller,
    )
    if retired is None:
        raise HTTPException(status_code=409, detail="Fact is no longer active")
    # Not the reason: it is the retiring person's own words.
    _audit_fact(
        "fact_retired", f"Standing fact {fact_id} retired" + _audit_subject(existing),
        {"fact_id": fact_id, **_audit_text(existing, subject=existing.subject, statement=existing.statement),
         "by": "principal" if principal else "teammate",
         **({} if principal else {"caller_person_id": caller})},
        actor="principal" if principal else "teammate",
    )
    return retired


@router.post("/memories/facts/{fact_id}/edit", response_model=Fact)
def edit_standing_fact(fact_id: int, body: FactEdit, request: Request) -> Fact:
    """Change what an active fact says. The edit is recorded as a correction
    that replaces it (same subject; the old wording kept as what it
    corrects), through ``record_fact``'s own checks. Who may: as for retire,
    the principal any fact, a teammate only their own."""
    principal = _caller_is_principal(request)
    existing = _readable_fact(fact_id, request)
    caller = None if principal else _caller_id(request)
    if not principal and not (existing is not None and _is_own_fact(existing, caller)):
        raise HTTPException(
            status_code=403,
            detail="Only the principal, or the teammate who recorded it, can edit a standing fact",
        )
    if existing is None or existing.kind == "profile":
        raise HTTPException(status_code=404, detail="Fact not found")
    if existing.status != "active":
        raise HTTPException(status_code=409, detail="Fact is no longer active")
    statement = " ".join(body.statement.split())
    if not statement or statement == existing.statement:
        raise HTTPException(status_code=422, detail="Say what the fact should say now")
    try:
        fact, replaced = record_fact(
            subject=existing.subject, statement=statement, source_quote=statement,
            kind="correction", previous_statement=existing.statement, replaces_fact_id=fact_id,
            source_channel="web",
            recorded_by_person_id=None if principal else caller,
            recorded_by_role="principal" if principal else "teammate",
            recorded_by_name="" if principal else existing.recorded_by_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    _audit_fact(
        "fact_edited", f"Standing fact {fact_id} edited" + _audit_subject(existing),
        {"fact_id": fact_id, "new_fact_id": fact.id,
         **_audit_text(existing, subject=existing.subject, before=existing.statement, after=fact.statement),
         "status": fact.status,
         "superseded_ids": [f.id for f in replaced],
         "by": "principal" if principal else "teammate",
         **({} if principal else {"caller_person_id": caller})},
        actor="principal" if principal else "teammate",
    )
    return fact


def _require_principal(request: Request, what: str) -> None:
    if not _caller_is_principal(request):
        raise HTTPException(status_code=403, detail=f"Only the principal can {what}")


@router.post("/memories/facts/{fact_id}/approve", response_model=Fact)
def approve_standing_fact(fact_id: int, request: Request) -> Fact:
    """Put a proposed fact in force: a decision copied in from mail or a chat
    app that waits for the principal's OK (``memory.facts.approve_fact``)."""
    _require_principal(request, "approve a fact")
    approved = approve_fact(fact_id)
    if approved is None:
        raise HTTPException(status_code=409, detail="Fact is not waiting for approval")
    fact, superseded = approved
    _audit_fact(
        "fact_reviewed", f"Standing fact {fact_id} approved: {fact.subject[:80]}",
        {"fact_id": fact_id, "decision": "approved", "subject": fact.subject,
         "statement": fact.statement, "superseded_ids": [f.id for f in superseded]},
        actor="principal",
    )
    return fact


@router.post("/memories/facts/{fact_id}/decline", response_model=Fact)
def decline_standing_fact(fact_id: int, request: Request, body: FactReview | None = None) -> Fact:
    """Drop a proposed fact; it is never used. Principal only."""
    _require_principal(request, "decline a fact")
    reason = " ".join(((body.reason if body else "") or "declined from the Memory page").split())
    declined = decline_fact(fact_id, reason=reason[:280])
    if declined is None:
        raise HTTPException(status_code=409, detail="Fact is not waiting for approval")
    _audit_fact(
        "fact_reviewed", f"Standing fact {fact_id} declined: {declined.subject[:80]}",
        {"fact_id": fact_id, "decision": "declined", "subject": declined.subject,
         "statement": declined.statement},
        actor="principal",
    )
    return declined


# --- People (peer memory) ---


def _caller_is_principal(request: Request) -> bool:
    from openexecutive.api.routes.people import caller_is_principal

    return caller_is_principal(request)


def _viewer(request: Request) -> int | None:
    """The caller's own People entry, or None (signed in but not on the
    roster, a service call, an unreadable roster)."""
    from openexecutive.api.routes.chat import _resolve_caller_person_id

    try:
        return _resolve_caller_person_id(request)
    except Exception:
        logger.exception("memories: caller lookup failed — showing no one's peer memory")
        return None


def _own_entries(people: list[PersonMemory], viewer: int | None) -> list[PersonMemory]:
    """The caller's own entry alone. Nobody sees another person's, the
    principal included: peer memory is drawn from each person's own
    conversations."""
    return [p for p in people if viewer is not None and p.person_id == viewer]


@router.get("/memories/people", response_model=PeopleMemory)
async def list_people_memory(
    request: Request, recent: int = Query(5, ge=1, le=50)
) -> PeopleMemory:
    """What peer memory knows about the caller: card, conclusion count,
    last-learned time and the ``recent`` newest conclusions. Read-only and
    LLM-free; ``status`` is ``disabled`` when peer memory is off.

    Only the caller's own entry: peer memory is drawn from each person's own
    conversations, so nobody else's is shown, the principal included. A
    caller who isn't on the roster sees none."""
    overview = await people_overview(recent=recent)
    people = _own_entries(overview.people, _viewer(request))
    return overview.model_copy(update={
        "people": people,
        "conclusion_total": sum(p.conclusion_count for p in people),
    })


@router.get("/memories/people/{person_id}/conclusions", response_model=PersonConclusionsPage)
async def list_person_conclusions(
    person_id: int,
    request: Request,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=PERSON_CONCLUSIONS_MAX_PAGE),
) -> PersonConclusionsPage:
    """One page of every conclusion peer memory holds about the caller,
    newest first. Read-only and LLM-free; 404 when the person is not on the
    roster or peer memory has no peer for them yet, and for anyone but the
    caller themselves, exactly as for an id that does not exist."""
    if _viewer(request) != person_id:
        raise HTTPException(status_code=404, detail="Person not found in peer memory")
    result = await person_conclusions(person_id, page=page, size=size)
    if result is None:
        raise HTTPException(status_code=404, detail="Person not found in peer memory")
    return result


@router.delete("/memories/people/{person_id}/conclusions/{conclusion_id}",
               status_code=status.HTTP_204_NO_CONTENT)
async def forget_conclusion(person_id: int, conclusion_id: str, request: Request) -> Response:
    """Remove one line of what peer memory concluded about the caller. Only
    their own, and only one found among their own conclusions: anything else
    is 404, as for an id that does not exist. 503 when peer memory can't be
    reached."""
    if _viewer(request) != person_id:
        raise HTTPException(status_code=404, detail="Not found")
    outcome = await forget_person_conclusion(person_id, conclusion_id)
    if outcome == "unavailable":
        raise HTTPException(status_code=503, detail="Memory is unavailable right now. Try again shortly.")
    if outcome != "forgotten":
        raise HTTPException(status_code=404, detail="Not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
