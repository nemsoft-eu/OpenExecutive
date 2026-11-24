"""Decisions API — the approve/reject/edit surface for gated Executive proposals.

This is the approve→execute bridge: proposals written by calendar_tools (and
future classes) sit in the trust ledger as `status='proposed'` until a human
acts here. What approving, rejecting and cancelling DO is per class: each class
resolved here has a ``DecisionClassSpec`` in ``DECISION_CLASSES`` (meeting
booking: create the calendar event on approve, delete it on cancel). An
instance of any other class is refused with 409, never carried out as some
other class.

Routes:
  GET  /decisions            — list instances (filter by class, status)
  GET  /decisions/{id}       — single instance detail
  POST /decisions/{id}/approve — approve (optionally with edited payload)
  POST /decisions/{id}/reject  — reject
  GET  /decisions/classes/meeting_scheduling — the class's autonomy mode
  PUT  /decisions/classes/meeting_scheduling — set it (principal only)
  GET  /audit/reliability    — per-class reliability card (also in audit.py router)
"""
from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from openexecutive.api import caller as api_caller
from openexecutive.memory.decision_ledger import (
    DECISION_ALERT_SOURCE,
    STATUS_APPROVED_UNCHANGED,
    STATUS_APPROVED_WITH_EDIT,
    STATUS_FAILED,
    STATUS_PROPOSED,
    STATUS_REJECTED,
    DecisionInstance,
    ReliabilityCard,
    aggregate_reliability,
    get_class_mode,
    get_decision_instance,
    list_instances,
    mark_resolved,
    mark_reversed,
    set_class_mode,
)

router = APIRouter()
logger = logging.getLogger(__name__)

_CALENDAR_CLASS = "meeting_scheduling"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ApproveBody(BaseModel):
    """Approve a proposal, optionally with an edited meeting payload.

    Any field present in `edits` overwrites the corresponding field in the
    original proposed_payload.  Missing fields retain their proposed values.
    Supported edit keys: title, start, end, attendee_emails, description.

    For a reply card (``delegation_reply``) nothing is edited here: the draft
    is sent as it is in Gmail. ``edits`` carries the person's second yes when
    something changed since the card was made: ``recipients`` (the list they
    were shown) and ``thread_moved_on: true``, and ``seen_caught_up_at``
    (the card's ``caught_up_at`` as shown, so a draft brought up to date
    since asks again); and Send + allow in training:
    ``allow: true``, with ``example: true`` to keep an edited draft as an
    example (``delegation.training``).

    For a meeting invite card (``delegation_invite``): ``answer``, ``accept``
    (the default) or ``decline``, answered in the person's own calendar; and
    ``allow: true`` with accept, in training, to accept that sender's
    invites on their own from now on (``delegation.invites``).
    """
    edits: dict[str, Any] | None = None


class RejectBody(BaseModel):
    reason: str = ""


ClassMode = Literal["propose", "auto_execute"]


class DecisionClassMode(BaseModel):
    """How the Executive handles a decision class: ``propose`` puts each one
    on the briefing for approval, ``auto_execute`` does it straight away."""
    decision_class: str
    mode: ClassMode


class DecisionClassModeUpdate(BaseModel):
    mode: ClassMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_payload(instance: DecisionInstance) -> dict[str, Any]:
    try:
        return json.loads(instance.proposed_payload_json)
    except (json.JSONDecodeError, TypeError):
        return {}


_HELD_CLASSES = frozenset({"department_action", "take_the_lead_action"})


def _served_json(raw: str | None) -> str | None:
    """A held step's stored payload (proposed or final) as served, or None
    when it has no tool input to clean."""
    from openexecutive.orchestrator import card_details

    try:
        payload = json.loads(raw or "")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("input"), dict):
        return None
    shown = card_details.served(payload)
    said = payload.get(SAID_INPUT)
    if isinstance(said, dict) and said and not isinstance(payload.get("steps"), list):
        # The step with what was said on its card, for the cards whose
        # Approve sends it (``edits.input``). ``details`` stay the step as
        # filed: what a plain Approve, in chat or on a page, carries out.
        said_details = card_details.served({**payload, "input": {**payload["input"], **said}})["details"]
        shown = {**shown, "said_details": said_details}
    return json.dumps(shown, ensure_ascii=False)


def _served(instance: DecisionInstance) -> DecisionInstance:
    """A held step as a page reads it: plain details, and no file's bytes
    (``orchestrator.card_details.served``), in what was proposed and in what
    was carried out (``final_payload_json`` copies the input). Approve reads
    the stored row."""
    if instance.decision_class not in _HELD_CLASSES:
        return instance
    update: dict[str, Any] = {}
    for field in ("proposed_payload_json", "final_payload_json"):
        shown = _served_json(getattr(instance, field, None))
        if shown is not None:
            update[field] = shown
    return instance.model_copy(update=update) if update else instance


def _payload_diff(original: dict[str, Any], final: dict[str, Any]) -> bool:
    """Return True if the payloads differ in any semantically meaningful field."""
    for key in ("title", "start", "end", "description"):
        if original.get(key) != final.get(key):
            return True
    orig_attendees = sorted(original.get("attendee_emails", []))
    final_attendees = sorted(final.get("attendee_emails", []))
    return orig_attendees != final_attendees


def _clear_decision_alert(
    instance_id: int, status: str, source: str | None = DECISION_ALERT_SOURCE
) -> None:
    """Clear the companion briefing alert when a decision is resolved.

    Calendar proposals are surfaced on the briefing as an alert linked by
    ``external_id='decision:{id}'`` (see calendar_tools._propose_via_decision_alert).
    Transition it out of the unread queue when the decision is approved
    (``ack``) or rejected/cancelled (``dismissed``). Best-effort: a missing
    alert (e.g. a decision created before the bridge) is a harmless no-op, and
    a failure here must not 500 a decision that already executed. ``source``
    is the class's ``alert_source``; a class that raises no alert passes None.
    """
    if source is None:
        return
    from openexecutive.alerts.store import set_status_by_external
    from openexecutive.memory.decision_ledger import decision_alert_external_id

    try:
        set_status_by_external(source, decision_alert_external_id(instance_id), status)
    except Exception:
        logger.exception(
            "decisions: failed to clear companion alert for instance %d", instance_id
        )


async def _execute_booking(
    instance: DecisionInstance,
    final_payload: dict[str, Any],
    *,
    by_principal: bool = False,
) -> dict[str, Any]:
    """Call the MCP to create the calendar event. Returns the gateway response.

    ``by_principal``: the approver is the principal, acting in the web app. An
    approval runs outside any chat turn, so without this the gateway would
    refuse a contact on the invite even though the principal asked for it.
    """
    from openexecutive.orchestrator.calendar_tools import _do_create_event
    from openexecutive.orchestrator.mcp_gateway import get_active_gateway
    from openexecutive.orchestrator.people_tools import grant_contact_egress

    gateway = get_active_gateway()
    if gateway is None:
        return {"error": "MCP gateway not running — cannot create calendar event"}
    if by_principal:
        with grant_contact_egress():
            return await _do_create_event(gateway, final_payload)
    return await _do_create_event(gateway, final_payload)


def _approver_is_principal(request: Request) -> bool:
    """Whether the caller resolves to the principal (a request with no
    ``x-caller-email`` does — the CLI, direct curl, local login). Fails
    closed."""
    from openexecutive.api.routes.people import caller_is_principal

    return caller_is_principal(request)


def _is_private(instance: DecisionInstance) -> bool:
    """Not everyone's to see: a class that is the principal's alone
    (``DecisionClassSpec.principal_only``) or its approver's alone
    (``approver_only``, decided before this is asked), or a payload marked
    ``private`` (a booking with one of the principal's contacts, per
    ``calendar_tools``)."""
    spec = DECISION_CLASSES.get(instance.decision_class)
    if spec is not None and (spec.principal_only or spec.approver_only):
        return True
    payload = _parse_payload(instance)
    return payload.get("private") is True


def _approver_only(instance: DecisionInstance) -> bool:
    spec = DECISION_CLASSES.get(instance.decision_class)
    return spec is not None and spec.approver_only


def _principal_and_approver(instance: DecisionInstance) -> bool:
    spec = DECISION_CLASSES.get(instance.decision_class)
    return spec is not None and spec.principal_and_approver


def _caller_is_approver(instance: DecisionInstance, request: Request) -> bool:
    """Whether the caller is the person ``instance`` went to for approval.
    Fails closed."""
    from openexecutive.api.routes.chat import _resolve_caller_person_id

    try:
        caller = _resolve_caller_person_id(request)
    except Exception:
        logger.exception("decisions: caller lookup failed — treating them as someone else")
        return False
    return caller is not None and caller == instance.approver_person_id


def _own_chat_ok(instance: DecisionInstance, request: Request) -> bool:
    """A step held from one person's own chat is theirs and its approver's
    (``alerts.own_chat``): nobody else on the team sees or decides it, the
    principal included. Fails closed."""
    from openexecutive.alerts import own_chat
    from openexecutive.api.routes.chat import _resolve_caller_person_id

    try:
        caller = _resolve_caller_person_id(request)
    except Exception:
        logger.exception("decisions: caller lookup failed — treating them as someone else")
        caller = None
    return own_chat.decision_shown_to(instance, caller)


def _own_chat_kept(instances: list[DecisionInstance], request: Request) -> list[DecisionInstance]:
    """``instances`` without the ones held from someone else's own chat
    (``_own_chat_ok``), with the caller and the chats looked up once for the
    whole list. Fails closed."""
    from openexecutive.alerts import own_chat
    from openexecutive.api.routes.chat import _resolve_caller_person_id

    try:
        caller = _resolve_caller_person_id(request)
    except Exception:
        logger.exception("decisions: caller lookup failed — treating them as someone else")
        caller = None
    only = own_chat.decision_readers(instances)
    return [
        i for i in instances
        if only.get(int(i.id)) is None or (caller is not None and caller in only[int(i.id)])
    ]


def _visible_instance(instance_id: int, request: Request) -> DecisionInstance:
    """The instance, or 404 — also for a private one when the caller is not
    the principal, and for one only its approver may see when the caller is
    anyone else (the principal included), so its existence (and the contact
    or mail on it) is not revealed."""
    instance = get_decision_instance(instance_id)
    if instance is None or not _own_chat_ok(instance, request):
        raise HTTPException(status_code=404, detail="Decision instance not found")
    if _approver_only(instance):
        if not _caller_is_approver(instance, request):
            raise HTTPException(status_code=404, detail="Decision instance not found")
        return instance
    if _principal_and_approver(instance):
        if not (_approver_is_principal(request) or _caller_is_approver(instance, request)):
            raise HTTPException(status_code=404, detail="Decision instance not found")
        return instance
    if _is_private(instance) and not _approver_is_principal(request):
        raise HTTPException(status_code=404, detail="Decision instance not found")
    return instance


_NOT_YOURS = (
    "Only the person this went to for approval, or the owner, can decide it."
)


def _resolver(instance: DecisionInstance, request: Request) -> int | None:
    """The caller's Person id when they may approve, reject or cancel
    ``instance``: the principal, or the person it went to for approval
    (``approver_person_id``) unless it is private to the principal. Anyone
    else gets 403. Fails closed: a roster that can't be read is a 403."""
    from openexecutive.api.routes.chat import _resolve_caller_person_id

    caller = _resolve_caller_person_id(request)
    if not _own_chat_ok(instance, request):
        pass  # from someone else's own chat: the 403 below
    elif _approver_only(instance):
        if caller is not None and caller == instance.approver_person_id:
            return caller
    elif _approver_is_principal(request):
        return caller
    elif _principal_and_approver(instance):
        if caller is not None and caller == instance.approver_person_id:
            return caller
    elif caller is not None and caller == instance.approver_person_id and not _is_private(instance):
        return caller
    logger.info(
        "decisions: caller may not resolve instance %d (class %s)",
        instance.id, instance.decision_class,
    )
    raise HTTPException(status_code=403, detail=_NOT_YOURS)


def _refresh(instance_id: int) -> DecisionInstance:
    updated = get_decision_instance(instance_id)
    if updated is None:
        raise HTTPException(status_code=500, detail="Instance vanished after update")
    return updated


# ---------------------------------------------------------------------------
# Decision classes
# ---------------------------------------------------------------------------

# (instance, body, request, the resolver's Person id) → the updated instance.
_Approve = Callable[
    [DecisionInstance, ApproveBody, Request, int | None], Awaitable[DecisionInstance]
]


@dataclass(frozen=True)
class DecisionClassSpec:
    """How the routes below handle one class of gated decision."""

    name: str
    # Only the principal may see or resolve any instance of it, whatever its
    # payload says.
    principal_only: bool
    # The companion briefing alert's source, cleared when an instance is
    # resolved; None when the class raises none.
    alert_source: str | None
    # Carries out an approval and records it (its own compare-and-set, with
    # the resolver); raises HTTPException to refuse. Called only on a
    # 'proposed' instance, and only for a caller who may resolve it.
    approve: _Approve
    # Best effort, after a reject has been recorded.
    after_reject: Callable[[DecisionInstance], Awaitable[None]] | None = None
    # Best effort, after the tidy-up let a waiting step lapse (True) or its
    # Undo put it back (False): the class's own close-out, like Skip's.
    after_lapse: Callable[[DecisionInstance, bool], None] | None = None
    # Undoes what an approval did; None when it can't be undone (409).
    cancel: Callable[[DecisionInstance], Awaitable[None]] | None = None
    # Only the person an instance went to for approval (``approver_person_id``)
    # may see or resolve it — not the principal either, unless it is theirs.
    approver_only: bool = False
    # Only the principal and the person it went to for approval may see or
    # resolve it; nobody else on the team, whatever its payload says.
    principal_and_approver: bool = False


def run_after_lapse(instance_id: int, *, lapsed: bool, db_path: Path | None = None) -> None:
    """Run a waiting step's class close-out for a lapse (or its Undo):
    ``alerts.lifecycle.settle`` and the card's reopen call it, so a lapse
    ends a step the way Skip does in each class's own records."""
    from openexecutive.memory.decision_ledger import get_decision_instance

    instance = get_decision_instance(instance_id, db_path=db_path)
    spec = DECISION_CLASSES.get(instance.decision_class) if instance is not None else None
    if instance is None or spec is None or spec.after_lapse is None:
        return
    spec.after_lapse(instance, lapsed)


def _spec_for(instance: DecisionInstance) -> DecisionClassSpec:
    spec = DECISION_CLASSES.get(instance.decision_class)
    if spec is None:
        raise HTTPException(
            status_code=409,
            detail=f"Decisions of class {instance.decision_class!r} can't be resolved here.",
        )
    return spec


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/decisions", response_model=list[DecisionInstance])
def get_decisions(
    request: Request,
    decision_class: str = _CALENDAR_CLASS,
    status: str | None = None,
    limit: int = 50,
) -> list[DecisionInstance]:
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be 1–500")
    spec = DECISION_CLASSES.get(decision_class)
    if spec is not None and (spec.approver_only or (spec.principal_and_approver and not _approver_is_principal(request))):
        from openexecutive.api.routes.chat import _resolve_caller_person_id

        caller = _resolve_caller_person_id(request)
        if caller is None:
            return []
        instances = list_instances(decision_class, status=status, limit=500)
        return [_served(i) for i in instances if i.approver_person_id == caller][:limit]
    if _approver_is_principal(request):
        # Read past ``limit``: steps held from other people's own chats drop
        # out below, and a page of them must not hide the rest.
        kept = _own_chat_kept(list_instances(decision_class, status=status, limit=500), request)
        return [_served(i) for i in kept[:limit]]
    spec = DECISION_CLASSES.get(decision_class)
    if spec is not None and spec.principal_only:
        return []
    instances = list_instances(decision_class, status=status, limit=500)
    return [_served(i) for i in _own_chat_kept(instances, request) if not _is_private(i)][:limit]


@router.get("/decisions/{instance_id}", response_model=DecisionInstance)
def get_decision(instance_id: int, request: Request) -> DecisionInstance:
    instance = _visible_instance(instance_id, request)
    served = _served(instance)
    if instance.decision_class not in _HELD_CLASSES or not served.proposed_payload_json:
        return served
    # Whether this reader may change its step (Edit, Or tell me), so a page
    # offers only what Approve and /say take from them.
    try:
        may = _may_change(instance, request, _resolver(instance, request))
    except HTTPException:
        may = False
    from openexecutive.api.routes.chat import _resolve_caller_person_id
    from openexecutive.orchestrator import card_details

    proposed = json.loads(served.proposed_payload_json)
    # The card's opening lines (who it waits on, who started it, why it
    # waits), as every app shows them.
    top = card_details.top(
        proposed, approver_id=instance.approver_person_id, viewer_id=_resolve_caller_person_id(request),
    )
    shown = {**proposed, "may_change": may, "top": top}
    return served.model_copy(update={"proposed_payload_json": json.dumps(shown, ensure_ascii=False)})


def _check_booking_edits(payload: dict[str, Any], edits: dict[str, Any]) -> None:
    """An edited booking keeps the shape the tool would have accepted: an
    edited title of 1-200 characters, and a start and end it can read (a time
    with no offset is the company's, as in ``calendar_tools._parse_iso``), end after start.
    422 otherwise."""
    from openexecutive.orchestrator.calendar_tools import _parse_iso

    if "title" in edits:
        title = edits["title"]
        if not isinstance(title, str) or not title.strip() or len(title) > 200:
            raise HTTPException(status_code=422, detail="Give the meeting a title of up to 200 characters.")
    raw_start, raw_end = payload.get("start"), payload.get("end")
    start = _parse_iso(raw_start) if isinstance(raw_start, str) else None
    end = _parse_iso(raw_end) if isinstance(raw_end, str) else None
    if start is None or end is None:
        raise HTTPException(status_code=422, detail="Start and end must be dates and times.")
    if end <= start:
        raise HTTPException(status_code=422, detail="The end has to be after the start.")


async def _approve_booking(
    instance: DecisionInstance, body: ApproveBody, request: Request, resolver: int | None
) -> DecisionInstance:
    """Meeting booking: create the calendar event, then record the approval."""
    return await _book(instance, body, by_principal=_approver_is_principal(request), resolver=resolver)


async def _book(
    instance: DecisionInstance, body: ApproveBody, *, by_principal: bool, resolver: int | None
) -> DecisionInstance:
    """Book ``instance``'s meeting and record the approval. Contacts go on
    the invite only when ``by_principal``."""
    instance_id = instance.id
    original_payload = _parse_payload(instance)
    final_payload = dict(original_payload)

    # Apply edits if provided.
    if body.edits:
        for key in ("title", "start", "end", "description"):
            if key in body.edits:
                final_payload[key] = body.edits[key]
        _check_booking_edits(final_payload, body.edits)
        if "attendee_emails" in body.edits:
            final_payload["attendee_emails"] = body.edits["attendee_emails"]

    # Re-run the backend's conflict check — advisory here, but we log a
    # warning if the slot is busy. (Google: multi-calendar freebusy; Microsoft:
    # the Executive's own calendar view — see integrations.workspace.)
    try:
        from openexecutive.integrations.workspace.registry import get_calendar_provider
        from openexecutive.orchestrator.mcp_gateway import get_active_gateway
        gw = get_active_gateway()
        if gw is not None:
            busy = await get_calendar_provider().has_conflicts(
                gw,
                final_payload["start"],
                final_payload["end"],
                list(final_payload.get("attendee_emails", [])),
            )
            if busy:
                logger.warning(
                    "decisions/approve: calendar reports conflict for instance %d — proceeding anyway (human override)",
                    instance_id,
                )
    except Exception:
        logger.debug("decisions/approve: conflict check skipped", exc_info=True)

    # Create the actual calendar event. Contacts on the invite only when the
    # principal is the one approving.
    result = await _execute_booking(instance, final_payload, by_principal=by_principal)
    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])

    external_event_id = result.get("event_id")
    # Persist the video-meeting link (Meet or Teams, if one was minted) so the
    # UI and any later reference can surface it.
    if result.get("meet_link"):
        final_payload["meet_link"] = result["meet_link"]
    edited = _payload_diff(original_payload, final_payload)
    outcome = STATUS_APPROVED_WITH_EDIT if edited else STATUS_APPROVED_UNCHANGED

    # Compare-and-set: if a concurrent approve already won, our event is a
    # duplicate and must be deleted before we return 409.
    recorded = mark_resolved(
        instance_id,
        outcome,
        final_payload=final_payload,
        resolver_person_id=resolver,
        external_event_id=external_event_id,
    )
    if not recorded:
        # A concurrent approve beat us.  Delete the event we just created to
        # avoid a ghost booking with no ledger row referencing it.
        if external_event_id:
            try:
                from openexecutive.orchestrator.calendar_tools import _do_delete_event
                gw2 = get_active_gateway()
                if gw2 is not None:
                    await _do_delete_event(gw2, external_event_id)
            except Exception:
                logger.exception(
                    "decisions/approve: failed to clean up leaked event %s for instance %d",
                    external_event_id, instance_id,
                )
        raise HTTPException(
            status_code=409,
            detail="A concurrent approval already processed this proposal.",
        )

    # Won the compare-and-set: clear the companion briefing alert.
    _clear_decision_alert(instance_id, "ack")
    return _refresh(instance_id)


async def _cancel_booking(instance: DecisionInstance) -> None:
    """Meeting booking: delete the event an approval created, if any."""
    if not instance.external_event_id:
        if instance.status == STATUS_PROPOSED:
            return
        # Booked, but the event id was never read: refuse rather than report
        # a cancel that leaves the meeting on everyone's calendar.
        from openexecutive.orchestrator.calendar_tools import _NO_EVENT_ID

        raise HTTPException(status_code=409, detail=_NO_EVENT_ID)
    from openexecutive.orchestrator.calendar_tools import _do_delete_event
    from openexecutive.orchestrator.mcp_gateway import get_active_gateway

    gw = get_active_gateway()
    if gw is None:
        raise HTTPException(status_code=502, detail="MCP gateway not running")
    result = await _do_delete_event(gw, instance.external_event_id)
    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])


_MEETING_BOOKING = DecisionClassSpec(
    name=_CALENDAR_CLASS,
    principal_only=False,
    alert_source=DECISION_ALERT_SOURCE,
    approve=_approve_booking,
    cancel=_cancel_booking,
)


async def _send_reply(
    instance: DecisionInstance, body: ApproveBody, request: Request, resolver: int | None
) -> DecisionInstance:
    """Send the reply's exact Gmail draft, on its person's tap
    (``delegation.reply_send``, which checks everything again first)."""
    from openexecutive.delegation.reply_send import SendRefused, send_approved_reply

    # ``edits.example`` keeps an edited draft as how they write (Drafts in
    # training). The rest is the second yes.
    confirm = dict(body.edits or {})
    example = confirm.pop("example", None)
    # ``edits.allow``: "on its own next time", In training (delegation.training).
    allow = confirm.pop("allow", None)
    try:
        await send_approved_reply(
            instance, caller=api_caller.caller(request), resolver=resolver, confirm=confirm,
            allow=allow is True, example=example is True,
        )
    except SendRefused as refused:
        raise HTTPException(
            status_code=refused.status,
            detail={"code": refused.code, "message": refused.message, **refused.extra},
        ) from None
    return _refresh(instance.id)


async def _dismiss_reply(instance: DecisionInstance) -> None:
    """Delete the reply's draft from Gmail when nobody edited it."""
    from openexecutive.delegation.replies import dismiss

    await dismiss(instance)


# A reply the inbox watcher drafted as someone (delegation.inbox): theirs
# alone, whatever its payload says (a team member's is hidden from the
# principal too), and never an alert.
_DELEGATED_REPLY = DecisionClassSpec(
    name="delegation_reply",
    principal_only=False,
    approver_only=True,
    alert_source=None,
    approve=_send_reply,
    after_reject=_dismiss_reply,
)

async def _carry_out_lead(
    instance: DecisionInstance, body: ApproveBody, request: Request, resolver: int | None
) -> DecisionInstance:
    """Take the lead: do the exact action that waited, once (claimed first),
    then record it in Recent activity.

    ``edits.input`` changes the card's editable fields first (the message,
    a meeting's title or time; ``take_the_lead.apply_edits``). ``edits.allow``
    on a training card also allows that action from now on, keeping any
    edit as an example of how it's wanted: the principal's alone, and only
    from a request the API can tie to them, like turning Take the lead on."""
    from openexecutive.orchestrator import take_the_lead

    payload = _parse_payload(instance)
    edits = body.edits or {}
    if _edits_beyond_said(payload, edits.get("input")):  # noqa: SIM102
        # Changing what the Executive says or books is the principal's: an
        # area approver says yes or no to what it would do, as it stands,
        # with what was said on its card.
        if not _approver_is_principal(request):
            raise HTTPException(status_code=403, detail="Only the account owner can change this before approving.")
    payload, changed = _with_edits(payload, edits.get("input"))
    grant = payload.get("allow") if isinstance(payload.get("allow"), dict) else None
    allowing = edits.get("allow") is True
    if allowing:
        if grant is None or payload.get("kind") != take_the_lead.TRAINING:
            raise HTTPException(status_code=422, detail="This one can't be allowed from its card.")
        if not _approver_is_principal(request) or not _provably_principal(request):
            raise HTTPException(
                status_code=403,
                detail="Only the account owner can allow this, from a signed-in session on this server.",
            )
        # Checked before anything is done, so a full list refuses the allow
        # instead of failing after the action went out.
        if not take_the_lead.room_for(str(grant.get("key"))):
            raise HTTPException(
                status_code=409,
                detail=f"It can learn at most {take_the_lead.ALLOWED_MAX} things. Remove one in Settings first.",
            )
    return await _run_lead(
        instance, payload, resolver, by_principal=_approver_is_principal(request), changed=changed,
        grant=grant if allowing else None,
    )


def _own_card(instance: DecisionInstance, resolver: int | None, *, by_principal: bool) -> bool:
    """Whether the person who asked is approving a step held on their own
    card. Checked the same way for both card kinds; ``take_the_lead.carry_out``
    decides which tools it covers (an extension's ``card`` tool, which only
    ever runs on their own device), and for any other tool it changes
    nothing."""
    payload = _parse_payload(instance)
    return (
        not by_principal
        and resolver is not None
        and resolver == instance.approver_person_id
        and resolver == payload.get("requested_by_person_id")
    )


async def _run_lead(
    instance: DecisionInstance, payload: dict[str, Any], resolver: int | None, *, by_principal: bool,
    changed: dict[str, str] | None = None, grant: dict[str, Any] | None = None,
) -> DecisionInstance:
    """Do a Take the lead card's action once (claimed first), as ``payload``
    has it, and allow it from now on when ``grant`` is given (checked by the
    caller)."""
    from openexecutive.memory.decision_ledger import claim_for_execution, finish_execution
    from openexecutive.orchestrator import take_the_lead

    if not claim_for_execution(instance.id, resolver_person_id=resolver):
        raise HTTPException(status_code=409, detail="Someone else resolved this decision first.")
    try:
        result = await take_the_lead.carry_out(
            {**payload, "tapped_by": resolver}, by_principal=by_principal,
            own_card=_own_card(instance, resolver, by_principal=by_principal),
        )
    except Exception as exc:
        logger.exception("decisions/approve: As me action %d failed", instance.id)
        finish_execution(instance.id, STATUS_FAILED, final_payload={**payload, "error": type(exc).__name__})
        take_the_lead.resolved(instance.id, "failed")
        _clear_decision_alert(instance.id, "dismissed")
        _keep_allowed(grant, instance.id, resolver, changed)
        raise HTTPException(status_code=502, detail="Something went wrong, so it may not have gone through. Check before trying again.") from None
    failed = take_the_lead.result_failed(result)
    finish_execution(
        instance.id,
        STATUS_FAILED if failed else STATUS_APPROVED_WITH_EDIT if changed else STATUS_APPROVED_UNCHANGED,
        final_payload={**payload, "result": result[:2000]},
    )
    take_the_lead.resolved(instance.id, "failed" if failed else "approved", payload=payload, result=result)
    _clear_decision_alert(instance.id, "dismissed" if failed else "ack")
    if failed:
        logger.warning("decisions/approve: step %d failed: %s", instance.id, result[:300])
        _keep_allowed(grant, instance.id, resolver, changed)
        raise HTTPException(
            status_code=502,
            detail=f"It couldn't do that. {take_the_lead.failure_words(str(payload.get('tool') or ''), result)}",
        )
    _keep_allowed(grant, instance.id, resolver, changed)
    return _refresh(instance.id)


def _keep_allowed(
    grant: dict[str, Any] | None, decision_id: int, resolver: int | None, changed: dict[str, str] | None,
) -> None:
    """Save an "on its own next time" tick. Kept whether or not the step
    went through: the tick is about the kind of step, and a step that failed
    (an app turned it down) shouldn't make the person teach it again."""
    if grant is None:
        return
    from openexecutive.orchestrator import take_the_lead

    try:
        take_the_lead.allow(
            str(grant.get("key")), str(grant.get("label") or ""), example=changed or None,
            created_by=f"person:{resolver}" if resolver is not None else "principal", decision_id=decision_id,
        )
    except take_the_lead.RuleError:
        # Only the learning failed (the list filled up since the check
        # before the step). Say so in the log rather than fail the card.
        logger.warning("decisions/approve: the next-time tick on card %d couldn't be saved", decision_id)
        return
    except Exception:
        logger.exception("decisions/approve: saving the next-time tick on card %d broke", decision_id)
        return
    _audit_allowed(str(grant.get("label") or ""), decision_id, bool(changed))


def _provably_principal(request: Request) -> bool:
    """Whether this request is tied to the principal (signed sign-ins or
    local login), as turning Take the lead on needs."""
    from openexecutive.api.routes.take_the_lead import _provably_theirs
    from openexecutive.people.store import find_principal_person

    try:
        principal = find_principal_person()
    except Exception:
        return False
    return principal is not None and _provably_theirs(request, principal)


def _audit_allowed(label: str, decision_id: int, edited: bool) -> None:
    from openexecutive.audit import log_event

    try:
        log_event(
            "take_the_lead_changed", f"Allowed from a card: {label}", actor="user",
            details={"allowed": label, "decision_id": decision_id, "with_example": edited},
        )
    except Exception:
        logger.warning("decisions: couldn't audit an allowance", exc_info=True)


async def _lead_declined(instance: DecisionInstance) -> None:
    from openexecutive.orchestrator import take_the_lead

    take_the_lead.resolved(instance.id, "declined")


def _lead_lapsed(instance: DecisionInstance, lapsed: bool) -> None:
    """The log reads it as lapsed, or waiting again on Undo; its text stays,
    so Undo brings the same step back."""
    from openexecutive.orchestrator import take_the_lead

    take_the_lead.mark_lapsed(instance.id, lapsed=lapsed)


# Take the lead: an action the gate held (orchestrator.take_the_lead), for
# the person whose authority covers it or the principal. What it would do
# quotes the Executive's context, so it is private to them.
_TAKE_THE_LEAD = DecisionClassSpec(
    name="take_the_lead_action",
    principal_only=False,
    principal_and_approver=True,
    alert_source=DECISION_ALERT_SOURCE,
    approve=_carry_out_lead,
    after_reject=_lead_declined,
    after_lapse=_lead_lapsed,
)

async def _carry_out_actions(
    instance: DecisionInstance, body: ApproveBody, request: Request, resolver: int | None
) -> DecisionInstance:
    """Do an action card's actions, exactly as stored, on its person's tap
    (``delegation.action_cards``, which checks the caller and each action
    again first). ``edits.only`` picks which of them; ``edits.allow``, In
    training, also allows everyone they reach."""
    from openexecutive.delegation.action_cards import ApproveRefused, approve

    try:
        await approve(
            instance, caller=api_caller.caller(request), resolver=resolver,
            only=(body.edits or {}).get("only"), allow=(body.edits or {}).get("allow") is True,
        )
    except ApproveRefused as refused:
        raise HTTPException(
            status_code=refused.status, detail={"code": refused.code, "message": refused.message},
        ) from None
    return _refresh(instance.id)


async def _actions_dismissed(instance: DecisionInstance) -> None:
    from openexecutive.delegation.action_cards import dismissed

    await dismissed(instance)


# A card of actions the Executive suggested on a turn about someone's mail
# (delegation.action_cards): theirs alone, like a reply card, and never an
# alert.
_DELEGATED_ACTIONS = DecisionClassSpec(
    name="delegation_actions",
    principal_only=False,
    approver_only=True,
    alert_source=None,
    approve=_carry_out_actions,
    after_reject=_actions_dismissed,
)

async def _answer_invite(
    instance: DecisionInstance, body: ApproveBody, request: Request, resolver: int | None
) -> DecisionInstance:
    """Accept or decline a meeting invite in its person's own calendar, on
    their tap (``delegation.invites``, which checks the caller and the card
    again first). ``edits.answer``: accept (the default) or decline;
    ``edits.allow`` with accept, In training, also allows that sender."""
    from openexecutive.delegation.invites import AnswerRefused, answer

    edits = body.edits or {}
    try:
        await answer(
            instance, caller=api_caller.caller(request), resolver=resolver,
            answer=edits.get("answer", "accept"), allow=edits.get("allow") is True,
        )
    except AnswerRefused as refused:
        raise HTTPException(
            status_code=refused.status, detail={"code": refused.code, "message": refused.message},
        ) from None
    return _refresh(instance.id)


async def _invite_dismissed(instance: DecisionInstance) -> None:
    from openexecutive.delegation.invites import dismissed

    await dismissed(instance)


# A meeting invite from someone the person knows (delegation.invites): theirs
# alone, like a reply card, and never an alert. Dismiss answers nothing.
_DELEGATED_INVITE = DecisionClassSpec(
    name="delegation_invite",
    principal_only=False,
    approver_only=True,
    alert_source=None,
    approve=_answer_invite,
    after_reject=_invite_dismissed,
)

async def _carry_out_department_action(
    instance: DecisionInstance, body: ApproveBody, request: Request, resolver: int | None
) -> DecisionInstance:
    """A step held for a yes (orchestrator.department_gate): do that exact
    call, once (claimed first), and record who approved it. ``edits.input``
    changes the card's editable fields first (an email's subject and text,
    a message; ``take_the_lead.apply_edits``): the owner's, or the card's
    own approver's."""
    payload = _parse_payload(instance)
    edits = body.edits or {}
    if edits.get("input") is not None:
        mine = resolver is not None and resolver == instance.approver_person_id
        if not (_approver_is_principal(request) or mine):
            raise HTTPException(status_code=403, detail="Only the person this card waits on can change it before approving.")
    payload, changed = _with_edits(payload, edits.get("input"))
    return await _run_department_action(
        instance, resolver, by_principal=_approver_is_principal(request), payload=payload, edited=bool(changed),
    )


def may_change_as(instance: DecisionInstance, *, principal: bool, approver: bool) -> bool:
    """Whether someone may change a held step's fields, as Approve's own
    ``edits.input`` allows, on the page and in a chat app alike: a Take the
    lead card's are the principal's, a held step's the principal's or the
    person it waits on (``approver``)."""
    return principal or (instance.decision_class == _DEPARTMENT_ACTION.name and approver)


def _may_change(instance: DecisionInstance, request: Request, resolver: int | None) -> bool:
    return may_change_as(
        instance, principal=_approver_is_principal(request),
        approver=resolver is not None and resolver == instance.approver_person_id,
    )


def _with_edits(
    payload: dict[str, Any], input_edits: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """A held step's payload with ``input_edits`` put in through
    ``take_the_lead.apply_edits`` (the step's own rules for a change), and
    the fields that changed. Raises HTTPException (422)."""
    from openexecutive.orchestrator import card_details, take_the_lead

    if input_edits is None:
        return {k: v for k, v in payload.items() if k != SAID_INPUT}, {}
    if len(card_details.held_steps(payload)) > 1:
        # One card for several calls: Approve runs each as shown.
        raise HTTPException(status_code=422, detail="This one can't be changed from its card.")
    source: dict[str, Any] = payload["input"] if isinstance(payload.get("input"), dict) else {}
    try:
        new_input, changed = take_the_lead.apply_edits(
            str(payload.get("tool") or ""), source, input_edits, mcp=bool(payload.get("mcp")),
        )
    except take_the_lead.EditError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    # What was said is in the input now: what was carried out shows as sent.
    done = {k: v for k, v in _edited(payload, new_input, changed).items() if k != SAID_INPUT}
    return done, changed


def _edited(payload: dict[str, Any], new_input: dict[str, Any], changed: dict[str, str] | None) -> dict[str, Any]:
    """The card's payload as the person changed it: what it does, and the
    line Activity and Done for you show, say what goes out, never what the
    card first said."""
    from openexecutive.orchestrator import take_the_lead

    out = {**payload, "input": new_input}
    if changed:
        try:
            out["summary"] = take_the_lead.summarize(
                str(payload.get("tool") or ""), new_input, mcp=bool(payload.get("mcp")),
            )
        except Exception:
            logger.warning("decisions/approve: couldn't restate an edited card", exc_info=True)
    return out


async def _run_department_action(
    instance: DecisionInstance, resolver: int | None, *, by_principal: bool,
    payload: dict[str, Any] | None = None, edited: bool = False,
) -> DecisionInstance:
    from openexecutive.audit import log_event
    from openexecutive.memory.decision_ledger import claim_for_execution, finish_execution
    from openexecutive.orchestrator import take_the_lead

    payload = payload if payload is not None else _parse_payload(instance)
    if not claim_for_execution(instance.id, resolver_person_id=resolver):
        raise HTTPException(status_code=409, detail="Someone else resolved this decision first.")
    try:
        result = await take_the_lead.carry_out(
            {**payload, "tapped_by": resolver}, by_principal=by_principal,
            own_card=_own_card(instance, resolver, by_principal=by_principal),
        )
    except Exception as exc:
        logger.exception("decisions/approve: department action %d failed", instance.id)
        finish_execution(instance.id, STATUS_FAILED, final_payload={**payload, "error": type(exc).__name__})
        _clear_decision_alert(instance.id, "dismissed")
        _audit_department_action(log_event, "department_action_failed", instance, payload, resolver)
        raise HTTPException(
            status_code=502,
            detail="Something went wrong, so it may not have gone through. Check before trying again.",
        ) from None
    failed = take_the_lead.result_failed(result)
    finish_execution(
        instance.id,
        STATUS_FAILED if failed else STATUS_APPROVED_WITH_EDIT if edited else STATUS_APPROVED_UNCHANGED,
        final_payload={**payload, "result": result[:2000]},
    )
    _clear_decision_alert(instance.id, "dismissed" if failed else "ack")
    _audit_department_action(
        log_event, "department_action_failed" if failed else "department_action_done",
        instance, payload, resolver,
    )
    if failed:
        logger.warning("decisions/approve: step %d failed: %s", instance.id, result[:300])
        raise HTTPException(
            status_code=502,
            detail=f"It couldn't do that. {take_the_lead.failure_words(str(payload.get('tool') or ''), result)}",
        )
    return _refresh(instance.id)


def _audit_department_action(
    log_event: Any, event_type: str, instance: DecisionInstance, payload: dict[str, Any],
    resolver: int | None, *, actor: str | None = None,
) -> None:
    verb = {
        "department_action_done": "Approved and done",
        "department_action_failed": "Approved, but it failed",
        "department_action_declined": "Declined",
        "department_action_lapsed": "Left to lapse",
        "department_action_waiting": "Waiting again",
    }[event_type]
    try:
        log_event(
            event_type,
            f"{verb}: {payload.get('summary') or payload.get('tool')}"[:240],
            actor=actor or (f"person:{resolver}" if resolver is not None else "user"),
            details={
                "decision_id": instance.id,
                "tool": payload.get("tool"),
                "requested_by_person_id": payload.get("requested_by_person_id"),
                "department": payload.get("department"),
                "resolved_by_person_id": resolver,
            },
            department=str(payload.get("department") or "") or None,
            private=True,
        )
    except Exception:
        logger.warning("decisions: couldn't audit department action %d", instance.id, exc_info=True)


async def _department_action_declined(instance: DecisionInstance) -> None:
    from openexecutive.audit import log_event

    _audit_department_action(
        log_event, "department_action_declined", instance, _parse_payload(instance),
        instance.resolver_person_id,
    )


def _department_action_lapsed(instance: DecisionInstance, lapsed: bool) -> None:
    from openexecutive.audit import log_event

    _audit_department_action(
        log_event, "department_action_lapsed" if lapsed else "department_action_waiting", instance,
        _parse_payload(instance), None, actor="executive" if lapsed else "user",
    )


# A send or edit a team member asked for in chat that their department's
# authority held (orchestrator.department_gate), for the approver it went to
# or the principal. What it says quotes the conversation, so nobody else on
# the team sees it.
_DEPARTMENT_ACTION = DecisionClassSpec(
    name="department_action",
    principal_only=False,
    principal_and_approver=True,
    alert_source=DECISION_ALERT_SOURCE,
    approve=_carry_out_department_action,
    after_reject=_department_action_declined,
    after_lapse=_department_action_lapsed,
)

async def _carry_out_app_action(
    instance: DecisionInstance, body: ApproveBody, request: Request, resolver: int | None
) -> DecisionInstance:
    """Something the person asked for in chat that waits for their tap
    (``orchestrator.app_control``: a delete, a share, turning on something
    that acts as them): do it with this request, the tapper's own, through
    the route its button calls, which checks them again. Claimed first, so it
    runs once; a card older than a day is turned down instead."""
    from openexecutive.audit import log_event
    from openexecutive.memory.decision_ledger import claim_for_execution, finish_execution
    from openexecutive.orchestrator import app_control

    payload = _parse_payload(instance)
    if app_control.card_stale(instance.created_at):
        mark_resolved(instance.id, STATUS_REJECTED, resolver_person_id=resolver)
        _clear_decision_alert(instance.id, "dismissed")
        raise HTTPException(
            status_code=409, detail="This was more than a day ago, so it wasn't done. Ask again in chat.",
        )
    if not claim_for_execution(instance.id, resolver_person_id=resolver):
        raise HTTPException(status_code=409, detail="Someone else resolved this decision first.")
    details = {"decision_id": instance.id, "action": payload.get("action"), "id": payload.get("id")}
    try:
        did = await app_control.carry_out(payload, request)
    except (app_control.Refused, HTTPException) as exc:
        reason = app_control.refusal_text(exc) if isinstance(exc, HTTPException) else str(exc)
        finish_execution(instance.id, STATUS_FAILED, final_payload={**payload, "error": reason})
        _clear_decision_alert(instance.id, "dismissed")
        log_event("app_action_refused", f"Card {instance.id} refused", actor="user", details=details)
        status_code = exc.status_code if isinstance(exc, HTTPException) and exc.status_code < 500 else 409
        raise HTTPException(status_code=status_code, detail=reason) from None
    except Exception:
        logger.exception("decisions/approve: app action %d failed", instance.id)
        finish_execution(instance.id, STATUS_FAILED, final_payload={**payload, "error": "failed"})
        _clear_decision_alert(instance.id, "dismissed")
        raise HTTPException(
            status_code=502, detail="That didn't go through. Check before trying again.",
        ) from None
    finish_execution(instance.id, STATUS_APPROVED_UNCHANGED, final_payload={**payload, "did": did})
    _clear_decision_alert(instance.id, "ack")
    log_event("app_action_done", did[:200], actor="user", details=details)
    return _refresh(instance.id)


# Something asked for in chat that waits for its person's tap
# (orchestrator.app_control): theirs alone. Its Needs you card, left when the
# chat couldn't show it, is cleared when it is answered.
_APP_ACTION = DecisionClassSpec(
    name="app_action",
    principal_only=False,
    approver_only=True,
    alert_source=DECISION_ALERT_SOURCE,
    approve=_carry_out_app_action,
)

DECISION_CLASSES: dict[str, DecisionClassSpec] = {
    spec.name: spec
    for spec in (
        _MEETING_BOOKING, _DELEGATED_REPLY, _TAKE_THE_LEAD, _DELEGATED_ACTIONS, _DEPARTMENT_ACTION,
        _APP_ACTION, _DELEGATED_INVITE,
    )
}


@router.post("/decisions/{instance_id}/approve", response_model=DecisionInstance)
async def approve_decision(
    instance_id: int, body: ApproveBody, request: Request
) -> DecisionInstance:
    instance = _visible_instance(instance_id, request)
    spec = _spec_for(instance)
    resolver = _resolver(instance, request)
    if instance.status != STATUS_PROPOSED:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot approve a decision with status={instance.status!r}",
        )
    return _served(await spec.approve(instance, body, request, resolver))


# What an approved step gave back that the chat's next turn reads.
_CHAT_RESULT_CHARS = 1500
# What the turn reads when Approve came from a page in another app, which
# never approves a card: the card still waits where it was.
HOST_APPROVE_NOTE = (
    "Approve came from a page in another app, which can't approve a card, so nothing was done. "
    "The card still waits where it was. Say so in a line."
)


async def approve_in_chat(instance_id: int, request: Request, input_edits: dict[str, Any] | None = None) -> str:
    """Approve tapped on a card shown under a reply in the person's own chat
    (web or phone), then what that chat's next turn reads, so the Executive
    carries on with what they asked instead of stopping at the card.

    The tap is the card's own Approve (``approve_decision``, with every check
    it makes on Today); ``input_edits`` are the card's changed fields, as its
    Edit sends them. Never raises: a refusal or a failure is said in the
    note, and the turn tells them."""
    from openexecutive.orchestrator.content_trust import wrap_untrusted

    try:
        body = ApproveBody(edits={"input": input_edits} if input_edits else None)
        instance = await approve_decision(instance_id, body, request)
    except HTTPException as exc:
        detail = exc.detail
        if isinstance(detail, dict):
            # A refusal says why in its `message` (department_gate).
            detail = detail.get("message")
        if not isinstance(detail, str) or not detail.strip():
            detail = "it couldn't be done"
        # Labelled as data: a failed step's words can carry the outside
        # service's text, as what a step gave back can.
        return (
            "They tapped Approve on a card in this chat, but it didn't go through. "
            "Say so in a line, then carry on with what you can.\nWhy:\n"
            + wrap_untrusted(detail[:300], source="tool_result")
        )
    except Exception:
        logger.exception("decisions: approving card %d from the chat failed", instance_id)
        return (
            "They tapped Approve on a card in this chat, but something went wrong, so it may not "
            "have gone through. Say so in a line; don't say it was done."
        )
    payload = _parse_payload(instance)
    try:
        final = json.loads(instance.final_payload_json or "{}")
    except ValueError:
        final = {}
    # Quoted: a step's summary can quote what it sends, words from outside.
    summary = json.dumps(" ".join(str(payload.get("summary") or "the step").split())[:200])
    note = (
        f"They tapped Approve on the card {summary} in this chat, and it was done. "
        "Carry on with what they asked in this chat, using what it gave back; "
        "don't ask them to approve it again."
    )
    # A held step says what it gave back in `result`; an app action in `did`.
    result = (final.get("result") or final.get("did")) if isinstance(final, dict) else None
    if isinstance(result, str) and result.strip():
        note += "\nWhat it gave back:\n" + wrap_untrusted(result[:_CHAT_RESULT_CHARS], source="tool_result")
    return note


class StepSayBody(BaseModel):
    """Words said on a held step's card (Take the lead, a teammate's held
    send): ``text``, and ``seen`` the fields as the card showed them, so a
    go is judged against what they read."""

    text: str = ""
    seen: dict[str, str] = {}


class StepFieldView(BaseModel):
    field: str
    label: str
    value: str
    long: bool = False


class StepSayResponse(BaseModel):
    # What the Executive said back.
    reply: str
    # The step's fields as they read now, with what was said put in.
    fields: list[StepFieldView]
    # The card once a go carried it out; None while it still waits.
    approved: DecisionInstance | None = None


SAID_INPUT = "said_input"


def _edits_beyond_said(payload: dict[str, Any], input_edits: object) -> bool:
    """True when ``input_edits`` changes the step past what was said on its card."""
    if not input_edits:
        return False
    said = payload.get(SAID_INPUT)
    said = said if isinstance(said, dict) else {}
    return not isinstance(input_edits, dict) or any(said.get(k) != v for k, v in input_edits.items())
SAY_MAX_CHARS = 500
_COULD_NOT_DO = "I couldn't do that. Tap to try again."
_PICK_TO_GO = "Here it is now. Tap Do it to go ahead."


def said_fields(payload: dict[str, Any]) -> list[StepFieldView]:
    """A held step's editable fields as they read now: as filed, with what
    was said on the card put in (``said_input``)."""
    from openexecutive.orchestrator import card_details, take_the_lead

    if len(card_details.held_steps(payload)) > 1:
        return []
    source: dict[str, Any] = payload["input"] if isinstance(payload.get("input"), dict) else {}
    said_raw = payload.get(SAID_INPUT)
    said: dict[str, Any] = said_raw if isinstance(said_raw, dict) else {}
    return [
        StepFieldView(
            field=str(f["field"]), label=str(f["label"]), long=bool(f.get("long")),
            value=str(said.get(f["field"], f["value"])),
        )
        for f in take_the_lead.editable_fields(str(payload.get("tool") or ""), source, mcp=bool(payload.get("mcp")))
    ]


def _said_line(text: str) -> str:
    line = " ".join((text or "").split())
    if not line:
        raise HTTPException(status_code=422, detail="Say something first.")
    if len(line) > SAY_MAX_CHARS:
        raise HTTPException(status_code=422, detail=f"Keep it under {SAY_MAX_CHARS} characters.")
    return line


@router.post("/decisions/{instance_id}/say", response_model=StepSayResponse)
async def say_on_step(instance_id: int, body: StepSayBody, request: Request) -> StepSayResponse:
    """Words said on a held step's card ("make it shorter", "yes, send it"):
    ``say_on_held``, for the signed-in caller, with Approve as a tap on
    the page."""
    _said_line(body.text)
    instance = _visible_instance(instance_id, request)
    if instance.decision_class not in _HELD_CLASSES:
        raise HTTPException(status_code=409, detail="Use this card's own buttons.")
    resolver = _resolver(instance, request)

    async def approve(fresh: DecisionInstance, edits: dict[str, str]) -> DecisionInstance:
        return await _spec_for(fresh).approve(fresh, ApproveBody(edits={"input": edits}), request, resolver)

    return await say_on_held(
        instance, body.text, resolver=resolver, may_change=_may_change(instance, request, resolver),
        caller_key=api_caller.caller(request).email, seen=body.seen, approve=approve,
    )


async def say_on_held(
    instance: DecisionInstance,
    text: str,
    *,
    resolver: int | None,
    may_change: bool,
    caller_key: str,
    seen: dict[str, str],
    approve: Callable[[DecisionInstance, dict[str, str]], Awaitable[DecisionInstance]],
) -> StepSayResponse:
    """Words said on a held step's card, by ``resolver``, who may answer it
    (checked by the caller): on the page, or in reply to its card in a chat
    app (``say_in_chat``). One fast call rewrites the step's fields from
    them; the card keeps the new fields, and Approve sends them as the edit,
    so the step's own rules for a change apply there as for a typed one.
    When they said to go ahead (judged from their words and the fields
    alone, ``replan.said_go``), ``approve`` carries it out at once with
    those edits, the same as a tap. ``may_change``: whether they may change
    its fields with Approve's own edits. ``seen``: the fields as the card
    showed them, when it says. What was said stays on the card, for them
    only."""
    from openexecutive.alerts import card_talk, replan
    from openexecutive.alerts.store import get_alert_by_external
    from openexecutive.memory.decision_ledger import decision_alert_external_id, update_open_payload

    line = _said_line(text)
    instance_id = instance.id
    if instance.decision_class not in _HELD_CLASSES:
        raise HTTPException(status_code=409, detail="Use this card's own buttons.")
    if instance.status != STATUS_PROPOSED:
        raise HTTPException(status_code=409, detail="This card was already answered.")
    payload = _parse_payload(instance)
    # Its wording is changed only by whoever may change it with Approve's
    # own edits, so what was said can go on any way the card is approved.
    fields = said_fields(payload) if may_change else []
    if not fields:
        raise HTTPException(status_code=409, detail="This one can't be changed from its card.")
    # What the card showed before these words, for a go when the page sent none.
    shown = {f.field: f.value for f in fields}
    try:
        new = await replan.rewrite_fields(
            str(payload.get("summary") or ""),
            [replan.StepField(field=f.field, label=f.label, value=f.value) for f in fields],
            line,
            caller_key=caller_key,
        )
    except replan.ReplanError as exc:
        code = {"empty": 422, "too_long": 422, "too_many": 429}.get(exc.code, 503)
        raise HTTPException(status_code=code, detail=exc.message) from exc

    # The fields as filed: what was said is kept as the change from these.
    original = {f.field: f.value for f in said_fields({**payload, SAID_INPUT: {}})}
    now = {**{f.field: f.value for f in fields}, **new.values}
    refused = ""
    if new.values:
        said = {k: v for k, v in now.items() if v != original.get(k)}
        # Kept only when the step's own rules for a change take it, so any
        # Approve of the card can carry it out.
        try:
            _with_edits(payload, said)
        except HTTPException as exc:
            refused = f"I couldn't change it that way: {str(exc.detail).rstrip('.')}. Nothing changed."
        if not refused:
            changed = {**payload, SAID_INPUT: said}
            if not update_open_payload(instance_id, changed, expected_json=instance.proposed_payload_json):
                raise HTTPException(status_code=409, detail="This card changed meanwhile. Look again.")
            payload = changed
            fields = said_fields(payload)

    alert = get_alert_by_external(DECISION_ALERT_SOURCE, decision_alert_external_id(instance_id))
    talk_id = alert.id if alert is not None else None
    if talk_id is not None:
        card_talk.add_turn(talk_id, "you", line, person_id=resolver)

    reply, approved = refused or new.reply, None
    if new.go and not refused:
        step = [replan.StepField(field=f.field, label=f.label, value=f.value) for f in fields]
        saw = seen or shown
        before = [replan.StepField(field=f.field, label=f.label, value=saw.get(f.field, "")) for f in fields]
        limit = ""
        try:
            go = await replan.said_go(
                replan.step_text(step), line, before=replan.step_text(before),
                before_max=replan.STEP_TEXT_MAX, caller_key=caller_key,
            )
        except replan.ReplanError as exc:
            go = False
            limit = exc.message if exc.code == "too_many" else ""
        if not go:
            # It said it was on it, but the go wasn't sure (or the hour's
            # limit was reached, which it says): the step waits.
            reply = limit or _PICK_TO_GO
        if go:
            edits = {f.field: f.value for f in fields if f.value != original.get(f.field)}
            fresh = get_decision_instance(instance_id)
            try:
                if fresh is None:
                    raise HTTPException(status_code=409, detail=_COULD_NOT_DO)
                approved = _served(await approve(fresh, edits))
            except HTTPException as exc:
                if exc.status_code >= 500:
                    # It was tried: say what happened, and show it as it is now.
                    reply = str(exc.detail)
                    approved = _served(_refresh(instance_id))
                else:
                    # Not tried: why, in its own words when it has them.
                    reply = exc.detail if isinstance(exc.detail, str) else _COULD_NOT_DO
    if talk_id is not None:
        card_talk.add_turn(talk_id, "me", reply, person_id=resolver)
    return StepSayResponse(reply=reply, fields=fields, approved=approved)


@router.post("/decisions/{instance_id}/reject", response_model=DecisionInstance)
async def reject_decision(instance_id: int, body: RejectBody, request: Request) -> DecisionInstance:
    instance = _visible_instance(instance_id, request)
    spec = _spec_for(instance)
    resolver = _resolver(instance, request)
    return _served(await _reject(instance, spec, resolver))


async def _reject(
    instance: DecisionInstance, spec: DecisionClassSpec, resolver: int | None
) -> DecisionInstance:
    instance_id = instance.id
    if instance.status != STATUS_PROPOSED:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot reject a decision with status={instance.status!r}",
        )
    if not mark_resolved(instance_id, STATUS_REJECTED, resolver_person_id=resolver):
        raise HTTPException(status_code=409, detail="Someone else resolved this decision first.")
    _clear_decision_alert(instance_id, "dismissed", spec.alert_source)
    updated = _refresh(instance_id)
    if spec.after_reject is not None:
        try:
            await spec.after_reject(updated)
        except Exception:
            logger.exception("decisions/reject: after-reject step failed for instance %d", instance_id)
    return updated


# The cards a chat app shows with buttons (integrations.channel_cards): the
# ones a chat reply leaves for the person chatting, as the web chat shows them
# under its reply, and the ones that wait for someone outside any chat (a
# reply to send, an invite, an action Take the lead or a department held),
# sent to their chat app.
CHAT_CARD_CLASSES = frozenset({
    _CALENDAR_CLASS, "delegation_actions", "app_action", "delegation_reply", "delegation_invite",
    "take_the_lead_action", "department_action",
})
# Of the app_action cards (orchestrator.app_control), the ones a chat app's
# tap may carry out: those whose action checks the tapper itself, not a
# signed request (a workflow change, workflow_change.run).
CHAT_APP_ACTIONS = frozenset({"change_workflow"})


def shown_in_chat(instance: DecisionInstance) -> bool:
    """Whether a chat app shows this card with Approve and Skip buttons."""
    if instance.decision_class not in CHAT_CARD_CLASSES:
        return False
    if instance.decision_class != _APP_ACTION.name:
        return True
    return _parse_payload(instance).get("action") in CHAT_APP_ACTIONS


class _ChatTapRequest:
    """What an app_action card's action reads on a chat app's tap: the
    proven tap, and no signed caller (``workflow_change._tapper``)."""

    def __init__(self, tap: Any) -> None:
        self.state = SimpleNamespace(chat_tap=tap)
        self.headers: dict[str, str] = {}
        self.query_params: dict[str, str] = {}
        self.client = None


def chat_card_is_theirs(instance: DecisionInstance, person: Any) -> bool:
    """Whether ``person``, proven in a chat app, may see and answer card
    ``instance`` there: the web chat's rule (``_resolver``). A card of
    actions is its person's alone; a meeting is the principal's, or its
    approver's unless it is the principal's private booking."""
    from openexecutive.people.store import find_principal_person

    if not shown_in_chat(instance) or person.id is None:
        return False
    principal = find_principal_person()
    is_principal = principal is not None and principal.id == person.id
    if _approver_only(instance):
        allowed = person.id == instance.approver_person_id
    else:
        allowed = is_principal or (person.id == instance.approver_person_id and not _is_private(instance))
    if allowed:
        # A step held from someone else's own chat isn't theirs either.
        from openexecutive.alerts import own_chat

        allowed = own_chat.decision_shown_to(instance, person.id)
    return allowed


async def resolve_in_chat(
    instance_id: int, person: Any, *, approve: bool, allow: bool = False, shown_at: str | None = None,
    edits: dict[str, Any] | None = None, shown: str | None = None,
) -> DecisionInstance:
    """Approve or skip a card tapped in a chat app, for ``person``: the one the
    adapter proved tapped it, when it is theirs (``chat_card_is_theirs``). ``allow``
    is the "on its own next time" button (In training): the same as the web
    card's tick. ``shown_at``: when the tapped card showed the card as it is
    (a chat app's tap, ``channel_cards.tap``); else when it was posted to
    the chat app. A held step goes as its chat card shows it: with what was
    said on it (``said_input``, ``channel_cards.card_text``), or with
    ``edits`` when given (a go said on it, ``say_in_chat``). ``shown``: the
    version of the card tapped; a held step that reads differently now
    (words said on it meanwhile) isn't done. Raises HTTPException, as the
    routes do."""
    from openexecutive.delegation.action_cards import ApproveRefused
    from openexecutive.delegation.action_cards import approve as approve_actions
    from openexecutive.delegation.gmail import normalize_email
    from openexecutive.delegation.verified import ChatTap
    from openexecutive.integrations.channel_cards import card_text
    from openexecutive.people.store import find_principal_person

    instance = get_decision_instance(instance_id)
    if instance is None or not chat_card_is_theirs(instance, person):
        # As the routes: a card that isn't theirs is not there.
        raise HTTPException(status_code=404, detail="This card is no longer there.")
    if instance.status == STATUS_PROPOSED and approve and card_text(instance.id) is None:
        # Not one a chat can show whole (``channel_cards.CARD_MAX``): what a
        # tap would send or do is more than any chat card showed.
        raise HTTPException(status_code=409, detail="This one is too long to approve here. Open it in the app.")
    spec = _spec_for(instance)
    principal = find_principal_person()
    is_principal = principal is not None and principal.id == person.id
    if not approve:
        return await _reject(instance, spec, person.id)
    if instance.status != STATUS_PROPOSED:
        raise HTTPException(status_code=409, detail="This card was already handled.")
    if instance.decision_class == _CALENDAR_CLASS:
        return await _book(instance, ApproveBody(), by_principal=is_principal, resolver=person.id)
    tap = ChatTap(normalize_email(person.email or ""))
    if instance.decision_class == _APP_ACTION.name:
        return await _carry_out_app_action(instance, ApproveBody(), _ChatTapRequest(tap), person.id)  # type: ignore[arg-type]
    if instance.decision_class == _DELEGATED_REPLY.name:
        return await _send_reply_from_chat(instance, tap, person.id, allow=allow, shown_at=shown_at)
    if instance.decision_class == _DELEGATED_INVITE.name:
        from openexecutive.delegation.invites import AnswerRefused, answer

        try:
            await answer(instance, caller=tap, resolver=person.id, answer="accept", allow=allow)
        except AnswerRefused as refused:
            raise HTTPException(status_code=refused.status, detail=refused.message) from None
        return _refresh(instance.id)
    if instance.decision_class in _HELD_CLASSES and shown is not None:
        from openexecutive.integrations.channel_cards import shows

        if not shows(instance, shown):
            raise HTTPException(status_code=409, detail=_CHANGED_MEANWHILE)
    if instance.decision_class == _TAKE_THE_LEAD.name:
        # As it stands (changing it is the web card's); "next time" allows
        # it from now on, the owner's alone, as the web card's tick does.
        # The web card's checks, in its order: a training card with a grant,
        # the owner's own tap (a ChatTap naming their address is as proven as
        # a signed sign-in, ``verified.caller_refusal``), and room on the
        # allowed list before anything is done.
        from openexecutive.delegation.verified import caller_refusal
        from openexecutive.orchestrator import take_the_lead

        payload = _parse_payload(instance)
        grant = payload.get("allow") if isinstance(payload.get("allow"), dict) else None
        if allow:
            if grant is None or payload.get("kind") != take_the_lead.TRAINING:
                raise HTTPException(status_code=422, detail="This one can't be allowed from its card.")
            if principal is None or not is_principal or caller_refusal(tap, normalize_email(principal.email or "")) is not None:
                raise HTTPException(status_code=403, detail="Only the account owner can allow this.")
            if not take_the_lead.room_for(str(grant.get("key"))):
                raise HTTPException(
                    status_code=409,
                    detail=f"It can learn at most {take_the_lead.ALLOWED_MAX} things. Remove one in Settings first.",
                )
        # As the chat card showed it: the step with what was said on it.
        payload, changed = _with_edits(payload, _said_input(payload) if edits is None else edits or None)
        return await _run_lead(
            instance, payload, person.id, by_principal=is_principal, changed=changed, grant=grant if allow else None,
        )
    if instance.decision_class == _DEPARTMENT_ACTION.name:
        payload = _parse_payload(instance)
        payload, changed = _with_edits(payload, _said_input(payload) if edits is None else edits or None)
        return await _run_department_action(
            instance, person.id, by_principal=is_principal, payload=payload, edited=bool(changed),
        )
    try:
        await approve_actions(instance, caller=tap, resolver=person.id, allow=allow)
    except ApproveRefused as refused:
        raise HTTPException(status_code=refused.status, detail=refused.message) from None
    return _refresh(instance.id)


def _said_input(payload: dict[str, Any]) -> dict[str, Any] | None:
    """What was said on a held step's card (``said_input``), as Approve's
    edits, or None when nothing was."""
    said = payload.get(SAID_INPUT)
    return dict(said) if isinstance(said, dict) and said else None


def as_said(payload: dict[str, Any]) -> dict[str, Any]:
    """A held step's payload with what was said on its card put in, and its
    line restated: what its chat card shows (``channel_cards.card_text``)
    and its tap there carries out (``resolve_in_chat``)."""
    said = _said_input(payload)
    if said is None or not isinstance(payload.get("input"), dict):
        return payload
    return _edited(payload, {**payload["input"], **said}, said)


_CHANGED_MEANWHILE = "This card changed meanwhile. Look again."


async def say_in_chat(instance_id: int, person: Any, text: str, *, shown: str) -> StepSayResponse:
    """Words said on a held step's card in a chat app (a reply to the card's
    message), by ``person``, the one the adapter proved said them, when the
    card is theirs (``chat_card_is_theirs``) and still says what the card
    of version ``shown`` said: ``say_on_held``, as on the page, with a go
    carried out as a tap on the chat card (``resolve_in_chat``). Raises
    HTTPException, as the routes do."""
    from openexecutive.delegation.gmail import normalize_email
    from openexecutive.integrations.channel_cards import shows
    from openexecutive.people.store import find_principal_person

    instance = get_decision_instance(instance_id)
    if instance is None or not chat_card_is_theirs(instance, person):
        raise HTTPException(status_code=404, detail="This card is no longer there.")
    if not shows(instance, shown):
        # The step their words would change is the one they read.
        raise HTTPException(status_code=409, detail=_CHANGED_MEANWHILE)
    principal = find_principal_person()
    may_change = may_change_as(
        instance, principal=principal is not None and principal.id == person.id,
        approver=person.id == instance.approver_person_id,
    )

    async def approve(fresh: DecisionInstance, edits: dict[str, str]) -> DecisionInstance:
        # The fields the go was judged on, not a newer read of the card.
        return await resolve_in_chat(fresh.id, person, approve=True, edits=edits)

    return await say_on_held(
        instance, text, resolver=person.id, may_change=may_change,
        caller_key=normalize_email(person.email or ""), seen={}, approve=approve,
    )


async def _send_reply_from_chat(
    instance: DecisionInstance, tap: Any, resolver: int, *, allow: bool = False, shown_at: str | None = None,
) -> DecisionInstance:
    """Send a reply card's draft on a tap in a chat app, as written. What
    would ask a second yes on the web card (recipients changed, newer mail,
    a change the card, posted at ``shown_at``, didn't show), or that card's
    questions still waiting for an answer, is turned down here with what to
    do instead. ``allow``: the "next time" button."""
    from openexecutive.delegation.answers import questions_left
    from openexecutive.delegation.inbox import card_payload
    from openexecutive.delegation.reply_send import SendRefused, send_approved_reply
    from openexecutive.integrations import channel_cards

    # The chat card doesn't show the questions, and the draft isn't finished.
    if questions_left(card_payload(instance)):
        raise HTTPException(
            status_code=409, detail="This reply still has a question for you. Open it in the app to answer it and send.",
        )
    if shown_at is None:
        shown_at = channel_cards.shown_at(instance.id)
    try:
        await send_approved_reply(
            instance, caller=tap, resolver=resolver, confirm={}, allow=allow, shown_at=shown_at,
        )
    except SendRefused as refused:
        message = refused.message
        if refused.code == "confirm":
            message = f"{message} Open it in the app to check and send it."
        raise HTTPException(status_code=refused.status, detail=message) from None
    return _refresh(instance.id)


@router.post("/decisions/{instance_id}/cancel", response_model=DecisionInstance)
async def cancel_decision(instance_id: int, request: Request) -> DecisionInstance:
    """Cancel an approved/executed event (reverse it)."""
    instance = _visible_instance(instance_id, request)
    spec = _spec_for(instance)
    _resolver(instance, request)
    if spec.cancel is None:
        raise HTTPException(status_code=409, detail="This decision can't be undone.")
    if instance.status not in (
        STATUS_APPROVED_UNCHANGED, STATUS_APPROVED_WITH_EDIT, STATUS_PROPOSED,
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel a decision with status={instance.status!r}",
        )
    await spec.cancel(instance)
    mark_reversed(instance_id, reason="cancelled_via_ui", decision_class=spec.name)
    _clear_decision_alert(instance_id, "dismissed", spec.alert_source)
    return _served(_refresh(instance_id))


def _class_mode_response(decision_class: str) -> DecisionClassMode:
    # An unknown stored value reads as the fail-safe default, like the gate.
    mode = get_class_mode(decision_class)
    return DecisionClassMode(
        decision_class=decision_class,
        mode="auto_execute" if mode == "auto_execute" else "propose",
    )


def _has_active_principal() -> bool:
    """Whether a non-archived principal is on the roster. Fails closed."""
    from openexecutive.people.store import find_principal_person

    try:
        return find_principal_person() is not None
    except Exception:
        logger.exception("decisions: principal lookup failed — treating as none")
        return False


@router.get(
    f"/decisions/classes/{_CALENDAR_CLASS}", response_model=DecisionClassMode
)
def get_meeting_class_mode() -> DecisionClassMode:
    """Whether the Executive proposes meetings for approval or books them."""
    return _class_mode_response(_CALENDAR_CLASS)


@router.put(
    f"/decisions/classes/{_CALENDAR_CLASS}", response_model=DecisionClassMode
)
def set_meeting_class_mode(
    request: Request, body: DecisionClassModeUpdate
) -> DecisionClassMode:
    """Set the meeting class mode. Principal only — the same rule as the
    workspace settings (``chat._caller_is_principal_or_unclaimed``): letting
    the Executive book meetings on its own is the principal's call. While no
    principal exists (when that rule lets anyone in) only ``propose`` can be
    set — ``auto_execute`` is a 409. Every change is audited."""
    from openexecutive.api.routes.chat import _caller_is_principal_or_unclaimed

    if not _caller_is_principal_or_unclaimed(request):
        raise HTTPException(
            status_code=403,
            detail="Only the principal can change how meetings are scheduled",
        )
    if body.mode == "auto_execute" and not _has_active_principal():
        # Before anyone is the principal the PUT is open to every caller (so
        # first-run setup is never locked out), and auto-booking would then
        # be switched on by nobody in particular. Propose stays allowed.
        raise HTTPException(
            status_code=409,
            detail=(
                "Meetings can only be booked automatically once someone is set "
                "as the principal (the owner) on the People page. Until then "
                "they are proposed for approval."
            ),
        )
    before = _class_mode_response(_CALENDAR_CLASS).mode
    if body.mode != before:
        set_class_mode(_CALENDAR_CLASS, body.mode)
        from openexecutive.audit import log_event as audit_log

        audit_log(
            "decision_class_mode_changed",
            f"Meeting scheduling mode changed: {before} → {body.mode}",
            actor=api_caller.actor(request),
            details={
                "decision_class": _CALENDAR_CLASS,
                "mode": {"from": before, "to": body.mode},
            },
        )
    return _class_mode_response(_CALENDAR_CLASS)


@router.get("/audit/reliability", response_model=ReliabilityCard)
def get_reliability(
    request: Request,
    decision_class: str = _CALENDAR_CLASS,
    days: int = 30,
) -> ReliabilityCard:
    """How a class's decisions went. The principal's alone: it counts every
    instance, private ones included."""
    if not _approver_is_principal(request):
        raise HTTPException(status_code=403, detail="Only the owner can see this.")
    if days < 1 or days > 365:
        raise HTTPException(status_code=400, detail="days must be 1–365")
    return aggregate_reliability(decision_class, window_days=days)
