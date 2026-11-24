"""What a Needs you card shows: its words and its buttons.

The one place that decides it. Every screen that shows a card draws it from
:func:`card_view`, so they all show the same card. Each keeps its own
layout; none works out a card's words or buttons from its raw fields.

Approve is never a button: it is the card's first option
(:class:`CardOption`, ``POST /alerts/{id}/approve`` with that option's plan,
or ``/decisions/{id}/approve`` for a booking). A button is one of a fixed
list of actions (:data:`Action`), each done the same way everywhere:

- ``edit_approve``: edit the text first, then approve exactly that.
- ``dismiss``: Dismiss, or Got it on a card that's only to know.
- ``snooze``: hide it until a time picked from the app's snooze choices.
- ``rerun``: queue a failed run again (``run_id``).
- ``open``: an app page (``href``, a web path).
- ``open_url``: an outside https address (``href``).
- ``chat``: start a chat with ``prompt`` (``memory`` is the line it keeps).
- ``call``: ``POST href`` with no body, then the card says ``done``.

``place`` says where it sits: ``primary`` (the one main button), ``menu``
(the ⋯ menu), ``side`` (apart, on the right: Dismiss) or ``answer`` (a row of
answers in the card's body).

``panel`` names a card answered on its own panel, which the app draws whole:
``new_sender`` (who is this?), ``lead`` (Take the lead: Approve, Edit,
Decline), ``held`` (a teammate's send a department held).

Every other card reads the same way (``alerts.card_talk``): ``happened``
(what happened), ``plan`` under "What I'll do", the ``options`` to pick from
before anything happens (the first is the Executive's own pick: an Approve
of that option's plan, or on a card that asks something, that answer), and
``tell_me``: a box where the person says anything ("ask Dan instead", "yes,
send it from you"), sent to ``POST /alerts/{id}/say``. What they said and
what the Executive said back show on the card (``turns``).
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import quote, urlsplit

from pydantic import BaseModel, Field

Action = Literal[
    "edit_approve", "dismiss", "snooze", "rerun", "open", "open_url", "chat", "call",
]
Place = Literal["primary", "menu", "side", "answer"]
Tone = Literal["neutral", "amber", "sky", "rose", "emerald"]
Panel = Literal["", "new_sender", "lead", "held"]
Kind = Literal[
    "new_sender", "lead", "held", "ask", "rerun", "document", "signal", "decision", "proposal",
]

# Tags the card logic reads. Each names where it's set.
ARTIFACT_TAG = "artifact"  # orchestrator draft_artifact
# browser/handover.py: a "Waiting on you" card from the browser.
BROWSER_HANDOVER_TAG = "browser-handover"
# orchestrator.take_the_lead: a training card, and a held Take the lead action.
LEAD_TRAINING_TAG = "take_the_lead:training"
LEAD_ACTION_TAG = "decision_class:take_the_lead_action"
# orchestrator.department_gate: a teammate's send their department held.
HELD_ACTION_TAG = "decision_class:department_action"

# A card body longer than this is folded to two lines while the card is closed.
LONG_BODY_CHARS = 90
MEMORY_LINE_MAX_CHARS = 300

# What the card needs from its reader (Alert.needs), as its top chip says it.
NEEDS_CHIP = {
    "answer": "Needs your answer",
    "choice": "Pick one",
    "ok": "Needs your OK",
    "fyi": "For you to know",
}

ARTIFACT_BADGE = {"html": "Web page", "docx": "Word doc", "xlsx": "Spreadsheet", "link": "Link"}


class CardChip(BaseModel):
    label: str
    tone: Tone = "neutral"
    # Shown on hover, where there is one.
    title: str = ""


class CardButton(BaseModel):
    action: Action
    label: str
    place: Place
    href: str = ""
    run_id: str = ""
    prompt: str = ""
    memory: str = ""
    # ``call``: what the card says once it worked, and while it runs.
    done: str = ""
    busy_label: str = ""


class CardOption(BaseModel):
    """One way to go, picked before anything happens. ``approve``: Approve
    with ``plan`` (the server checks it is one the card offers). ``answer``:
    send ``label`` as the answer (``POST /alerts/{id}/answer``)."""

    action: Literal["approve", "answer"]
    label: str
    plan: str = ""


class CardLine(BaseModel):
    """One line of the reader's conversation on the card: theirs (``you``)
    or the Executive's (``me``)."""

    who: Literal["you", "me"]
    text: str


class CardReview(BaseModel):
    """The review's one line: "Since you last looked", "Updated", "Likely stale"."""

    title: str
    text: str
    tone: Tone = "neutral"


class CardView(BaseModel):
    kind: Kind
    panel: Panel = ""
    # Who it's routed to, by full name, when it is.
    routed_to: str = ""
    # The top row: what it needs from you, then As me or where it came from.
    need: str = ""
    as_me: bool = False
    source: str = ""
    # Where a watch's signal came from, in the Executive's words: "I noticed
    # this on the Acme pricing page I'm watching." Home's row and the open
    # card say it under the change. "" on every other card.
    noticed: str = ""
    # The chip under the title: a deadline or "Likely stale".
    status: CardChip | None = None
    # The text under the title, and whether it folds while the card is closed.
    description: str = ""
    long: bool = False
    # What happened, in one line (alerts.card_talk); "" until it is written,
    # and the card shows ``description`` meanwhile.
    happened: str = ""
    # What the Executive will do (the plan its first option carries out),
    # under ``plan_title`` ("What I'll do", or "What I'll do · Changed by
    # Dana"); shown when ``plan_title`` is set.
    plan_title: str = ""
    plan: str = ""
    # The ways to go, the Executive's own pick first.
    options: list[CardOption] = Field(default_factory=list)
    # Whether "Or tell me…" shows: the person can say anything on the card.
    tell_me: bool = False
    # What was said on the card, in order.
    turns: list[CardLine] = Field(default_factory=list)
    waiting_because: str = ""
    # Opened card.
    radar: str = ""
    review: CardReview | None = None
    chips: list[CardChip] = Field(default_factory=list)
    surfaced: str = ""
    tags: list[CardChip] = Field(default_factory=list)
    document: bool = False
    reviewed: str = ""
    buttons: list[CardButton] = Field(default_factory=list)
    # A held step's card opens with these, on every app
    # (``orchestrator.card_details.top``): waiting_on, started, why, waits,
    # title. Empty on every other card.
    top: dict[str, str] = Field(default_factory=dict)


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(UTC)


def _parse(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        at = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return None
    return at if at.tzinfo else at.replace(tzinfo=UTC)


def age_label(iso: str | None, now: datetime | None = None) -> str:
    """"5m", "3h", "2d"; "" when unreadable."""
    at = _parse(iso)
    if at is None:
        return ""
    mins = max(0, round((_now(now) - at).total_seconds() / 60))
    if mins < 60:
        return f"{mins}m"
    hours = round(mins / 60)
    if hours < 48:
        return f"{hours}h"
    return f"{round(hours / 24)}d"


def days_until(iso: str | None, now: datetime | None = None) -> int | None:
    """Whole days to a deadline: 0 within the day, negative once past (floor,
    so an 11-hour-old deadline is -1, overdue, never due today)."""
    at = _parse(iso)
    if at is None:
        return None
    return int((at - _now(now)).total_seconds() // 86400)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def memory_line(action: str, subject: str) -> str:
    """The line a chat started from a card keeps of it."""
    s = _clip(" ".join(subject.split()), MEMORY_LINE_MAX_CHARS)
    return f'{action}.\n"{s}"' if s else f"{action}."


def safe_https(url: str | None) -> str:
    """An https address with no name or password in it, else ""."""
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        return ""
    return url


def _host(url: str) -> str:
    try:
        return urlsplit(url).hostname or url
    except ValueError:
        return url


def is_signal(item: Any) -> bool:
    """A signal from outside (the watch list, an app's new entries, a watched
    stock's move): for the reader to know."""
    return (
        item.category == "monitoring"
        or bool(item.surfaced_reason)
        or any(t.startswith("external:") for t in item.topic_tags or [])
    )


def noticed_line(watch_title: str | None) -> str:
    """Where a watch's signal came from, said the same for a watch someone
    asked for and one the Executive started itself."""
    name = " ".join((watch_title or "").split())
    if not name:
        return "I noticed this on something I'm watching."
    return f"I noticed this on the {name} I'm watching."


def status_chip(item: Any, now: datetime | None = None) -> CardChip | None:
    """The chip a closed card shows: its deadline, else "Likely stale"."""
    due = days_until(item.due_at, now)
    if due is not None:
        if due < 0:
            return CardChip(label="Overdue", tone="rose")
        return CardChip(label="Due today" if due == 0 else f"Due in {due}d", tone="amber")
    if item.review_verdict == "likely_stale":
        return CardChip(label="Likely stale", tone="amber")
    return None


PLAN_TITLE = "What I'll do"


# Over what a card that asks something wants to know.
ASKS_TITLE = "What I need to know"


def plan_title(changed_by: str | None = None) -> str:
    """The heading over what the card will do: it names who changed it when
    that wasn't the reader, so nobody approves someone else's plan as the
    Executive's own."""
    return f"{PLAN_TITLE} · Changed by {changed_by}" if changed_by else PLAN_TITLE


def _description(item: Any, artifact: bool, monitoring: bool) -> str:
    if artifact:
        return (item.suggested_action or "").strip()
    if monitoring:
        return ""
    body = (item.body or "").strip()
    headline = (item.headline or "").strip()
    if not body or body == headline:
        return ""
    if headline and body.startswith(headline):
        rest = body[len(headline):].lstrip(" \t\n.:;,—–-")
        return rest[:1].upper() + rest[1:]
    return body


def _lifecycle(item: Any, now: datetime | None) -> tuple[CardReview | None, list[CardChip], str]:
    verdict = item.review_verdict or ""
    stale = verdict == "likely_stale"
    review = None
    # A card the review rewrote already says the new text, so its note would
    # only say it again: the chip below marks it instead.
    if item.review_note and verdict != "changed":
        title = "Likely stale" if stale else "Since you last looked"
        review = CardReview(title=title, text=item.review_note, tone="amber" if stale else "neutral")
    chips: list[CardChip] = []
    due = days_until(item.due_at, now)
    if item.why_now or due is not None:
        when = "" if due is None else "overdue" if due < 0 else "due today" if due == 0 else f"due in {due}d"
        chips.append(CardChip(label=" · ".join(p for p in (item.why_now, when) if p), tone="amber"))
    if stale and not item.review_note:
        chips.append(CardChip(label="Likely stale", tone="amber"))
    if (item.superseded_count or 0) > 0:
        chips.append(CardChip(label=f"{item.superseded_count} folded in"))
    if verdict == "changed":
        chips.append(CardChip(label="Updated by the Executive", tone="sky"))
    if verdict == "drafted":
        chips.append(CardChip(label="Draft ready in your queue", tone="amber"))
    reviewed = ""
    if item.last_reviewed_at:
        reviewed = f"Reviewed {age_label(item.last_reviewed_at, now)} ago"
        if verdict and verdict != "likely_stale":
            reviewed += f" · {verdict}"
    return review, chips, reviewed


def card_view(
    item: Any,
    *,
    assignee_name: str = "",
    talk: Any = None,
    now: datetime | None = None,
    viewer_id: int | None = None,
) -> CardView:
    """What one card (a ``ProposalItem``) shows.

    ``assignee_name`` is the full name of the person it's routed to, when it
    is. ``talk`` is the card's line, options and the reader's
    conversation on it (``alerts.card_talk.CardTalk``), when it has any.
    """
    tags = list(item.topic_tags or [])
    decision = item.decision_instance_id is not None
    if item.roster_request is not None:
        return CardView(kind="new_sender", panel="new_sender", need="Who is this?")
    if getattr(item, "source", "") == "roster_request":
        # A new sender's card whose request was already answered: only
        # Dismiss. What a stranger wrote never seeds a chat through Approve.
        return CardView(kind="new_sender", need="Who is this?", buttons=[
            CardButton(action="dismiss", label="Dismiss", place="side"),
        ])
    # A held step's card (Take the lead, a teammate's held send) carries out
    # one exact step: its panel shows the step, its options and "Or tell
    # me…" (POST /decisions/{id}/say), and what was said on it.
    said = [CardLine(who=t.who, text=t.text) for t in (getattr(talk, "turns", None) or [])]
    lead = decision and (LEAD_TRAINING_TAG in tags or LEAD_ACTION_TAG in tags)
    if lead or (decision and HELD_ACTION_TAG in tags):
        from openexecutive.orchestrator.card_details import held_top

        top = held_top(int(item.decision_instance_id), viewer_id) or {}
        # Whose OK it waits for, and who started it: the same words on
        # every app (``card_details.top``).
        need = f"Waiting on {top['waiting_on']}" if top.get("waiting_on") else "Needs your OK"
        return CardView(
            kind="lead" if lead else "held", panel="lead" if lead else "held", need=need,
            source=top.get("started", ""), tell_me=True, turns=said, top=top,
        )

    artifact = ARTIFACT_TAG in tags
    monitoring = item.category == "monitoring"
    signal = is_signal(item)
    stale = item.review_verdict == "likely_stale"
    failed_run = item.failed_run_id or ""
    needs = item.needs or ""
    asks = needs in ("answer", "choice")
    handed_over = BROWSER_HANDOVER_TAG in tags
    # No step to do is nothing to approve. Approve does the card's step
    # (``suggested_action``); a card without one is for them to know, unless
    # it asks, nudges, suggests a workflow or keeps its own buttons.
    stepless = not (item.suggested_action or "").strip() and not asks and needs != "ok" and not (
        artifact or monitoring or failed_run or handed_over or decision
        or item.recommended_move in ("nudge", "suggest_workflow")
    )
    # A signal from outside is for them to know until it waits for an OK
    # (it was filed that way, or someone asked it for a step).
    no_approve = asks or needs == "fyi" or (signal and needs != "ok") or stepless
    got_it = needs == "fyi" or (signal and not needs) or stepless
    first = assignee_name.split(" ")[0] if assignee_name else ""

    def button(action: Action, label: str, **kw: Any) -> CardButton:
        return CardButton(action=action, label=label, place="menu", **kw)

    nudge = (
        button(
            "chat", f"Nudge {first or 'owner'}",
            prompt=(
                f"Nudge {assignee_name or 'the owner'} about this item — it has gone quiet: "
                f'"{_clip(" ".join(item.headline.split()), MEMORY_LINE_MAX_CHARS)}"'
                "\n\nSend a short, friendly check-in via message_person "
                "and tell me what you sent."
            ),
            memory=memory_line(f"Nudge {assignee_name or 'the owner'} about this item", item.headline),
        )
        if item.recommended_move == "nudge" and not artifact and not monitoring
        else None
    )
    from openexecutive.workflows import workflow_label

    workflow = (
        button("open", f"Run {workflow_label(item.suggested_workflow)}", href=f"/jobs/{quote(item.suggested_workflow, safe='')}")
        if item.recommended_move == "suggest_workflow" and item.suggested_workflow
        and not artifact and not monitoring
        else None
    )
    rerun = button("rerun", "Run again", run_id=failed_run) if failed_run else None
    open_run = (
        button("open", "Open the run", href=f"/jobs/runs/{quote(failed_run, safe='')}") if failed_run else None
    )
    # The card's own option names the step ("Reply to Dana") when the card
    # has its words (``action_label``), else Approve; None when it has no
    # step to approve.
    label = (getattr(item, "action_label", "") or "").strip()
    approve = (
        ("Done" if handed_over else label or "Approve")
        if not no_approve and not artifact and not failed_run
        else None
    )
    # Every card that takes words says what to change in "Or tell me…";
    # one that doesn't keeps Edit & approve.
    tell_me = not decision and not handed_over and (
        asks or bool(item.watch_slug)
        or (bool(item.plan_editable) and not failed_run and not artifact)
    )
    edit_approve = (
        button("edit_approve", "Edit & approve")
        if not handed_over and not no_approve and not monitoring and not artifact
        and not decision and not failed_run and not tell_me
        else None
    )
    dismiss = button("dismiss", "Got it" if got_it else "Skip" if tell_me or approve is not None else "Dismiss")
    open_doc = (
        button("open", "Open document", href="/artifacts/" + quote(f"alert:{item.alert_id}", safe=""))
        if artifact
        else None
    )
    link = safe_https(item.artifact_url) if artifact and item.artifact_format == "link" else ""
    open_in_app = button("open_url", f"Open in app ({_host(link)})", href=link) if link else None
    snooze = None if artifact else button("snooze", "Snooze")

    plan = (item.suggested_action or "").strip()
    # The ways to go: Approve of the card's plan first (or of another plan
    # it offers), or on a card that asks, its answers.
    own_options = list(getattr(talk, "options", None) or [])
    options: list[CardOption] = []
    if asks:
        labels = [o.label for o in own_options] or [c for c in (item.choices or []) if c.strip()]
        options = [CardOption(action="answer", label=x) for x in labels]
    elif approve is not None:
        if own_options and own_options[0].plan == plan:
            options = [CardOption(action="approve", label=o.label, plan=o.plan) for o in own_options]
        else:
            options = [CardOption(action="approve", label=approve, plan=plan)]

    primary: CardButton | None
    if artifact:
        primary = open_doc
    elif asks:
        primary = None
    elif got_it:
        primary = dismiss
    elif monitoring:
        primary = None
    elif stale:
        primary = dismiss
    else:
        primary = rerun or nudge or workflow

    buttons: list[CardButton] = []
    if primary is not None:
        buttons.append(primary.model_copy(update={"place": "primary"}))
    for b in (rerun, open_run, nudge, workflow, edit_approve, snooze, open_in_app):
        if b is not None and b is not primary:
            buttons.append(b)
    if dismiss is not primary:
        buttons.append(dismiss.model_copy(update={"place": "side"}))

    # The top row: what it needs, then where it came from (As me when the
    # Executive is the one who'd do it).
    if artifact:
        need = ARTIFACT_BADGE.get(item.artifact_format or "", "Document")
    elif stepless:
        need = NEEDS_CHIP["fyi"]
    elif needs in NEEDS_CHIP:
        need = NEEDS_CHIP[needs]
    elif failed_run:
        need = "Didn't finish"
    elif got_it:
        need = "For you to know"
    elif approve:
        need = "Needs your OK"
    else:
        need = ""
    source = (
        "Workflow" if failed_run
        else "Watch list" if signal
        else ""
    )

    kind = _kind(item, artifact, failed_run, asks, decision)
    # What the card will do shows on every card with a step to approve, in
    # full: the option's few words ("Reply to Dana") never stand in for the
    # step it carries out (who it copies, what it attaches). A browser
    # hand-over's step is the Executive's own note.
    show_plan = bool(plan) and approve is not None and not handed_over
    question = (getattr(item, "question", "") or "").strip() if asks else ""
    review, chips, reviewed = _lifecycle(item, now)
    description = _description(item, artifact, monitoring)
    turns = [
        CardLine(who=t.who, text=t.text)
        for t in (getattr(talk, "turns", None) or [])
    ]
    return CardView(
        kind=kind,
        panel="",
        routed_to=assignee_name if item.routed_to_person_id is not None else "",
        need=need,
        as_me=not source,
        source=source,
        # Every change a watch raised says it, a big one pulled into the
        # action lane included; nothing else claims a watch.
        noticed=noticed_line(getattr(item, "watch_title", None)) if getattr(item, "watch_slug", None) else "",
        status=None if monitoring else status_chip(item, now),
        description=description,
        long=len(description) > LONG_BODY_CHARS,
        happened=(getattr(talk, "happened", "") or "") if not artifact else "",
        # A card that asks something shows what it asks where a plan goes.
        plan_title=ASKS_TITLE if question else plan_title(item.plan_changed_by) if show_plan else "",
        plan=question or (plan if approve is not None else ""),
        options=options,
        tell_me=tell_me,
        turns=turns,
        waiting_because=item.waiting_because if approve is not None else "",
        radar=item.body if monitoring and item.body and item.body != item.headline else "",
        review=review,
        chips=chips,
        surfaced=item.surfaced_reason or "",
        # Only a person's own words for a topic: "external:query" and the
        # like are how the card was made, and the source line already says it.
        tags=[CardChip(label=t, tone="neutral") for t in tags if t != ARTIFACT_TAG and ":" not in t],
        document=artifact,
        reviewed=reviewed,
        buttons=buttons,
    )


def _kind(
    item: Any, artifact: bool, failed_run: str, asks: bool, decision: bool,
) -> Kind:
    """What sort of card it is, for the lists that group cards by it."""
    if decision:
        return "decision"
    if failed_run:
        return "rerun"
    if artifact:
        return "document"
    if item.category == "monitoring":
        return "signal"
    if asks:
        return "ask"
    return "proposal"


def attach_views(
    items: list[Any],
    names: dict[int, str],
    *,
    talk: dict[int, Any] | None = None,
    now: datetime | None = None,
    viewer_id: int | None = None,
) -> None:
    """Set ``view`` on every card. ``names`` maps person id to full name;
    ``talk`` card id to its ``alerts.card_talk.CardTalk``; ``viewer_id`` is
    who reads them."""
    for item in items:
        routed = item.routed_to_person_id
        item.view = card_view(
            item,
            assignee_name=names.get(routed, "") if routed is not None else "",
            talk=(talk or {}).get(item.alert_id),
            now=now,
            viewer_id=viewer_id,
        )
