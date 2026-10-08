"""Delivered-brief state: what the last morning/EoD brief covered.

The principal briefs used to re-list the whole unread queue every day and
call the "activity" block "since last brief" when it was really just the
latest N rows. This module gives each recurring brief kind a memory of what
was last *delivered* so the workflow can:

- bound "what changed" to the window since the previous delivery (`since`),
- split proposals into NEW (created inside that window) vs CARRIED OVER,
- list what the Executive's alert review handled inside the window, and
- skip the model call entirely when the fingerprint of the inputs is
  unchanged, sending a one-line "nothing new" instead.

Storage reuses the ``briefing_narrative`` table (``briefing/narrative_cache``)
under a ``brief:<kind>`` scope: ``input_hash`` holds the fingerprint,
``narrative_text`` the delivered artifact and ``generated_at`` the delivery
time. Scopes are only ever read by exact key, so the namespace cannot
collide with the per-viewer header cache. Only the scheduler records a
delivery (after a successful send), so manual workflow runs never advance
the window.

Each run's outcome, sent or not, goes under ``brief_delivery:<kind>``
(``record_delivery_outcome``): the run's aggregate reason in ``input_hash``
and, in ``narrative_text``, a JSON object holding the channel that carried it
and ONE ENTRY PER RECIPIENT (``RecipientOutcome``: person id, name, reason,
the channel that reached them and the one their plan tried first — channel
names, never addresses). ``narrative_text`` is a TEXT column
(``briefing/narrative_cache``), so carrying the list there needs no schema
change; a row written before this shape existed holds the bare channel name
and reads back with ``recipients=None``.

The Briefing's "not sent" notice and the Setup status page read it back
through ``outstanding_problems``, which answers with one ``(reason, names)``
per outstanding problem rather than one reason for the run. ``send_failed``
comes from the record per recipient, ``no_channel`` from the roster as it
stands now (``scheduler.runner.unreachable_principals``, so a channel
connected since clears it), and ``not_written`` stays run-level because the
artifact did not exist for anyone. A brief that reached one founder and not
the other therefore names the one it missed, on both surfaces, instead of
telling the founder holding it that it reached nobody.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from openexecutive.alerts.lifecycle import parse_aware
from openexecutive.briefing import narrative_cache

if TYPE_CHECKING:
    from openexecutive.people.models import Person

logger = logging.getLogger(__name__)

SCOPE_PREFIX = "brief:"
DEFAULT_WINDOW = timedelta(hours=24)

SUPPRESSED_TEMPLATE = (
    "Nothing new since yesterday's brief — {n} item{s} still waiting on you."
)

# Audit event types the alert review job emits for autonomous moves. The
# brief's "handled overnight" block is built from these (see handled_since).
REVIEW_EVENT_TYPES: tuple[str, ...] = (
    "alert_review_closed",
    "alert_review_routed",
    "alert_review_nudged",
    "alert_review_escalated",
    "alert_review_drafted",
    "alert_review_merged",
    "alert_review_suggested_workflow",
    "alert_review_changed",
)

# Review events that are bookkeeping on a still-open alert, not a completed
# move. A `changed` verdict rewrites the card's text in place — the alert
# stays in "Needs you" and the card itself shows the note — so listing it
# under "handled" double-reports it and mislabels it as done. It renders in
# the brief's REWRITTEN block instead (see rewritten_since).
_NOT_HANDLED_EVENT_TYPES: frozenset[str] = frozenset({"alert_review_changed"})

# Every audit event type the handled block reads, mapped to the short kind
# the brief and the /today rail render. The research watch policy's
# autonomous moves ride alongside the alert review's.
HANDLED_EVENT_KINDS: dict[str, str] = {
    **{
        t: t.removeprefix("alert_review_")
        for t in REVIEW_EVENT_TYPES
        if t not in _NOT_HANDLED_EVENT_TYPES
    },
    "watchlist_research_added": "watching",
    "watchlist_auto_disabled": "stopped_watching",
}

# The nudge audit summary opens with "[alert N] " so `alerts.review` can count
# delivered nudges per alert with a text query; it is bookkeeping, not prose.
_ALERT_MARKER_RE = re.compile(r"^\[alert \d+\]\s*")


def scope_for(kind: str) -> str:
    return f"{SCOPE_PREFIX}{kind}"


def last_delivered(kind: str) -> narrative_cache.BriefingNarrative | None:
    """The last delivered brief of this kind, or None on a cold store."""
    try:
        return narrative_cache.get(scope_for(kind))
    except Exception:
        logger.exception("brief_state: read failed for %s", kind)
        return None


def since_for(kind: str, now: datetime | None = None) -> datetime:
    """Start of the "what changed" window: the previous delivery, else 24 h ago.

    Bounded to at most 7 days back so a brief that stopped firing for a while
    does not replay a month of history when it resumes.
    """
    now = now or datetime.now(UTC)
    prev = last_delivered(kind)
    delivered_at = parse_aware(prev.generated_at) if prev else None
    if delivered_at is None:
        return now - DEFAULT_WINDOW
    # Never in the future (clock skew / a scheduler `now` earlier than the
    # recorded delivery would otherwise empty the window and suppress).
    return min(now, max(delivered_at, now - timedelta(days=7)))


def record_delivered(kind: str, fingerprint: str, text: str) -> None:
    """Persist the delivered brief so the next run can diff against it."""
    try:
        narrative_cache.put(narrative_cache.BriefingNarrative(
            scope=scope_for(kind),
            input_hash=fingerprint,
            narrative_text=text,
            generated_at=narrative_cache.utc_now_iso(),
        ))
    except Exception:
        logger.exception("brief_state: write failed for %s", kind)


# How a brief's latest run ended: sent, or why not (``not_written`` — the run
# failed or produced nothing; the others are scheduler.runner.
# PrincipalDelivery.reason).
DeliveryReason = Literal["delivered", "no_owner", "no_channel", "send_failed", "not_written"]
# The reasons that are a PROBLEM to report. `delivered` is the one reason with
# no copy in `DELIVERY_PROBLEMS`, so keeping it out of this alias is what makes
# "there is nothing to say about a delivered brief" a fact the type checker
# enforces rather than a `KeyError` waiting in a renderer.
DeliveryProblem = Literal["no_owner", "no_channel", "send_failed", "not_written"]
_DELIVERY_REASONS: frozenset[str] = frozenset(get_args(DeliveryReason))
DELIVERY_SCOPE_PREFIX = "brief_delivery:"
# The daily briefs (their send times show on the Setup status page).
BRIEF_KINDS: tuple[str, ...] = ("principal_brief_morning", "principal_brief_eod")
# Every recurring message to the owner whose runs are recorded: the daily
# briefs, and solo mode's weekly review.
DELIVERY_KINDS: tuple[str, ...] = (*BRIEF_KINDS, "principal_weekly_review")
# Delivery channels (scheduler.runner.delivery_order) as the app names them.
CHANNEL_NAMES: dict[str, str] = {
    "email": "email",
    "slack_dm": "Slack",
    "discord_dm": "Discord",
    "telegram": "Telegram",
}
# Why a brief didn't reach the owner, and what to do about it, in the user's
# words.
DELIVERY_PROBLEMS: dict[DeliveryProblem, tuple[str, str]] = {
    "no_owner": (
        "there's no owner on the People list to send it to",
        "Finish setup so you're on the People list as the owner.",
    ),
    "no_channel": (
        "nothing is set up to send it to you",
        "Connect Gmail, or add your Slack, Telegram or Discord to your People profile.",
    ),
    "send_failed": (
        "every way of sending it failed",
        "The Setup status page shows which connection needs attention.",
    ),
    "not_written": (
        "it couldn't be written",
        "The Setup status page shows which part needs attention — often the AI model.",
    ),
}


_NAMELESS = "someone on the People list"


def _and_list(names: Sequence[str]) -> str:
    """``"Ada"``, ``"Ada and Grace"``, ``"Ada, Grace and Lin"``.

    A blank name is DESCRIBED, never dropped: ``full_name`` has no
    ``min_length``, and filtering the blanks out would quietly shorten the
    list — telling a founder about one unreached co-principal when there are
    two is the same silent omission this whole surface exists to prevent.
    """
    items = [n.strip() or _NAMELESS for n in names] or [_NAMELESS]
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


# Per-person phrasings, ``{who}`` being an `_and_list` of the people the
# problem is about. Every line in `DELIVERY_PROBLEMS` is written for the only
# recipient there is — "nothing is set up to send it to *you*", "every way of
# sending it failed" — so on a co-founded roster it tells whichever founder is
# reading that the brief reaches nobody, about one they receive every day.
_PARTIAL_PROBLEMS: dict[DeliveryProblem, tuple[str, str]] = {
    "no_channel": (
        "nothing is set up to send it to {who}",
        "Connect Gmail, or add their Slack, Telegram or Discord to their People profile.",
    ),
    "send_failed": (
        "every way of sending it to {who} failed",
        "The Setup status page shows which of their connections needs attention.",
    ),
}


def partial_delivery_problem(
    names: Sequence[str], reason: DeliveryProblem = "no_channel"
) -> tuple[str, str]:
    """``(problem, fix)`` for a problem that is about some principals, not all.

    Names the ones the brief isn't reaching rather than telling the reader it
    reaches nobody; a nameless row is described rather than given an id, which
    is not something to show a reader.

    ``not_written`` and ``no_owner`` have no per-person phrasing because they
    are facts about the run or the roster rather than about a person, so they
    fall back to ``DELIVERY_PROBLEMS``.
    """
    template = _PARTIAL_PROBLEMS.get(reason)
    if template is None:
        return DELIVERY_PROBLEMS[reason]
    problem, fix = template
    return problem.format(who=_and_list(names)), fix


def brief_name(kind: str) -> str:
    """The brief's name in the app ("morning brief"): the scheduler's own label."""
    from openexecutive.scheduler.action_phrasing import KIND_LABEL

    return KIND_LABEL.get(kind, "brief")


def channel_phrase(channel: str) -> str:
    """As in "sent to you by email" or "sent to you on Slack"."""
    return "by email" if channel == "email" else f"on {CHANNEL_NAMES.get(channel, channel)}"


@dataclass(frozen=True)
class RecipientOutcome:
    """How a fan-out went for ONE of its recipients."""

    # Who it was sent to. The rendered name comes from the LIVE Person row at
    # read time, so a recipient archived since is dropped and a renamed one is
    # named correctly; `name` is who they were when it ran, kept so the stored
    # row says something on its own.
    person_id: int | None
    name: str
    reason: DeliveryReason
    # The channel that reached them ("email", "slack_dm", ...), if one did.
    channel: str | None
    # The channel their plan STARTED with on the run that was recorded, if
    # they had a plan. Stored rather than recomputed, because "reached on
    # email while the plan starts at Slack" only means "Slack failed" if
    # Slack was in the plan AT THE TIME: a channel connected since the run
    # would otherwise manufacture "Slack didn't work for Ada" about a channel
    # nothing ever tried. See `backup_channel_problem`.
    first_tried: str | None = None


@dataclass(frozen=True)
class DeliveryOutcome:
    kind: str
    # The run's aggregate reason and channel (`delivery_summary`). The channel
    # is named only when every delivery used the same one, so it is never one
    # founder's standing in for everyone's.
    reason: DeliveryReason
    channel: str | None
    at: datetime
    # One entry per recipient the fan-out tried, or None for a row written
    # before this shape existed (a legacy row says nothing per person, so its
    # reason is reported run-level). An EMPTY tuple is a different fact: the
    # run is known to have reached nobody.
    recipients: tuple[RecipientOutcome, ...] | None = None


# Failure reasons worst first. Which one summarises a fan-out must not depend
# on roster order — `failed[0]` let person id decide which founder's problem
# anyone could see — and the aggregate is no longer where attribution lives,
# since `recipients` carries that, so a fixed severity is enough.
_REASON_SEVERITY: tuple[DeliveryReason, ...] = (
    "not_written", "send_failed", "no_channel", "no_owner", "delivered",
)


def delivery_summary(
    recipients: Sequence[RecipientOutcome],
) -> tuple[DeliveryReason, str | None]:
    """The ``(reason, channel)`` summarising a whole fan-out.

    No recipients at all is ``no_owner`` — there was no principal to send to.
    Otherwise the worst reason any recipient got.

    The channel is named only when the whole run succeeded on the SAME one, so
    on a one-principal roster it is that person's and otherwise it is None —
    never one founder's channel passing for everyone's, and never a channel at
    all for a run that also failed somewhere.

    That aggregate channel has no consumer for a row this writes: the only
    reader (``backup_channel_problem``) takes the per-recipient channel and
    falls back to this one only for a LEGACY row, which by definition was not
    written here. It is kept because a single aggregate is what the stored
    shape has always carried and the field is the one thing a legacy reader
    can still understand.

    Never raises: the severity list covers every ``DeliveryReason``, and the
    default is there so a value from some future writer degrades to
    ``send_failed`` rather than ``StopIteration`` out of a module whose whole
    contract is that it does not raise.
    """
    if not recipients:
        return "no_owner", None
    reasons = {r.reason for r in recipients}
    reason: DeliveryReason = next(
        (r for r in _REASON_SEVERITY if r in reasons), "send_failed"
    )
    channels = {r.channel for r in recipients}
    return reason, channels.pop() if reason == "delivered" and len(channels) == 1 else None


def record_delivery_outcome(
    kind: str,
    *,
    reason: DeliveryReason,
    channel: str | None,
    recipients: Sequence[RecipientOutcome] | None = None,
) -> None:
    """Remember how this brief's latest run ended. Never raises.

    ``reason``/``channel`` are the run's aggregate (``delivery_summary``).
    ``recipients`` is one entry per person the fan-out tried; pass ``[]`` for a
    run that reached nobody (no principal at all, or an artifact that was never
    written), which is a different fact from ``None`` — ``None`` writes the
    pre-per-recipient shape, whose reason can only be reported run-level.
    """
    try:
        if recipients is None:
            text = channel or ""
        else:
            text = json.dumps({
                "channel": channel,
                "recipients": [
                    {
                        "person_id": r.person_id,
                        "name": r.name,
                        "reason": r.reason,
                        "channel": r.channel,
                        "first_tried": r.first_tried,
                    }
                    for r in recipients
                ],
            })
        narrative_cache.put(narrative_cache.BriefingNarrative(
            scope=f"{DELIVERY_SCOPE_PREFIX}{kind}",
            input_hash=reason,
            narrative_text=text,
            generated_at=narrative_cache.utc_now_iso(),
        ))
    except Exception:
        logger.exception("brief_state: delivery outcome write failed for %s", kind)


def _read_outcome_payload(
    text: str,
) -> tuple[str | None, tuple[RecipientOutcome, ...] | None]:
    """``(channel, recipients)`` from a stored ``narrative_text``.

    A bare channel name is the pre-per-recipient shape and reads back with
    ``recipients=None``. A payload that opens with ``{`` but will not parse (a
    truncated write, a hand-edited row) is read as unreadable rather than
    raised on: the reason in ``input_hash`` still stands, and reporting it
    run-level is what a legacy row does anyway. One unreadable ENTRY is
    skipped while the others still count — dropping the whole list would hide
    the founders it could still name.
    """
    if not text.startswith("{"):
        return (text or None), None
    try:
        payload = json.loads(text)
        raw = payload["recipients"]
        channel = payload.get("channel")
        if not isinstance(raw, list) or not (channel is None or isinstance(channel, str)):
            raise ValueError("unexpected brief_delivery payload shape")
    except Exception:
        logger.warning("brief_state: unreadable delivery payload, read as legacy")
        return None, None
    out: list[RecipientOutcome] = []
    for entry in raw:
        if not isinstance(entry, dict) or entry.get("reason") not in _DELIVERY_REASONS:
            continue
        pid = entry.get("person_id")
        # Only channels the app knows, since both reach a sentence a founder
        # reads ("went on …", "Slack didn't work for …").
        entry_channel = entry.get("channel")
        first_tried = entry.get("first_tried")
        out.append(RecipientOutcome(
            # A JSON `true` is an int in Python, and would then match person 1.
            person_id=pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
            name=str(entry.get("name") or ""),
            reason=cast(DeliveryReason, entry["reason"]),  # checked above
            channel=entry_channel if entry_channel in CHANNEL_NAMES else None,
            first_tried=first_tried if first_tried in CHANNEL_NAMES else None,
        ))
    return (channel or None), tuple(out)


def last_delivery_outcome() -> DeliveryOutcome | None:
    """The latest run of any recorded message (``DELIVERY_KINDS``: the two
    briefs and the weekly review), or None when none has run yet (or the
    store can't be read). Never raises."""
    latest: DeliveryOutcome | None = None
    for kind in DELIVERY_KINDS:
        try:
            row = narrative_cache.get(f"{DELIVERY_SCOPE_PREFIX}{kind}")
        except Exception:
            logger.exception("brief_state: delivery outcome read failed for %s", kind)
            continue
        at = parse_aware(row.generated_at) if row is not None else None
        if row is None or at is None or row.input_hash not in _DELIVERY_REASONS:
            continue
        channel, recipients = _read_outcome_payload(row.narrative_text)
        outcome = DeliveryOutcome(
            kind=kind,
            reason=cast(DeliveryReason, row.input_hash),  # checked above
            channel=channel,
            at=at,
            recipients=recipients,
        )
        if latest is None or outcome.at > latest.at:
            latest = outcome
    return latest


def _names_unless_all(
    people: Sequence[Person], roster: Sequence[Person]
) -> tuple[str, ...]:
    """The names to attribute a problem to, or ``()`` when it covers the whole
    roster — "to you" is then true for whoever is reading, and shorter.

    ``people`` must be a DUPLICATE-FREE subset of ``roster`` — both callers
    build it from the roster itself, and the record-derived one is keyed by
    person id for exactly this reason. Counting is then enough; a duplicate
    would push the count to the roster's and turn one founder's problem into
    everyone's.
    """
    if len(people) >= len(roster):
        return ()
    return tuple(p.full_name for p in people)


def outstanding_problems(
    outcome: DeliveryOutcome | None,
    roster: Sequence[Person],
    *,
    email_ready: bool,
) -> list[tuple[DeliveryProblem, tuple[str, ...]]]:
    """Everything still keeping the latest brief from a principal, worst
    first, one ``(reason, names)`` per problem. Empty when there is nothing to
    report.

    ``names`` is who the problem is about, or EMPTY when it covers every
    principal there is — the one case in which ``DELIVERY_PROBLEMS``' second
    person ("nothing is set up to send it to you", "every way of sending it
    failed") is honest for whoever is reading. With a co-founder still
    receiving the brief it is not, so a partial names them
    (``partial_delivery_problem``).

    Two problems at once is the ordinary case this exists for: one founder's
    sends failing while another has no channel used to collapse into one
    stored reason, so roster order decided which of them anyone could see and
    the other was rendered nowhere.

    Where each reason comes from:

    * ``no_owner`` — the roster, and nothing else is actionable while it
      holds, so it is returned alone. Not reported for a ``delivered``
      record: that brief went out, and the next run records ``no_owner``
      itself.
    * ``send_failed`` — the RECORD, per recipient, intersected with the live
      roster by person id (the rule ``deliver_to_each_principal`` applies when
      it sends, so the two cannot disagree about who a recipient is). A
      recipient archived since is dropped; a founder added since was not in
      that run, so nothing is claimed about them. A record with no
      per-recipient list — written before that shape existed, or by a run
      whose delivery raised before any recipient — reports it run-level.
    * ``no_channel`` — the roster as it stands NOW
      (``scheduler.runner.unreachable_principals``), never the record, so a
      channel connected since clears it and a founder added since counts.
    * ``not_written`` — run-level by nature: the artifact did not exist for
      anyone, so it is legitimately single-valued.
    """
    from openexecutive.scheduler.runner import unreachable_principals

    if not roster and (outcome is None or outcome.reason != "delivered"):
        return [("no_owner", ())]
    problems: list[tuple[DeliveryProblem, tuple[str, ...]]] = []
    if outcome is not None:
        if outcome.reason == "not_written":
            problems.append(("not_written", ()))
        if outcome.recipients:
            by_id = {p.id: p for p in roster if p.id is not None}
            # Keyed by person id, so a row that somehow lists the same
            # recipient twice names them once: a duplicate would otherwise
            # read as "Ada and Ada", or push the count past the roster's and
            # make `_names_unless_all` call a partial problem roster-wide.
            failing = list({
                r.person_id: by_id[r.person_id]
                for r in outcome.recipients
                if r.reason == "send_failed"
                and r.person_id is not None
                and r.person_id in by_id
            }.values())
            # A failing entry with NO person id is a failure the record
            # asserts and cannot attribute, which is a different thing from a
            # recipient who has since left the roster (an id that simply is
            # not on it any more). The second is genuinely nothing to report;
            # the first must not shrink the list or vanish from it, so it
            # forces the whole problem run-level — the conservative reading,
            # since we can no longer say it is only some of them.
            if any(
                r.reason == "send_failed" and r.person_id is None
                for r in outcome.recipients
            ):
                problems.append(("send_failed", ()))
            elif failing:
                problems.append(("send_failed", _names_unless_all(failing, roster)))
        elif outcome.reason == "send_failed":
            problems.append(("send_failed", ()))
    unreachable = unreachable_principals(roster, email_ready=email_ready)
    if unreachable:
        problems.append(("no_channel", _names_unless_all(unreachable, roster)))
    return problems


def render_problems(
    problems: Sequence[tuple[DeliveryProblem, Sequence[str]]],
) -> tuple[str, str]:
    """One ``(problem, fix)`` pair for a whole ``outstanding_problems`` list.

    The clauses are joined with "; " and the fixes de-duplicated in order, so
    two founders with the same problem get one instruction and two different
    problems get both.

    Shared by the Briefing notice and the Setup light so the same problem is
    always worded the same way. Not a guarantee that the two surfaces always
    say the same thing overall: the Setup light has earlier branches of its
    own (a missing owner, a roster nothing can reach) that return before the
    problem list is consulted.
    """
    clauses: list[str] = []
    fixes: list[str] = []
    for reason, names in problems:
        problem, fix = (
            partial_delivery_problem(names, reason) if names else DELIVERY_PROBLEMS[reason]
        )
        clauses.append(problem)
        if fix not in fixes:
            fixes.append(fix)
    return "; ".join(clauses), " ".join(fixes)


def backup_channel_problem(
    outcome: DeliveryOutcome | None,
    roster: Sequence[Person],
    *,
    email_ready: bool,
    owner: Person | None = None,
) -> tuple[str, str] | None:
    """``(summary, broken_channel)`` when the last brief got through only on a
    LATER channel than the one a recipient's own plan starts with — so that
    first channel is broken and every brief is going by the backup — else
    None.

    Judged per recipient, from what the RUN recorded: the channel that
    reached them against ``RecipientOutcome.first_tried``, the one their plan
    started with at the time. Comparing against ``delivery_order`` as it
    stands now would call a channel broken that nothing ever tried — connect
    Slack after a brief that went by email and the light would announce
    "Slack didn't work for Ada". Only a recorded attempt can support the
    claim, so only a recorded attempt is used.

    ``owner`` is the row a LEGACY record (no per-recipient list) is attributed
    to, and is only consulted then — a single stored channel for however many
    recipients is unambiguous on a one-principal roster and nowhere else, so
    that is where that branch stays. It still compares against the plan as it
    stands, which is the behaviour it has always had; a legacy row carries no
    attempt to compare with.

    None when the recipients' broken first channels are not all the SAME
    channel. The summary is one line and its fix points at ONE light, and
    there is no honest way to name two: the per-channel lights beside it show
    an integration that is down but not, say, one founder's wrong Slack id, so
    this is a known gap rather than a case the page covers elsewhere.
    """
    from openexecutive.scheduler.runner import delivery_order

    if outcome is None:
        return None
    label = brief_name(outcome.kind)
    if outcome.recipients is None:
        if len(roster) > 1 or owner is None or not outcome.channel:
            return None
        plan = delivery_order(owner, email_ready=email_ready)
        if not plan or plan[0] == outcome.channel:
            return None
        return (
            f"Your last {label} went {channel_phrase(outcome.channel)}, "
            f"because {CHANNEL_NAMES[plan[0]]} didn't work.",
            plan[0],
        )
    by_id = {p.id: p for p in roster if p.id is not None}
    # person id -> (person, the channel that reached them, the one the run
    # tried first). Keyed, so a row that somehow lists the same recipient
    # twice names them once rather than as "Ada and Ada".
    by_person: dict[int, tuple[Person, str, str]] = {}
    for r in outcome.recipients:
        person = by_id.get(r.person_id) if r.person_id is not None else None
        if person is None or r.reason != "delivered" or not r.channel:
            continue
        if r.first_tried and r.first_tried != r.channel:
            by_person[cast(int, r.person_id)] = (person, r.channel, r.first_tried)
    on_backup = list(by_person.values())
    broken = {first for _, _, first in on_backup}
    if len(broken) != 1:
        return None
    first = broken.pop()
    if len(on_backup) == 1 and len(roster) == 1:
        _, carried, _ = on_backup[0]
        return (
            f"Your last {label} went {channel_phrase(carried)}, "
            f"because {CHANNEL_NAMES[first]} didn't work.",
            first,
        )
    who = _and_list([p.full_name for p, _, _ in on_backup])
    return (
        f"{CHANNEL_NAMES[first]} didn't work for {who}, so the last {label} "
        "reached them on a backup channel.",
        first,
    )


def split_proposals(
    proposals: list[dict[str, Any]], since: datetime | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(new, carried)``: proposals created at/after ``since`` vs earlier.

    With ``since`` None everything is "new" (the legacy single-list view).
    """
    if since is None:
        return list(proposals), []
    new: list[dict[str, Any]] = []
    carried: list[dict[str, Any]] = []
    for p in proposals:
        created = parse_aware(p.get("created_at"))
        (new if created is None or created >= since else carried).append(p)
    return new, carried


def rewritten_since(
    proposals: list[dict[str, Any]], since: datetime | None
) -> list[dict[str, Any]]:
    """Open proposals the review rewrote (verdict ``changed``) at/after ``since``.

    These are the alerts that dropped out of the handled block: still open,
    text or severity refreshed by the Executive. The brief reports them under
    "what changed", never as done. Empty when ``since`` is None.
    """
    if since is None:
        return []
    out: list[dict[str, Any]] = []
    for p in proposals:
        if p.get("review_verdict") != "changed":
            continue
        # Fail open like split_proposals: a rewrite with no readable stamp is
        # reported rather than dropped from every block.
        reviewed = parse_aware(p.get("last_reviewed_at"))
        if reviewed is None or reviewed >= since:
            out.append(p)
    return out


def rewritten_lines(
    proposals: list[dict[str, Any]], since: datetime | None, limit: int = 10
) -> list[str]:
    """Bullet lines for the briefs' REWRITTEN block (one per rewritten open
    proposal): ``- <headline> — <review note>``. Shared by the morning brief
    and the end-of-day digest so the two never drift."""
    return [
        f"- {str(p.get('headline', ''))[:160]} — {str(p.get('review_note', ''))[:120]}"
        for p in rewritten_since(proposals, since)[:limit]
    ]


def handled_since(
    since: datetime, limit: int = 20, *, include_private: bool = False
) -> list[dict[str, Any]]:
    """Completed autonomous moves recorded in the audit log since ``since``.

    Each item: ``{"kind": event_type sans prefix, "event_type": str,
    "summary": str, "at": iso, "alert_id": int | None, "details": dict}``,
    newest first. ``summary`` has the nudge bookkeeping marker stripped;
    ``details`` is the audit row's structured payload (headline, target
    person, evidence ref, new status …) for callers that render more than
    one line. Empty when the audit store is unavailable.

    Rows private to the principal (one that names their contact, say) are
    left out unless ``include_private``: only the principal's own ``/today``
    asks for them. The briefs don't, as a written brief can be read by others.
    """
    try:
        from openexecutive.audit.logger import get_audit_logger

        logger_ = get_audit_logger()
        out: list[dict[str, Any]] = []
        for event_type, kind in HANDLED_EVENT_KINDS.items():
            for ev in logger_.query(
                event_type=event_type,
                since=since.isoformat(),
                limit=limit,
                include_private=include_private,
            ):
                details = ev.details if isinstance(ev.details, dict) else {}
                out.append({
                    "kind": kind,
                    "event_type": event_type,
                    "summary": _ALERT_MARKER_RE.sub("", ev.summary or ""),
                    "at": ev.ts,
                    "alert_id": details.get("alert_id"),
                    "details": details,
                })
        out.sort(key=lambda e: e["at"], reverse=True)
        return out[:limit]
    except Exception:
        logger.debug("brief_state: handled_since unavailable", exc_info=True)
        return []


def build_brief_fingerprint(
    *,
    today_data: dict[str, Any],
    activity: list[dict[str, Any]],
    handled: list[dict[str, Any]],
    since: datetime | None,
    pending_watch_suggestions: int = 0,
    mode: str = "team",
    live_keys: dict[str, Any] | None = None,
    reflection_flags: str = "",
    teammate_changes: str = "",
    owner_notes: list[Any] | None = None,
) -> str:
    """Stable hash of everything the brief would say. Deliberately free of
    dates and timestamps so an unchanged day yields the same fingerprint
    tomorrow (activity is keyed by kind + summary, never by its stamp).

    ``mode="solo"`` leaves out who is awaiting (the solo brief never says)
    and carries the mode, so switching mode never suppresses the first brief
    in the new one as "unchanged". Team fingerprints are unchanged.

    Solo also carries ``due_soon`` (the DUE THIS WEEK items) as
    ``(loop_id, state)`` pairs — no dates — so a new item, a closed one, or
    one that falls due today or goes overdue un-suppresses the brief. And
    the TOP THREE TODAY items by key (in order — no dates, no slot times),
    plus only a coarse hash of today's calendar (``top_three.calendar_hash``)
    when one was read, so an unchanged day still suppresses.

    ``live_keys`` (``LiveSignals.keys`` — who wrote and about what, what is
    stuck, the calendar's shape; no times) and ``reflection_flags`` make the
    principal's actual world count: a brief is "unchanged" only when no mail
    came in, nothing got stuck and the day looks the same. Both are carried
    only when non-empty, so a caller that passes neither keeps its old
    fingerprint. So is ``teammate_changes`` (the TEAMMATE CORRECTIONS block):
    a correction a teammate made since the last brief un-suppresses it, since
    the principal hears of it nowhere else. ``owner_notes`` (the FROM YOUR
    NOTES keys, ``history_brief.NotesBlock.keys``: note ids and states, no
    dates) likewise, only when non-empty."""
    new, carried = split_proposals(today_data.get("proposals", []), since)
    payload = {
        "new": sorted(int(p.get("alert_id") or 0) for p in new),
        "carried": sorted(int(p.get("alert_id") or 0) for p in carried),
        "likely_stale": sum(1 for p in carried if p.get("review_verdict") == "likely_stale"),
        "activity": sorted(
            (str(a.get("kind", "")), str(a.get("summary", ""))[:80]) for a in activity
        ),
        "handled": sorted((h["kind"], h["summary"][:80]) for h in handled),
        # A rewrite alone must still un-suppress the brief now that it no
        # longer rides in `handled` (keyed on the note, never the stamp). Same
        # list the REWRITTEN block renders: carried items only — a new item
        # already moves the fingerprint by id.
        "rewritten": sorted(
            (int(p.get("alert_id") or 0), str(p.get("review_note", ""))[:80])
            for p in rewritten_since(carried, since)
        ),
        "depts": sorted(
            (d.get("slug", ""), d.get("at_risk_count", 0), d.get("off_track_count", 0))
            for d in today_data.get("departments", [])
            if d.get("at_risk_count", 0) or d.get("off_track_count", 0)
        ),
        "awaiting": sorted(
            int(p.get("id", 0)) for p in today_data.get("people", []) if p.get("awaiting_count", 0)
        ),
        "watch_suggestions": int(pending_watch_suggestions),
    }
    if mode == "solo":
        payload.pop("awaiting")
        payload["mode"] = mode
        due = today_data.get("due_soon") or []
        if due:
            # Only when present, so a solo fingerprint with nothing due is
            # exactly what it was before this block existed.
            payload["due_soon"] = sorted(
                (int(d.get("loop_id") or 0), str(d.get("state", ""))) for d in due
            )
        top = today_data.get("top_three") or []
        if top:
            payload["top_three"] = [str(t.get("key", "")) for t in top]
        calendar = today_data.get("today_calendar")
        if isinstance(calendar, dict) and calendar.get("hash"):
            payload["calendar"] = str(calendar["hash"])
    if live_keys and any(live_keys.values()):
        payload["live"] = live_keys
    if reflection_flags:
        payload["reflection_flags"] = reflection_flags
    if teammate_changes:
        payload["teammate_changes"] = teammate_changes
    if owner_notes:
        payload["owner_notes"] = [list(k) if isinstance(k, tuple) else k for k in owner_notes]
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


_FLAGGED_RE = re.compile(
    r"\*\*Flagged for the brief:?\*\*:?\s*(.*?)(?=\n\s*\*\*[^*\n]+:?\*\*|\n---|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_REFLECTION_FLAGS_MAX = 800


def reflection_flags_since(since: datetime) -> str:
    """The "Flagged for the brief" bullets of the latest executive reflection
    that finished at/after ``since``, or "".

    The reflection runs just before the morning brief and is told to write
    what the principal should see there; nothing carried it across, so the
    brief only ever saw an activity line with the run's title. Cut to a few
    hundred characters. Never raises."""
    try:
        from openexecutive.workflows import persistence

        for run in persistence.list_runs(
            workflow_name="executive_reflection", status="done", limit=3, visible_to=None,
        ):
            finished = parse_aware(run.get("updated_at"))
            if finished is None or finished < since:
                continue
            full = persistence.get_run(str(run["run_id"])) or {}
            match = _FLAGGED_RE.search(str(full.get("artifact") or ""))
            if match is None:
                return ""
            text = match.group(1).strip()
            if len(text) > _REFLECTION_FLAGS_MAX:
                text = text[: _REFLECTION_FLAGS_MAX - 1].rstrip() + "…"
            return text
    except Exception:
        logger.debug("brief_state: reflection flags unavailable", exc_info=True)
    return ""


def suppress_unchanged_enabled() -> bool:
    """`PRINCIPAL_BRIEF_SUPPRESS_UNCHANGED`, defaulting to on when settings
    cannot be built (bare test DB with no env)."""
    try:
        from openexecutive.config import get_settings

        return bool(get_settings().principal_brief_suppress_unchanged)
    except Exception:
        return True


def suppressed_line(n_waiting: int) -> str:
    return SUPPRESSED_TEMPLATE.format(n=n_waiting, s="" if n_waiting == 1 else "s")


def pending_watch_suggestions() -> int:
    """Research watch suggestions awaiting the principal on /watchlist.
    Zero when the monitoring store is unavailable."""
    try:
        from openexecutive.monitoring import store as monitoring_store

        return len(monitoring_store.list_pending_suggestions())
    except Exception:
        logger.debug("brief_state: pending_watch_suggestions unavailable", exc_info=True)
        return 0


__all__ = [
    "BRIEF_KINDS",
    "CHANNEL_NAMES",
    "DELIVERY_KINDS",
    "DELIVERY_PROBLEMS",
    "HANDLED_EVENT_KINDS",
    "REVIEW_EVENT_TYPES",
    "SUPPRESSED_TEMPLATE",
    "DeliveryOutcome",
    "DeliveryReason",
    "RecipientOutcome",
    "backup_channel_problem",
    "brief_name",
    "build_brief_fingerprint",
    "channel_phrase",
    "delivery_summary",
    "handled_since",
    "last_delivered",
    "last_delivery_outcome",
    "outstanding_problems",
    "partial_delivery_problem",
    "pending_watch_suggestions",
    "record_delivered",
    "record_delivery_outcome",
    "render_problems",
    "rewritten_lines",
    "rewritten_since",
    "scope_for",
    "since_for",
    "split_proposals",
    "suppress_unchanged_enabled",
    "suppressed_line",
]
