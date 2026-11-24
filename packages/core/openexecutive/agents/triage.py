from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from typing import TYPE_CHECKING, Any

from openexecutive.agents.base import BaseAgent
from openexecutive.alerts.models import (
    AlertChannel,
    AlertEvent,
    AlertSeverity,
    TriageDecision,
)
from openexecutive.alerts.preferences import matches_mute
from openexecutive.audit.usage import log_model_usage
from openexecutive.providers import get_provider

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)

_TRIAGE_TIMEOUT = 60.0
_MAX_EVENT_CHARS = 8000


TRIAGE_TOOL: dict[str, Any] = {
    "name": "emit_alert_decision",
    "description": (
        "Emit the triage decision for this inbound event. Always call this tool; "
        "never reply in plain text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "alert": {
                "type": "boolean",
                "description": "True if the user should be alerted; false if the event is not worth surfacing.",
            },
            "severity": {
                "type": "string",
                "enum": ["low", "medium", "high", "urgent"],
            },
            "channels": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "web", "slack_dm", "email", "persisted",
                        "department_channel", "company_broadcast",
                    ],
                },
                "description": (
                    "Which channels to deliver on. Always include 'persisted'. "
                    "Use 'department_channel' for dept-scoped alerts (sets "
                    "department_slug + broadcast_integration); 'company_broadcast' "
                    "for company-wide alerts (sets broadcast_integration). "
                    "Never use either for board/comp/legal — those are DM-only."
                ),
            },
            "headline": {
                "type": "string",
                "description": "<= 100 chars. The single most important fact.",
            },
            "body": {
                "type": "string",
                "description": "1-3 sentence summary with implication.",
            },
            "suggested_action": {
                "type": "string",
                "description": (
                    "The single concrete action the Executive will take on approval "
                    "(first-person imperative, <= 1 sentence). Empty string if none: "
                    "a card with nothing to approve is for the reader to know, or asks them."
                ),
            },
            "action_label": {
                "type": "string",
                "description": (
                    "The Approve button's words: suggested_action in 2-5 words, a verb "
                    "first, naming who or what (\"Reply to Dana\", \"Book the Leo call\", "
                    "\"Check Stripe's pricing\"). Empty when suggested_action is empty."
                ),
            },
            "topic_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-4 lowercase topic tags.",
            },
            "dedup_key": {
                "type": "string",
                "description": "Stable summary key. Same underlying event must produce the same key.",
            },
            "reason_if_suppressed": {
                "type": "string",
                "description": "Reason when alert=false ('duplicate', 'muted: <pattern>', 'low_signal'). Empty otherwise.",
            },
            "department_slug": {
                "type": "string",
                "description": (
                    "Required when channels includes 'department_channel'. "
                    "The slug of the department whose team room to post to "
                    "(e.g. 'marketing', 'finance'). Empty otherwise."
                ),
            },
            "broadcast_integration": {
                "type": "string",
                "enum": ["slack", "discord", "telegram"],
                "description": (
                    "Required when channels includes 'department_channel' or "
                    "'company_broadcast'. The integration to broadcast on. "
                    "Omit the field entirely when no broadcast channel is "
                    "in `channels` — do NOT emit an empty string."
                ),
            },
        },
        "required": [
            "alert",
            "severity",
            "channels",
            "headline",
            "body",
            "dedup_key",
        ],
    },
}


_EVENT_CLOSE_RE = re.compile(r"<\s*/\s*event", re.IGNORECASE)


def _format_event_block(event: AlertEvent) -> str:
    body = (event.body or "")[:_MAX_EVENT_CHARS]
    parts = [f"source: {event.source}", f"external_id: {event.external_id}"]
    if event.subject:
        parts.append(f"subject: {event.subject}")
    if event.from_:
        parts.append(f"from: {event.from_}")
    if event.channel:
        parts.append(f"channel: {event.channel}")
    if event.user:
        parts.append(f"sender: {event.user}")
    if event.title:
        parts.append(f"title: {event.title}")
    parts.append("---")
    parts.append(body)
    # Every field is sender-written (a mail body, a chat line, a watched
    # page's text): none may close the <event> block and speak outside it.
    # NFKC first, so a fullwidth "＜/event＞" is caught as the ASCII one.
    text = unicodedata.normalize("NFKC", "\n".join(parts))
    return _EVENT_CLOSE_RE.sub(r"<\\/event", text)


def _format_recent_alerts(recent: list[dict]) -> str:
    if not recent:
        return "(none)"
    lines = []
    for a in recent[:20]:
        lines.append(
            f"- [{a.get('severity', '?')}] {a.get('headline', '')}"
            f" | dedup_key={a.get('dedup_key', '')}"
            f" | tags={','.join(a.get('topic_tags') or [])}"
        )
    return "\n".join(lines)


def _format_mutes(patterns: list[str]) -> str:
    if not patterns:
        return "(none)"
    return "\n".join(f"- {p}" for p in patterns)


def _format_initiatives(initiatives: list[Any]) -> str:
    if not initiatives:
        return "(none)"
    lines = []
    for i in initiatives:
        title = getattr(i, "title", None) or (i.get("title") if isinstance(i, dict) else "")
        status = getattr(i, "status", None) or (i.get("status") if isinstance(i, dict) else "")
        summary = getattr(i, "summary", None) or (i.get("summary") if isinstance(i, dict) else "")
        lines.append(f"- {title} [{status}]: {summary}")
    return "\n".join(lines)


class TriageAgent(BaseAgent):
    name = "triage"
    domain = "triage"

    @property
    def model(self) -> str:  # type: ignore[override]
        # Cheap and fast. Triage runs on every inbound event. Uses the
        # configured routing model (like utility_fast) so an Anthropic-free
        # deployment routes triage to its local / OpenRouter model too,
        # rather than forcing a hardcoded Claude model.
        from openexecutive.config import get_settings

        return get_settings().routing_model

    def get_system_prompt(self) -> str:
        from openexecutive.prompts.triage_prompt import TRIAGE_PROMPT

        return TRIAGE_PROMPT

    def _build_user_content(
        self,
        event: AlertEvent,
        recent: list[dict],
        mute_patterns: list[str],
        initiatives: list[Any],
    ) -> str:
        return (
            f"<event>\n{_format_event_block(event)}\n</event>\n\n"
            f"<recent_alerts>\n{_format_recent_alerts(recent)}\n</recent_alerts>\n\n"
            f"<muted_topics>\n{_format_mutes(mute_patterns)}\n</muted_topics>\n\n"
            f"<active_initiatives>\n{_format_initiatives(initiatives)}\n</active_initiatives>"
        )

    async def triage(
        self,
        event: AlertEvent,
        *,
        recent_alerts: list[dict] | None = None,
        mute_patterns: list[str] | None = None,
        active_initiatives: list[Any] | None = None,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> TriageDecision:
        """Run triage and return a structured decision. Never raises. When
        review makes no whole decision (the call failed, or its answer was
        cut off or missing), a card the Executive raised itself is kept as
        raised, unless a muted topic names it; any other event gets a
        low-severity persisted-only decision so it is still recorded."""
        # The optional ``client`` parameter is a legacy hook for callers that
        # need to inject a specific Anthropic SDK instance (evals/runner.py
        # and a few unit tests). When unset, we route through the provider
        # registry so a Council override on this agent's model takes effect.
        caller = client if client is not None else get_provider(self.effective_model())

        user_content = self._build_user_content(
            event,
            recent_alerts or [],
            mute_patterns or [],
            active_initiatives or [],
        )

        # Both code paths use the same Anthropic-shape kwargs; ``client``
        # exposes ``messages.create`` while ``provider`` exposes
        # ``messages_create`` — branch only on the call shape.
        create_kwargs: dict[str, Any] = {
            "model": self.effective_model(),
            # Room for the whole decision: a body of up to 2000 characters
            # plus the headline, action and tags.
            "max_tokens": 1536,
            "timeout": _TRIAGE_TIMEOUT,
            "system": self.effective_system_prompt(),
            "tools": [TRIAGE_TOOL],
            "tool_choice": {"type": "tool", "name": "emit_alert_decision"},
            "messages": [{"role": "user", "content": user_content}],
        }

        try:
            if client is not None:
                message = await caller.messages.create(**create_kwargs)  # type: ignore[union-attr]
            else:
                message = await caller.messages_create(**create_kwargs)  # type: ignore[union-attr]
        except Exception:
            logger.exception("Triage LLM call failed for event %s", event.external_id)
            return _no_decision(event, mute_patterns or [], "triage_error")
        log_model_usage(message, model=self.effective_model(), actor="triage")

        # An answer cut off at max_tokens is a partial decision: its missing
        # fields would read as "no card". If it got as far as saying to raise
        # a card, the card shows in the event's own words; otherwise it is
        # handled as no decision.
        cut_off = getattr(message, "stop_reason", None) == "max_tokens"
        for block in message.content:
            if block.type == "tool_use" and block.name == "emit_alert_decision":
                if not cut_off:
                    return _parse_decision(block.input)
                if isinstance(block.input, dict) and block.input.get("alert") is True:
                    said_tags = [str(t).lower() for t in block.input.get("topic_tags") or []]
                    kept = _no_decision(event, mute_patterns or [], "cut_off", raise_it=True, tags=said_tags)
                    return _with_partial_delivery(kept, block.input, event)
                if isinstance(block.input, dict) and block.input.get("alert") is False:
                    # It had decided not to raise it: that decision stands.
                    return _parse_decision(block.input)

        logger.warning(
            "Triage gave no whole decision for event %s", event.external_id,
        )
        return _no_decision(event, mute_patterns or [], "no_decision")


def _no_decision(
    event: AlertEvent, mute_patterns: list[str], why: str, *,
    raise_it: bool = False, tags: list[str] | None = None,
) -> TriageDecision:
    """What stands when review made no decision. A card the Executive chose
    to raise (``AlertEvent.from_executive``) is kept as raised: a review that
    couldn't run never loses it, only a decision it made can. A muted topic
    it names stays muted. Anything else is recorded quietly, as before, so
    an outage doesn't turn every chat line and newsletter into a card.
    ``raise_it`` is a cut-off review that had already said to raise one:
    any event is kept then."""
    headline = (event.subject or event.title or (event.body or "").strip().split("\n", 1)[0] or event.source)[:200]
    # A kept card carries the tags a cut-off review gave, matched the way
    # the pipeline matches a reviewed card's. The department or room it came
    # from holds it back only when that very topic was muted ("mute this
    # topic" stores department:<slug>), so a short mute never hits it.
    topic_tags = [*(tags or [])]
    own = set()
    if event.department:
        own.add(f"department:{event.department.lower()}")
    if event.channel:
        own.add(event.channel.lower())
    # A cut-off review that said to raise it had seen the mutes: only its
    # tags count then. With no review at all, the words stand in.
    muted = (
        matches_mute(topic_tags, mute_patterns)
        or any(p.strip().lower() in own for p in mute_patterns)
        or (not raise_it and _names_muted_topic(event, mute_patterns))
    )
    if muted:
        why = "muted"
    if (raise_it or event.from_executive) and not muted:
        return TriageDecision(
            alert=True,
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.WEB, AlertChannel.PERSISTED],
            headline=headline,
            # Outside text the review never rewrote shows as a short preview.
            body=(event.body or "")[: 2000 if event.from_executive else 280],
            dedup_key=_kept_key(event),
            topic_tags=topic_tags[:8],
        )
    return TriageDecision(
        alert=False,
        severity=AlertSeverity.LOW,
        channels=[AlertChannel.PERSISTED],
        headline=headline,
        body=(event.body or "")[:280],
        dedup_key=f"{why}-{event.source}-{event.external_id}",
        reason_if_suppressed=why,
    )


def _with_partial_delivery(kept: TriageDecision, raw: dict, event: AlertEvent) -> TriageDecision:
    """A kept card keeps the severity and channels a cut-off review had
    already given, so an urgent one is still delivered as urgent. A room or
    company post only stays for a card the Executive raised: the kept card
    shows the event's own words, and outside mail nobody finished reviewing
    is not posted to a room."""
    if not kept.alert:
        return kept
    said = _parse_decision(raw)
    update: dict[str, Any] = {}
    if str(raw.get("severity") or "").lower() in AlertSeverity._value2member_map_:
        update["severity"] = said.severity
    rooms = {AlertChannel.DEPARTMENT_CHANNEL, AlertChannel.COMPANY_BROADCAST}
    channels = [c for c in said.channels if event.from_executive or c not in rooms]
    if raw.get("channels") and channels != [AlertChannel.PERSISTED]:
        # Web stays, so a room post missing its room (cut off before the
        # slug) still leaves the card on Home.
        update["channels"] = [*dict.fromkeys([AlertChannel.WEB, *channels])]
        if any(c in rooms for c in channels):
            update["department_slug"] = said.department_slug
            update["broadcast_integration"] = said.broadcast_integration
    return kept.model_copy(update=update) if update else kept


def _kept_key(event: AlertEvent) -> str:
    """The key a kept card folds on. ``create_alert`` needs no external id,
    so without one the card's own words make the key: two different cards
    never fold into one, and the same card raised again still does."""
    if event.external_id:
        return f"kept-{event.source}-{event.external_id}"
    text = f"{event.source}\n{event.subject}\n{event.body}"
    return f"kept:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def _names_muted_topic(event: AlertEvent, mute_patterns: list[str]) -> bool:
    """Whether a mute pattern is a word (or its plural) in the event's
    subject or body: with no review there are no topic tags to match, so
    the words stand in. "newsletter" holds "newsletters" back, but a short
    mute like "ai" never hits "email" or "air"."""
    text = f"{event.subject}\n{event.title or ''}\n{event.body}".lower()
    return any(
        p.strip() and re.search(rf"(?<!\w){re.escape(p.strip().lower())}(?:e?s)?(?!\w)", text)
        for p in mute_patterns
    )


def _parse_decision(raw: dict) -> TriageDecision:
    """Tolerantly coerce the tool input into a TriageDecision."""
    try:
        severity_str = (raw.get("severity") or "low").lower()
        severity = AlertSeverity(severity_str) if severity_str in AlertSeverity._value2member_map_ else AlertSeverity.LOW

        channels: list[AlertChannel] = []
        for c in raw.get("channels") or []:
            c_str = str(c).lower()
            if c_str in AlertChannel._value2member_map_:
                channels.append(AlertChannel(c_str))
        if AlertChannel.PERSISTED not in channels:
            channels.append(AlertChannel.PERSISTED)

        topic_tags = [str(t).lower() for t in (raw.get("topic_tags") or [])]

        # Shift 3: broadcast routing context. Coerce defensively — the
        # model may omit them entirely (default "") or emit a value
        # outside the integration enum (drop it silently in that case,
        # the dispatcher returns False on missing context and the
        # broadcast becomes a no-op rather than a crash).
        dept_slug = str(raw.get("department_slug") or "")[:64]
        bi_raw = str(raw.get("broadcast_integration") or "")
        broadcast_integration = (
            bi_raw if bi_raw in ("slack", "discord", "telegram") else ""
        )

        return TriageDecision(
            alert=bool(raw.get("alert", False)),
            severity=severity,
            channels=channels,
            headline=(raw.get("headline") or "")[:200],
            body=(raw.get("body") or "")[:2000],
            suggested_action=(raw.get("suggested_action") or "")[:500],
            action_label=str(raw.get("action_label") or "")[:80],
            topic_tags=topic_tags[:8],
            dedup_key=(raw.get("dedup_key") or "")[:120],
            reason_if_suppressed=(raw.get("reason_if_suppressed") or "")[:200],
            department_slug=dept_slug,
            broadcast_integration=broadcast_integration,
        )
    except Exception:
        logger.exception("Failed to parse triage decision: %r", raw)
        return TriageDecision(
            alert=False,
            severity=AlertSeverity.LOW,
            channels=[AlertChannel.PERSISTED],
            headline="(parse error)",
            body="",
            dedup_key="parse-error",
            reason_if_suppressed="parse_error",
        )
