"""Approval cards in a chat app, with Approve and Skip buttons.

The web chat shows the cards a reply leaves (a meeting to book, a card of
actions from ``propose_actions``) under that reply, for the person chatting
to tap. A chat app that can prove who taps a button shows them the same way:
the adapter collects the reply's cards (``Executive.chat(cards=...)``), sends
each as a short message with two buttons, and passes a tap to
``api.routes.decisions.resolve_in_chat``, which applies the web chat's rule
for who may decide it.

Where a tap proves who made it, the same bar as a verified speaker there
(``people_tools._is_verified_speaker_surface``):

- Telegram: a private chat (the chat is the person; ``from.id`` must equal
  the chat id) on a webhook only Telegram can call (``TELEGRAM_WEBHOOK_SECRET``
  set and valid).
- Slack: a DM, with ``SLACK_CARD_BUTTONS`` on (Interactivity turned on in
  the Slack app). Slack signs the tap (Socket Mode, or the signing secret
  over HTTP), and it counts only from the user whose DM with the bot it is.
- Discord: a DM. The tap comes over the bot's own gateway connection, and
  counts only from the user whose DM with the bot it is.
- Google Chat: a DM between the Chat app and one person alone
  (``singleUserBotDm``). The tap comes to the webhook with Google's token,
  counts only from a human on People in such a DM, and only once Chat
  itself, read as the app, confirms the message, the space and its one
  person (``google_chat.verify_tap``), since the token isn't bound to the
  event's body. Not yet sent cards
  that wait outside any chat (``chat_for``): the app has no stored DM to reach
  them in.

Anywhere else no card is shown and the Executive is told the card is on Today.

A card that waits for someone outside any chat (a reply the inbox watcher
drafted, a meeting invite, an action Take the lead or a department held) is
sent to their chat app once it has waited ``PUSH_AFTER``
(:func:`push_waiting`, from the scheduler's tick): the first chat app in
their reach order that proves taps, outside their quiet hours. A reply card
has Send and Skip, an invite Accept and Skip.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# A button's data (Telegram callback_data, a Slack button value, a Discord
# custom_id, a Google Chat button's action parameter): "card:<decision id>:a" (Approve), ":s" (Skip) or ":n"
# (Approve, and on its own next time: the In training tick, shown only on
# a card that can teach, ``allow_label``).
_PREFIX = "card"
APPROVE = "a"
SKIP = "s"
ALLOW = "n"
# A reply card's other sender: switch who it goes from, and show it again
# to send (``send_from``). It sends nothing.
SWITCH = "x"
_CHOICES = (APPROVE, SKIP, ALLOW, SWITCH)
# Each choice in a word, for a tap's audit row.
CHOICE_NAMES = {APPROVE: "approve", SKIP: "skip", ALLOW: "allow", SWITCH: "switch"}
_VERSION = re.compile(r"^[0-9a-f]{8}$")
# The card's two parts, as on the app's cards.
HAPPENED = "What happened"
WILL_DO = "What I'll do"
WHY_IT_WAITS = "Why it waits"
# A reply leaves at most this many cards (action_cards.CARDS_PER_TURN plus a meeting).
MAX_CARDS = 4

# What a second tap gets while the first is still deciding the card; the
# card itself is left for the first tap to update.
BUSY = "Already on it."
# Cards a tap is deciding right now (one process serves a company's taps).
_resolving: set[int] = set()

_CLASS_LABEL = {
    "meeting_scheduling": "Meeting to book",
    "delegation_actions": "Waiting for your approval",
    "app_action": "Waiting for your OK",
    "delegation_reply": "Reply ready to send",
    "delegation_invite": "Meeting invite",
    "take_the_lead_action": "Waiting for your OK",
    "department_action": "Waiting for your approval",
}
# The approve button's label and what the card says once it is done, where
# they aren't Approve and Approved.
_APPROVE_LABEL = {
    "delegation_invite": "Accept", "take_the_lead_action": "Do it", "department_action": "Do it",
}
_DONE_TEXT = {"delegation_reply": "Sent.", "delegation_invite": "Accepted."}
# These cards are shown whole or not at all: a tap sends or does exactly what
# is stored, so a card cut short would approve what it never showed. This fits
# every chat app's message (Discord's 2000 is the smallest).
CARD_MAX = 1800
_WHOLE_CLASSES = frozenset({"delegation_actions", "delegation_reply", "take_the_lead_action", "department_action"})


def telegram_proves_taps() -> bool:
    """Whether a Telegram update can only come from Telegram."""
    from openexecutive.config import get_settings

    s = get_settings()
    return bool(s.telegram_bot_token and s.telegram_webhook_secret and s.telegram_webhook_secret_valid)


def _dm_proves_taps(channel: str) -> bool:
    """Whether a Slack, Discord or Google Chat DM here can show cards with
    buttons."""
    from openexecutive.config import get_settings

    s = get_settings()
    if channel == "slack":
        return bool(s.slack_bot_token and s.slack_card_buttons)
    if channel == "google_chat":
        # The Chat app's webhook: its token says Google sent the event, and
        # a tap counts only once Chat itself confirms it
        # (``google_chat.verify_tap``).
        return bool(s.google_chat_project_number and (
            s.google_chat_service_account_file or s.google_chat_service_account_email
        ))
    return bool(s.discord_bot_token)


def shows_cards(session: Any) -> bool:
    """Whether this turn's chat shows its cards with buttons: a private
    Telegram chat with someone on the roster, on a webhook only Telegram can
    call, a Slack or Discord DM with someone on the roster, or a Google Chat
    DM between the app and someone on the roster alone (the adapter marks
    only that one private). Fails closed."""
    channel = str(getattr(session, "origin_channel", "") or "") if session is not None else ""
    if getattr(session, "caller_person_id", None) is None:
        return False
    try:
        if channel == "telegram":
            ref = str(getattr(session, "origin_channel_ref", "") or "")
            return ref.isdigit() and telegram_proves_taps()
        if channel in ("slack", "discord", "google_chat"):
            from openexecutive.delegation.settings import private_conversation

            return private_conversation(session) and _dm_proves_taps(channel)
    except Exception:
        logger.warning("channel_cards: couldn't read the %s settings", channel, exc_info=True)
    return False


def card_here(session: Any) -> bool:
    """Whether a card left on this turn is shown in the chat it was left in:
    the web chat (and the phone app, which uses it), or a chat app that
    ``shows_cards``."""
    return bool(getattr(session, "from_web_chat", False)) or shows_cards(session)


# What a card shows only in part (``card_text`` lists the first ten, and
# not who is To and who Cc): all of it goes into its version, so a change
# there reads as a change.
_SHOWN_IN_PART = ("draft_to", "draft_spec", "attendee_emails")


def version(decision_id: int, text: str, instance: Any = None) -> str:
    """What card ``decision_id`` says as ``text``, in short, with everyone
    it goes to: a tap carries the version of the card it was on, so it does
    only what that card showed (``tap``). ``instance``: the card as already
    read, else it is read now."""
    from openexecutive.memory.decision_ledger import get_decision_instance

    instance = instance if instance is not None else get_decision_instance(decision_id)
    try:
        payload = json.loads(instance.proposed_payload_json or "{}") if instance is not None else {}
    except ValueError:
        payload = {}
    whole = {k: payload.get(k) for k in _SHOWN_IN_PART} if isinstance(payload, dict) else {}
    return hashlib.sha256(json.dumps([text, whole], sort_keys=True, default=str).encode()).hexdigest()[:8]


def shows(instance: Any, shown: str) -> bool:
    """Whether card ``instance``, as read, still says what the card of
    version ``shown`` said: what is done is then what that card showed."""
    text = card_text(instance.id, instance)
    return text is not None and version(instance.id, text, instance) == shown


def callback_data(decision_id: int, choice: str, shown: str) -> str:
    """A button's data: the card, the choice, and the version of ``shown``,
    the card's words as the button's message shows them."""
    return f"{_PREFIX}:{decision_id}:{choice}:{version(decision_id, shown)}"


@dataclass(frozen=True)
class Choice:
    decision_id: int
    choice: str
    version: str


def choice_of(data: str) -> str:
    """The choice in one of ``card_options``' callback data."""
    return data.split(":")[2]


def parse_choice(data: Any) -> Choice | None:
    """The card, choice and version from a button's callback data, or None."""
    if not isinstance(data, str):
        return None
    parts = data.split(":")
    if len(parts) != 4 or parts[0] != _PREFIX or parts[2] not in _CHOICES:
        return None
    if not parts[1].isdigit() or len(parts[1]) > 12 or not _VERSION.fullmatch(parts[3]):
        return None
    return Choice(int(parts[1]), parts[2], parts[3])


def _when(start: Any, end: Any) -> str:
    try:
        s = datetime.fromisoformat(str(start))
        e = datetime.fromisoformat(str(end))
    except ValueError:
        return ""
    day = s.strftime("%a %d %b").replace(" 0", " ")
    return f"{day}, {s.strftime('%H:%M')}-{e.strftime('%H:%M')}"


def card_text(decision_id: int, instance: Any = None) -> str | None:
    """What the card says, or None when it isn't one a chat shows or isn't
    waiting any more. ``instance``: the card as already read, else it is
    read now."""
    from openexecutive.api.routes.decisions import shown_in_chat
    from openexecutive.memory.decision_ledger import STATUS_PROPOSED, get_decision_instance

    instance = instance if instance is not None else get_decision_instance(decision_id)
    if instance is None or not shown_in_chat(instance):
        return None
    if instance.status != STATUS_PROPOSED:
        return None
    try:
        payload = json.loads(instance.proposed_payload_json or "{}")
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    lines = [_CLASS_LABEL[instance.decision_class]]
    if instance.decision_class == "meeting_scheduling":
        title = str(payload.get("title") or "Meeting").strip()[:200]
        lines.append(title)
        when = _when(payload.get("start"), payload.get("end"))
        if when:
            lines.append(when)
        who = [str(a) for a in payload.get("attendee_emails") or [] if a][:10]
        if who:
            lines.append("With " + ", ".join(who))
    elif instance.decision_class == "delegation_reply":
        from openexecutive.delegation import send_from

        sender = str(payload.get("from_name") or payload.get("from_email") or "").strip()[:120]
        gist = str(payload.get("gist") or "").strip()[:400]
        lines += ["", HAPPENED, gist or (f"{sender} wrote to you." if sender else "An email came in.")]
        to = [str(a) for a in payload.get("draft_to") or [] if a][:10]
        to_whom = ", ".join(to) or sender or "them"
        lines += ["", WILL_DO]
        if send_from.is_from_executive(payload):
            lines.append(f"Send {to_whom} this reply from {send_from.executive_name()}, with you on cc.")
        else:
            lines.append(f"Send {to_whom} this reply from you.")
        subject = str(payload.get("draft_subject") or payload.get("subject") or "").strip()[:200]
        if subject:
            lines.append(subject)
        body = str(payload.get("draft_body") or "").strip()
        if body:
            lines.append("")
            lines.append(body)
    elif instance.decision_class == "delegation_invite":
        lines.append(str(payload.get("title") or "Meeting").strip()[:200])
        when = _when(payload.get("start"), payload.get("end"))
        if when:
            lines.append(when)
        sender = str(payload.get("from_name") or payload.get("from_email") or "").strip()[:120]
        if sender:
            lines.append(f"From {sender}")
        if payload.get("clashes"):
            lines.append("It clashes with something on your calendar.")
    elif instance.decision_class in ("take_the_lead_action", "department_action"):
        # Everything the call says, as the web card shows it before Approve
        # (``card_details.served``: the file by name, rows as lines), with
        # what was said on it put in: what a tap here carries out.
        from openexecutive.api.routes.decisions import as_said
        from openexecutive.orchestrator import card_details

        if not card_details.held_steps(payload):
            return None
        shown = card_details.served(as_said(payload))
        # It opens as on every app (``card_details.top``), read by the
        # person it waits on: who started it and why it waits.
        top = card_details.top(
            as_said(payload), approver_id=instance.approver_person_id, viewer_id=instance.approver_person_id,
        )
        lines.append(str(top["title"] or shown.get("summary") or payload.get("tool") or "").strip()[:400])
        lines.append(top["started"] + ".")
        lines += ["", WHY_IT_WAITS, top["waits"][:300], "", WILL_DO]
        for detail in shown.get("details") or []:
            value = str(detail.get("value") or "")
            if detail.get("href"):
                value = f"{value} ({detail['href']})"
            lines.append(f"{detail['field']}: {value}")
    elif instance.decision_class == "app_action":
        # Written by the server from the change itself (workflow_change.card_text).
        # Never cut: workflow_change stages only a card that fits.
        lines.append(str(payload.get("card") or "").strip())
    else:
        why = str(payload.get("why") or "").strip()[:300]
        if why:
            lines.append(why)
        for action in payload.get("actions") or []:
            if isinstance(action, dict) and action.get("summary"):
                lines.append(f"- {str(action['summary'])}")
                # As the web card shows it: a message's words, the files and
                # who else can open their folder. Quoted line by line, so
                # words the model wrote can't pass for another action.
                said = str(action.get("text") or "").strip()
                if said:
                    lines.extend(f"> {line}" for line in said.splitlines())
    text = "\n".join(lines)
    if instance.decision_class in _WHOLE_CLASSES and len(text) > CARD_MAX:
        # Too long to show whole: it waits in the app instead.
        return None
    return text


# A button's label is cut to this (Slack's limit for a button's text).
LABEL_MAX = 75


def card_options(decision_id: int, shown: str) -> tuple[tuple[str, str], ...]:
    """A card's buttons as (label, callback data) for the card's words
    ``shown`` (``card_text``), the Executive's pick first and
    Skip last, as on the app's card: Do it on a held step, Send from me or
    from the Executive (and the other sender, ``SWITCH``) on a reply,
    Accept on an invite, Approve on the rest; and, on a card that can teach
    (In training, ``allow_label``), the pick and on its own next time."""
    from openexecutive.delegation import send_from
    from openexecutive.memory.decision_ledger import get_decision_instance

    instance = get_decision_instance(decision_id)
    kind = instance.decision_class if instance is not None else ""
    label = _APPROVE_LABEL.get(kind, "Approve")
    other: tuple[str, str] | None = None
    if kind == "delegation_reply" and instance is not None:
        from openexecutive.delegation.inbox import card_payload

        me, executive = "Send from me", f"Send from {send_from.executive_name()}, cc me"
        from_executive = _from_executive(instance)
        label = executive if from_executive else me
        # A follow-up goes from them alone, as on the app's card.
        if card_payload(instance).get("source") != "follow_up":
            other = (me if from_executive else executive, SWITCH)
    options = [(label, APPROVE)]
    if instance is not None and allow_label(instance):
        options.append((f"{label}, and do this on its own next time", ALLOW))
    if other is not None:
        options.append(other)
    options.append(("Skip", SKIP))
    return tuple((text[:LABEL_MAX], callback_data(decision_id, choice, shown)) for text, choice in options)


def _from_executive(instance: Any) -> bool:
    from openexecutive.delegation import send_from
    from openexecutive.delegation.inbox import card_payload

    try:
        return send_from.is_from_executive(card_payload(instance))
    except Exception:
        return False


def allow_label(instance: Any) -> str | None:
    """What approving with "on its own next time" would allow on this
    card, in words, or None when the card can't teach: In training is off
    (``delegation.training`` for a person's own, ``take_the_lead.Lead`` for
    the Executive's), or everything on it is allowed already. Never raises."""
    try:
        payload = json.loads(instance.proposed_payload_json or "{}")
        if not isinstance(payload, dict):
            return None
        kind = instance.decision_class
        person_id = instance.approver_person_id
        if kind == "take_the_lead_action":
            from openexecutive.orchestrator.take_the_lead import TRAINING

            grant = payload.get("allow")
            if payload.get("kind") == TRAINING and isinstance(grant, dict) and grant.get("key"):
                return str(grant.get("label") or "this")
            return None
        if person_id is None:
            return None
        from openexecutive.delegation import training

        if not training.in_training(person_id):
            return None
        if kind == "delegation_reply":
            if not training.can_allow_reply(person_id, payload, training.get(person_id)):
                return None
            return ", ".join(training.reply_goes_to(payload))
        if kind == "delegation_invite":
            sender = str(payload.get("from_email") or "")
            if payload.get("can_answer") is False or not sender or training.allowed_person(person_id, sender):
                return None
            return sender
        if kind == "delegation_actions":
            from openexecutive.delegation.action_cards import _addresses

            addresses = [a for a in _addresses(payload.get("actions") or []) if a]
            if not addresses or training.all_allowed(person_id, addresses) is not None:
                return None
            return ", ".join(sorted(set(addresses)))
        return None
    except Exception:
        logger.warning("channel_cards: couldn't tell whether card %s can teach", getattr(instance, "id", "?"),
                       exc_info=True)
        return None


def outcome_text(approved: bool, error: str | None, decision_class: str = "") -> str:
    if error:
        return f"Not done: {error}"
    return _DONE_TEXT.get(decision_class, "Approved.") if approved else "Skipped."


def outcome_of(line: str) -> str:
    """``done``, ``busy``, ``changed`` (shown again, nothing done) or
    ``refused``, for the audit row of a tap."""
    if line == BUSY:
        return "busy"
    if line == CHANGED:
        return "changed"
    return "refused" if line.startswith("Not done") else "done"


@dataclass(frozen=True)
class Tapped:
    """What a tap on a card's button did. ``line`` goes under the card; a
    card that still waits (``options``) is shown again as ``text`` with
    those buttons, else it keeps what it said, with ``line`` under it."""

    line: str
    text: str = ""
    options: tuple[tuple[str, str], ...] = ()
    # Words said on it carried it out (``say``): it waits no more.
    settled: bool = False


CHANGED = "This card changed. Here it is now."
SWITCHED = "Changed who it goes from. Check it, then send it."


def with_line(shown: str, line: str) -> str:
    """A card's message after a tap: what it shows, then ``line``. A line an
    earlier tap left when it showed the card again gives way to the new
    one, so a card never says two outcomes; and how to say words on it
    (``TELL_ME``) goes once it doesn't wait for them."""
    body = shown or ""
    while earlier := next((e for e in (CHANGED, SWITCHED, TELL_ME) if body == e or body.endswith(f"\n\n{e}")), ""):
        body = body[: -len(earlier)].rstrip("\n")
    return f"{body}\n\n{line}" if body else line


def with_tell_me(decision_id: int, text: str) -> str:
    """A card's words, then how to say words on it (``TELL_ME``) when it
    takes them (``takes_words``): for a chat app that shows the card as
    words alone."""
    return f"{text}\n\n{TELL_ME}" if takes_words(decision_id) else text


async def tap(tapped: Choice, person: Any) -> Tapped:
    """A tap on one of a card's buttons (``card_options``) by ``person``,
    the proven tapper: approve, approve and allow from now on, skip, or
    (a reply) switch who it goes from, which sends nothing and shows the
    card again with its new words. A tap on a card that says something
    else now (switched meanwhile, changed in the app, an older copy of it)
    does nothing but show the card as it is now, so what is done is always
    what the tapped card showed. Skip is a skip, whatever it showed."""
    decision_id, choice = tapped.decision_id, tapped.choice
    # Taken before the card is read: a change saved after it isn't shown.
    read_at = datetime.now(UTC).isoformat()
    answer, shown, executive = await asyncio.to_thread(_look, tapped, person)
    if answer is not None:
        return answer
    if choice == SWITCH:
        return await _switch(decision_id, person, executive=bool(executive))
    line = await resolve(
        decision_id, person, approve=choice != SKIP, allow=choice == ALLOW,
        # It showed the card as it was read.
        shown_at=read_at if shown else None, shown=tapped.version if shown else None,
    )
    return Tapped(line)


def _look(tapped: Choice, person: Any) -> tuple[Tapped | None, bool, bool | None]:
    """The card ``tapped`` is on, read once (off the chat app's loop): an
    answer when the tap is to do nothing, whether it showed the card as it
    is now, and (a Switch) whether the card it showed goes from the
    Executive next. Skip is a skip, whatever it showed."""
    decision_id = tapped.decision_id
    if tapped.choice == SKIP:
        return None, False, None
    # Read before the card: a switch saved after it changes what the card
    # reads, so the version no longer matches and nothing is switched.
    executive = _switches_to_executive(decision_id) if tapped.choice == SWITCH else None
    current = card_text(decision_id)
    if current is None and not _waits(decision_id):
        return None, False, None  # waits no more: answered as handled
    if current is not None and version(decision_id, current) == tapped.version:
        return None, True, executive
    # Answered here, not done: only to its own person, as anywhere else.
    if not _theirs(decision_id, person):
        return Tapped("Not done: this card is no longer there."), False, None
    if current is None:
        # Its words no longer fit a chat: nothing to check the tap against.
        return Tapped(f"Not done: this card changed. Open it in the app: {_app_link(decision_id)}"), False, None
    return Tapped(CHANGED, current, card_options(decision_id, current)), False, None


# Under a card that takes words, where a chat app shows it: how to say them.
TELL_ME = "Or tell me what to change: reply to this card."
_TALKS = frozenset({"delegation_reply", "take_the_lead_action", "department_action"})


def takes_words(decision_id: int) -> bool:
    """Whether card ``decision_id`` takes words said in reply to it in a
    chat app (``say``): a reply, or a held step with fields to change, as
    on its page. Never raises: a card it can't tell about takes none."""
    from openexecutive.api.routes.decisions import said_fields
    from openexecutive.memory.decision_ledger import STATUS_PROPOSED, get_decision_instance

    try:
        instance = get_decision_instance(decision_id)
        if instance is None or instance.status != STATUS_PROPOSED or instance.decision_class not in _TALKS:
            return False
        if instance.decision_class == "delegation_reply":
            return True
        payload = json.loads(instance.proposed_payload_json or "{}")
        return isinstance(payload, dict) and bool(said_fields(payload))
    except Exception:
        logger.warning("channel_cards: couldn't tell whether card %d takes words", decision_id, exc_info=True)
        return False


async def say(said: Choice, person: Any, text: str) -> Tapped | None:
    """Words said in a chat app in reply to a card's message ("Or tell
    me…"), by ``person``, the proven speaker; ``said`` is that message's
    card and version (any of its buttons). As a tap: only on the card as
    that message showed it, and only their own. The card takes them its own
    way, as on its page: a held step rewrites its fields, and a go carries
    it out (``decisions.say_in_chat``); a reply rewrites its draft, or its
    work takes them (``answers.take``), and a go shows it to send. What the
    Executive said back goes under the card, shown again with its buttons
    while it waits. None when the card takes no words here, waits no more or
    isn't theirs: the words are then an ordinary message."""
    decision_id = said.decision_id
    answer, go_on = await asyncio.to_thread(_look_said, said, person)
    if not go_on:
        return answer
    if decision_id in _resolving:
        return Tapped(BUSY)
    _resolving.add(decision_id)
    try:
        line = await _take_words(decision_id, person, text, said.version)
    finally:
        _resolving.discard(decision_id)
    return line if isinstance(line, Tapped) else await asyncio.to_thread(_shown_again, decision_id, line)


def _look_said(said: Choice, person: Any) -> tuple[Tapped | None, bool]:
    decision_id = said.decision_id
    if not takes_words(decision_id) or not _theirs(decision_id, person):
        return None, False
    current = card_text(decision_id)
    if current is None:
        return Tapped(f"Not done: this card changed. Open it in the app: {_app_link(decision_id)}"), False
    if version(decision_id, current) != said.version:
        return Tapped(CHANGED, current, card_options(decision_id, current)), False
    return None, True


async def _take_words(decision_id: int, person: Any, text: str, shown: str) -> str | Tapped:
    """What was said back, or (a held step carried out) the card's outcome."""
    from fastapi import HTTPException

    from openexecutive.api.routes.decisions import say_in_chat
    from openexecutive.delegation.reply_send import SendRefused
    from openexecutive.memory.decision_ledger import get_decision_instance

    try:
        instance = await asyncio.to_thread(get_decision_instance, decision_id)
        if instance is None:
            return Tapped("Not done: this card is no longer there.")
        if instance.decision_class == "delegation_reply":
            return await _said_on_reply(instance, person, text)
        back = await say_in_chat(decision_id, person, text, shown=shown)
    except HTTPException as exc:
        return Tapped(outcome_text(True, exc.detail if isinstance(exc.detail, str) else "it couldn't be done."))
    except SendRefused as refused:
        return Tapped(outcome_text(True, refused.message))
    except Exception:
        logger.exception("channel_cards: words said on card %d failed", decision_id)
        return Tapped(outcome_text(True, "something went wrong. Check Today before trying again."))
    # Carried out: the card keeps what it said, with this under it.
    return Tapped(back.reply, settled=True) if back.approved is not None else back.reply


async def _said_on_reply(instance: Any, person: Any, text: str) -> str | Tapped:
    from openexecutive.delegation import answers, card_work
    from openexecutive.delegation.gmail import normalize_email
    from openexecutive.delegation.inbox import card_payload
    from openexecutive.delegation.verified import ChatTap
    from openexecutive.memory.decision_ledger import get_decision_instance

    def notes(card: Any) -> list[dict[str, Any]]:
        return [t for t in answers.conversation(card_payload(card)) if t.get("kind") == "note"]

    before = len(notes(instance))
    taken = await answers.take(
        instance, person=person, caller=ChatTap(normalize_email(person.email or "")), resolver=person.id, text=text,
    )
    if not taken.started:
        if card_work.running(instance.id):
            return Tapped("Not done: I'm still finishing the last step. Try again in a moment.")
        return Tapped("Not done: this card is no longer there.")
    if taken.worked:
        return f"I'm on it. Follow along in the app: {_app_link(instance.id)}"
    fresh = await asyncio.to_thread(get_decision_instance, instance.id)
    said = notes(fresh) if fresh is not None else []
    if len(said) <= before:
        # Nothing was written for these words: the card changed under them.
        return _shown_again(instance.id, "Not done: this card changed meanwhile. Look again.")
    return str(said[-1].get("text") or "") or "I changed your reply."


def _shown_again(decision_id: int, line: str) -> Tapped:
    text = card_text(decision_id)
    if text is None:
        if _waits(decision_id):
            return Tapped(f"{line}\n\nIt waits for your OK in the app: {_app_link(decision_id)}")
        return Tapped(line)
    return Tapped(line, text, card_options(decision_id, text))


def _switches_to_executive(decision_id: int) -> bool:
    from openexecutive.delegation import send_from
    from openexecutive.delegation.inbox import card_payload
    from openexecutive.memory.decision_ledger import get_decision_instance

    instance = get_decision_instance(decision_id)
    return instance is not None and not send_from.is_from_executive(card_payload(instance))


def _theirs(decision_id: int, person: Any) -> bool:
    from openexecutive.api.routes.decisions import chat_card_is_theirs
    from openexecutive.memory.decision_ledger import get_decision_instance

    instance = get_decision_instance(decision_id)
    return instance is not None and chat_card_is_theirs(instance, person)


async def _switch(decision_id: int, person: Any, *, executive: bool) -> Tapped:
    from openexecutive.delegation import send_from
    from openexecutive.delegation.gmail import normalize_email
    from openexecutive.delegation.reply_send import SendRefused
    from openexecutive.delegation.verified import ChatTap
    from openexecutive.memory.decision_ledger import get_decision_instance

    if decision_id in _resolving:
        return Tapped(BUSY)
    # Held before anything waits, as ``resolve`` holds it.
    _resolving.add(decision_id)
    try:
        instance = await asyncio.to_thread(get_decision_instance, decision_id)
        if instance is None or instance.decision_class != "delegation_reply":
            return Tapped("Not done: this card is no longer there.")
        # As a tap on the web card: its checks, and its rows kept to the person.
        await send_from.switch(
            instance, caller=ChatTap(normalize_email(person.email or "")), resolver=person.id,
            # Who the tapped card said it would go from next, not a new read.
            executive=executive,
        )
    except SendRefused as refused:
        return Tapped(f"Not done: {refused.message}")
    except Exception:
        logger.exception("channel_cards: switching who card %d goes from failed", decision_id)
        return Tapped("Not done: something went wrong. Check Today before trying again.")
    finally:
        _resolving.discard(decision_id)
    return await asyncio.to_thread(_switched, decision_id)


def _switched(decision_id: int) -> Tapped:
    text = card_text(decision_id)
    if text is None:
        return Tapped(f"Changed who it goes from. It waits for your OK in the app: {_app_link(decision_id)}")
    return Tapped(SWITCHED, text, card_options(decision_id, text))


def _waits(decision_id: int) -> bool:
    from openexecutive.memory.decision_ledger import STATUS_PROPOSED, get_decision_instance

    instance = get_decision_instance(decision_id)
    return instance is not None and instance.status == STATUS_PROPOSED


def _app_link(decision_id: int) -> str:
    from openexecutive.config import get_settings

    return f"{get_settings().ui_base_url.rstrip('/')}/cards/{decision_id}"


async def resolve(
    decision_id: int, person: Any, *, approve: bool, allow: bool = False, shown_at: str | None = None,
    shown: str | None = None,
) -> str:
    """Approve or skip a card for ``person``, the proven tapper; ``allow``
    is the "next time" button (approve, and allow it from now on);
    ``shown_at``: when the tapped card showed what it is now, and ``shown``
    its version. Returns the line to show under the card."""
    from fastapi import HTTPException

    from openexecutive.api.routes.decisions import resolve_in_chat
    from openexecutive.memory.decision_ledger import get_decision_instance

    # The buttons stay until the card is edited after this returns, so a
    # second tap can land while the first is still booking: it is turned
    # away rather than racing it to a second invite.
    if decision_id in _resolving:
        return BUSY
    try:
        instance = get_decision_instance(decision_id)
    except Exception:
        logger.exception("channel_cards: reading card %d failed", decision_id)
        return outcome_text(approve, "something went wrong. Check Today before trying again.")
    kind = instance.decision_class if instance is not None else ""
    _resolving.add(decision_id)
    try:
        await resolve_in_chat(decision_id, person, approve=approve, allow=allow, shown_at=shown_at, shown=shown)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else "it couldn't be done."
        return outcome_text(approve, detail)
    except Exception:
        logger.exception("channel_cards: resolving card %d failed", decision_id)
        return outcome_text(approve, "something went wrong. Check Today before trying again.")
    finally:
        _resolving.discard(decision_id)
    return outcome_text(approve, None, kind)


# --------------------------------------------------------------------------- #
# Cards that wait outside any chat, sent to the person's chat app
# --------------------------------------------------------------------------- #

# A card waits this long first, so one handled on the web right away is not
# sent too, and is sent only while it is this young.
PUSH_AFTER = timedelta(minutes=2)
PUSH_WITHIN = timedelta(hours=24)
SCAN_EVERY = timedelta(minutes=1)
# At most this many cards to one person in a pass; the rest go on the next.
PUSH_PER_PASS = 5
# A send that fails (a rate limit, the chat app down) is tried again on the
# next passes, this many times in all.
SEND_TRIES = 3
# A send still marked in progress after this long was cut off (a restart
# mid-send) and counts as failed.
SEND_STALE = timedelta(minutes=5)
# The claims a pass may take again: failed, or cut off mid-send.
_RETRYABLE = "(state = 'failed' OR (state = 'sending' AND pushed_at < ?)) AND tries < ?"

_TABLE = """
    CREATE TABLE IF NOT EXISTS chat_card_pushes (
        decision_id INTEGER PRIMARY KEY,
        person_id INTEGER NOT NULL,
        channel TEXT NOT NULL,
        pushed_at TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'sending',
        tries INTEGER NOT NULL DEFAULT 1
    )
"""

_last_scan_at: datetime | None = None
_task: asyncio.Task[int] | None = None


def _connect(db_path: Path | None) -> sqlite3.Connection:
    from openexecutive.memory import episodic

    conn = sqlite3.connect(str(db_path or episodic.DB_PATH))
    conn.execute(_TABLE)
    return conn


def _pushed(decision_ids: list[int], now: datetime, db_path: Path | None) -> set[int]:
    if not decision_ids:
        return set()
    conn = _connect(db_path)
    try:
        marks = ",".join("?" * len(decision_ids))
        rows = conn.execute(
            f"SELECT decision_id FROM chat_card_pushes WHERE decision_id IN ({marks}) AND NOT ({_RETRYABLE})",
            [*decision_ids, (now - SEND_STALE).isoformat(), SEND_TRIES],
        ).fetchall()
    finally:
        conn.close()
    return {int(r[0]) for r in rows}


def _mark_pushed(decision_id: int, person_id: int, channel: str, now: datetime, db_path: Path | None) -> bool:
    """Claim the card for a send: new, or failed (or cut off) fewer than
    ``SEND_TRIES`` times. False when it is sent, being sent, or out of tries."""
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO chat_card_pushes (decision_id, person_id, channel, pushed_at) VALUES (?,?,?,?)",
            (decision_id, person_id, channel, now.isoformat()),
        )
        if cur.rowcount != 1:
            cur = conn.execute(
                "UPDATE chat_card_pushes SET state = 'sending', tries = tries + 1, person_id = ?, channel = ?,"
                f" pushed_at = ? WHERE decision_id = ? AND {_RETRYABLE}",
                (person_id, channel, now.isoformat(), decision_id, (now - SEND_STALE).isoformat(), SEND_TRIES),
            )
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()


def shown_at(decision_id: int, db_path: Path | None = None) -> str:
    """When card ``decision_id`` was posted to its person's chat app (an ISO
    time), so a tap on it is checked against what it showed
    (``card_drafts.asks_again``). "" when it wasn't, or can't be read."""
    try:
        conn = _connect(db_path)
        try:
            row = conn.execute(
                "SELECT pushed_at FROM chat_card_pushes WHERE decision_id = ? AND state = 'sent'", (decision_id,),
            ).fetchone()
        finally:
            conn.close()
    except Exception:
        logger.warning("channel_cards: couldn't read when card %d was posted", decision_id, exc_info=True)
        return ""
    return str(row[0]) if row else ""


def _mark_sent(decision_id: int, sent: bool, db_path: Path | None) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE chat_card_pushes SET state = ? WHERE decision_id = ?", ("sent" if sent else "failed", decision_id),
        )
        conn.commit()
    finally:
        conn.close()


def chat_for(person: Any) -> str | None:
    """The chat app a card for ``person`` goes to: the first in their reach
    order (``people.reach``) whose taps prove who made them, or None."""
    from openexecutive.people.reach import reach_order

    for channel in reach_order(person, email=False):
        try:
            if channel == "telegram" and str(person.telegram_chat_id or "").isdigit() and telegram_proves_taps():
                return channel
            if channel in ("slack", "discord") and _dm_proves_taps(channel):
                return channel
        except Exception:
            logger.warning("channel_cards: couldn't read the %s settings", channel, exc_info=True)
    return None


def _waiting(now: datetime) -> list[Any]:
    """Open cards made outside any chat that have waited ``PUSH_AFTER`` and
    are younger than ``PUSH_WITHIN``, oldest first."""
    from openexecutive.api.routes.decisions import CHAT_CARD_CLASSES, shown_in_chat
    from openexecutive.memory.decision_ledger import STATUS_PROPOSED, list_instances

    cards = []
    for decision_class in sorted(CHAT_CARD_CLASSES):
        for card in list_instances(decision_class, status=STATUS_PROPOSED, limit=200):
            # A card made on its own person's turn was shown there; a
            # department's is made on the asker's turn, never its approver's.
            shown = card.originating_session_id and decision_class != "department_action"
            if shown or card.gate_mode != "propose" or not shown_in_chat(card):
                continue
            try:
                made = datetime.fromisoformat(card.created_at)
            except ValueError:
                continue
            if made.tzinfo is None:
                made = made.replace(tzinfo=UTC)
            if now - PUSH_WITHIN <= made <= now - PUSH_AFTER:
                cards.append(card)
    return sorted(cards, key=lambda c: c.id)


async def _send(
    channel: str, person: Any, decision_id: int, text: str, options: tuple[tuple[str, str], ...],
) -> None:
    from openexecutive.config import get_settings

    if channel == "telegram":
        from openexecutive.integrations.telegram_bot import send_card

        await send_card(
            str(get_settings().telegram_bot_token or ""), int(person.telegram_chat_id), decision_id, text,
            options=options, raise_on_error=True,
        )
    elif channel == "slack":
        from openexecutive.integrations.slack_bot import send_dm_card

        await send_dm_card(str(person.slack_user_id), decision_id, text, options)
    elif channel == "discord":
        from openexecutive.integrations.discord_bot import send_dm_card as discord_card

        await discord_card(str(person.discord_user_id), decision_id, text, options)


async def push_waiting(now: datetime | None = None, *, db_path: Path | None = None) -> int:
    """Send each card waiting outside any chat to its person's chat app, once.
    Returns how many went. Never raises."""
    from openexecutive.audit import log_event
    from openexecutive.people.channel import is_within_availability
    from openexecutive.people.store import find_principal_person, get_person

    now = now or datetime.now(UTC)
    sent = 0
    try:
        cards = _waiting(now)
        done = _pushed([c.id for c in cards], now, db_path)
        principal = find_principal_person() if any(c.approver_person_id is None for c in cards) else None
    except Exception:
        logger.warning("channel_cards: couldn't read the waiting cards", exc_info=True)
        return 0
    per_person: dict[int, int] = {}
    for card in cards:
        if card.id in done:
            continue
        person = get_person(card.approver_person_id) if card.approver_person_id is not None else principal
        if person is None or person.id is None or person.archived:
            continue
        if per_person.get(person.id, 0) >= PUSH_PER_PASS:
            continue
        channel = chat_for(person)
        if channel is None or not is_within_availability(person, now=now, quiet_hours=True):
            continue
        text = card_text(card.id)
        if not text or not _mark_pushed(card.id, person.id, channel, now, db_path):
            continue
        try:
            await _send(channel, person, card.id, text, card_options(card.id, text))
        except Exception:
            logger.warning("channel_cards: couldn't send card %d to %s", card.id, channel, exc_info=True)
            _settle(card.id, False, db_path)
            continue
        _settle(card.id, True, db_path)
        sent += 1
        per_person[person.id] = per_person.get(person.id, 0) + 1
        try:
            log_event(
                "chat_card_sent", f"Sent card {card.id} to {channel}", actor="executive",
                details={"channel": channel, "decision_id": card.id, "person_id": person.id},
                private=True, private_to_person=person.id,
            )
        except Exception:
            logger.warning("channel_cards: couldn't audit card %d", card.id, exc_info=True)
    return sent


def _settle(decision_id: int, sent: bool, db_path: Path | None) -> None:
    try:
        _mark_sent(decision_id, sent, db_path)
    except Exception:
        logger.warning("channel_cards: couldn't record card %d's send", decision_id, exc_info=True)


def maybe_push(now: datetime) -> bool:
    """Start a pass of :func:`push_waiting` when ``SCAN_EVERY`` has passed and
    none is running. True when one started. Never raises."""
    global _last_scan_at, _task
    try:
        if _task is not None and not _task.done():
            return False
        if _last_scan_at is not None and now - _last_scan_at < SCAN_EVERY:
            return False
        _last_scan_at = now
        _task = asyncio.get_running_loop().create_task(push_waiting(now))
        return True
    except Exception:
        logger.warning("channel_cards: couldn't start a pass", exc_info=True)
        return False


__all__ = [
    "MAX_CARDS",
    "ALLOW",
    "APPROVE",
    "CHOICE_NAMES",
    "SKIP",
    "SWITCH",
    "Tapped",
    "allow_label",
    "callback_data",
    "card_here",
    "card_options",
    "card_text",
    "chat_for",
    "Choice",
    "choice_of",
    "version",
    "with_line",
    "with_tell_me",
    "maybe_push",
    "parse_choice",
    "push_waiting",
    "resolve",
    "say",
    "shows_cards",
    "tap",
    "takes_words",
    "telegram_proves_taps",
]
