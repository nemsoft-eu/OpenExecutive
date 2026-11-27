"""Act as me: where a reply the inbox watcher wrote lives, and changing it
from its card.

Two places, by the person's own choice on the Draft replies to my inbox
card (``InboxWatch.mailbox_drafts``):

- On the card alone (the default). Nothing is saved in their mailbox, so
  their Drafts folder doesn't fill up with replies they may never send. The
  card keeps what a draft would hold (``draft_spec``: who it goes to, the
  subject, the thread headers) and the words (``draft_body``). Send makes
  the draft from exactly that, at that moment, and sends it through the same
  checked path as any other (``reply_send``); a refusal before it went takes
  the draft back out of their mailbox.
- Also in their mailbox's Drafts, as before: they can open it there, and
  Send sends it exactly as it is there.

Either way the words can be changed on the card (``edit``) before Send, by
hand or by answering what it left them to decide (``delegation.answers``,
which rewrites them through ``edit`` and its checks). A
reply in their mailbox is rewritten there too (``update_draft_text``), and
only while it is still the version the card last saw: an edit made in the
mailbox meanwhile is never overwritten. Who it goes to can be changed the
same way (``set_recipients``), between the two lists the watcher fixed when
it wrote the card: the sender alone, or everyone the email went to. No
address comes from the request. Only the card's own person, from a
request the API can tie to them, can change it, and never while it is being
sent on its own. Nothing here sends anything.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# The longest reply that can be typed on a card.
MAX_TEXT_CHARS = 20_000

# Card payload keys for a reply kept on its card alone.
CARD_ONLY = "card_only"
DRAFT_SPEC = "draft_spec"
DRAFT_KEY = "draft_key"
# Set once the person changed the words on the card.
EDITED_ON_CARD = "edited_on_card"
# A reply in their mailbox: its version after the last edit on the card.
EDITED_VERSION = "edited_version"
# Set once the person changed the words by hand: what is left of "Decide
# before sending" no longer holds Send back (``delegation.answers``).
QUESTIONS_SETTLED = "questions_settled"
# When a reply asked for in chat last replaced the card's (``from_chat``). A
# card shown before then sends only once it was shown again (``version``).
CHANGED_IN_CHAT = "changed_in_chat_at"
# The card before the last change they asked for on it (an answer, their
# words in its box, its work): its words and the questions answered then
# (``undo_point``). ``undo`` puts them back. Any other change of the words,
# a switch of who it goes from, or work that changed the files drops it.
UNDO_KEY = "before_change"
NOTE_UNDONE = "I put back the words from before that change."

# Cards whose draft is being made right now in this process, so two taps
# never make two.
_MAKING: set[int] = set()
# Cards whose words are being changed right now in this process: one edit at
# a time, so a second can't rewrite the mailbox draft and then lose the card.
_EDITING: set[int] = set()


def in_mailbox(person_id: int) -> bool:
    """Whether ``person_id``'s replies are also saved in their mailbox's
    Drafts. Unreadable is no: nothing lands in their mailbox unasked."""
    from openexecutive.delegation.inbox import get_watch

    return get_watch(person_id).mailbox_drafts


async def save(person_id: int, client: Any, spec: Any) -> Any:
    """Save ``spec`` in ``person_id``'s Drafts when they keep replies there;
    otherwise nothing is saved and the returned draft has no id (the card
    keeps the reply, ``card_fields``)."""
    from openexecutive.delegation.gmail import CreatedDraft

    if in_mailbox(person_id):
        return await client.create_draft(spec)
    return CreatedDraft(draft_id="", message_id="", thread_id=spec.thread_id or "")


def card_key(message_id: str) -> str:
    """The ``delegation_drafts`` row id of a reply kept on its card, until it
    becomes a real draft (``drafts.rename``)."""
    return f"card-{message_id}"


def card_fields(spec: Any, key: str) -> dict[str, Any]:
    """What a card needs to make the draft itself when it is sent."""
    return {
        CARD_ONLY: True,
        DRAFT_KEY: key,
        DRAFT_SPEC: {
            "to": list(spec.to),
            "cc": list(spec.cc),
            "subject": spec.subject,
            "thread_id": spec.thread_id,
            "in_reply_to": spec.in_reply_to,
            "references": spec.references,
            "from_name": spec.from_name,
            "from_addr": spec.from_addr,
        },
    }


def is_card_only(payload: dict[str, Any]) -> bool:
    """A reply kept on its card that hasn't become a draft yet."""
    return payload.get(CARD_ONLY) is True and not payload.get("draft_id")


def spec_for(payload: dict[str, Any], files: list[tuple[str, str, bytes]]) -> Any:
    """The draft ``payload`` describes, carrying ``files``."""
    from dataclasses import replace

    return replace(_spec(payload), attachments=list(files))


def _spec(payload: dict[str, Any], decision_id: int | None = None) -> Any:
    """The draft a card-only reply makes at Send, with the files the
    Executive left on card ``decision_id`` (``delegation.card_work``)."""
    from openexecutive.delegation.gmail import DraftSpec

    stored = payload.get(DRAFT_SPEC)
    stored = stored if isinstance(stored, dict) else {}

    def text(key: str) -> str | None:
        value = stored.get(key)
        return value if isinstance(value, str) and value else None

    def addresses(key: str) -> list[str]:
        value = stored.get(key)
        return [a for a in value if isinstance(a, str) and a] if isinstance(value, list) else []

    return DraftSpec(
        to=addresses("to"),
        cc=addresses("cc"),
        subject=text("subject") or str(payload.get("draft_subject") or ""),
        body=str(payload.get("draft_body") or ""),
        thread_id=text("thread_id") or str(payload.get("thread_id") or "") or None,
        in_reply_to=text("in_reply_to"),
        references=text("references"),
        from_name=text("from_name") or "",
        from_addr=text("from_addr"),
        attachments=_card_files(decision_id) if decision_id is not None else [],
    )


def _card_files(decision_id: int) -> list[tuple[str, str, bytes]]:
    from openexecutive.delegation.card_work import files_for

    return files_for(decision_id)


async def make_draft(instance: Any, person_id: int, client: Any) -> dict[str, Any]:
    """Send, for a reply kept on its card: save it in the person's Drafts
    exactly as the card shows it, and record that on the card, so the send
    path checks and sends it like any other (and the reconciler can settle
    it). Returns the card's payload as it now is. Raises ``SendRefused``."""
    from openexecutive.delegation import drafts
    from openexecutive.delegation.gmail import GmailError
    from openexecutive.delegation.inbox import card_payload
    from openexecutive.delegation.reply_send import SendRefused
    from openexecutive.memory.decision_ledger import get_decision_instance, update_open_payload

    if instance.id in _MAKING or instance.id in _EDITING:
        raise SendRefused(409, "busy", "Your change to this reply is still being saved. Try again in a moment.")
    _MAKING.add(instance.id)
    try:
        current = get_decision_instance(instance.id)
        payload = card_payload(current) if current is not None else {}
        if current is None or not is_card_only(payload):
            # Made meanwhile (another tap) or no longer waiting.
            raise SendRefused(409, "already_handled", "This reply was already sent or dismissed.")
        spec = _spec(payload, instance.id)
        if not spec.to or not spec.body.strip():
            raise SendRefused(409, "no_recipients", "This reply isn't addressed to anyone, so it wasn't sent.")
        try:
            created = await client.create_draft(spec)
        except GmailError as exc:
            raise SendRefused(
                502, "gmail_error", "Couldn't reach your mailbox to send it. Nothing was sent; try again in a moment.",
            ) from exc
        made = {**payload, "draft_id": created.draft_id, "draft_message_id": created.message_id}
        if not update_open_payload(instance.id, made, expected_json=current.proposed_payload_json):
            await _delete_quietly(client, created.draft_id)
            raise SendRefused(409, "already_handled", "This reply was already sent or dismissed.")
        try:
            drafts.rename(person_id, str(payload.get(DRAFT_KEY) or ""), created.draft_id, created.message_id)
        except Exception:
            logger.warning("delegation.card_drafts: couldn't record the draft", exc_info=True)
        return made
    finally:
        _MAKING.discard(instance.id)


async def take_back(instance: Any, person_id: int, payload: dict[str, Any], client: Any) -> None:
    """A send refused after ``make_draft``: take the draft back out of the
    person's mailbox and keep the reply on its card again. Best-effort; a
    card that closed meanwhile only loses the draft."""
    from openexecutive.delegation import drafts
    from openexecutive.delegation.inbox import card_payload
    from openexecutive.memory.decision_ledger import get_decision_instance, update_open_payload

    draft_id = str(payload.get("draft_id") or "")
    if not draft_id:
        return
    await _delete_quietly(client, draft_id)
    current = get_decision_instance(instance.id)
    if current is None:
        return
    now = card_payload(current)
    if now.get("draft_id") != draft_id:
        return
    back = {**now, "draft_id": "", "draft_message_id": ""}
    if update_open_payload(instance.id, back, expected_json=current.proposed_payload_json):
        try:
            drafts.rename(person_id, draft_id, str(payload.get(DRAFT_KEY) or ""), "")
        except Exception:
            logger.warning("delegation.card_drafts: couldn't record the draft taken back", exc_info=True)


async def _delete_quietly(client: Any, draft_id: str) -> None:
    try:
        await client.delete_draft(draft_id)
    except Exception:
        logger.warning("delegation.card_drafts: couldn't take a draft back out of the mailbox", exc_info=True)


def _check_words(text: str) -> str:
    from openexecutive.delegation.reply_send import SendRefused

    words = text.rstrip()
    if not words.strip():
        raise SendRefused(422, "empty", "Write something before you save it.")
    if len(words) > MAX_TEXT_CHARS:
        raise SendRefused(422, "too_long", f"That's too long to send from here (at most {MAX_TEXT_CHARS:,} characters).")
    return words


# Rewrites the words for ``edit``: given the person and the card's payload,
# the new words and any payload fields to keep with them.
Rewrite = Callable[[Any, dict[str, Any]], Awaitable[tuple[str, dict[str, Any]]]]


async def set_recipients(
    instance: Any, *, caller: Any, resolver: int | None, everyone: bool, gmail: Any = None,
) -> dict[str, Any]:
    """Send the reply on ``instance`` to the sender alone, or to everyone the
    email went to (``inbox.REPLY_OTHERS``), on the person's tap: on the card,
    and in their mailbox when it is there too. The words stay. Returns the
    card's payload as it now is; raises ``SendRefused``. Sends nothing."""
    return await edit(instance, caller=caller, resolver=resolver, gmail=gmail, everyone=everyone)


def recipients_for(payload: dict[str, Any], everyone: bool) -> tuple[list[str], list[str]]:
    """To and Cc for the Reply to choice: the sender, plus everyone else the
    email went to when ``everyone``. Both lists come from the card itself."""
    from openexecutive.delegation.inbox import REPLY_OTHERS
    from openexecutive.delegation.reply_send import SendRefused

    sender = str(payload.get("from_email") or "")
    others = payload.get(REPLY_OTHERS)
    others = [a for a in others if isinstance(a, str) and a and a != sender] if isinstance(others, list) else []
    if not sender or not others or payload.get("source") == "follow_up":
        raise SendRefused(409, "no_choice", "This reply can only go to the people it's addressed to.")
    return [sender], (others if everyone else [])


def owner(instance: Any, *, caller: Any, resolver: int | None) -> Any:
    """The person whose reply card ``instance`` is, when ``resolver`` is
    them and the signed caller is them (``reply_send._check_caller``, the
    rule Send uses). Raises ``SendRefused``. Every change to a card, by
    hand, by rewrite or by its work, and Stop on its work, passes this."""
    from openexecutive.audit import rows_for_person
    from openexecutive.delegation.gmail import normalize_email
    from openexecutive.delegation.inbox import card_payload
    from openexecutive.delegation.reply_send import _NOT_YOURS, SendRefused, _check_caller
    from openexecutive.delegation.settings import can_delegate
    from openexecutive.people.store import get_person

    payload = card_payload(instance)
    person = get_person(int(payload.get("person_id") or 0))
    if (
        person is None or person.id is None or not person.email or not can_delegate(person)
        or instance.approver_person_id != person.id or resolver != person.id
    ):
        raise SendRefused(403, "not_yours", _NOT_YOURS)
    with rows_for_person(person.id):
        _check_caller(caller, normalize_email(person.email))
    return person


def check(instance: Any, *, caller: Any, resolver: int | None) -> Any:
    """``owner``, and the card is still open to change: not sent, not being
    sent on its own, As you on. Returns the person; raises ``SendRefused``.
    A change by hand, a rewrite and new words for the card's work all pass
    this one place."""
    from openexecutive.delegation.inbox import SENDING
    from openexecutive.delegation.reply_send import SendRefused
    from openexecutive.delegation.settings import is_enabled
    from openexecutive.memory.decision_ledger import STATUS_PROPOSED

    person = owner(instance, caller=caller, resolver=resolver)
    if instance.status != STATUS_PROPOSED or instance.id in SENDING:
        raise SendRefused(409, "already_handled", "This reply was already sent or dismissed.")
    if getattr(instance, "gate_mode", "") == "auto_execute":
        raise SendRefused(409, "already_handled", "This reply is being sent on its own right now.")
    if not is_enabled(person.id):
        raise SendRefused(409, "inbox_off", "Turn on As you to change this reply here.")
    return person


async def edit(
    instance: Any, *, caller: Any, resolver: int | None, text: str = "", gmail: Any = None,
    rewrite: Rewrite | None = None, everyone: bool | None = None,
) -> dict[str, Any]:
    """Change the words of the reply on ``instance`` to ``text``, on the
    person's tap. Returns the card's payload as it now is; raises
    ``SendRefused``. Sends nothing.

    With ``rewrite`` (answering its open questions, ``delegation.answers``)
    the words come from it instead, once every check has passed and while
    the card is held for this edit. A change by hand settles the card's open
    questions (``QUESTIONS_SETTLED``): the person wrote what goes. With
    ``everyone`` the words stay and who it goes to changes (``set_recipients``)."""
    from openexecutive.audit import rows_for_person
    from openexecutive.delegation.inbox import card_payload
    from openexecutive.delegation.reply_send import SendRefused

    payload = card_payload(instance)
    person = check(instance, caller=caller, resolver=resolver)
    with rows_for_person(person.id):
        recipients = recipients_for(payload, everyone) if everyone is not None else None
        if recipients is not None:
            words = str(payload.get("draft_body") or "")
        else:
            words = _check_words(text) if rewrite is None else ""
        if instance.id in _EDITING or instance.id in _MAKING:
            raise SendRefused(409, "busy", "This reply is being saved or sent right now. Try again in a moment.")
        _EDITING.add(instance.id)
        try:
            # Words changed by hand: the card's summary no longer says what
            # the draft says, so the card falls back to what they wrote.
            extra: dict[str, Any] = {} if recipients is not None else {QUESTIONS_SETTLED: True, "gist": ""}
            if recipients is not None and can_undo(payload):
                # Who it goes to only: the words' change can still be taken back.
                extra[UNDO_KEY] = payload[UNDO_KEY]
            if rewrite is not None:
                # The card's own words, rewritten for whoever sends them.
                written, extra = await rewrite(person, payload)
                words = _check_words(written)
                before = str(payload.get("draft_body") or "")
                if words != before:
                    # Their change can be taken back (``undo``). A question
                    # answered after it can't: Undo shows under the last
                    # note, so it only ever takes back what that note says.
                    extra[UNDO_KEY] = undo_point(payload)
            return await _edit(instance, person, payload, words, gmail, extra, recipients)
        finally:
            _EDITING.discard(instance.id)


async def _remade(
    instance: Any, person: Any, payload: dict[str, Any], changed: dict[str, Any], gmail: Any,
    recipients: tuple[list[str], list[str]] | None,
) -> dict[str, Any]:
    """``changed`` saved, with the draft in their mailbox made again from it:
    its words, who it goes to and the card's files, the old draft taken out.
    Raises ``SendRefused``."""
    from dataclasses import replace

    from openexecutive.delegation import drafts
    from openexecutive.delegation.card_work import files_for
    from openexecutive.delegation.gmail import (
        BLOCKING_CODES,
        STATUS_MESSAGES,
        GmailError,
        gmail_for,
        gmail_status,
        normalize_email,
    )
    from openexecutive.delegation.inbox import CLOSED, _audit, _close_card
    from openexecutive.delegation.reply_send import SendRefused
    from openexecutive.memory.decision_ledger import update_open_payload

    old = str(payload.get("draft_id") or "")
    email = normalize_email(person.email)
    client = gmail if gmail is not None else gmail_for(email)
    status = await gmail_status(email, gmail=client)
    if status != "connected":
        raise SendRefused(409, BLOCKING_CODES.get(status, "gmail_error"), STATUS_MESSAGES[status])
    # The same checks as a change to it (``_edit``): never over a draft they
    # changed, sent or deleted in their mailbox meanwhile.
    try:
        there = await client.get_draft(old)
    except GmailError as exc:
        raise SendRefused(502, "gmail_error", "Couldn't save your change in your mailbox. Try again in a moment.") from exc
    if there is None:
        _close_card(person.id, instance.id, str(payload.get("message_id") or ""), "draft_gone", CLOSED)
        raise SendRefused(409, "draft_gone", "That draft isn't in your mailbox any more: it was sent or deleted there.")
    if there.message.id != str(payload.get(EDITED_VERSION) or payload.get("draft_message_id") or ""):
        raise SendRefused(
            409, "draft_changed",
            "This draft was changed in your mailbox, so it wasn't changed here. Change it there, or dismiss it.",
        )
    spec = spec_for(changed, files_for(instance.id))
    if recipients is not None:
        spec = replace(spec, to=list(recipients[0]), cc=list(recipients[1]))
    try:
        created = await client.create_draft(spec)
    except GmailError as exc:
        raise SendRefused(502, "gmail_error", "Couldn't save your change in your mailbox. Try again in a moment.") from exc
    changed["draft_id"], changed["draft_message_id"] = created.draft_id, created.message_id
    changed.pop(EDITED_VERSION, None)
    if not update_open_payload(instance.id, changed, expected_json=instance.proposed_payload_json):
        await _delete_quietly(client, created.draft_id)
        raise SendRefused(409, "already_handled", "This reply changed or was sent meanwhile. Look at it again.")
    await _delete_quietly(client, old)
    try:
        drafts.rename(person.id, old, created.draft_id, created.message_id)
    except Exception:
        logger.warning("delegation.card_drafts: couldn't record the new draft", exc_info=True)
    _audit("delegation_reply_edited", f"Person {person.id} changed a reply on its card", {
        "person_id": person.id, "decision_id": instance.id, "in_mailbox": True,
    })
    return changed


def undo_point(payload: dict[str, Any]) -> dict[str, Any]:
    """What ``undo`` puts back: the card's words, the questions answered and
    whether they were settled, as they are in ``payload`` before a change."""
    from openexecutive.delegation.answers import ANSWERED_KEY

    answered = payload.get(ANSWERED_KEY)
    return {
        "draft_body": str(payload.get("draft_body") or ""),
        ANSWERED_KEY: dict(answered) if isinstance(answered, dict) else {},
        QUESTIONS_SETTLED: payload.get(QUESTIONS_SETTLED) is True,
    }


def can_undo(payload: dict[str, Any]) -> bool:
    point = payload.get(UNDO_KEY)
    return isinstance(point, dict) and bool(str(point.get("draft_body") or "").strip())


async def undo(instance: Any, *, caller: Any, resolver: int | None, gmail: Any = None) -> dict[str, Any]:
    """Take back the last change they asked for on the card: its words and
    the questions answered as they were before it (``UNDO_KEY``), saved like
    any change, also in their mailbox's draft, with a line
    in the card's conversation. The same checks as ``edit``. Returns the
    card's payload as it now is; raises ``SendRefused``. Sends nothing."""
    from openexecutive.audit import rows_for_person
    from openexecutive.delegation.answers import (
        ANSWERED_KEY,
        CONVERSATION_KEY,
        CONVERSATION_MAX,
        conversation,
    )
    from openexecutive.delegation.inbox import card_payload
    from openexecutive.delegation.reply_send import SendRefused

    payload = card_payload(instance)
    person = check(instance, caller=caller, resolver=resolver)
    if not can_undo(payload):
        raise SendRefused(409, "nothing_to_undo", "There's no change to take back. Look at it again.")
    with rows_for_person(person.id):
        if instance.id in _EDITING or instance.id in _MAKING:
            raise SendRefused(409, "busy", "This reply is being saved or sent right now. Try again in a moment.")
        _EDITING.add(instance.id)
        try:
            turns = [*conversation(payload), {
                "kind": "note", "text": NOTE_UNDONE, "at": datetime.now(UTC).isoformat(),
            }]
            point = payload[UNDO_KEY]
            extra = {
                CONVERSATION_KEY: turns[-CONVERSATION_MAX:], "gist": "", ANSWERED_KEY: point[ANSWERED_KEY],
                QUESTIONS_SETTLED: point[QUESTIONS_SETTLED],
            }
            return await _edit(
                instance, person, payload, _check_words(str(point["draft_body"])), gmail, extra, by_hand=False,
            )
        finally:
            _EDITING.discard(instance.id)


async def redraft(instance: Any, person: Any, gmail: Any, rewrite: Rewrite) -> dict[str, Any]:
    """Rewrite the reply on ``instance`` on the watcher's own turn, not a
    tap: a newer email in its conversation (``answers.catch_up``). The same
    card checks as ``edit`` except the caller's, since nobody tapped; the
    words stay the card's until Send. Returns the payload as it now is;
    raises ``SendRefused``. Sends nothing."""
    from openexecutive.delegation.inbox import SENDING, card_payload
    from openexecutive.delegation.reply_send import SendRefused
    from openexecutive.delegation.settings import is_enabled
    from openexecutive.memory.decision_ledger import STATUS_PROPOSED

    payload = card_payload(instance)
    if instance.status != STATUS_PROPOSED or instance.id in SENDING:
        raise SendRefused(409, "already_handled", "This reply was already sent or dismissed.")
    if getattr(instance, "gate_mode", "") == "auto_execute" or not is_enabled(person.id):
        raise SendRefused(409, "already_handled", "This reply isn't waiting on its card.")
    if instance.id in _EDITING or instance.id in _MAKING:
        raise SendRefused(409, "busy", "This reply is being saved or sent right now.")
    _EDITING.add(instance.id)
    try:
        written, extra = await rewrite(person, payload)
        return await _edit(instance, person, payload, _check_words(written), gmail, extra, by_hand=False)
    finally:
        _EDITING.discard(instance.id)


def version(payload: dict[str, Any]) -> str:
    """The card's version as the person last saw it changed without them:
    caught up with newer mail (``answers.CAUGHT_UP_KEY``), changed from
    chat (``CHANGED_IN_CHAT``) and switched to or from the Executive
    (``send_from``). The card hands it out as ``caught_up_at``
    and Send gets it back as ``seen_caught_up_at``."""
    from openexecutive.delegation.answers import CAUGHT_UP_KEY

    caught_up = str(payload.get(CAUGHT_UP_KEY) or "")
    chat = changed_since_shown(payload)
    return f"{caught_up}|{chat}" if chat else caught_up


def changed_since_shown(payload: dict[str, Any]) -> str:
    """The part of ``version`` a card shown before must have seen: changed
    from chat (``CHANGED_IN_CHAT``) and who it goes from switched, either
    way (``send_from.SWITCHED_AT`` / ``SWITCHED_BACK_AT``). "" when none
    happened."""
    from openexecutive.delegation.send_from import SWITCHED_AT, SWITCHED_BACK_AT

    chat = str(payload.get(CHANGED_IN_CHAT) or "")
    to = str(payload.get(SWITCHED_AT) or "")
    back = str(payload.get(SWITCHED_BACK_AT) or "")
    return f"{chat}+{to}+{back}" if to or back else chat


def _shows_now(payload: dict[str, Any], confirm: dict[str, Any], shown_at: str) -> bool:
    """Whether the card that was tapped shows what goes now. A web or phone
    card hands back the version it showed (``seen_caught_up_at``). A chat
    app's card is shown once, when it is posted (``shown_at``): it shows
    what goes when nothing was changed from chat, nor who it goes from
    switched, since."""
    from openexecutive.delegation.send_from import SWITCHED_AT, SWITCHED_BACK_AT

    changed = changed_since_shown(payload)
    if not changed or str(confirm.get("seen_caught_up_at") or "").partition("|")[2] == changed:
        return True
    stamps = [str(payload.get(k) or "") for k in (CHANGED_IN_CHAT, SWITCHED_AT, SWITCHED_BACK_AT)]
    return bool(shown_at) and all(s <= shown_at for s in stamps if s)


def asks_again(
    payload: dict[str, Any], confirm: dict[str, Any], *, moved_on: bool, shown_at: str = "",
) -> list[str]:
    """Why a Send must be shown again before it goes, whoever's mailbox it
    goes from: newer mail from someone else (``moved_on``) or a reply brought
    up to date with it (``answers.CAUGHT_UP_KEY``) that the tapped card
    didn't show, unless they said yes to it (``thread_moved_on``); and a
    change made without them (from chat, or who it goes from) that it
    didn't show (``_shows_now``)."""
    from openexecutive.delegation.answers import CAUGHT_UP_KEY

    reasons: list[str] = []
    said_yes = confirm.get("thread_moved_on") is True
    caught_up = str(payload.get(CAUGHT_UP_KEY) or "")
    seen_caught_up = str(confirm.get("seen_caught_up_at") or "").partition("|")[0]
    if not said_yes and (moved_on or (caught_up and seen_caught_up != caught_up)):
        reasons.append("thread_moved_on")
    if not _shows_now(payload, confirm, shown_at):
        reasons.append("changed_in_chat")
    return reasons


async def from_chat(
    instance: Any, person: Any, gmail: Any, *, words: str, to: list[str], cc: list[str],
) -> dict[str, Any]:
    """Put a reply the person asked for in chat (``ghostwrite_email``) onto
    the reply card already waiting for that email, in place of a second
    draft: its words, and who it goes to. The same card checks as
    ``redraft``; the card's open questions are settled, since the person
    said what goes. Stamped ``CHANGED_IN_CHAT``, so a card shown before
    (a chat app's, a page not reloaded) asks again at Send, and it doesn't go
    on its own. Returns the payload as it now is; raises ``SendRefused``.
    Sends nothing."""
    from openexecutive.delegation import disclosure as ai_disclosure
    from openexecutive.delegation import send_from
    from openexecutive.delegation.answers import CONVERSATION_KEY, CONVERSATION_MAX, conversation
    from openexecutive.delegation.inbox import SENDING, card_payload
    from openexecutive.delegation.reply_send import SendRefused
    from openexecutive.delegation.settings import is_enabled
    from openexecutive.memory.decision_ledger import STATUS_PROPOSED

    payload = card_payload(instance)
    if instance.approver_person_id != person.id or payload.get("person_id") != person.id:
        raise SendRefused(403, "not_yours", "That card isn't this person's.")
    if instance.status != STATUS_PROPOSED or instance.id in SENDING:
        raise SendRefused(409, "already_handled", "This reply was already sent or dismissed.")
    if getattr(instance, "gate_mode", "") == "auto_execute" or not is_enabled(person.id):
        raise SendRefused(409, "already_handled", "This reply isn't waiting on its card.")
    if instance.id in _EDITING or instance.id in _MAKING:
        raise SendRefused(409, "busy", "This reply is being saved or sent right now.")
    _EDITING.add(instance.id)
    try:
        signed = ai_disclosure.sign(person.id, words, on_its_own=False)
        # What they asked for in chat goes as them: the card goes from them again.
        now = datetime.now(UTC).isoformat()
        extra: dict[str, Any] = {
            QUESTIONS_SETTLED: True, "gist": "", CHANGED_IN_CHAT: now, send_from.FROM_EXECUTIVE: False,
        }
        if send_from.is_from_executive(payload):
            # Said on the card, as a switch on the card is.
            turns = [*conversation(payload), {"kind": "note", "text": send_from.NOTE_TO_YOU, "at": now}]
            extra.update({CONVERSATION_KEY: turns[-CONVERSATION_MAX:], send_from.SWITCHED_BACK_AT: now})
        return await _edit(
            instance, person, payload, _check_words(signed), gmail, extra, (list(to), list(cc)), by_hand=False,
        )
    finally:
        _EDITING.discard(instance.id)


async def _edit(
    instance: Any, person: Any, payload: dict[str, Any], words: str, gmail: Any, extra: dict[str, Any],
    recipients: tuple[list[str], list[str]] | None = None, *, by_hand: bool = True,
) -> dict[str, Any]:
    """``edit`` once its checks passed, one at a time per card. Not
    ``by_hand`` (``redraft``): the words stay the Executive's, so Send and
    training don't take them for the person's change. ``words`` are the
    card's one version, for whoever sends it (``send_from``). A draft in
    their mailbox is brought up to date only while it goes from them: from
    the Executive it never goes from there. ``UNDO_KEY`` is kept only when
    ``extra`` sets it (a change they asked for): any other change of the
    words can't be taken back to before it."""
    from openexecutive.delegation import send_from
    from openexecutive.delegation.gmail import (
        BLOCKING_CODES,
        STATUS_MESSAGES,
        DraftChanged,
        GmailError,
        GmailNotFound,
        gmail_for,
        gmail_status,
        normalize_email,
    )
    from openexecutive.delegation.inbox import CLOSED, _audit, _close_card
    from openexecutive.delegation.reply_send import SendRefused
    from openexecutive.memory.decision_ledger import update_open_payload

    changed = {**payload, **extra, "draft_body": words, **({EDITED_ON_CARD: True} if by_hand else {})}
    if UNDO_KEY not in extra:
        changed.pop(UNDO_KEY, None)
    if recipients is not None:
        to, cc = recipients
        changed["draft_to"] = [*to, *cc]
        spec = payload.get(DRAFT_SPEC)
        if isinstance(spec, dict):
            changed[DRAFT_SPEC] = {**spec, "to": to, "cc": cc}
    # Their mailbox's draft goes from them: it never holds the Executive's words.
    draft_id = "" if send_from.is_from_executive(changed) else str(payload.get("draft_id") or "")
    if draft_id and send_from.is_from_executive(payload):
        # From them again: the draft waited unchanged while it went from the
        # Executive, so it is made again from the card, words and files.
        return await _remade(instance, person, payload, changed, gmail, recipients)
    if draft_id:
        email = normalize_email(person.email)
        client = gmail if gmail is not None else gmail_for(email)
        status = await gmail_status(email, gmail=client)
        if status != "connected":
            raise SendRefused(409, BLOCKING_CODES.get(status, "gmail_error"), STATUS_MESSAGES[status])
        seen = str(payload.get(EDITED_VERSION) or payload.get("draft_message_id") or "")
        try:
            if recipients is not None:
                version = await client.update_draft_text(draft_id, seen, words, to=recipients[0], cc=recipients[1])
            else:
                version = await client.update_draft_text(draft_id, seen, words)
        except GmailNotFound:
            _close_card(person.id, instance.id, str(payload.get("message_id") or ""), "draft_gone", CLOSED)
            raise SendRefused(
                409, "draft_gone", "That draft isn't in your mailbox any more: it was sent or deleted there.",
            ) from None
        except DraftChanged as exc:
            if exc.reason == "not_plain":
                raise SendRefused(
                    409, "draft_not_plain", "This draft has files or formatting, so change it in your mailbox.",
                ) from None
            raise SendRefused(
                409, "draft_changed",
                "This draft was changed in your mailbox, so it wasn't changed here. Change it there, or dismiss it.",
            ) from None
        except GmailError as exc:
            raise SendRefused(
                502, "gmail_error", "Couldn't save your change in your mailbox. Try again in a moment.",
            ) from exc
        changed[EDITED_VERSION] = version
        if not by_hand and not payload.get(EDITED_ON_CARD):
            # Still the Executive's own draft, however many times it caught
            # up: the mailbox's new version is the one Send compares with to
            # tell whether the person changed it.
            changed["draft_message_id"] = version
    if not update_open_payload(instance.id, changed, expected_json=instance.proposed_payload_json):
        raise SendRefused(409, "already_handled", "This reply changed or was sent meanwhile. Look at it again.")
    _audit("delegation_reply_edited", f"Person {person.id} changed a reply on its card", {
        "person_id": person.id, "decision_id": instance.id, "in_mailbox": bool(draft_id),
        **({"recipients": len(changed["draft_to"])} if recipients is not None else {}),
    })
    return changed
