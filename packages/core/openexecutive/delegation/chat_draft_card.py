"""Act as me: the card under a chat reply that holds an email written as you
(``ghostwrite_email``), read and changed in place.

A chat draft waits in the person's own Drafts until they send it. Its card
shows it as it is there now (``state``), and changes it right there, on the
person's tap:

- by hand (``set_text``): their own words replace the draft's;
- by asking (``change``): what to change, in their words or a tone pill
  ("Make it softer"), rewritten by the same model and checks as a reply
  card's change (``delegation.answers.rewrite``).

Either way the draft in their mailbox is rewritten in place
(``update_draft_text``), only while it is still the version the card last
saw: an edit they made in their mailbox meanwhile is never overwritten, and
a draft with files or formatting is left for them to change there. The row
then holds the new version, so Send under the chat sends what the card
shows. Who it goes to and its subject never change here. Only the draft's
own person, from a request the API can tie to them, can change it. Nothing
here sends anything.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# The card's state of a draft (``state``).
OPEN = "open"            # waiting in their Drafts, as the card last saw it
CHANGED = "changed"      # changed in their mailbox since: theirs to send there
SENT = "sent"
REPLACED = "replaced"    # retired for a newer version (``replaced_by``)
GONE = "gone"            # no longer in their Drafts (sent or deleted there)

# Rows being changed right now, in this process: one change at a time per draft.
_CHANGING: set[int] = set()


def _row(person: Any, row_id: int) -> Any:
    from openexecutive.delegation import drafts
    from openexecutive.delegation.reply_send import _NOT_YOURS, SendRefused
    from openexecutive.delegation.settings import can_delegate

    if person is None or person.id is None or not person.email or not can_delegate(person):
        raise SendRefused(403, "not_yours", _NOT_YOURS)
    row = drafts.get_row(row_id)
    if row is None or row.person_id != person.id or row.source != drafts.SOURCE_CHAT:
        # Someone else's draft reads the same as none at all.
        raise SendRefused(404, "draft_not_found", "There's no draft of yours with that id.")
    return row


def _closed(row: Any) -> dict[str, Any] | None:
    """The state of a draft that isn't waiting any more, from the database
    alone; None while it waits."""
    if row.sent_message_id:
        return {"status": SENT}
    if row.gone_at:
        return {"status": REPLACED if row.replaced_by else GONE}
    return None


async def _client(person: Any, gmail: Any) -> Any:
    from openexecutive.delegation.gmail import (
        BLOCKING_CODES,
        STATUS_MESSAGES,
        gmail_for,
        gmail_status,
        normalize_email,
    )
    from openexecutive.delegation.reply_send import SendRefused

    email = normalize_email(person.email)
    client = gmail if gmail is not None else gmail_for(email)
    status = await gmail_status(email, gmail=client)
    if status != "connected":
        raise SendRefused(409, BLOCKING_CODES.get(status, "gmail_error"), STATUS_MESSAGES[status])
    return client


def _shown(row: Any, draft: Any, files: list[str] | None = None) -> dict[str, Any]:
    from openexecutive.delegation.send_from import executive_name

    message = draft.message
    if message.id != row.message_id:
        return {"status": CHANGED}
    return {
        "status": OPEN,
        "to": list(message.to),
        "cc": list(message.cc),
        "subject": message.subject,
        "body": message.text,
        # The names of the files attached to it, so the card shows them.
        "files": list(files or []),
        # Who it goes from: the person, or (switched on the card) the Executive.
        "from_executive": row.from_executive,
        # The version shown: Send sends it back, so a card shown before a
        # change (on another screen) sends nothing it didn't show.
        "version": message.id,
        "executive_name": executive_name(),
    }


async def _file_names(client: Any, message: Any) -> list[str]:
    """The names of the files attached to the draft: Gmail lists them with
    the message, Outlook only when asked. [] when they can't be read: the
    card still shows the draft."""
    from openexecutive.delegation.ghostwriter import one_line

    try:
        attached = list(message.attachments)
        if not attached and message.has_attachments:
            attached = list(await client.list_attachments(message.id))
        return [one_line(a.name, 120) or "attachment" for a in attached]
    except Exception:
        logger.warning("chat draft card: couldn't list the draft's files", exc_info=True)
        return []


async def _read(row: Any, client: Any, now: datetime) -> dict[str, Any]:
    from openexecutive.delegation import drafts
    from openexecutive.delegation.gmail import GmailError
    from openexecutive.delegation.reply_send import SendRefused

    try:
        draft = await client.get_draft(row.draft_id)
    except GmailError as exc:
        raise SendRefused(502, "gmail_error", "Couldn't read the draft in your mailbox. Try again in a moment.") from exc
    if draft is None:
        drafts.mark_gone(row.id, now)
        fresh = drafts.get_row(row.id)
        return {"status": REPLACED if fresh is not None and fresh.replaced_by else GONE}
    if draft.message.id != row.message_id:
        return _shown(row, draft)
    return _shown(row, draft, await _file_names(client, draft.message))


async def state(
    person: Any, row_id: int, *, caller: Any, gmail: Any = None, now: datetime | None = None,
) -> dict[str, Any]:
    """The draft ``row_id`` as its card shows it: ``status`` (``OPEN``,
    ``CHANGED``, ``SENT``, ``REPLACED`` or ``GONE``), and while it waits as
    the card last saw it, who it goes to, its subject and its words, read
    from their mailbox. Raises ``SendRefused``. Changes nothing but marking
    a draft found missing. Like a change, only for a caller the API can tie
    to the draft's person: it reads their own mail."""
    from openexecutive.audit import rows_for_person
    from openexecutive.delegation.gmail import normalize_email
    from openexecutive.delegation.reply_send import _check_caller

    row = _row(person, row_id)
    with rows_for_person(person.id):
        _check_caller(caller, normalize_email(person.email))
        closed = _closed(row)
        if closed is not None:
            return closed
        return await _read(row, await _client(person, gmail), now or datetime.now(UTC))


async def _save(
    person: Any, row_id: int, *, caller: Any, gmail: Any, now: datetime | None, words_for: Any,
) -> tuple[dict[str, Any], str]:
    """Rewrite the draft with the words ``words_for(person, current)``
    returns (and a line on what changed, and who it goes from now: True for
    the Executive, False for the person, None for no change), once every
    check passed and while the draft is held for it. Returns its state and
    that line."""
    from openexecutive.audit import rows_for_person
    from openexecutive.delegation import drafts
    from openexecutive.delegation.card_drafts import _check_words
    from openexecutive.delegation.gmail import (
        DraftChanged,
        GmailError,
        GmailNotFound,
        normalize_email,
    )
    from openexecutive.delegation.inbox import _audit
    from openexecutive.delegation.reply_send import _CHAT_SENDING, SendRefused, _check_caller
    from openexecutive.delegation.settings import is_enabled

    now = now or datetime.now(UTC)
    row = _row(person, row_id)
    with rows_for_person(person.id):
        _check_caller(caller, normalize_email(person.email))
        if not is_enabled(person.id):
            raise SendRefused(409, "act_as_me_off", "Turn on As you to change it here, or change it in your Drafts.")
        closed = _closed(row)
        if closed is not None:
            raise SendRefused(409, "draft_gone", "That draft isn't in your Drafts any more.", state=closed)
        if row.id in _CHANGING or row.id in _CHAT_SENDING:
            raise SendRefused(409, "busy", "This draft is being changed or sent right now. Try again in a moment.")
        _CHANGING.add(row.id)
        try:
            client = await _client(person, gmail)
            current = await _read(row, client, now)
            if current["status"] == CHANGED:
                raise SendRefused(
                    409, "draft_changed", "This draft was changed in your mailbox, so change it there.", state=current,
                )
            if current["status"] != OPEN:
                raise SendRefused(409, "draft_gone", "That draft isn't in your Drafts any more.", state=current)
            written, note, executive = await words_for(person, current)
            if executive == current.get("from_executive"):
                executive = None
            if written == str(current.get("body") or "") and executive is None:
                # Nothing to change (a question, answered in the note): the
                # draft in their mailbox stays as it is.
                return {**current}, note
            words = _check_words(written)
            try:
                version = await client.update_draft_text(row.draft_id, row.message_id, words)
            except GmailNotFound:
                drafts.mark_gone(row.id, now)
                raise SendRefused(409, "draft_gone", "That draft isn't in your Drafts any more.") from None
            except DraftChanged as exc:
                if exc.reason == "not_plain":
                    raise SendRefused(
                        409, "draft_not_plain", "This draft has files or formatting, so change it in your Drafts.",
                    ) from None
                raise SendRefused(
                    409, "draft_changed", "This draft was changed in your mailbox, so change it there.",
                ) from None
            except GmailError as exc:
                raise SendRefused(
                    502, "gmail_error", "Couldn't save the change in your mailbox. Try again in a moment.",
                ) from exc
            if version == row.message_id:
                # The mailbox answered with the version it already had: the
                # edit didn't land, and Send would send the old words.
                raise SendRefused(
                    502, "gmail_error", "Couldn't save the change in your mailbox. Try again in a moment.",
                )
            if not drafts.set_version(row.id, row.message_id, version, executive=executive):
                raise SendRefused(409, "draft_gone", "That draft was sent or changed meanwhile. Look at it again.")
            _audit("delegation_chat_draft_edited", f"Person {person.id} changed a chat draft on its card", {
                "person_id": person.id, "draft_row": row.id, "by_hand": not note,
                **({"from_executive": executive} if executive is not None else {}),
            })
            going = current.get("from_executive") if executive is None else executive
            return {**current, "body": words, "from_executive": going, "version": version}, note
        finally:
            _CHANGING.discard(row.id)


async def set_text(
    person: Any, row_id: int, text: str, *, caller: Any, gmail: Any = None, now: datetime | None = None,
) -> dict[str, Any]:
    """Put the person's own words ``text`` in the draft ``row_id``, on their
    tap. Returns its state; raises ``SendRefused``. Sends nothing."""

    async def mine(_person: Any, _current: dict[str, Any]) -> tuple[str, str, bool | None]:
        return text, "", None

    saved, _ = await _save(person, row_id, caller=caller, gmail=gmail, now=now, words_for=mine)
    return saved


def _note_and_forward(current: dict[str, Any]) -> tuple[str, str]:
    """The draft's words to write, and a forward's message under them that
    stays as it is ("" for any other email)."""
    from openexecutive.delegation.email_html import split_forward

    body = str(current.get("body") or "")
    # Only a forward's note: a reply's quoted header stays the model's to see.
    is_forward = str(current.get("subject") or "").strip().lower().startswith(("fw:", "fwd:"))
    return split_forward(body) if is_forward else (body, "")


def _payload(current: dict[str, Any], words: str) -> dict[str, Any]:
    """The draft as a card payload, for the shared writers
    (``answers.rewrite``, ``send_from``): its words, who it goes to, and
    who it goes from."""
    from openexecutive.delegation.send_from import FROM_EXECUTIVE

    return {
        "draft_body": words,
        "draft_to": [*current.get("to", []), *current.get("cc", [])],
        FROM_EXECUTIVE: current.get("from_executive") is True,
    }


def _rejoined(current: dict[str, Any], note: str, words: str, forwarded: str) -> str:
    if words == note:
        return str(current.get("body") or "")  # unchanged, forward and all
    return f"{words.rstrip()}\n\n{forwarded}" if forwarded else words


async def switch(
    person: Any, row_id: int, *, executive: bool, caller: Any, gmail: Any = None, now: datetime | None = None,
) -> dict[str, Any]:
    """Send the draft ``row_id`` from the Executive (``executive``), with the
    person on cc, or from the person again, on their tap: its words written
    once for the new sender (``send_from.rewritten_for``, the same rewrite a
    reply card's switch makes) and saved in their Drafts like any change, so
    the card and their Drafts keep one version. Returns its state; raises
    ``SendRefused``. Sends nothing."""
    from openexecutive.delegation.send_from import NOTE_TO_EXECUTIVE, NOTE_TO_YOU, rewritten_for

    async def written(who: Any, current: dict[str, Any]) -> tuple[str, str, bool | None]:
        if (current.get("from_executive") is True) == executive:
            return str(current.get("body") or ""), "", None
        note, forwarded = _note_and_forward(current)
        words = await rewritten_for(who, _payload(current, note), executive=executive)
        return _rejoined(current, note, words, forwarded), NOTE_TO_EXECUTIVE if executive else NOTE_TO_YOU, executive

    saved, note = await _save(person, row_id, caller=caller, gmail=gmail, now=now, words_for=written)
    return {**saved, "note": note} if note else saved


async def change(
    person: Any, row_id: int, text: str, *, caller: Any, gmail: Any = None, now: datetime | None = None,
) -> tuple[dict[str, Any], list[str], str]:
    """Rewrite the draft ``row_id`` the way the person asked, in their words
    (``text``: "drop Lisa's name", or a tone pill's "Make it shorter"), on
    their tap. Returns its state, the sentences that are new, and a short
    line on what changed. Raises ``SendRefused``. Sends nothing.

    The words are written for whoever sends it (the person, or the
    Executive once switched). Asking for it to go from the other one
    ("send it from the Executive") switches it, written once for them, as
    the card's From does (``switch``). A forward keeps the message it
    forwards: only the note above it is rewritten."""
    from openexecutive.delegation.answers import (
        _SEND_FROM_ASKED,
        CONVERSATION_KEY,
        NeedsWork,
        SendFrom,
        changed_sentences,
        rewrite,
    )
    from openexecutive.delegation.reply_send import SendRefused
    from openexecutive.delegation.send_from import (
        NOTE_TO_EXECUTIVE,
        NOTE_TO_YOU,
        executive_name,
        rewritten_for,
    )

    new: list[str] = []

    async def rewritten(who: Any, current: dict[str, Any]) -> tuple[str, str, bool | None]:
        note, forwarded = _note_and_forward(current)
        payload = _payload(current, note)
        try:
            words, extra, _ = await rewrite(who, payload, [], change=text, now=now)
        except SendFrom as only:
            # Only who it goes from, and no change of words.
            words, extra = note, {_SEND_FROM_ASKED: "executive" if only.executive else "me"}
        turns = extra.get(CONVERSATION_KEY)
        last = turns[-1] if isinstance(turns, list) and turns else {}
        line = str(last.get("text") or "") if isinstance(last, dict) and last.get("kind") == "note" else ""
        sender = extra.get(_SEND_FROM_ASKED)
        executive = sender == "executive" if isinstance(sender, str) else None
        if executive is not None and executive != (current.get("from_executive") is True):
            # Who it goes from, asked for too: the new words written once for them.
            words = await rewritten_for(who, {**payload, "draft_body": words}, executive=executive)
            line = NOTE_TO_EXECUTIVE if executive else NOTE_TO_YOU
        else:
            if executive is not None and words == note and not line:
                # Asked for the sender it already has: say so, nothing changes.
                line = f"It already goes from {executive_name() if executive else 'you'}."
            executive = None
        new[:] = changed_sentences(note, words)
        return _rejoined(current, note, words, forwarded), line or "I changed the email.", executive

    try:
        saved, note = await _save(person, row_id, caller=caller, gmail=gmail, now=now, words_for=rewritten)
    except NeedsWork as work:
        # Looking things up takes the Executive's tools, which this card
        # doesn't have: the draft stays as it is.
        what = "answer that" if work.question else "do that"
        raise SendRefused(
            409, "needs_work", f"I need to look things up to {what}. Ask me in the chat, and the draft stays as it is.",
        ) from None
    return saved, new, note


async def discard(
    person: Any, row_id: int, *, caller: Any, gmail: Any = None, now: datetime | None = None,
) -> dict[str, Any]:
    """Delete the draft ``row_id`` from the person's Drafts, on their tap.
    It is read first, and one changed in their mailbox since the row's
    version is theirs to delete there. Outlook also checks the version as
    it deletes; Gmail has no such check, so an edit made in the instant
    between that read and the delete is not caught there. Returns its state
    (``GONE`` once deleted, or the state it already had); raises
    ``SendRefused``. Sends nothing."""
    from openexecutive.audit import rows_for_person
    from openexecutive.delegation import drafts
    from openexecutive.delegation.gmail import GmailError, normalize_email
    from openexecutive.delegation.inbox import _audit
    from openexecutive.delegation.reply_send import _CHAT_SENDING, SendRefused, _check_caller
    from openexecutive.delegation.settings import is_enabled

    now = now or datetime.now(UTC)
    row = _row(person, row_id)
    with rows_for_person(person.id):
        _check_caller(caller, normalize_email(person.email))
        closed = _closed(row)
        if closed is not None:
            return closed
        if not is_enabled(person.id):
            raise SendRefused(409, "act_as_me_off", "Turn on As you to delete it here, or delete it in your Drafts.")
        if row.id in _CHANGING or row.id in _CHAT_SENDING:
            raise SendRefused(409, "busy", "This draft is being changed or sent right now. Try again in a moment.")
        _CHANGING.add(row.id)
        try:
            client = await _client(person, gmail)
            current = await _read(row, client, now)
            if current["status"] == CHANGED:
                raise SendRefused(
                    409, "draft_changed", "This draft was changed in your mailbox, so delete it there.", state=current,
                )
            if current["status"] != OPEN:
                return current
            try:
                deleted = await client.delete_draft(row.draft_id)
            except GmailError as exc:
                raise SendRefused(
                    502, "gmail_error", "Couldn't delete it in your mailbox. Try again in a moment.",
                ) from exc
            if not deleted:
                # Already gone, or (Outlook) left alone because it changed or
                # stopped being a draft meanwhile: say what it is now, and
                # never call it deleted.
                after = await _read(row, client, now)
                if after["status"] == CHANGED:
                    raise SendRefused(
                        409, "draft_changed", "This draft was changed in your mailbox, so delete it there.",
                        state=after,
                    )
                if after["status"] == OPEN:
                    raise SendRefused(
                        502, "gmail_error", "Couldn't delete it in your mailbox. Try again in a moment.",
                    )
                return after
            drafts.mark_gone(row.id, now)
            _audit("delegation_chat_draft_deleted", f"Person {person.id} deleted a chat draft on its card", {
                "person_id": person.id, "draft_row": row.id,
            })
            return {"status": GONE}
        finally:
            _CHANGING.discard(row.id)
