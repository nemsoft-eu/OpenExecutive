"""``propose_actions``: leave the person you are speaking with a card of
actions to approve with one tap (``delegation.action_cards``).

It rides with Act as me's tools (``delegation_tools.DELEGATION_TOOLS``), so it
is offered and re-checked the same way (``_writer``: the pin and the verified
surface). It opens no mailbox and does nothing itself, so it stays on after a
turn read mail, when ``message_person``, the calendar tools and roster writes
are withheld: that is what it is for. Nothing on the card happens until the
person approves it, signed in on the web: in the chat it was left in (its chip
carries the card's id) or on Today; or with its Approve button in a chat app
that shows cards (``integrations.channel_cards``). The exception is a card
their Handle it for me and Approvals let go (``action_cards.run_on_its_own``):
carried out at once, with the daily cap bounding it, and below Hands-off
reaching only people settled on their roster.

A turn another AI app asked (``Session.from_mcp_client``) is the signed-in
person's own and gets this tool the same way; the card names the app.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

PROPOSE_ACTIONS = "propose_actions"

PROPOSE_ACTIONS_TOOL: dict[str, Any] = {
    "name": PROPOSE_ACTIONS,
    "description": (
        "Leave the person you are speaking with a card of actions to approve: they "
        "see each action exactly as it will happen and tap Approve (in this "
        "chat, or on Today in the web app) to do it. Nothing happens until they do. Use it "
        "when an email of theirs calls for a meeting, a message to someone, or a new "
        "contact, and you can't do it yourself on this turn. Up to 5 actions a card. "
        "Kinds: `invite` (title, start, end as local date-times, attendee_person_ids "
        "from the roster, optional description), `message` (person_id from the "
        "roster, text exactly as it will be sent), `add_contact` (full_name, email, "
        "optional role; the owner's cards only), `drive_files` (rename and/or move "
        "Google Drive files: `files` with each file_id and its new name, and "
        "`move_to`, a folder id, to move them; up to 25 files; the owner's cards "
        "only), for a rename or move they asked for that the Drive tools refused, "
        "`save_files` (save files into a Drive folder people outside the team can "
        "open, when save_file says it needs a card: `folder`, the folder id, and "
        "`files`, each with the file's handle as `file` and the `name` to save "
        "under; one action a folder; the owner's cards only). "
        "Tell them the card is waiting; never say it is done."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "why": {
                "type": "string",
                "description": "One line on what this is for, e.g. 'From Dana's email about the pilot'.",
            },
            "actions": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["invite", "message", "add_contact", "drive_files", "save_files"]},
                        "title": {"type": "string"},
                        "start": {"type": "string", "description": "Local date and time, e.g. 2026-10-10T14:00."},
                        "end": {"type": "string", "description": "Local date and time, e.g. 2026-10-10T14:30."},
                        "attendee_person_ids": {"type": "array", "items": {"type": "integer"}},
                        "description": {"type": "string"},
                        "person_id": {"type": "integer"},
                        "text": {"type": "string"},
                        "full_name": {"type": "string"},
                        "email": {"type": "string"},
                        "role": {"type": "string"},
                        "files": {
                            "type": "array",
                            "maxItems": 25,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file_id": {"type": "string", "description": "drive_files: the Drive file's id."},
                                    "file": {"type": "string", "description": "save_files: the file's handle."},
                                    "name": {
                                        "type": "string",
                                        "description": "drive_files: the new name; leave out to only move it. save_files: the name to save under.",
                                    },
                                },
                            },
                        },
                        "move_to": {"type": "string", "description": "Drive folder id to move the files into."},
                        "folder": {"type": "string", "description": "save_files: the Drive folder id to save into."},
                    },
                    "required": ["kind"],
                },
            },
        },
        "required": ["why", "actions"],
    },
}


def _error(message: str) -> str:
    return json.dumps({"error": message})


async def handle_propose_actions(tool_input: dict[str, Any]) -> str:
    from openexecutive.audit import rows_via
    from openexecutive.delegation import action_cards
    from openexecutive.integrations.channel_cards import card_here, card_text, shows_cards
    from openexecutive.orchestrator.delegation_tools import _writer
    from openexecutive.orchestrator.schedule_tools import current_session

    session = current_session.get()
    writer = _writer(PROPOSE_ACTIONS)
    if isinstance(writer, str):
        return writer
    person, pinned = writer.person, writer.pinned
    if getattr(pinned, "action_cards", 0) >= action_cards.CARDS_PER_TURN:
        return _error(f"That's {action_cards.CARDS_PER_TURN} cards this turn. Put the rest on one of them.")
    checked = action_cards.check(tool_input.get("actions"), person, now=datetime.now(UTC))
    if isinstance(checked, str):
        return _error(checked)
    unshown = await action_cards.describe_drive_files(checked)
    if unshown is None:
        unshown = await action_cards.describe_save_files(checked, getattr(session, "session_id", None))
    if unshown is not None:
        return _error(unshown)
    # Asked from another AI app: the card says which.
    via = (rows_via() or "an AI app") if getattr(session, "from_mcp_client", False) is True else None
    try:
        if len(action_cards.open_cards(person.id)) >= action_cards.OPEN_CARDS_MAX:
            return _error(f"They already have {action_cards.OPEN_CARDS_MAX} cards waiting. Ask them to clear some first.")
        pinned.action_cards = getattr(pinned, "action_cards", 0) + 1
        # Handle it for me on and Approvals above Careful: carried out now.
        on_its_own = action_cards.on_its_own_refusal(person, checked, now=datetime.now(UTC)) is None
        decision_id = action_cards.propose(
            person, checked, str(tool_input.get("why") or ""),
            session_id=getattr(session, "session_id", None), on_its_own=on_its_own,
            via=via,
        )
    except Exception:
        logger.warning("propose_actions: couldn't store the card", exc_info=True)
        return _error("Couldn't leave the card just now. Try again in a moment.")
    action_cards.audit("delegation_actions_proposed", f"Left person {person.id} an action card", {
        "person_id": person.id, "decision_id": decision_id,
        "kinds": [a["kind"] for a in checked],
        **({"via": via} if via else {}),
    })
    if on_its_own:
        try:
            results = await action_cards.run_on_its_own(decision_id, person)
        except Exception:
            logger.warning("propose_actions: carrying out the card on its own failed", exc_info=True)
            results = None
        if results is not None:
            return json.dumps({
                "status": "done_on_its_own",
                "decision_id": decision_id,
                "actions": [
                    {"action": checked[r["index"]]["summary"], "status": r["status"], "detail": r.get("detail", "")}
                    for r in results
                ],
                "note": (
                    "Their settings (Approvals) let these go without a tap, so they were carried out now. "
                    "Tell them what happened."
                ),
            })
    return json.dumps({
        "status": "waiting_for_approval",
        "decision_id": decision_id,
        "actions": [a["summary"] for a in checked],
        "note": (
            # A chat app shows a card whole or not at all (card_text).
            "Nothing has happened yet. It is on Today in the web app for them to approve; it's too long to show in this chat."
            if shows_cards(session) and card_text(decision_id) is None else
            "Nothing has happened yet. It is in this chat and on Today in the web app for them to approve."
            if card_here(session) or not getattr(session, "origin_channel", "") else
            "Nothing has happened yet. It is on Today in the web app for them to approve; this chat can't show it."
        ),
    })


ACTION_CARD_TOOLS: list[dict[str, Any]] = [PROPOSE_ACTIONS_TOOL]
ACTION_CARD_TOOL_HANDLERS: dict[str, Any] = {PROPOSE_ACTIONS: handle_propose_actions}
