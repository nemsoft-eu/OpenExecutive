"""Who is asking on a chat turn, and whether only they read it.

Every tool that acts for "the person asking" asks here, on every surface: the
web chat (and the phone app, which uses it), a chat app's direct message, a
call with the Executive, another AI app signed in as them. The answer comes
from the turn's flags in this one place, so a surface added later works for
every tool at once instead of tool by tool.

- ``speaker``: the person on People talking on this turn, on a surface that
  verified it is them (``people_tools``), or None (an email from someone
  else, an unattended run, a sender on nobody's People entry).
- ``own_private_speaker``: that person when nobody else reads this
  conversation (``delegation.settings.private_conversation``): their chats,
  tasks, notes and settings are shown and changed there, never in a shared
  chat or a channel.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# The end of every refusal the person asking hears: said to them plainly, and
# passed to no one. A message or card about it to the owner would put their
# ask on someone else's list, so no refusal tells the Executive that someone
# else "needs to" do it.
PASS_ON_NOTHING = (
    "Answer only the person who asked (on email, your reply to them is that answer); "
    "send nothing to anyone else on their behalf: no message, card or alert about it. "
    "A run nobody is watching still says in its own result that this step didn't happen."
)


def _session() -> Any:
    from openexecutive.orchestrator.schedule_tools import current_session

    return current_session.get()


def speaker(session: Any = None) -> Any:
    """The verified speaker's People row (principal or teammate), or None.
    ``session`` defaults to the turn's own (``current_session``). Fails
    closed. Never raises."""
    from openexecutive.orchestrator.people_tools import (
        is_principal_on_verified_surface,
        teammate_on_verified_surface,
    )
    from openexecutive.people.store import find_principal_person, get_person
    from openexecutive.utils.deployment import is_local_login

    session = session if session is not None else _session()
    if (
        session is None
        or getattr(session, "unattended", False) is True
        or getattr(session, "private_to_principal", False) is True
    ):
        return None
    if (
        getattr(session, "from_web_chat", False) is True or getattr(session, "from_mcp_client", False) is True
    ) and not (getattr(session, "web_caller_signed_in", False) is True or is_local_login()):
        # A request that only holds the shared secret is not a sign-in.
        return None
    try:
        if is_principal_on_verified_surface(session):
            caller = getattr(session, "caller_person_id", None)
            return get_person(int(caller)) if caller is not None else find_principal_person()
        return teammate_on_verified_surface(session)
    except Exception:
        logger.warning("speaker: lookup failed", exc_info=True)
        return None


def own_private_speaker(session: Any = None) -> Any:
    """``speaker`` when nobody but them (and the Executive) reads this
    conversation, else None. Never raises."""
    from openexecutive.delegation.settings import private_conversation

    session = session if session is not None else _session()
    if (
        session is None
        or getattr(session, "meeting_call", False) is True
        or getattr(session, "from_meeting", False) is True
    ):
        # A call with others in it is never one person's own conversation.
        return None
    person = speaker(session)
    if person is None:
        return None
    try:
        return person if private_conversation(session) else None
    except Exception:
        logger.warning("speaker: privacy check failed", exc_info=True)
        return None
