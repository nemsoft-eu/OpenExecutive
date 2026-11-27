"""The Executive hands a browser task over to the person.

Some steps on a website are the person's alone: signing in, a code from their
phone, a passkey, a password or card field (the extension never types into
those), a "prove you're human" check, or a choice only they should make. The
Executive calls ``browser_hand_over`` (``browser.tools``) with what it
needs from them. Then:

- the extension brings that tab to the front with a bar at the top, "Waiting
  on you: <what>", with Done and Stop (``packages/extension``);
- in the chat they are typing in, the turn asks there first, with Done and
  Stop (``reserve``, the ``hand_over`` stream event); a Needs you card says
  the same (``SOURCE``) and reaches them wherever their cards do only when
  the chat's ask goes unanswered for ``CARD_AFTER_S``, or at once on a turn
  with no chat to ask in;
  in a workflow, the run pauses on it (``extensions.workflow_wait``) and its
  creator is told, as for any pause;
- they do it, then press Done on the bar or the card. When the page they
  were handed was a sign-in form, the extension also sees it go and says
  Done by itself.

The hand-over owns its card: however it ends (Done, Stop, the chat turn
stopped, a newer one pushing it out), the card closes with it (``finish``,
``_drop``), and cards left by a restart close at start-up
(``close_leftover_cards``).

The Executive looks at the page again only after Done. Password, card and
code fields never show their values to it, in the page's text or its
screenshot (the extension blanks them for the picture). A turn waits ``WAIT_S`` for it; past that the turn ends
saying so. Then Done on the bar only takes it down: Done on the card
carries the work on in a new chat (the card's approve path), where the person
sees what it will do; a paused workflow run is picked up by either. Stop ends
it: the turn is told to stop, a waiting run stops (``resumer.stop_waiting``).
The extension's own Done (a sign-in page left) is a hint from the page, so it
only answers a turn or run that is waiting, on a hand-over whose page had a
sign-in form when it was handed over (``Handover.sign_in``).

This is not a check: Approvals, Your rules and In training apply to what the
Executive does next exactly as before. Hand-overs are held in this process,
like the relay's steps: after a restart, Done on an old bar says it's gone.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

Outcome = Literal["done", "stop"]
OUTCOMES: frozenset[str] = frozenset({"done", "stop"})
# The extension saw a sign-in page left: a hint from the page, so it only
# answers a turn or run that is waiting (``api/routes/browser.py``).
AUTO = "auto"

# The card's source and tag (the web card says Done, not Approve).
SOURCE = "browser_handover"
TAG = "browser-handover"
# How long a chat turn waits for Done before it ends saying so.
WAIT_S = 300.0
# How long the chat's own ask waits for Done or Stop before the Needs you
# card goes out too.
CARD_AFTER_S = 60.0
# What a hand-over says it needs, at most.
MAX_NEED = 140
# What the Executive was in the middle of, kept for carrying on later.
MAX_TASK = 600
# Open hand-overs per person: more are left over (oldest go first).
MAX_PER_PERSON = 4
# How long after a workflow's hand-over Done may find its run not yet saved
# as waiting (``reopen``).
FRESH_S = 120.0
# A hand-over nobody finished is forgotten after this (a run waits 24 h).
TTL_S = 25 * 3600.0


@dataclass
class Handover:
    id: str
    person_id: int
    need: str
    card_id: int | None = None
    # The paused workflow run's wait key; "" on a chat turn.
    workflow_key: str = ""
    # The page had a sign-in form when it was handed over (the extension's
    # answer to the hand_over step). Only then is its own Done ("auto")
    # taken: a page left later can't say Done for any other kind of step.
    sign_in: bool = False
    # The chat the person is typing in shows the ask with Done and Stop
    # (``reserve``): the card waits ``CARD_AFTER_S``.
    in_chat: bool = False
    # How it ended ("done" or "stop"), "" while open.
    outcome: str = ""
    # Done or Stop the chat's ask got before this started (``answer_early``).
    early: str = ""
    created: float = field(default_factory=time.monotonic)
    # Set while a turn waits for the answer.
    future: asyncio.Future[str] | None = None
    # The loop it was started on: a sync route that closes its card runs in
    # a worker thread and hands the answer back through it.
    loop: asyncio.AbstractEventLoop | None = None


# Read from worker threads too (card_closed): iterate over a copy, never the
# dict itself, so an insert on the loop thread can't break the loop.
_open: dict[str, Handover] = {}
# Ids the chat was already shown with the ask, by (person, chat, need),
# taken by the hand-over that asks it (``reserve``, ``start``). They last
# the round they were shown in, however long its other steps take: the
# chat's next round clears what that one left (``new_round``).
_reserved: dict[tuple[int, str, str], list[str]] = {}
# Done or Stop pressed on the chat's ask before its hand-over started (an
# earlier step of the round was still running), by id: the hand-over takes
# it when it starts (``answer_early``).
_early: dict[str, str] = {}


def one_line(text: object, cap: int) -> str:
    return " ".join(str(text or "").split())[:cap].strip()


def _drop(handover: Handover) -> None:
    """Forget a hand-over that was left over, as a closed card does: a turn
    still waiting on it is told to stop now rather than sitting out its wait,
    and its bar comes down, since it can no longer be answered."""
    _open.pop(handover.id, None)
    _ended(handover, "stop")

    def _close() -> None:
        future = handover.future
        if future is not None and not future.done():
            future.set_result("stop")
        end_bar(handover)

    loop = handover.loop
    try:
        here = asyncio.get_running_loop()
    except RuntimeError:
        here = None
    if loop is None or loop is here:
        _close()
    elif not loop.is_closed():
        loop.call_soon_threadsafe(_close)


def _prune(now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
    for h in [h for h in list(_open.values()) if now - h.created >= TTL_S]:
        _drop(h)


def _close_card(card_id: int | None, outcome: str) -> None:
    """Close a hand-over's card the way it ended: Done = done, anything else
    = dismissed. Only while it is still open (Done on it already closed it)."""
    if card_id is None:
        return
    from openexecutive.alerts import store

    try:
        store.set_status_if_open(card_id, "ack" if outcome == "done" else "dismissed")
    except Exception:
        logger.warning("browser hand-over: couldn't close card %s", card_id, exc_info=True)


def _ended(handover: Handover, outcome: str) -> None:
    """It ended: its card, if it has one yet, closes with it (``place_card``
    closes one made after this)."""
    if handover.outcome:
        return
    handover.outcome = "done" if outcome == "done" else "stop"
    _close_card(handover.card_id, handover.outcome)


def place_card(handover: Handover, card_id: int | None) -> None:
    """Its Needs you card was made: a hand-over that ended meanwhile closes it
    at once, so no card outlives what it asks about. Safe from a worker
    thread."""
    handover.card_id = card_id
    if handover.outcome:
        _close_card(card_id, handover.outcome)


def close_leftover_cards() -> int:
    """At start-up: hand-overs live in this process, so a card one left open
    before a restart can no longer be answered. Close them all. How many.
    An install runs one API process (the browser relay assumes it too), so
    no other process is still waiting on one of these."""
    from openexecutive.alerts import store

    try:
        return store.close_open_by_source(SOURCE, "dismissed")
    except Exception:
        logger.warning("browser hand-over: couldn't close left-over cards", exc_info=True)
        return 0


def new_round(person_id: int, chat: str) -> None:
    """Clear what ``person_id``'s chat ``chat`` reserved in an earlier round:
    every hand-over of that round has run (or the turn stopped first), so
    none is left to take it."""
    for key in [k for k in list(_reserved) if k[0] == person_id and k[1] == chat]:
        for hid in _reserved.pop(key, []):
            _early.pop(hid, None)


def answer_early(handover_id: str, person_id: int, outcome: str) -> bool:
    """Done or Stop on the chat's ask of ``person_id`` whose hand-over hasn't
    started yet: kept for it to take when it does (``start``). False when no
    such ask of theirs is waiting to start."""
    if outcome not in OUTCOMES:
        return False
    if not any(k[0] == person_id and handover_id in ids for k, ids in list(_reserved.items())):
        return False
    _early[handover_id] = outcome
    return True


def reserve(person_id: int, chat: str, need: str) -> str:
    """The id the hand-over asking ``need`` of ``person_id`` in their chat
    ``chat`` will take, so the chat can show the ask with Done and Stop
    before it starts. The hand-over then waits ``CARD_AFTER_S`` before its
    card goes out."""
    hid = uuid.uuid4().hex
    # Several with the same words in one round are taken in order.
    _reserved.setdefault((person_id, chat, one_line(need, MAX_NEED)), []).append(hid)
    while len(_reserved) > 64:
        for old in _reserved.pop(next(iter(_reserved))):
            _early.pop(old, None)
    return hid


def _take_reserved(person_id: int, chat: str, need: str) -> str | None:
    queue = _reserved.get((person_id, chat, need))
    if not queue:
        return None
    hid = queue.pop(0)
    if not queue:
        _reserved.pop((person_id, chat, need), None)
    return hid


def start(
    person_id: int, need: str, *, card_id: int | None = None, workflow_key: str = "", in_chat: str = ""
) -> Handover:
    """A new open hand-over for ``person_id``. ``in_chat``: the chat of
    theirs it was asked in, so it takes the id that chat was shown this
    round (``reserve``)."""
    _prune()
    mine = sorted((h for h in list(_open.values()) if h.person_id == person_id), key=lambda h: h.created)
    for old in mine[: max(0, len(mine) - MAX_PER_PERSON + 1)]:
        _drop(old)
    try:
        loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    need = one_line(need, MAX_NEED)
    reserved = _take_reserved(person_id, in_chat, need) if in_chat and not workflow_key else None
    handover = Handover(
        id=reserved or uuid.uuid4().hex, person_id=person_id, need=need,
        card_id=card_id, workflow_key=workflow_key, in_chat=reserved is not None, loop=loop,
        early=_early.pop(reserved, "") if reserved else "",
    )
    _open[handover.id] = handover
    return handover


def get(handover_id: str) -> Handover | None:
    _prune()
    return _open.get(str(handover_id or ""))


def by_card(card_id: int) -> Handover | None:
    _prune()
    return next((h for h in list(_open.values()) if h.card_id == card_id), None)


def waited_on(handover: Handover) -> bool:
    """Whether a turn is waiting for this hand-over right now."""
    return handover.future is not None and not handover.future.done()


def workflow_key(person_id: int, handover_id: str) -> str:
    """What a workflow run paused on this hand-over waits on."""
    return f"handover:{person_id}:{handover_id}"


def arm(handover: Handover) -> asyncio.Future[str]:
    """Get ready for Done or Stop before the bar is shown, so one pressed
    straight away (or the page's own Done) isn't lost before ``wait`` starts:
    ``finish`` then answers it, and ``wait`` returns it at once."""
    handover.future = asyncio.get_running_loop().create_future()
    return handover.future


async def wait(handover: Handover, timeout: float | None = None) -> Outcome | None:
    """Wait for Done or Stop on ``handover`` (``WAIT_S`` at most, unless
    ``timeout`` says); None when it didn't come in time (it stays open, for
    a Done later)."""
    timeout = WAIT_S if timeout is None else timeout
    future = handover.future if handover.future is not None else arm(handover)
    try:
        got = await asyncio.wait_for(asyncio.shield(future), timeout)
    except TimeoutError:
        return None
    finally:
        handover.future = None
    return "stop" if got == "stop" else "done"


def finish(handover_id: str, person_id: int, outcome: str) -> tuple[Handover | None, bool]:
    """Close ``person_id``'s hand-over with ``outcome``: (it, whether a turn
    was waiting for it, which now has the answer). (None, False) when there
    is no such open hand-over of theirs."""
    handover = get(handover_id)
    if handover is None or handover.person_id != person_id or outcome not in OUTCOMES:
        return None, False
    _open.pop(handover.id, None)
    future = handover.future
    waited = future is not None and not future.done()
    if waited or outcome == "stop":
        # Done with no turn waiting leaves the card open: Done on it is
        # what carries the work on (its approve path) once the turn ended.
        # A workflow's hand-over has no card (the run waits on the bar).
        _ended(handover, outcome)
    if waited and future is not None:
        future.set_result(outcome)
        return handover, True
    return handover, False


def reopen(handover: Handover) -> bool:
    """Put back a hand-over ``finish`` closed when Done found nothing to carry
    on yet (a workflow run not saved as waiting): the bar stays, and a later
    Done tries again. Only while it is new (``FRESH_S``): past that the run
    has ended (stopped from its card, or timed out), so it stays closed.
    Whether it was put back."""
    if time.monotonic() - handover.created >= FRESH_S:
        return False
    _open[handover.id] = handover
    return True


def end_bar(handover: Handover) -> None:
    """Take the bar down in their browser: they said Done or Stop somewhere
    else (the card). Best effort, in the background; never raises."""
    from openexecutive.browser import relay as browser_relay

    async def _end() -> None:
        try:
            await browser_relay.run(handover.person_id, "hand_over_end", {"id": handover.id}, timeout=15.0)
        except Exception:
            # Not connected, or gone: the bar's own Done finds it closed.
            return

    try:
        task = asyncio.get_running_loop().create_task(_end())
    except RuntimeError:
        return
    _ending.add(task)
    task.add_done_callback(_ending.discard)


_ending: set[asyncio.Task[None]] = set()


def card_closed(card_id: int) -> bool:
    """Its card was dismissed or cleared without Done: the same as Stop. A
    turn waiting for it is told to stop and the bar comes down. Safe from a
    worker thread (the sync ack routes). False when no hand-over is open on
    that card."""
    handover = by_card(card_id)
    if handover is None:
        return False
    _open.pop(handover.id, None)
    handover.outcome = handover.outcome or "stop"
    loop = handover.loop

    def _close() -> None:
        future = handover.future
        if future is not None and not future.done():
            future.set_result("stop")
        end_bar(handover)

    if loop is None or loop.is_closed():
        return True
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        _close()
    else:
        loop.call_soon_threadsafe(_close)
    return True


def _reset_for_tests() -> None:
    _open.clear()
    _reserved.clear()
    _early.clear()
