"""One loop for a model pass that calls tools until its work is done.

The passes that run with nobody watching (the morning review and Take the
lead, research's routing and watch passes, the chat review) drive the model
through here, as does any other chat with tools that isn't a turn of the
Executive's. Each round sends what the pass has so
far, runs the calls the model wrote and sends their results back, read the
way a chat turn reads them (orchestrator.reading): sizes come from the
model, and a long result comes back in parts, a repeat of the same call
bringing the next part without running the tool again.

A pass ends when the model answers without a call. It writes up from what
it has, with one more call that offers no tools, when the same call came
back the same ``REPEAT_LIMIT`` times, when its reads fill what its model
holds, when it has run the most rounds its caller gives it (``max_rounds``),
or when the pass itself says it has had enough (``enough``).

A round cut off at the model's output limit carries on, as a chat turn
does: the call it was still writing never runs and is answered with
``CUT_OFF_RESULT``, and cut text is followed by ``CUT_OFF_NOTE`` so the
model goes on from where it stopped. ``MAX_EMPTY_CUT_ROUNDS`` cut rounds in
a row that get nowhere end the pass.

Each round reads the history before it from the prompt cache: the newest
tool result carries the loop's one cache breakpoint
(``reading.apply_loop_cache_marker``, as in a chat turn), when caching is
on and the pass's system and tools leave a breakpoint free.
"""
from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from openexecutive.orchestrator.reading import (
    CUT_OFF_NOTE,
    CUT_OFF_RESULT,
    FULL_NOTE,
    MAX_EMPTY_CUT_ROUNDS,
    REPEATING_NOTE,
    STEPS_NOTE,
    TurnReadingBudget,
    apply_loop_cache_marker,
    last_written_block,
    says_something,
)

logger = logging.getLogger(__name__)

Ended = Literal["done", "repeating", "full", "steps", "enough", "cut"]


@dataclass
class Rounds:
    """How a pass went: the text of its closing answer (what the model wrote
    in the call that ended it; text beside its tool calls is it thinking
    out loud; with the text of any rounds it wrote that were cut off at the
    output limit before it), how many model calls it made, and why it ended."""

    text: str
    rounds: int
    ended: Ended


def _block(block: Any) -> dict[str, Any] | None:
    """A response block as the next request sends it back, or None."""
    from openexecutive.providers.translator import reasoning_replay_block

    if isinstance(block, dict):
        return block
    kind = getattr(block, "type", "")
    if kind == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if kind == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", ""),
            "name": getattr(block, "name", ""),
            "input": getattr(block, "input", {}) or {},
        }
    # OpenRouter reasoning continuity across rounds.
    return reasoning_replay_block(block)


# The API takes at most four cache breakpoints in a request; the loop's
# marker is one, so it goes on only when the system and tools leave one.
_MAX_BREAKPOINTS = 4


def _breakpoints(blocks: Any) -> int:
    if not isinstance(blocks, list):
        return 0
    return sum(1 for b in blocks if isinstance(b, dict) and b.get("cache_control"))


def _caching() -> bool:
    try:
        from openexecutive.config import get_settings

        return bool(get_settings().enable_caching)
    except Exception:
        return False


def text_of(blocks: list[dict[str, Any]]) -> str:
    return "\n\n".join(b["text"] for b in blocks if b.get("type") == "text" and b.get("text")).strip()


async def run_rounds(
    llm: Any,
    *,
    model: str,
    max_tokens: int,
    system: Any,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    run_call: Callable[[dict[str, Any]], Awaitable[Any]],
    actor: str,
    max_rounds: int,
    session_id: str | None = None,
    on_response: Callable[[Any, int], Any] | None = None,
    enough: Callable[[], str | None] | None = None,
    sent: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    on_result: Callable[[dict[str, Any], Any], Any] | None = None,
) -> Rounds:
    """Run the model on ``messages`` until it answers without a call.

    ``run_call`` runs one tool call (``{"id", "name", "input"}``) and returns
    its result; it is never called for a repeat that brings the next part of
    a long result. Each response is recorded as ``actor``'s usage
    (audit.usage), and ``on_response`` sees it with its round number. ``enough``, asked after each round, returns a note when
    the pass wants the model to write up now ("" to stop at once, with no
    write-up). ``sent`` picks what of ``messages`` each request sends (a chat
    that keeps only its newest turns). ``on_result`` sees each call's result
    as the model was sent it: a part of a long one, never what it has not
    read. After ``max_rounds`` rounds of calls it writes up. ``messages``
    grows with each round. A
    provider error is raised to the caller.
    """
    from openexecutive.audit.usage import log_model_usage
    from openexecutive.providers.registry import reading_limits_for

    reading = TurnReadingBudget(reading_limits_for(model))
    # Only what this loop appends is marked: the caller's messages are its own.
    first = len(messages)
    caching = _caching() and _breakpoints(system) + _breakpoints(tools) < _MAX_BREAKPOINTS
    ending: Ended | None = None
    rounds = 0
    # Text of rounds cut off at the output limit, which the next carries on.
    cut_text = ""
    empty_cuts = 0
    while True:
        rounds += 1
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "tools": tools,
            "messages": sent(messages) if sent else messages,
        }
        if ending is not None:
            # Writing up: nothing more runs.
            kwargs["tool_choice"] = {"type": "none"}
        response = await llm.messages_create(**kwargs)
        log_model_usage(response, model=model, actor=actor, iteration=rounds, session_id=session_id)
        if on_response is not None:
            seen = on_response(response, rounds)
            if inspect.isawaitable(seen):
                await seen
        reading.saw(getattr(response, "usage", None))
        blocks = [b for b in (_block(x) for x in getattr(response, "content", None) or []) if b is not None]
        uses = [b for b in blocks if b.get("type") == "tool_use"]
        cut_off = getattr(response, "stop_reason", None) == "max_tokens"
        last = last_written_block(blocks) if cut_off else None
        # The call the round was still writing: its input is unfinished, so
        # it never runs, and its half-written input is not sent again.
        cut = last if last is not None and last.get("type") == "tool_use" else None
        if cut is not None:
            cut["input"] = {}
            uses = [u for u in uses if u is not cut]
        if cut_off:
            got_somewhere = bool(uses) or (
                cut is None and last is not None and says_something(str(last.get("text", "")))
            )
            empty_cuts = 0 if got_somewhere else empty_cuts + 1
            if not uses and cut is None:
                # Cut in its text, or wrote nothing: carry on from there.
                logger.warning("%s: round %d cut off at max_tokens=%d; carrying on", model, rounds, max_tokens)
                if blocks:
                    messages.append({"role": "assistant", "content": blocks})
                cut_text += "".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text")
                if empty_cuts > MAX_EMPTY_CUT_ROUNDS:
                    return Rounds(text=cut_text.strip(), rounds=rounds, ended="cut")
                if reading.window_full:
                    return Rounds(text=cut_text.strip(), rounds=rounds, ended="full")
                messages.append({"role": "user", "content": [{"type": "text", "text": CUT_OFF_NOTE}]})
                continue
        else:
            empty_cuts = 0
        if ending is not None or (cut is None and not uses):
            # Writing up runs nothing, even a call a provider wrote anyway.
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            return Rounds(text=(cut_text + text_of(blocks)).strip(), rounds=rounds, ended=ending or "done")
        cut_text = ""
        messages.append({"role": "assistant", "content": blocks})
        replays = {str(u["id"]): r for u in uses if (r := reading.replay(u)) is not None}
        reading.ran(uses)
        results: list[dict[str, Any]] = []
        for use in uses:
            replayed = replays.get(str(use["id"]))
            if replayed is not None:
                content: Any = replayed
            else:
                out = await run_call(use)
                reading.came_back(str(use["name"]), use["input"], out)
                content = reading.cap(str(use["name"]), use["input"], str(out), str(use["id"]))
            if on_result is not None:
                on_result(use, content)
            results.append({"type": "tool_result", "tool_use_id": use["id"], "content": content})
        if cut is not None:
            logger.warning("%s: round %d cut off while writing a call; asking for it again", model, rounds)
            results.append({"type": "tool_result", "tool_use_id": cut["id"], "content": CUT_OFF_RESULT})
            if empty_cuts > MAX_EMPTY_CUT_ROUNDS:
                messages.append({"role": "user", "content": results})
                return Rounds(text="", rounds=rounds, ended="cut")
        if not cut_off and getattr(response, "stop_reason", None) != "tool_use":
            # Calls beside a finished answer: they ran, and the answer stands.
            messages.append({"role": "user", "content": results})
            return Rounds(text=text_of(blocks), rounds=rounds, ended="done")
        note = enough() if enough is not None else None
        if note == "":
            messages.append({"role": "user", "content": results})
            return Rounds(text="", rounds=rounds, ended="enough")
        if note is not None:
            ending = "enough"
        elif reading.repeating:
            logger.warning("%s: a call came back the same several times; writing up", model)
            note, ending = REPEATING_NOTE, "repeating"
        elif reading.full:
            logger.info("%s: the pass has read all its model holds; writing up", model)
            note, ending = FULL_NOTE, "full"
        elif rounds >= max_rounds:
            logger.info("%s: the pass has run its %d rounds; writing up", model, max_rounds)
            note, ending = STEPS_NOTE, "steps"
        if note:
            results.append({"type": "text", "text": note})
        messages.append({"role": "user", "content": results})
        if caching:
            apply_loop_cache_marker(messages, first)
