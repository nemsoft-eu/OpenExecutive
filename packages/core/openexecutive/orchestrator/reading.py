"""How much a turn reads into its prompt, and the rest of a long result.

Every tool result stays in the prompt for the rest of the turn. The sizes
come from the turn's model (providers.registry.reading_limits_for): one read
fits whole unless it is longer than a part, and then it comes back in parts,
each repeat of the same call bringing the next one without running the tool
again. Only when the turn has read all its model can hold does a result stop
short, and its note says so plainly. The Executive's loop and a workflow's
action step both read through here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import zlib
from collections.abc import Awaitable
from contextvars import ContextVar
from typing import Any, TypeVar

from openexecutive.orchestrator import tool_groups, turn_progress
from openexecutive.providers.feature_gate import _CHARS_PER_TOKEN, ReadingLimits
from openexecutive.workflows import step_script, tool_catalog

logger = logging.getLogger(__name__)

# The same call with the same result this many times is no step of the work
# (a write that keeps failing the same way, a retry that finds nothing new):
# the turn then writes its answer from what it has and runs no more tools.
# A turn that gets anywhere never meets it; it replaces a count of rounds.
REPEAT_LIMIT = 4
REPEATING_NOTE = (
    "You have made the same call several times and it came back the same "
    "each time. Do not call any more tools. Write your answer now from what "
    "you have: say what worked, what kept failing and why, and what would fix it."
)
# Once a turn has read all its model holds (TurnReadingBudget.full), more
# reads would push the prompt past the window: it writes its answer from
# what it has read and offers to carry on, the same way a repeat ends.
# The most rounds of tool calls a chat turn runs (the owner's choice,
# 11-18). At the last one it writes its answer from what it has.
MAX_STEPS = 40
STEPS_NOTE = (
    "This question has used all the steps one message gets. Do not call any "
    "more tools. Write your answer now from what you have, say plainly what "
    "is still left to do, and offer to carry on with it in a new message."
)
FULL_NOTE = (
    "This question has now read all it can hold at once. Do not call any "
    "more tools. Write your answer now from what you have read, say plainly "
    "what you have not read yet, and offer to carry on with the rest in a new message."
)

# Answers a tool call the model was still writing when its round was cut off
# at its output limit: the call never ran (its input is unfinished), and the
# turn or pass carries on so it can make it again in a shorter form.
CUT_OFF_RESULT = (
    "That call was cut off at the length limit before you finished writing "
    "it, so it did not run. Make it again in a shorter form, or in smaller "
    "steps (for example, fewer rows or columns first, then the rest)."
)

# Said after a round cut off at its output limit with no tool call left to
# answer (cut in its reply text, or a cut call the provider dropped whole).
CUT_OFF_NOTE = (
    "Your last answer was cut off at the length limit before it finished. If "
    "you were writing a tool call, it did not run: make it again in a shorter "
    "form, or in smaller steps. If you were writing your reply, carry on from "
    "where it stopped, without repeating what you already wrote."
)

# Cut rounds in a row that got nowhere after which a turn or pass ends: a round that
# wrote nothing at all (a model spending its whole answer thinking), or one
# cut while writing its only call (a document too long for one call, asked
# for again just as long), which never runs. A round whose cut fell in text,
# or that closed a call before the cut, is a step of the work, so this never
# stops a job that is getting written; without it the loop would bill a full
# answer each round, forever.
MAX_EMPTY_CUT_ROUNDS = 3

# A cut round's text that packs this small is a model repeating itself
# ("... ... ...", one line over and over), no step of the work. Prose and
# tables pack to a quarter or more of their length.
_STUCK_TEXT_MIN = 2_000
_STUCK_TEXT_RATIO = 0.05


def says_something(text: str) -> bool:
    """Whether cut text is a step of the work: not empty, and not one thing
    said over and over (it packs to almost nothing)."""
    if not text.strip():
        return False
    if len(text) < _STUCK_TEXT_MIN:
        return True
    raw = text.encode()
    return len(zlib.compress(raw, 6)) / len(raw) >= _STUCK_TEXT_RATIO


def last_written_block(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The last text or tool call a round wrote. When the round was cut off
    at its output limit and this is a call, that call is the one it was
    still writing: its input is unfinished, so it never runs. A reasoning
    block a translated provider (OpenRouter, OpenAI) puts last is skipped,
    and a call followed by text closed before the cut."""
    return next((b for b in reversed(blocks) if b.get("type") in ("text", "tool_use")), None)


def apply_loop_cache_marker(
    messages: list[dict[str, Any]], start: int = 0
) -> bool:
    """Move a tool loop's cache breakpoint to the newest tool result, in
    place. The Executive's turn and every background pass (run_rounds)
    mark their history this way.

    Without this the loop caches only the static tools+system prefix, so
    every iteration re-sends the whole accumulated transcript at full
    input price and ``cache_read_input_tokens`` stays pinned flat while
    the prompt grows. Marking the newest tool result makes each iteration
    read the previous iteration's write and extend it, so hits accrue.

    The marker MOVES rather than accumulates: a per-iteration marker left
    in place would reach one breakpoint per round and blow the
    4-breakpoint API limit. Sweeping first also makes this idempotent and
    self-healing — there is no bookkeeping index to drift out of sync.

    Placement is the newest non-empty ``tool_result`` in a user message.
    That is normally the last block of the last user message; when the
    newest results are empty it falls back to the newest non-empty one
    before them, so an all-empty iteration still gets a breakpoint. Only
    the empty blocks then sit outside the cached prefix, which costs
    nothing.

    The marker must stay at the ``tool_result`` block's own top level:
    ``providers.feature_gate._strip_cache_control`` removes markers one
    level deep only, so a marker nested inside ``tool_result["content"]``
    would survive the gate and reach a provider that rejects the field.

    ``start`` bounds the sweep to messages this loop appended. The
    Executive's ``current_messages`` is a SHALLOW copy, so the caller still owns the
    dicts before that index; without the bound a caller-supplied
    ``tool_result`` would be mutated in place. Today no caller builds one,
    but making that structural beats guarding it with a comment.
    """
    # Sweep every marker, and remember every markable block in order. The
    # newest one wins, but keeping the whole list means an iteration whose
    # last results are all empty falls back to the newest earlier result
    # rather than shipping with no breakpoint at all — losing the marker
    # costs the entire accumulated transcript at full price for that step.
    markable: list[dict[str, Any]] = []
    for msg in messages[start:]:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        is_user = msg.get("role") == "user"
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            block.pop("cache_control", None)
            # An empty result carries nothing worth caching and risks an
            # upstream 400 on the translated typed block. Non-str content
            # (a typed/image result) is markable — it is real bytes in the
            # prefix — but is left to the translator to render.
            body = block.get("content")
            if is_user and body:
                markable.append(block)
    if markable:
        markable[-1]["cache_control"] = {"type": "ephemeral"}
    return bool(markable)


# Ids and times in a result (a request id, a timestamp, a clock, time
# elapsed or left) don't make it a different answer: they are left out of
# call_key. Other numbers stay, so a status read whose progress moves
# (20, 40, 60) is getting somewhere, and one whose only change is "42s
# elapsed" is not. Every pattern here matches in one pass over the text,
# however long and whatever it holds: a result is outside text, and a
# pattern that rescans a run of "-" from each of its characters would hold
# the server for minutes on one long page title.
_VARYING_IN_A_RESULT = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
    r"|(?<![0-9A-Za-z:])\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?![0-9A-Za-z:])"
    r"|(?<![0-9A-Za-z.])\d+(?:\.\d+)?\s?(?:ms|s|secs?|seconds?|mins?|minutes?|h|hrs?|hours?)(?![0-9A-Za-z])"
    # A whole word of letters, digits, "_" and "-": judged by _an_id.
    r"|(?P<word>(?<![0-9A-Za-z_-])[0-9A-Za-z_-]{8,})"
)
# A hex id (a hash, a UUID): hex digits with a letter among them.
_HEX_ID = re.compile(r"(?=[0-9-]*[a-fA-F])[0-9a-fA-F]{8,}(?:-[0-9a-fA-F]{4,})*")


def _an_id(word: str) -> bool:
    """Whether a word in a result is an id: a hex one, or 12 or more
    characters mixing letters and digits."""
    if _HEX_ID.fullmatch(word):
        return True
    return len(word) >= 12 and any(c.isdigit() for c in word) and any(c.isalpha() for c in word)


def _varying(match: re.Match[str]) -> str:
    word = match.group("word")
    return "#" if word is None or _an_id(word) else word


def call_key(name: str, tool_input: Any, result: Any) -> str:
    """One call with what it returned, for REPEAT_LIMIT: the same error with
    a new request id or time each try is still the same result."""
    same = _VARYING_IN_A_RESULT.sub(_varying, str(result))
    return hashlib.sha256(
        json.dumps([name, tool_input, same], sort_keys=True, default=str).encode()
    ).hexdigest()


# Smallest cap for which the never-exceeds-limit guarantee holds: the marker
# is reserved inside the limit, and it is well under this.
_MIN_USEFUL_CAP = 1_000

# Longest tool name echoed into a truncation marker. Real names are well
# under this (the longest in the MCP surface is ~40 chars); the bound exists
# so a model-supplied name cannot inflate the marker past its own budget.
_TOOL_NAME_MARKER_MAX = 80
# Once a turn's reads fill what its model can hold (ReadingLimits.per_turn),
# a further tool result still shows this much, so the model sees what it got.
_TURN_BUDGET_FLOOR = 3_000

# Results the turn's reading budget counts but never cuts (see
# _turn_result_limit).
_NEVER_CUT_BY_TURN_BUDGET = frozenset(
    {step_script.RUN_SCRIPT_TOOL, tool_groups.OPEN_TOOLS, "search_tools"}
)

# Calls that change nothing a held result came from, though their names
# don't start with a read verb (tool_catalog.read_only_label): asking a
# specialist, finding or opening tools, and showing the person progress.
_READS_NOTHING = frozenset(
    {"consult_specialist", tool_groups.OPEN_TOOLS, "search_tools", turn_progress.SHOW_PROGRESS}
)


def _safe_tool_name(tool_name: Any) -> str:
    # ``tool_name`` comes from the model's tool_use block, so it is
    # attacker-influenceable via prompt injection. Two consequences:
    #   * it is never passed through ``str.format`` — a name containing
    #     braces would otherwise be interpreted as a field reference,
    #     substituting our locals or raising and killing the turn;
    #   * it is length-bounded, so the marker cannot outgrow the budget
    #     it is supposed to fit inside.
    return tool_name[:_TOOL_NAME_MARKER_MAX] if isinstance(tool_name, str) else "?"


# Once a turn's prompt holds this share of its model's window (the calls it
# wrote, its text and its reads, as the model counted them), it answers.
_PROMPT_FULL_SHARE = 0.8

# Scripts written about one token a character (Chinese, Japanese, Korean)
# start here; the limits count about three characters a token, so such text
# counts three times its length (text_weight).
_DENSE_SCRIPT_FROM = 0x2E80
_DENSE_SAMPLE = 20_000

# The reading limits of the turn whose tool is running (with_turn_limits),
# for a tool that sizes its own parts (orchestrator.document_tools).
_TURN_LIMITS: ContextVar[ReadingLimits | None] = ContextVar("turn_reading_limits", default=None)
_T = TypeVar("_T")


async def with_turn_limits(limits: ReadingLimits, call: Awaitable[_T]) -> _T:
    """Run one tool call of a turn that reads with ``limits``: the tool reads
    them back with ``turn_limits``. Bound for that call only, so nothing
    else in the task sees them after it."""
    token = _TURN_LIMITS.set(limits)
    try:
        return await call
    finally:
        _TURN_LIMITS.reset(token)


def turn_limits() -> ReadingLimits | None:
    return _TURN_LIMITS.get()


def text_weight(text: str) -> float:
    """How many plain characters each character of ``text`` counts as: 1 for
    text written in letters, up to 3 for text written about one token a
    character, from a sample of it."""
    sample = text[:_DENSE_SAMPLE]
    if not sample:
        return 1.0
    dense = sum(1 for ch in sample if ord(ch) >= _DENSE_SCRIPT_FROM)
    return 1.0 + 2.0 * dense / len(sample)


def _part_marker(name: str, *, start: int, end: int, total: int) -> str:
    """Said after a part of a result that goes on past ``end``."""
    return (
        f"\n\n[PART of a long result from `{name}`: this shows characters "
        f"{start + 1:,} to {end:,} of {total:,}. This is NOT the whole result. "
        "To read the next part, call the tool again with exactly the same "
        "input: that repeat does not run the tool again, it brings back the "
        "next part. Read on until you have what the request needs, and read "
        "every part before you edit, rewrite or replace anything from it.]"
    )


def _last_part_marker(name: str, *, start: int, total: int) -> str:
    return f"\n\n[LAST PART of the result from `{name}`: characters {start + 1:,} to {total:,} of {total:,}.]"


def _full_marker(name: str, *, shown: int, total: int) -> str:
    """Said after a result cut because the turn holds all it can."""
    return (
        f"\n\n[This question has read all it can hold at once: this shows the "
        f"first {shown:,} of {total:,} characters from `{name}`. This is NOT "
        "the whole result. Answer from what you have read, say plainly what "
        "you have not read yet, and offer to carry on with the rest in a new "
        "message. Never edit, rewrite or replace anything from this cut result.]"
    )


def _cap_tool_result(text: Any, *, tool_name: str, limit: int, budget_spent: bool = False) -> Any:
    """The first part of one tool result, sized to fit before it enters the
    prompt.

    ``limit`` comes from the model (providers.registry.reading_limits_for),
    so an ordinary result, a long document included, fits whole. A longer
    one comes back in parts: this is the first, and its marker says how to
    get the next (``TurnReadingBudget.replay``). ``budget_spent`` says the
    turn has read all its model can hold: the marker then says so plainly
    and offers to carry on, rather than another part.

    The marker is load-bearing. Truncating silently is worse than not
    truncating: the model reads the cut text as the whole answer and
    confabulates the rest.

    Non-``str`` results pass through untouched. The result never exceeds
    ``limit`` (at least ``_MIN_USEFUL_CAP``): the marker is reserved inside it.
    """
    if not isinstance(text, str):
        return text
    if len(text) <= limit:
        return text
    name = _safe_tool_name(tool_name)
    # Reserve against the widest marker: ``shown`` never exceeds ``limit``.
    widest = (
        _full_marker(name, shown=limit, total=len(text))
        if budget_spent
        else _part_marker(name, start=0, end=limit, total=len(text))
    )
    shown = max(0, limit - len(widest))
    logger.warning(
        "tool_result in parts tool=%s original_chars=%d shown_chars=%d limit=%d budget_spent=%s",
        name, len(text), shown, limit, budget_spent,
    )
    marker = (
        _full_marker(name, shown=shown, total=len(text))
        if budget_spent
        else _part_marker(name, start=0, end=shown, total=len(text))
    )
    return text[:shown] + marker


def _turn_result_limit(name: str, *, per_result: int, turn_budget: int, used: int) -> tuple[int, bool]:
    """The size of one tool result's first part, given what the turn has read.

    Every tool result stays in the prompt for each later call of the turn,
    so ``turn_budget`` (ReadingLimits.per_turn, a share of what the model
    holds) bounds what a turn's results add together: once ``used`` reaches
    it, a further result is cut to ``_TURN_BUDGET_FLOOR``. Returns (limit,
    whether the budget rather than ``per_result`` set it). A specialist's
    analysis is the answer itself, not a read, so the budget never cuts it;
    it fills the window like a read, so it counts (``TurnReadingBudget.cap``). A built tool's result counts but is never cut below
    ``per_result``: it is already bounded (step_script), and it lists the
    writes that already ran, which a short cut would hide and the model would
    then repeat. Tool discovery (``open_tools``, ``search_tools``) is treated
    the same way: its result is the schema for the next call, and a cut one
    would leave the model unable to make it.
    """
    if turn_budget <= 0 or name == "consult_specialist":
        return per_result, False
    if name in _NEVER_CUT_BY_TURN_BUDGET:
        return per_result, False
    remaining = turn_budget - used
    if remaining >= per_result:
        return per_result, False
    if remaining >= _MIN_USEFUL_CAP:
        # Room is left, if less than a whole part: the next part is smaller.
        return remaining, False
    return _TURN_BUDGET_FLOOR, True


class TurnReadingBudget:
    """What one turn's reads have put in the prompt, and the rest of each
    result that came back in parts.

    Sizes come from the turn's model (providers.registry.reading_limits_for).
    A result longer than a part is kept here whole; when the model repeats
    the call, same tool and same input, the loop answers it with the next
    part (``replay``) and never runs the tool again. Replaying rather than
    re-running matters: any tool's result can be long, and a repeated write
    (an event, a message) would happen twice.

    A held read is the one in hand when it ran. Once a round runs a call
    that may change things (``ran``: anything not plainly a read), the parts
    held for reads are dropped, so a repeat after a click or an edit reads
    what is there now. The parts of a call that may change things are kept
    until read to the end, so its repeat never runs it again. Text written
    about one token a character counts three times its length
    (``text_weight``). The content check's notice on a
    result (``notice``) goes on every one of its parts.
    """

    def __init__(self, limits: ReadingLimits) -> None:
        self.limits = limits
        self._prompt_tokens = 0
        # ``used`` when the prompt was last counted: reads since then are
        # not in that count yet.
        self._used_at_prompt = 0
        self.per_result = limits.per_read
        self.turn_budget = limits.per_turn
        self.used = 0
        # Each result that came back in parts, by call: its whole text, where
        # the next part starts, the notice each part carries, and whether the
        # call only read (so a later change drops it).
        self._parts: dict[str, tuple[str, int, str, bool]] = {}
        # tool_use ids answered by ``replay``: already sized, not cut again.
        self._replayed: set[str] = set()
        # How often each call came back with the same result (call_key).
        self._repeats: dict[str, int] = {}

    def came_back(self, name: str, tool_input: Any, result: Any) -> int:
        """Note what one call returned; how many times it has now come back
        the same (``repeating`` at REPEAT_LIMIT)."""
        key = call_key(name, tool_input, result)
        self._repeats[key] = seen = self._repeats.get(key, 0) + 1
        return seen

    @property
    def repeating(self) -> bool:
        """A call came back the same REPEAT_LIMIT times: the turn should
        write up from what it has."""
        return max(self._repeats.values(), default=0) >= REPEAT_LIMIT

    def prompt_was(self, tokens: int) -> None:
        """Note how big the turn's last prompt was, as the model counted it:
        everything in it (the calls the turn wrote, its text, its reads)."""
        if tokens >= self._prompt_tokens:
            self._prompt_tokens = tokens
            self._used_at_prompt = self.used

    def saw(self, usage: Any) -> None:
        """``prompt_was`` from a response's usage: all the model read, cached
        or not. Nothing when the provider gave no usage."""
        if usage is not None:
            self.prompt_was(
                sum(
                    int(getattr(usage, k, 0) or 0)
                    for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
                )
            )

    def _used(self) -> int:
        """What counts as read for sizing the next part: the turn's reads, or
        more when the prompt (system, history, attachments, what the turn
        wrote) leaves less room before most of the window than that."""
        window = self.limits.window_tokens
        if not window or not self._prompt_tokens:
            return self.used
        room = int(window * _PROMPT_FULL_SHARE - self._prompt_tokens) * _CHARS_PER_TOKEN
        room -= self.used - self._used_at_prompt
        return max(self.used, self.turn_budget - room)

    @property
    def full(self) -> bool:
        """The turn has read all its model holds, or its prompt has filled
        most of the window (``window_full``): it should answer now."""
        return self.used >= self.turn_budget or self.window_full

    @property
    def window_full(self) -> bool:
        """The prompt has filled most of the window (``_PROMPT_FULL_SHARE``):
        another round of writing would not fit."""
        window = self.limits.window_tokens
        return bool(window and self._prompt_tokens >= window * _PROMPT_FULL_SHARE)

    @staticmethod
    def _key(name: str, tool_input: Any) -> str:
        try:
            return name + "\x00" + json.dumps(tool_input, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return name + "\x00" + repr(tool_input)

    def replay(self, tool_use: dict[str, Any]) -> str | None:
        """The next part if ``tool_use`` repeats a call whose result came back
        in parts; the caller answers it with this text instead of running
        it. ``None`` otherwise."""
        key = self._key(tool_use["name"], tool_use["input"])
        held = self._parts.get(key)
        if held is None:
            return None
        text, start, notice, reads = held
        name = _safe_tool_name(tool_use["name"])
        limit, budget_spent = _turn_result_limit(
            tool_use["name"], per_result=self.per_result, turn_budget=self.turn_budget, used=self._used()
        )
        weight = text_weight(text[start:])
        limit = max(_MIN_USEFUL_CAP, int(limit / weight) - len(notice))
        if budget_spent:
            # Nothing more fits: say so, and keep the rest for no one.
            del self._parts[key]
            self._replayed.add(tool_use["id"])
            return (
                f"[This question has read all it can hold at once. The result from `{name}` "
                f"was read up to character {start:,} of {len(text):,}. Answer from what you "
                "have read, say plainly what you have not read yet, and offer to carry on "
                "with the rest in a new message.]"
            )
        last = _last_part_marker(name, start=start, total=len(text))
        if len(text) - start + len(last) <= limit:
            del self._parts[key]
            part = text[start:] + last
        else:
            room = max(1, limit - len(_part_marker(name, start=start, end=start + limit, total=len(text))))
            end = start + room
            self._parts[key] = (text, end, notice, reads)
            part = text[start:end] + _part_marker(name, start=start, end=end, total=len(text))
        self._replayed.add(tool_use["id"])
        self.used += int(len(part) * weight)
        return notice + part

    @staticmethod
    def _only_reads(name: str, tool_input: Any) -> bool:
        inner = tool_input.get("name") if isinstance(tool_input, dict) else None
        if name == "call_tool" and isinstance(inner, str):
            name = inner  # a connected app's tool, through the gateway: judged by its own name
        return name in _READS_NOTHING or bool(tool_catalog.read_only_label(name, {}))

    def ran(self, tool_uses: list[dict[str, Any]]) -> None:
        """Note the calls a round ran: one that may have changed something
        (not plainly a read) drops the parts held for reads, so the next
        repeat reads afresh. Next parts handed back by ``replay`` ran
        nothing."""
        for tu in tool_uses:
            if tu.get("id") in self._replayed:
                continue
            if self._only_reads(str(tu.get("name") or ""), tu.get("input")):
                continue
            self._parts = {k: v for k, v in self._parts.items() if not v[3]}
            return

    def cap(self, name: str, tool_input: Any, text: Any, tool_use_id: str = "", notice: str = "") -> Any:
        """``text`` sized for the prompt (its first part when it is long),
        counted against the turn's reading. ``notice`` is the content check's
        notice ``text`` starts with, for its later parts too."""
        if tool_use_id and tool_use_id in self._replayed:
            return text  # sized and counted by ``replay``
        weight = text_weight(text) if isinstance(text, str) else 1.0
        limit, budget_spent = _turn_result_limit(
            name, per_result=self.per_result, turn_budget=self.turn_budget, used=self._used()
        )
        limit = max(_MIN_USEFUL_CAP, int(limit / weight)) if not budget_spent else limit
        content = _cap_tool_result(text, tool_name=name, limit=limit, budget_spent=budget_spent)
        if not budget_spent and isinstance(text, str) and content is not text:
            # Where the next part starts: what the first part showed.
            start = content.rindex("\n\n[PART")
            reads = self._only_reads(name, tool_input)
            if notice and text.startswith(notice) and start > len(notice):
                self._parts[self._key(name, tool_input)] = (text[len(notice):], start - len(notice), notice, reads)
            else:
                self._parts[self._key(name, tool_input)] = (text, start, "", reads)
        if isinstance(content, str):
            # A specialist's analysis is never cut, but it fills the window
            # like any read, so it counts toward ``full``.
            self.used += int(len(content) * weight)
        return content
