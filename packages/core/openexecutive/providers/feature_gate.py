"""Spec-driven stripping of Anthropic-only request features.

When a non-Claude model is selected (or a Claude model is routed via a
backend that doesn't honor Anthropic-only features), we strip the
unsupported fields before the request leaves our process. The gate is
applied inside the OpenRouter provider so any future re-routing keeps
the same guarantees — a caller cannot accidentally ship a request with
``cache_control`` to a model that would 400 on it.
"""
from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureSpec:
    """What a given model is allowed to receive.

    All flags default to True for Claude family; the registry's
    ``MODEL_SPECS`` table flips them off for non-Claude models so a
    misconfigured caller can't bypass the gate.
    """

    supports_cache_control: bool = True
    supports_thinking: bool = True
    supports_web_search: bool = True
    supports_tool_use: bool = True
    # Whether a PDF ``document`` block may be sent. Anthropic reads it
    # natively and OpenRouter parses one for any model; a self-hosted
    # OpenAI-compatible server usually cannot (LOCAL_PDF_INPUT says it can).
    supports_pdf_input: bool = True
    # The most output a request may ask for (``max_tokens``); None = no cap.
    # A self-hosted server can refuse a request whose prompt plus
    # ``max_tokens`` exceeds its context (LOCAL_MAX_OUTPUT_TOKENS).
    max_output_tokens: int | None = None


# The most one answer may write: every Executive round, the committee
# revision and a specialist's analysis. Claude takes it; every other model
# gets at most NON_CLAUDE_MAX_OUTPUT_TOKENS, and a self-hosted one
# LOCAL_MAX_OUTPUT_TOKENS. An Executive round that reaches it carries on in
# the next round (orchestrator/executive.py, CUT_OFF_NOTE), so a long file or
# answer is written in parts rather than stopped.
MAX_OUTPUT_TOKENS = 32_000

# The most a specialist's analysis may write (the owner's choice, 11-18).
# The Executive's own answers keep MAX_OUTPUT_TOKENS.
SPECIALIST_MAX_TOKENS = 8_000


# What a non-Claude model is asked to write at most. Hosted models cap their
# output at different limits (gpt-4o at 16,384) and refuse a larger request.
# 16,000 is what the PDF reader and the fixture generator already ask of them,
# so the cap never lowers a request those callers made before.
NON_CLAUDE_MAX_OUTPUT_TOKENS = 16_000


# How much each model can hold (its context window, in tokens). Claude from
# 4.7 on, and every Claude 5, holds 1,000,000; older Claude is taken at
# 200,000 (4.6 serves 1M only where a deployment turns it on). Another
# hosted model is taken at 128,000 (gpt-4o's window, and the smallest of the
# hosted models offered); a self-hosted one holds LOCAL_CONTEXT_TOKENS.
CLAUDE_CONTEXT_TOKENS = 1_000_000
OLDER_CLAUDE_CONTEXT_TOKENS = 200_000
NON_CLAUDE_CONTEXT_TOKENS = 128_000

# Claude ids and OpenRouter slugs with their version (``claude-opus-5-5``,
# ``anthropic/claude-haiku-4.5``, ``claude-sonnet-4-6-20260115``).
_CLAUDE_VERSION_RE = re.compile(
    r"^(?:anthropic/)?claude-[a-z]+-(?P<major>\d{1,3})(?:[-.](?P<minor>\d{1,2}))?(?:-\d{8})?$"
)

# What one turn's reads may fill: a share of the model's window, the rest
# left for the instructions, the conversation and the answer. Counted at 3
# characters a token, under what English text takes (about 4), so a dense
# read (figures, code, another language) still fits.
_READ_SHARE = 0.4
_CHARS_PER_TOKEN = 3

# The most one turn's reads add up to, whatever the window (the owner's
# choice, 11-18). At it the turn writes its answer from what it has.
MAX_TURN_READ_CHARS = 200_000


def context_tokens(model: str, *, local_models: Any = (), local_context_tokens: int = 0) -> int:
    """How many tokens ``model`` can hold.

    ``local_models`` / ``local_context_tokens`` are the self-hosted models
    and what their server holds (LOCAL_MODELS, LOCAL_CONTEXT_TOKENS).
    """
    if local_context_tokens and model in local_models:
        return local_context_tokens
    m = _CLAUDE_VERSION_RE.match(model.lower())
    if m is None:
        return NON_CLAUDE_CONTEXT_TOKENS
    major, minor = int(m.group("major")), int(m.group("minor") or 0)
    if major >= 5 or (major == 4 and minor >= 7):
        return CLAUDE_CONTEXT_TOKENS
    return OLDER_CLAUDE_CONTEXT_TOKENS


@dataclass(frozen=True)
class ReadingLimits:
    """How much a turn may read, in characters, from its model's window.

    ``per_read`` is one part of one read (a file, a page, a tool's answer):
    a longer one comes back in parts. ``per_turn`` is what all of a turn's
    reads may add together, since each stays in the prompt for the rest of
    the turn. ``window_tokens`` is the model's whole window (0: unknown).
    """

    per_read: int
    per_turn: int
    window_tokens: int = 0


def reading_limits(window_tokens: int) -> ReadingLimits:
    """The reading limits for a model that holds ``window_tokens``."""
    per_turn = min(int(window_tokens * _READ_SHARE) * _CHARS_PER_TOKEN, MAX_TURN_READ_CHARS)
    return ReadingLimits(per_read=per_turn // 2, per_turn=per_turn, window_tokens=window_tokens)


# Claude models that reject a forced ``tool_choice`` (``any`` or a named
# ``tool``) with HTTP 400: Opus 5.5, Sonnet 5.5, Fable 5.1 and Mythos 5.1.
# Matched as an Anthropic id (``claude-opus-5-5``, optionally date-pinned)
# or an OpenRouter slug (``anthropic/claude-opus-5.5``).
_NO_FORCED_TOOL_CHOICE_RE = re.compile(
    r"^(?:anthropic/)?claude-(?:(?:opus|sonnet)-5[-.]5|(?:fable|mythos)-5[-.]1)(?:-\d{8})?$"
)


def rejects_forced_tool_choice(model: str) -> bool:
    """Whether ``model`` 400s on ``tool_choice`` ``any`` / ``tool``."""
    return bool(_NO_FORCED_TOOL_CHOICE_RE.match(model.lower()))


def relax_forced_tool_choice(model: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Turn a forced ``tool_choice`` into ``auto`` for models that reject it.

    Callers that force a tool (the emit-a-structured-result pattern) already
    name that tool in their prompt and handle a reply without a ``tool_use``
    block, so ``auto`` keeps them working where the forced form would 400.
    ``disable_parallel_tool_use`` is kept. Returns ``kwargs`` itself when
    nothing changes, else a shallow copy — the caller's dict is never mutated.
    """
    choice = kwargs.get("tool_choice")
    if not (
        isinstance(choice, dict)
        and choice.get("type") in ("any", "tool")
        and rejects_forced_tool_choice(model)
    ):
        return kwargs
    relaxed: dict[str, Any] = {"type": "auto"}
    if "disable_parallel_tool_use" in choice:
        relaxed["disable_parallel_tool_use"] = choice["disable_parallel_tool_use"]
    return {**kwargs, "tool_choice": relaxed}


# Claude ids and OpenRouter slugs with their major version, for the
# per-generation request rules below (``claude-haiku-5-5``,
# ``anthropic/claude-haiku-5.5``, ``claude-opus-5-20260315``).
_CLAUDE_MAJOR_RE = re.compile(
    r"^(?:anthropic/)?claude-[a-z]+-(?P<major>\d{1,3})(?:[-.]\d{1,3})?(?:-\d{8})?$"
)
# Claude models from the 5 generation on reject a non-default ``temperature``
# / ``top_p`` / ``top_k`` with HTTP 400; the defaults are what they run on
# anyway, so the fields are dropped rather than sent.
_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")
# Haiku 5.5 thinks unless a request turns it off, where Haiku 4.5 thought
# only when asked. The helpers that run on the routing model are sized for a
# reply without thinking (a title gets 32 tokens), so a request that says
# nothing about thinking gets it turned off. Anthropic accepts that only up
# to effort ``high``.
_THINKS_UNLESS_TOLD_RE = re.compile(r"^(?:anthropic/)?claude-haiku-5[-.]5(?:-\d{8})?$")
_EFFORTS_THAT_KEEP_THINKING = frozenset({"xhigh", "max"})


def fit_claude_generation(model: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Shape a request to the rules of the Claude generation it goes to.

    Drops sampling fields a 5-generation model would reject, and turns
    thinking off on a model that otherwise thinks by default when the caller
    set none. Returns ``kwargs`` itself when nothing changes, else a shallow
    copy; the caller's dict is never mutated.
    """
    lowered = model.lower()
    m = _CLAUDE_MAJOR_RE.match(lowered)
    if m is None:
        return kwargs
    out = kwargs
    if int(m.group("major")) >= 5 and any(k in kwargs for k in _SAMPLING_PARAMS):
        out = {k: v for k, v in kwargs.items() if k not in _SAMPLING_PARAMS}
    if "thinking" not in out and _THINKS_UNLESS_TOLD_RE.match(lowered):
        output_config = out.get("output_config")
        effort = output_config.get("effort") if isinstance(output_config, dict) else None
        if effort not in _EFFORTS_THAT_KEEP_THINKING:
            out = {**out, "thinking": {"type": "disabled"}}
    return out


# What a PDF becomes for a model that cannot read one: said, never dropped.
PDF_OMITTED_NOTE = "[PDF omitted: this model can't read PDF files]"


def apply_feature_gates(spec: FeatureSpec, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a new kwargs dict with unsupported features removed.

    Never mutates the input — the caller's dict is reused on retry paths
    and elsewhere, and an in-place strip would corrupt those.
    """
    if (
        spec.supports_cache_control
        and spec.supports_thinking
        and spec.supports_web_search
        and spec.supports_tool_use
        and spec.supports_pdf_input
        and _fits_output_cap(spec, kwargs)
    ):
        # Fast path: nothing to strip.
        return kwargs

    out = deepcopy(kwargs)

    if not spec.supports_cache_control:
        _strip_cache_control(out)

    if not spec.supports_thinking:
        out.pop("thinking", None)
        out.pop("output_config", None)

    if not spec.supports_web_search:
        _strip_web_search_tools(out)

    if not spec.supports_tool_use:
        out.pop("tools", None)
        out.pop("tool_choice", None)

    if not spec.supports_pdf_input:
        _replace_documents(out)

    if not _fits_output_cap(spec, out):
        out["max_tokens"] = spec.max_output_tokens

    return out


def _fits_output_cap(spec: FeatureSpec, kwargs: dict[str, Any]) -> bool:
    asked = kwargs.get("max_tokens")
    return spec.max_output_tokens is None or not isinstance(asked, int) or asked <= spec.max_output_tokens


def _strip_cache_control(kwargs: dict[str, Any]) -> None:
    """Remove ``cache_control`` from system blocks and tool entries.

    Non-Claude OpenRouter models 400 on unknown fields, and even where
    they tolerate them, the field has no semantic effect — so dropping
    it is the right call. Anthropic-routed callers never hit this path.
    """
    system = kwargs.get("system")
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                block.pop("cache_control", None)

    tools = kwargs.get("tools")
    if isinstance(tools, list):
        for t in tools:
            if isinstance(t, dict):
                t.pop("cache_control", None)

    # cache_control also appears on user-turn content blocks — notably the
    # agent loop's intra-turn marker, which rides on a tool_result block.
    # Strip it there too. One level deep is enough because every marker we
    # emit sits at a content block's own top level, never nested inside a
    # tool_result's own content list.
    messages = kwargs.get("messages")
    if isinstance(messages, list):
        for m in messages:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block.pop("cache_control", None)


def _strip_web_search_tools(kwargs: dict[str, Any]) -> None:
    """Remove Anthropic server-side web_search / web_fetch tool entries.

    These use a ``type: "web_search_…"`` field instead of ``input_schema``;
    OpenRouter has no equivalent server tool, so they must be dropped.
    """
    tools = kwargs.get("tools")
    if not isinstance(tools, list):
        return
    kwargs["tools"] = [
        t
        for t in tools
        if not (
            isinstance(t, dict)
            and isinstance(t.get("type"), str)
            and t["type"].startswith(("web_search_", "web_fetch_"))
        )
    ]
    if not kwargs["tools"]:
        kwargs.pop("tools", None)


def _replace_documents(kwargs: dict[str, Any]) -> None:
    """Swap every ``document`` block in the messages for a text note, so the
    model is told a PDF was there instead of the block vanishing in
    translation."""
    for message in kwargs.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        message["content"] = [
            {"type": "text", "text": PDF_OMITTED_NOTE}
            if isinstance(block, dict) and block.get("type") == "document"
            else block
            for block in content
        ]
