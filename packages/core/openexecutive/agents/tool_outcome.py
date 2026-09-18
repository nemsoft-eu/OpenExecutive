"""What a client-side tool handler returns when it needs to report failure.

Most handlers return a plain string and never fail in a way the model must
be told about. A handler that can — today the SearXNG ``web_search`` — returns
a ``ToolOutcome`` instead, and the two tool loops that dispatch handlers (the
Executive's and ``BaseAgent.analyze_with_tools``) both go through
``unwrap_tool_outcome`` so they cannot disagree on what a result means.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolOutcome:
    """A handler's result plus whether it reports a failure.

    The content is always a string: the OpenAI-compatible translator only
    converts string or text-block ``tool_result`` content, so a dict here
    would be silently flattened on the local path.

    Callers stamp ``is_error`` onto the outer Anthropic ``tool_result``
    block. Be clear about what that buys: the OpenAI-compatible translator
    forwards only ``tool_call_id`` and ``content``, so on the local path the
    flag is dropped and the model learns of the failure from the wording of
    ``content`` alone. Error text must therefore read unmistakably as an
    error on its own; the flag is correct Anthropic shape, not the signal.
    """

    content: str
    is_error: bool = False


def unwrap_tool_outcome(result: str | ToolOutcome) -> tuple[str, bool]:
    """``(content, is_error)`` for any handler result, ToolOutcome or string."""
    if isinstance(result, ToolOutcome):
        return result.content, result.is_error
    return result, False
