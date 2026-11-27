"""Shared helpers for workflows that run an Executive tool-use loop.

Both ``executive_reflection`` and ``executive_research`` drive the
Executive through a tool-use loop with the full outbound toolkit.
The two helpers below are the bits both loops share:

- ``execute_tool_call`` runs one tool call against the pass's handlers
  and returns its summary (tool, input_preview, result_preview, ok) with
  the whole result for the model. Failures are caught + summarized so the
  loop survives a single tool crash. The loop itself is
  ``orchestrator.rounds.run_rounds``.
- ``extract_artifact_from_response`` pulls the concatenated text-block
  content from a provider response — the model's narrative summary
  emitted alongside or after its tool calls.

Promoting these out of ``executive_reflection`` removes the private-
import tripwire ``executive_research`` would otherwise carry — a future
refactor renaming or moving them could silently break the other
workflow.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _skipped(name: str, tool_input: Any, why: str, *, ran: bool = False) -> tuple[dict[str, Any], str]:
    return {"tool": name, "input_preview": str(tool_input)[:120], "result_preview": why, "ok": False, "ran": ran}, why


async def execute_tool_call(
    name: str,
    tool_input: dict[str, Any],
    handlers: dict[str, Any],
    *,
    over_budget: bool = False,
) -> tuple[dict[str, Any], Any]:
    """Run one tool call of a pass (``orchestrator.rounds.run_rounds``'s
    ``run_call``).

    Returns its summary for the pass's record and artifact
    (``{tool, input_preview, result_preview, ok, ran}``; ``ran`` is whether
    the handler ran, error or not) and the result the model
    reads back, whole: the pass's reading sizes it (orchestrator.reading).
    A failure is caught and summarized, so one tool crash does not end the
    pass. ``over_budget`` refuses the call without running it (a pass's own
    count of the outbound calls it may make).
    """
    if over_budget:
        return _skipped(name, tool_input, "over budget — skipped")
    handler = handlers.get(name)
    if handler is None:
        return _skipped(name, tool_input, "unknown tool — skipped")
    try:
        result = await handler(tool_input)
    except Exception as exc:
        logger.exception("synthesis: tool %s raised", name)
        return _skipped(name, tool_input, f"raised: {exc}"[:160], ran=True)
    ok = True
    try:
        parsed = json.loads(result) if isinstance(result, str) else result
        if isinstance(parsed, dict) and "error" in parsed:
            ok = False
    except (ValueError, TypeError):
        # Non-JSON result — treat as success since we have no
        # signal otherwise (same as action_chips).
        pass
    return {
        "tool": name,
        "input_preview": str(tool_input)[:120],
        "result_preview": str(result)[:160],
        "ok": ok,
        "ran": True,
    }, result


def extract_artifact_from_response(response: Any) -> str:
    """Pull the final text summary from an Anthropic response.

    Returns the concatenated text of all text blocks. Empty string when
    the model only emitted tool_use blocks — callers fall back to a
    synthesized "no summary" artifact.
    """
    if not getattr(response, "content", None):
        return ""
    chunks: list[str] = []
    for block in response.content:
        if getattr(block, "type", "") == "text":
            text = getattr(block, "text", "")
            if text:
                chunks.append(text)
    return "\n\n".join(chunks).strip()


__all__ = ["execute_tool_call", "extract_artifact_from_response"]
