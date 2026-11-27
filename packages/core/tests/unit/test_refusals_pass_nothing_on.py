"""Every refusal the person asking hears is said plainly and passed to no one.

A refusal that told the Executive "the owner needs to make this change" or
"tell whoever asked that the principal needs to do it" led it to message the
owner with the request, so one person's ask landed on someone else's list.
Refusals end with ``speaker.PASS_ON_NOTHING`` instead; this keeps the old
wording from coming back anywhere in the Executive's tools.
"""
from __future__ import annotations

import ast
import asyncio
import json
import re
from pathlib import Path

import pytest

import openexecutive

_HANDS_IT_ON = re.compile(
    r"(owner|principal) needs to (make|do|add|record|tell|change)", re.IGNORECASE
)


def test_no_refusal_tells_the_executive_someone_else_needs_to_do_it() -> None:
    """Checked on whole string constants (``ast`` joins the literals a long
    refusal is split across), in every module of the package; docstrings
    aren't said to anyone."""
    root = Path(openexecutive.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        docstrings = {
            id(n.body[0].value)
            for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and n.body and isinstance(n.body[0], ast.Expr)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and _HANDS_IT_ON.search(" ".join(node.value.split()))
            ):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == []


def test_the_line_answers_the_asker_and_sends_nothing_to_anyone_else() -> None:
    """On email the answer to the asker is a reply the Executive sends, so
    the line must never forbid that: only messages to anyone else."""
    from openexecutive.orchestrator.speaker import PASS_ON_NOTHING

    assert PASS_ON_NOTHING.startswith("Answer only the person who asked")
    assert "send nothing to anyone else on their behalf" in PASS_ON_NOTHING
    assert "to anyone about it" not in PASS_ON_NOTHING


def test_the_decision_and_goal_refusals_answer_only_the_asker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Called the way a turn that may not do it calls them."""
    from openexecutive.orchestrator import decision_tools, department_tools
    from openexecutive.orchestrator.speaker import PASS_ON_NOTHING

    for module in (decision_tools, department_tools):
        monkeypatch.setattr(module, "_audit", lambda *a, **k: None)
        monkeypatch.setattr(module, "_principal_asked", lambda: False)

    outcome = json.loads(asyncio.run(decision_tools.handle_record_decision_outcome(
        {"decision_id": 1, "outcome": "It worked."}
    )))
    goal = json.loads(asyncio.run(department_tools.handle_create_goal(
        {"area": "sales", "title": "Close ten deals"}
    )))
    assert PASS_ON_NOTHING in outcome["error"]
    assert PASS_ON_NOTHING in goal["error"]


def test_the_unverified_roster_refusal_answers_only_the_asker() -> None:
    """An inbound email or a channel nobody is verified on."""
    from types import SimpleNamespace

    from openexecutive.orchestrator.people_tools import _roster_refusal_reason
    from openexecutive.orchestrator.speaker import PASS_ON_NOTHING

    for session in (None, SimpleNamespace(origin_channel="email", email_from="x@example.com", caller_person_id=None)):
        reason = _roster_refusal_reason(session)
        assert reason is not None and PASS_ON_NOTHING in reason
