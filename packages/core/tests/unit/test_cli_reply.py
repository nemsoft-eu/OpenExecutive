"""The terminal marks a reply that was written again, rather than showing two."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from openexecutive import cli
from openexecutive.orchestrator.executive import Executive


async def _stream(*chunks: str | dict[str, Any]) -> AsyncIterator[str | dict[str, Any]]:
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_a_rewritten_reply_is_marked_and_returned_alone(capsys: pytest.CaptureFixture[str]) -> None:
    final = await cli.print_reply(
        _stream(Executive._THINKING, "I saved the doc.", {"type": "rewrite"}, "I couldn't save the doc.")
    )
    out = capsys.readouterr().out
    assert final == "I couldn't save the doc."
    assert "Corrected reply (replaces the one above)" in out
    assert out.index("I saved the doc.") < out.index("Corrected") < out.index("I couldn't save")


@pytest.mark.asyncio
async def test_a_plain_reply_prints_as_is(capsys: pytest.CaptureFixture[str]) -> None:
    final = await cli.print_reply(_stream("Hello", {"type": "status", "text": "x"}, " there"))
    assert final == "Hello there"
    assert "Corrected" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_a_kept_first_reply_prints_once(capsys: pytest.CaptureFixture[str]) -> None:
    final = await cli.print_reply(_stream("I saved the doc.", {"type": "rewrite"}, "I saved ", "the doc."))
    out = capsys.readouterr().out
    assert final == "I saved the doc."
    assert out.count("I saved the doc.") == 1 and "Corrected" not in out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunks", "final", "corrected"),
    [
        (["Hi there.", {"type": "rewrite"}], "Hi there.", False),
        (["I saved the doc.", {"type": "rewrite"}, "I saved"], "I saved", True),
        (["I saved the doc.", {"type": "rewrite"}, "I saved ", "nothing."], "I saved nothing.", True),
        (["I sent it.", {"type": "rewrite"}, "I ", {"type": "rewrite"}, "I sent it."], "I sent it.", False),
        (["A.", {"type": "rewrite"}, "B.", {"type": "rewrite"}, "C."], "C.", True),
    ],
)
async def test_what_the_terminal_shows_is_what_it_returns(
    chunks: list[str | dict[str, Any]], final: str, corrected: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    assert await cli.print_reply(_stream(*chunks)) == final
    out = capsys.readouterr().out
    assert ("Corrected" in out) is corrected
    assert out.rstrip().endswith(final)


@pytest.mark.asyncio
async def test_a_kept_reply_that_began_with_a_newline_prints_once(capsys: pytest.CaptureFixture[str]) -> None:
    """The replayed first reply is the same text, so nothing is corrected."""
    from openexecutive.orchestrator.executive import _kept_if_unwritten

    stream = _kept_if_unwritten(_stream("\nI saved the doc.", {"type": "rewrite"}), Executive._THINKING)
    assert await cli.print_reply(stream) == "\nI saved the doc."
    assert "Corrected" not in capsys.readouterr().out
