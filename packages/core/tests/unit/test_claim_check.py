"""Do what it says (orchestrator.claim_check): a reply that claims a change no
tool call made gets one more round."""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from openexecutive.orchestrator import claim_check
from openexecutive.orchestrator.executive import Executive
from tests.unit._agent_loop_fakes import FinalMsg, ScriptedProvider, TextBlock, ToolUseBlock


@pytest.fixture(autouse=True)
def _claim_check_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIM_CHECK_ENABLED", "true")
    monkeypatch.setattr("openexecutive.orchestrator.executive.audit_log", lambda *a, **k: None)


def _run(provider: ScriptedProvider, message: str = "Remove it from the brief") -> list[Any]:
    async def _go() -> list[Any]:
        items: list[Any] = []
        with patch("openexecutive.orchestrator.executive.get_provider", return_value=provider):
            async for item in Executive()._stream_agent_loop(
                system_blocks=[],
                messages=[{"role": "user", "content": message}],
                model="claude-test",
            ):
                items.append(item)
        return items

    return asyncio.run(_go())


def _verdict(payload: dict[str, Any]):
    calls: list[str] = []

    async def fake(model: str, turn: str) -> dict[str, Any]:
        calls.append(turn)
        return payload

    return fake, calls


# --- the pure parts ----------------------------------------------------------


def test_word_check_skips_replies_that_claim_nothing() -> None:
    assert not claim_check.may_claim("Revenue was up 4% on last quarter.")
    assert not claim_check.may_claim("   ")
    assert not claim_check.may_claim("Let me know if you want the numbers.")
    assert claim_check.may_claim("Done.")
    assert claim_check.may_claim("I'll treat the draft as closed.")
    assert claim_check.may_claim("Removed it from the brief")
    # A repeat or a watch promised in so many words is checked too.
    assert claim_check.may_claim("Your recap goes out every Friday at 4pm.")
    assert claim_check.may_claim("Any email from Sam gets flagged; I'll ping you right away.")
    assert claim_check.may_claim("Keeping an eye out for Sam's email.")
    assert claim_check.may_claim("I’ll flag it the moment it lands.")
    for timetable in ("Your recap goes out each Friday.", "It goes out weekly on Fridays.",
                      "It goes out every other Friday.", "You get it every two weeks.",
                      "The board pack goes out quarterly.", "I'll check every 30 minutes."):
        assert claim_check.may_claim(timetable), timetable
    # Saying a message went to someone is checked, however it is put.
    for sent in ("I just messaged her that you've offered to help.", "Emailed Pat the numbers.",
                 "I let Kendall know you're on it.", "I texted Robin about the heater.",
                 "Reached out to Jamie this morning.", "I forwarded it to Sam.",
                 "I already asked the supplier for a quote."):
        assert claim_check.may_claim(sent), sent


def test_judge_is_told_a_message_backs_only_a_real_send() -> None:
    assert "only a call that sent that message to that person" in claim_check._SYSTEM
    assert "never that it went" in claim_check._SYSTEM


def test_nudge_sends_what_was_asked_and_offers_the_rest() -> None:
    text = claim_check.nudge("I just messaged her")
    assert "if the person asked for it, make it now" in text
    assert "offer it instead" in text
    assert "say plainly you couldn't" in text
    # A send already on a card is not made a second time.
    assert "don't make it again: say it's waiting for their OK" in text


def test_judge_is_told_a_call_backs_only_what_it_made() -> None:
    # A one-time step is no repeat, and a note in memory is no watch.
    assert "scheduled once does not back a claim that something is sent or run" in claim_check._SYSTEM
    assert "only the timetable that" in claim_check._SYSTEM
    # A standing change still backs "from now on": that rule is about timetables.
    assert 'still backs \"from now on\"' in claim_check._SYSTEM
    assert "kept in memory does not back a claim that something" in claim_check._SYSTEM
    # What does back them is named, so a true repeat or watch isn't redone.
    for backs in ("workflow saved switched on", "add_watchlist_entry", "propose_watch"):
        assert backs in claim_check._SYSTEM


def test_claim_must_be_in_the_reply_word_for_word() -> None:
    reply = "Done — I’ll treat the draft as closed."
    assert claim_check.accept_claim({"unbacked": True, "claim": "I'll treat the draft as closed"}, reply)
    assert claim_check.accept_claim({"unbacked": True, "claim": "I sent the email"}, reply) is None
    assert claim_check.accept_claim({"unbacked": False, "claim": "Done"}, reply) is None
    assert claim_check.accept_claim({"unbacked": True, "claim": ""}, reply) is None
    assert claim_check.accept_claim({"unbacked": True, "claim": "x" * 300}, "x" * 300) is None


def test_last_person_message_skips_tool_result_turns() -> None:
    messages = [
        {"role": "user", "content": "I sent it"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]},
    ]
    assert claim_check.last_person_message(messages) == "I sent it"


def test_check_is_off_without_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIM_CHECK_ENABLED", "false")
    fake, calls = _verdict({"unbacked": True, "claim": "Done"})
    with patch.object(claim_check, "_call_model", fake):
        assert asyncio.run(claim_check.unbacked_claim("x", "Done", [])) is None
    assert calls == []


def test_a_failed_check_lets_the_reply_through() -> None:
    async def boom(model: str, turn: str) -> dict[str, Any]:
        raise RuntimeError("provider down")

    with patch.object(claim_check, "_call_model", boom):
        assert asyncio.run(claim_check.unbacked_claim("x", "Done", [])) is None


def test_the_check_sees_the_turns_calls() -> None:
    fake, calls = _verdict({"unbacked": False, "claim": ""})
    with patch.object(claim_check, "_call_model", fake):
        asyncio.run(claim_check.unbacked_claim(
            "close it", "Closed it.", [("close_open_loop", '{"status": "closed"}')],
        ))
    assert "close_open_loop" in calls[0]
    assert "<assistant_reply>\nClosed it." in calls[0]


# --- in the agent loop -------------------------------------------------------


def test_unbacked_claim_gets_one_more_round_with_the_nudge() -> None:
    provider = ScriptedProvider([
        FinalMsg([TextBlock("Done.")], "end_turn"),
        FinalMsg([TextBlock("I can't remove items from the brief yet.")], "end_turn"),
    ])
    fake, calls = _verdict({"unbacked": True, "claim": "Done"})
    with patch.object(claim_check, "_call_model", fake):
        items = _run(provider)

    assert len(provider.calls) == 2
    second = provider.calls[1]["messages"]
    assert second[-2] == {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}
    nudge = second[-1]["content"][0]["text"]
    assert "“Done”" in nudge and "no tool call" in nudge and "write your whole reply again" in nudge
    # The retry replaces the first reply wherever it is read, web and phone
    # included: they drop what streamed on ``rewrite``.
    assert {"type": "rewrite"} in items
    # The break between the two replies comes from the next round's text
    # (test_reply_rounds.py); the retry itself streams nothing of its own.
    assert "\n\n" not in items
    # Checked once a turn: the second reply ends it without another check.
    assert len(calls) == 1


def test_backed_or_plain_reply_ends_the_turn() -> None:
    provider = ScriptedProvider([FinalMsg([TextBlock("Done.")], "end_turn")])
    fake, calls = _verdict({"unbacked": False, "claim": ""})
    with patch.object(claim_check, "_call_model", fake):
        _run(provider)
    assert len(provider.calls) == 1
    assert len(calls) == 1


def test_reply_without_claim_words_makes_no_check_call() -> None:
    provider = ScriptedProvider([FinalMsg([TextBlock("Revenue was up 4%.")], "end_turn")])
    fake, calls = _verdict({"unbacked": True, "claim": "Revenue"})
    with patch.object(claim_check, "_call_model", fake):
        _run(provider, "How did revenue do?")
    assert calls == []
    assert len(provider.calls) == 1


def test_tool_calls_this_turn_reach_the_check() -> None:
    provider = ScriptedProvider([
        FinalMsg([ToolUseBlock("t1", "list_open_loops", {})], "tool_use"),
        FinalMsg([TextBlock("Closed it.")], "end_turn"),
    ])
    fake, calls = _verdict({"unbacked": False, "claim": ""})

    async def handler(_input: dict[str, Any]) -> str:
        return '{"loops": []}'

    with (
        patch.object(claim_check, "_call_model", fake),
        patch.dict(
            "openexecutive.orchestrator.executive._ALL_SKILL_HANDLERS",
            {"list_open_loops": handler},
        ),
    ):
        _run(provider, "I sent it")
    assert len(calls) == 1
    assert "- list_open_loops:" in calls[0]


def test_the_turns_calls_reach_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """``calls_out`` hands the turn's calls to the conversation review's record."""
    monkeypatch.setenv("CLAIM_CHECK_ENABLED", "false")
    provider = ScriptedProvider([
        FinalMsg([ToolUseBlock("t1", "list_open_loops", {})], "tool_use"),
        FinalMsg([TextBlock("Nothing is open.")], "end_turn"),
    ])

    async def handler(_input: dict[str, Any]) -> str:
        return '{"loops": []}'

    calls: list[tuple[str, str]] = []

    async def _go() -> None:
        with patch("openexecutive.orchestrator.executive.get_provider", return_value=provider):
            async for _ in Executive()._stream_agent_loop(
                system_blocks=[], messages=[{"role": "user", "content": "anything open?"}],
                model="claude-test", calls_out=calls,
            ):
                pass

    with patch.dict("openexecutive.orchestrator.executive._ALL_SKILL_HANDLERS",
                    {"list_open_loops": handler}):
        asyncio.run(_go())
    assert calls == [("list_open_loops", '{"loops": []}')]


def test_the_check_sees_what_each_call_was_given() -> None:
    # A sheet write's result says only "updated range A9:B10"; the row it
    # moved and the formula it wrote are in what it was given.
    label = claim_check.call_label("call_tool", {
        "name": "google_workspace__modify_sheet_values",
        "arguments": {"range_name": "A9:B10", "values": [["Supplies", 600], ["Total", "=SUM(B2:B9)"]]},
    })
    assert label.startswith("google_workspace__modify_sheet_values ")
    assert "=SUM(B2:B9)" in label
    assert claim_check.call_label("list_open_loops", {}) == "list_open_loops"
    long = claim_check.call_label("note", {"text": "x" * 5000})
    assert len(long) < claim_check.CALL_INPUT_CHARS + 40
    rendered = claim_check.render_calls([(label, "Successfully updated range 'A9:B10'")])
    assert "=SUM(B2:B9)" in rendered


def test_every_call_of_a_long_turn_is_shown() -> None:
    """A save early in a long turn still backs "I saved it": no call is cut,
    the lines get shorter instead."""
    calls = [("save_file {\"name\": \"Logistics pack\"}", "Saved to Documents")]
    calls += [(f"update_department_plan {{\"step_id\": {i}}}", "ok " * 200) for i in range(40)]
    rendered = claim_check.render_calls(calls)
    lines = rendered.splitlines()
    assert len(lines) == 41 and "Logistics pack" in lines[0] and "Saved to Documents" in lines[0]
    assert all(len(line) <= max(claim_check.CALL_MIN_CHARS, claim_check.CALLS_BUDGET_CHARS // 41) for line in lines)


def test_a_claim_about_before_this_reply_is_never_unbacked() -> None:
    reply = "Earlier today I also sent Sam approval cards for the other quotes. I sent the quote just now."
    earlier = {
        "unbacked": True,
        "claim": "Earlier today I also sent Sam approval cards for the other quotes.",
        "when": claim_check.WHEN_BEFORE,
    }
    assert claim_check.accept_claim(earlier, reply) is None
    now = {"unbacked": True, "claim": "I sent the quote just now.", "when": claim_check.WHEN_THIS_REPLY}
    assert claim_check.accept_claim(now, reply) == "I sent the quote just now."
    # The judge must say when; a time word in the claim alone decides nothing.
    assert claim_check._TOOL["input_schema"]["required"] == ["unbacked", "claim", "when"]
    noted = {"unbacked": True, "claim": "I saved the notes from yesterday's call.", "when": claim_check.WHEN_THIS_REPLY}
    assert claim_check.accept_claim(noted, "I saved the notes from yesterday's call.") is not None


def test_a_very_long_turn_drops_its_middle_calls() -> None:
    """Past the budget, with no reads to drop, the middle goes, never the
    start or the end."""
    calls = [("save_file {\"name\": \"Logistics pack\"}", "Saved to Documents")]
    calls += [(f"update_step {{\"id\": {i}}}", "ok " * 200) for i in range(500)]
    calls += [("send_email {\"to\": \"sam@example.com\"}", "sent")]
    rendered = claim_check.render_calls(calls)
    lines = rendered.splitlines()
    most = claim_check.CALLS_BUDGET_CHARS // claim_check.CALL_MIN_CHARS
    assert len(lines) == most
    assert "Logistics pack" in lines[0] and "sam@example.com" in lines[-1]
    assert any(f"{len(calls) - most + 1} more calls left out" in line for line in lines)
    assert len(rendered) <= claim_check.CALLS_BUDGET_CHARS + 2 * claim_check.CALL_MIN_CHARS + len(lines)


def test_a_long_turn_drops_reads_before_any_change() -> None:
    """A save in the middle of 150 calls still backs "I saved it": the reads
    go first."""
    calls = [(f"list_open_loops {{\"page\": {i}}}", "x " * 100) for i in range(150)]
    calls.insert(60, ("save_file {\"name\": \"Dock door plan\"}", "Saved to Documents"))
    rendered = claim_check.render_calls(calls)
    lines = rendered.splitlines()
    assert len(lines) == claim_check.CALLS_BUDGET_CHARS // claim_check.CALL_MIN_CHARS
    assert "Dock door plan" in rendered and "Saved to Documents" in rendered
    assert sum("more calls left out" in line for line in lines) == 1

