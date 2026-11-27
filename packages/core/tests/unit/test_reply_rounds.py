"""A turn's text across tool rounds (orchestrator/executive.py): each round's
text starts a new paragraph, so the model's words before a tool call and its
answer never run together ("...stock.Yes. ITEM-1..."). A short working note
before a lookup ("Now the live data.") leaves the reply once an answer follows;
anything longer stays. Found by evals against a live install (2026-10-19) and
a simulated week (2026-11-19)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from openexecutive.orchestrator import web_search_tool
from openexecutive.orchestrator.executive import Executive, _round_separator
from openexecutive.orchestrator.schedule_tools import set_session
from openexecutive.orchestrator.session import Session

from ._agent_loop_fakes import FakeStream, FinalMsg, ScriptedProvider, TextBlock, ToolUseBlock


class StreamingFake(FakeStream):
    """Streams each text block of the final message as a text delta, as the
    API does, before the final message."""

    def __init__(self, final_msg: FinalMsg) -> None:
        super().__init__(final_msg)
        self._deltas = [
            SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=b.text))
            for b in final_msg.content
            if getattr(b, "type", "") == "text"
        ]

    async def __anext__(self) -> Any:
        if not self._deltas:
            raise StopAsyncIteration
        return self._deltas.pop(0)


class StreamingProvider(ScriptedProvider):
    def messages_stream(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        return StreamingFake(self._final_msgs.pop(0))


def _reply(rounds: list[FinalMsg]) -> str:
    provider = StreamingProvider(rounds)
    text: list[str] = []

    async def go() -> None:
        with (
            patch("openexecutive.orchestrator.executive.get_provider", return_value=provider),
            patch("openexecutive.orchestrator.executive.audit_log", lambda *_a, **_k: None),
            set_session(Session()),
        ):
            async for item in Executive()._stream_agent_loop(
                system_blocks=[], messages=[{"role": "user", "content": "x"}],
                model="claude-test", workspace_mode="team", principal_role_tag="", turn_id="t-1",
            ):
                if isinstance(item, str) and item != Executive._THINKING:
                    text.append(item)
                elif item == {"type": "rewrite"}:
                    # As every reader does: the retry replaces what came before.
                    text.clear()

    asyncio.run(go())
    return "".join(text)


@pytest.mark.parametrize(("tail", "text", "separator"), [
    ("stock.", "Yes.", "\n\n"),
    ("stock.\n", "Yes.", "\n"),
    ("stock.", "\nYes.", "\n"),
    ("stock.\n\n", "Yes.", ""),
    ("stock.", "\n\nYes.", ""),
])
def test_a_new_round_starts_a_new_paragraph(tail: str, text: str, separator: str) -> None:
    assert _round_separator(tail, text) == separator


def test_a_turn_keeps_every_round_in_its_own_paragraph() -> None:
    reply = _reply([
        FinalMsg([TextBlock("Check actual stock."), ToolUseBlock("tu1", "no_such_tool", {})], "tool_use"),
        FinalMsg([TextBlock("Yes. ITEM-1 has 12 in Stores.")], "end_turn"),
    ])
    assert reply == "Check actual stock.\n\nYes. ITEM-1 has 12 in Stores."


def test_a_working_note_before_a_lookup_leaves_the_answer() -> None:
    # Seen in a research reply: "Nothing on the carrier's surcharge schedule
    # in the mailbox. Now the live data." stayed at the top of the answer.
    reply = _reply([
        FinalMsg([TextBlock("Now to find the Finance folder."),
                  ToolUseBlock("tu1", "call_tool", {"name": "search_drive_files", "arguments": {}})], "tool_use"),
        FinalMsg([TextBlock("Nothing in the mailbox. Now the live data."), ToolUseBlock("tu2", "search_tools", {})],
                 "tool_use"),
        FinalMsg([TextBlock("Midwest diesel is $5.97, down 23 cents.")], "end_turn"),
    ])
    assert reply == "Nothing in the mailbox.\n\nMidwest diesel is $5.97, down 23 cents."


@pytest.mark.parametrize(("text", "kept"), [
    ("Yes, Acme signed it. Let me pull the signed copy.", "Yes, Acme signed it."),
    ("No, the NDA was never signed. Let me find the draft.", "No, the NDA was never signed."),
    ("First, the good news: Acme renewed.", "First, the good news: Acme renewed."),
    ("Now that Acme signed, the deal is closed.", "Now that Acme signed, the deal is closed."),
    ("Let me be blunt: we should not sign this.", "Let me be blunt: we should not sign this."),
    ("Looking at the contract, renewal is automatic.", "Looking at the contract, renewal is automatic."),
    ("Searching the CRM turned up nothing on Acme.", "Searching the CRM turned up nothing on Acme."),
    ("Now the bad news: Acme churned.", "Now the bad news: Acme churned."),
    ("Acme renewed. Now the bad news is that Beta churned.", "Acme renewed. Now the bad news is that Beta churned."),
    ("Time to hire has doubled since last year.", "Time to hire has doubled since last year."),
    ("Nothing on Acme yet. Checking the CRM now.", "Nothing on Acme yet."),
])
def test_only_the_next_step_sentences_leave(text: str, kept: str) -> None:
    reply = _reply([
        FinalMsg([TextBlock(text), ToolUseBlock("tu1", "search_drive_files", {})], "tool_use"),
        FinalMsg([TextBlock("Here's the signed copy.")], "end_turn"),
    ])
    assert reply == kept + "\n\nHere's the signed copy."


def test_a_note_with_nothing_after_it_is_the_reply() -> None:
    reply = _reply([
        FinalMsg([TextBlock("Checking the sheet now."), ToolUseBlock("tu1", "read_sheet", {})], "tool_use"),
        FinalMsg([], "end_turn"),
    ])
    assert reply == "Checking the sheet now."


def test_an_answer_written_before_an_action_stays() -> None:
    # "What's our runway? Email it to Dana.": the one-line answer comes
    # before the send, and the person still gets it.
    reply = _reply([
        FinalMsg([TextBlock("Runway is 14 months at the current burn."),
                  ToolUseBlock("tu1", "send_email", {})], "tool_use"),
        FinalMsg([TextBlock("Sent to Dana.")], "end_turn"),
    ])
    assert reply == "Runway is 14 months at the current burn.\n\nSent to Dana."


def test_a_short_answer_with_a_figure_before_a_lookup_stays() -> None:
    reply = _reply([
        FinalMsg([TextBlock("Revenue was $1.2M last quarter."), ToolUseBlock("tu1", "search_knowledge", {})],
                 "tool_use"),
        FinalMsg([TextBlock("Costs were $800k.")], "end_turn"),
    ])
    assert reply == "Revenue was $1.2M last quarter.\n\nCosts were $800k."


def test_a_short_answer_before_a_lookup_stays() -> None:
    # "Is the Acme NDA signed?": the yes comes before finding the file.
    reply = _reply([
        FinalMsg([TextBlock("Yes, Acme signed it last week."),
                  ToolUseBlock("tu1", "call_tool", {"name": "search_drive_files", "arguments": {}})], "tool_use"),
        FinalMsg([TextBlock("Here's the signed copy.")], "end_turn"),
    ])
    assert reply == "Yes, Acme signed it last week.\n\nHere's the signed copy."


def test_the_rest_of_a_cut_sentence_stays() -> None:
    # A round cut mid-answer carries on in the next; that short rest is
    # the end of the sentence, not a note.
    reply = _reply([
        FinalMsg([TextBlock("Revenue grew")], "max_tokens"),
        FinalMsg([TextBlock(" 12% in Q3."), ToolUseBlock("tu1", "search_knowledge", {})], "tool_use"),
        FinalMsg([TextBlock("Margins held.")], "end_turn"),
    ])
    assert reply == "Revenue grew 12% in Q3.\n\nMargins held."


class SearchBlock:
    """A server-side web search block as the API returns it."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)

    def model_dump(self, **_k: Any) -> dict[str, Any]:
        return dict(self.__dict__)


def _search(id_: str, query: str, error: str | None = None) -> list[Any]:
    content: Any = (
        {"type": "web_search_tool_result_error", "error_code": error}
        if error else [{"type": "web_search_result", "title": "EIA weekly diesel", "url": "https://example.com"}]
    )
    return [
        SearchBlock(type="server_tool_use", id=id_, name="web_search", input={"query": query}),
        SearchBlock(type="web_search_tool_result", tool_use_id=id_, content=content),
    ]


def test_a_working_note_before_a_web_search_in_the_same_round_leaves() -> None:
    # A web search runs inside the round, so the note before it and the
    # answer after it come in one response.
    reply = _reply([
        FinalMsg([TextBlock("Nothing in the mailbox. Now the live data."), *_search("s1", "diesel"),
                  TextBlock("Midwest diesel is $5.97"), TextBlock(", down 23 cents.")], "end_turn"),
    ])
    assert reply == "Nothing in the mailbox.\n\nMidwest diesel is $5.97, down 23 cents."


def test_an_answer_before_a_web_search_in_the_same_round_stays() -> None:
    reply = _reply([
        FinalMsg([TextBlock("Acme renewed. "), *_search("s1", "acme"), TextBlock("Beta churned in March.")],
                 "end_turn"),
    ])
    assert reply == "Acme renewed. Beta churned in March."


def test_the_turn_shares_one_search_budget_across_its_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    # Six facts to look up: the rounds share the turn's budget, the tools
    # stay the same every round (the cached prompt starts with them), the
    # round that spends it is told to finish, and the turn ends on its reply
    # with no extra "search again" round.
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "true")
    monkeypatch.setenv("WEB_SEARCH_TURN_BUDGET", "3")
    provider = StreamingProvider([
        FinalMsg([*_search("s1", "usps"), *_search("s2", "ups"), *_search("s3", "fedex"),
                  ToolUseBlock("tu1", "no_such_tool", {})], "tool_use"),
        FinalMsg([*_search("s4", "rates"), TextBlock("All three dates and the rate.")], "end_turn"),
    ])
    text: list[str] = []

    async def go() -> None:
        with (
            patch("openexecutive.orchestrator.executive.get_provider", return_value=provider),
            patch("openexecutive.orchestrator.executive.audit_log", lambda *_a, **_k: None),
            set_session(Session()),
        ):
            async for item in Executive()._stream_agent_loop(
                system_blocks=[], messages=[{"role": "user", "content": "x"}],
                model="claude-test", workspace_mode="team", principal_role_tag="", turn_id="t-1",
                search=web_search_tool.select_web_search_tool("claude-test"),
            ):
                if isinstance(item, str) and item != Executive._THINKING:
                    text.append(item)
                elif item == {"type": "rewrite"}:
                    text.clear()

    asyncio.run(go())
    assert len(provider.calls) == 2
    uses = [
        next(t["max_uses"] for t in call["tools"] if t.get("name") == "web_search") for call in provider.calls
    ]
    assert uses == [3, 3]
    assert provider.calls[0]["tools"] == provider.calls[1]["tools"]
    told = [b for b in provider.calls[1]["messages"][-1]["content"] if b.get("type") == "text"]
    assert [b["text"] for b in told] == [web_search_tool.SEARCHES_SPENT_NOTE]
    assert "".join(text).endswith("All three dates and the rate.")


def test_a_turn_that_searches_past_its_budget_writes_up(monkeypatch: pytest.MonkeyPatch) -> None:
    # Told once that the searches are spent, it searched again: the next
    # round is the write-up, and a tool it asks for there never runs.
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "true")
    monkeypatch.setenv("WEB_SEARCH_TURN_BUDGET", "2")
    provider = StreamingProvider([
        FinalMsg([*_search("s1", "a"), *_search("s2", "b"), ToolUseBlock("tu1", "no_such_tool", {})], "tool_use"),
        FinalMsg([*_search("s3", "c"), ToolUseBlock("tu2", "no_such_tool", {})], "tool_use"),
        FinalMsg([ToolUseBlock("tu3", "no_such_tool", {})], "tool_use"),
        FinalMsg([TextBlock("never asked for")], "end_turn"),
    ])
    text: list[str] = []

    async def go() -> None:
        with (
            patch("openexecutive.orchestrator.executive.get_provider", return_value=provider),
            patch("openexecutive.orchestrator.executive.audit_log", lambda *_a, **_k: None),
            set_session(Session()),
        ):
            async for item in Executive()._stream_agent_loop(
                system_blocks=[], messages=[{"role": "user", "content": "x"}],
                model="claude-test", workspace_mode="team", principal_role_tag="", turn_id="t-1",
                search=web_search_tool.select_web_search_tool("claude-test"),
            ):
                if isinstance(item, str) and item != Executive._THINKING:
                    text.append(item)

    asyncio.run(go())
    assert len(provider.calls) == 3
    notes = [
        b["text"] for m in provider.calls[-1]["messages"] if m["role"] == "user" and isinstance(m["content"], list)
        for b in m["content"] if b.get("type") == "text"
    ]
    assert notes[1:] == [web_search_tool.SEARCHES_SPENT_NOTE, web_search_tool.SEARCHES_OVER_NOTE]
    assert "".join(text).endswith(web_search_tool.SEARCHES_EMPTY_REPLY)


def test_a_round_a_long_search_paused_carries_on(monkeypatch: pytest.MonkeyPatch) -> None:
    # The API paused the round mid-search (pause_turn): it is sent back as it
    # is and finishes, instead of the turn ending on half a reply.
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "true")
    provider = StreamingProvider([
        FinalMsg([TextBlock("USPS: Dec 17. "), *_search("s1", "ups")], "pause_turn"),
        FinalMsg([*_search("s2", "fedex"), TextBlock("UPS: Dec 18.")], "end_turn"),
    ])
    text: list[str] = []

    async def go() -> None:
        with (
            patch("openexecutive.orchestrator.executive.get_provider", return_value=provider),
            patch("openexecutive.orchestrator.executive.audit_log", lambda *_a, **_k: None),
            set_session(Session()),
        ):
            async for item in Executive()._stream_agent_loop(
                system_blocks=[], messages=[{"role": "user", "content": "x"}],
                model="claude-test", workspace_mode="team", principal_role_tag="", turn_id="t-1",
            ):
                if isinstance(item, str) and item != Executive._THINKING:
                    text.append(item)
                elif item == {"type": "rewrite"}:
                    text.clear()

    asyncio.run(go())
    assert len(provider.calls) == 2
    assert provider.calls[1]["messages"][1]["role"] == "assistant"
    assert "".join(text) == "USPS: Dec 17. UPS: Dec 18."


def test_a_round_paused_past_the_budget_writes_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "true")
    monkeypatch.setenv("WEB_SEARCH_TURN_BUDGET", "1")
    provider = StreamingProvider([
        FinalMsg([*_search("s1", "ups")], "pause_turn"),
        FinalMsg([ToolUseBlock("tu1", "no_such_tool", {})], "tool_use"),
        FinalMsg([TextBlock("never asked for")], "end_turn"),
    ])
    text: list[str] = []

    async def go() -> None:
        with (
            patch("openexecutive.orchestrator.executive.get_provider", return_value=provider),
            patch("openexecutive.orchestrator.executive.audit_log", lambda *_a, **_k: None),
            set_session(Session()),
        ):
            async for item in Executive()._stream_agent_loop(
                system_blocks=[], messages=[{"role": "user", "content": "x"}],
                model="claude-test", workspace_mode="team", principal_role_tag="", turn_id="t-1",
                search=web_search_tool.select_web_search_tool("claude-test"),
            ):
                if isinstance(item, str) and item != Executive._THINKING:
                    text.append(item)

    asyncio.run(go())
    assert len(provider.calls) == 2
    last = provider.calls[1]["messages"][-1]
    assert last["content"] == [{"type": "text", "text": web_search_tool.SEARCHES_OVER_NOTE}]
    assert "".join(text).endswith(web_search_tool.SEARCHES_EMPTY_REPLY)


def test_one_round_is_untouched_and_a_silent_round_adds_nothing() -> None:
    assert _reply([FinalMsg([TextBlock("ITEM-1 has 12.")], "end_turn")]) == "ITEM-1 has 12."
    # A round that only calls a tool writes nothing, so no break before the answer.
    reply = _reply([
        FinalMsg([ToolUseBlock("tu1", "no_such_tool", {})], "tool_use"),
        FinalMsg([TextBlock("ITEM-1 has 12.")], "end_turn"),
    ])
    assert reply == "ITEM-1 has 12."


def test_a_claim_check_retry_replaces_the_first_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    # The retry writes the reply again, whole, with no break or trace of the
    # first: it never trails it as a correction.
    from openexecutive.orchestrator import claim_check

    async def unbacked(model: str, turn: str) -> dict[str, Any]:
        return {"unbacked": True, "claim": "Done"}

    monkeypatch.setenv("CLAIM_CHECK_ENABLED", "true")
    monkeypatch.setattr(claim_check, "_call_model", unbacked)
    reply = _reply([
        FinalMsg([TextBlock("Done.")], "end_turn"),
        FinalMsg([TextBlock("I haven't removed it yet.")], "end_turn"),
    ])
    assert reply == "I haven't removed it yet."


def test_a_collected_reply_is_rewritten_not_corrected(monkeypatch: pytest.MonkeyPatch) -> None:
    # An email reply is read whole, so a claim-check retry while
    # Executive.chat collects replaces the first reply instead of trailing a
    # "Correction:" after its sign-off.
    from openexecutive.orchestrator import claim_check

    async def unbacked(model: str, turn: str) -> dict[str, Any]:
        return {"unbacked": True, "claim": "Added Avery"}

    monkeypatch.setenv("CLAIM_CHECK_ENABLED", "true")
    monkeypatch.setattr(claim_check, "_call_model", unbacked)
    provider = StreamingProvider([
        FinalMsg([TextBlock("I've added Avery. — OE")], "end_turn"),
        FinalMsg([TextBlock("\n\nAvery is on this week's plan as a reviewer. — OE")], "end_turn"),
    ])
    out: list[Any] = []

    async def go() -> None:
        with (
            patch("openexecutive.orchestrator.executive.get_provider", return_value=provider),
            patch("openexecutive.orchestrator.executive.audit_log", lambda *_a, **_k: None),
            set_session(Session()),
        ):
            async for item in Executive()._stream_agent_loop(
                system_blocks=[], messages=[{"role": "user", "content": "x"}],
                model="claude-test", workspace_mode="team", principal_role_tag="", turn_id="t-1",
            ):
                out.append(item)

    asyncio.run(go())
    marker = out.index({"type": "rewrite"})
    after = "".join(i for i in out[marker:] if isinstance(i, str) and i != Executive._THINKING)
    assert "I've added Avery" not in after and "on this week's plan" in after
    nudge = provider.calls[1]["messages"][-1]["content"][0]["text"]
    assert "write your whole reply again" in nudge



def test_a_rewrite_that_writes_nothing_keeps_the_first_reply() -> None:
    """An empty retry (or a stop before its first word) never leaves a blank
    reply: the first one streams again, so it is saved and shown."""
    from openexecutive.orchestrator.executive import _kept_if_unwritten

    async def collect(items: list[Any]) -> list[Any]:
        async def stream() -> Any:
            for item in items:
                yield item

        return [i async for i in _kept_if_unwritten(stream(), Executive._THINKING)]

    rewrite = {"type": "rewrite"}
    out = asyncio.run(collect(["I sent ", "it.", rewrite, Executive._THINKING]))
    assert out[-1] == "I sent it."
    # A retry that wrote something stands as it is.
    out = asyncio.run(collect(["I sent it.", rewrite, "It waits on approval."]))
    assert out[-1] == "It waits on approval." and out.count("I sent it.") == 1
