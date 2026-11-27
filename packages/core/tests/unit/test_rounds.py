"""The one loop a background pass runs its tool calls in (orchestrator.rounds).

It ends when the model answers, and writes up (one more call with no tools)
when a call keeps coming back the same, when its reads fill what the turn
may read, when it has run its ``max_rounds`` rounds, or when the pass says
it has had enough. A long
result comes back in parts, as in a chat turn.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from openexecutive.audit import usage
from openexecutive.orchestrator import rounds
from openexecutive.orchestrator.reading import (
    CUT_OFF_NOTE,
    CUT_OFF_RESULT,
    FULL_NOTE,
    MAX_EMPTY_CUT_ROUNDS,
    REPEAT_LIMIT,
    REPEATING_NOTE,
    STEPS_NOTE,
)
from openexecutive.providers import feature_gate, registry


@pytest.fixture(autouse=True)
def _usage(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Each response's usage, recorded here rather than in a database."""
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(usage, "log_model_usage", lambda _response, **kw: seen.append(kw))
    return seen


def _use(name: str, args: dict[str, Any], id: str) -> Any:
    return SimpleNamespace(type="tool_use", name=name, input=args, id=id)


def _calls(*uses: Any) -> Any:
    return SimpleNamespace(content=list(uses), stop_reason="tool_use")


def _answer(text: str) -> Any:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn")


class _Model:
    """Answers each request with ``reply(request number, request)``; keeps a
    copy of every request."""

    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.requests: list[dict[str, Any]] = []

    async def messages_create(self, **kw: Any) -> Any:
        self.requests.append({**kw, "messages": list(kw["messages"])})
        return self.reply(len(self.requests), kw)


async def _run(model: _Model, run_call: Any, *, max_rounds: int = 100, **kw: Any) -> rounds.Rounds:
    return await rounds.run_rounds(
        model, model="claude-opus-5", max_tokens=1000, system="s", tools=[],
        messages=[{"role": "user", "content": "go"}], run_call=run_call, actor="a pass",
        max_rounds=max_rounds, **kw,
    )


@pytest.mark.asyncio
async def test_a_pass_keeps_going_until_its_work_is_done() -> None:
    ran: list[str] = []

    async def run_call(use: dict[str, Any]) -> str:
        ran.append(use["input"]["n"])
        return f"row {use['input']['n']}"

    model = _Model(lambda n, _kw: _calls(_use("read", {"n": n}, f"t{n}")) if n <= 40 else _answer("All 40 read."))
    done = await _run(model, run_call)
    assert len(ran) == 40 and done.text == "All 40 read." and done.ended == "done"
    assert done.rounds == 41
    assert all("tool_choice" not in r for r in model.requests)


@pytest.mark.asyncio
async def test_a_pass_writes_up_after_its_rounds() -> None:
    ran: list[int] = []

    async def run_call(use: dict[str, Any]) -> str:
        ran.append(use["input"]["n"])
        return f"row {use['input']['n']}"

    def reply(n: int, kw: dict[str, Any]) -> Any:
        if kw.get("tool_choice") == {"type": "none"}:
            return _answer("Read 5; the rest is left.")
        return _calls(_use("read", {"n": n}, f"t{n}"))

    model = _Model(reply)
    done = await _run(model, run_call, max_rounds=5)
    assert done.ended == "steps" and done.text == "Read 5; the rest is left."
    assert ran == [1, 2, 3, 4, 5]
    assert len(model.requests) == 6
    assert model.requests[-1]["messages"][-1]["content"][-1] == {"type": "text", "text": STEPS_NOTE}


@pytest.mark.asyncio
async def test_a_call_that_keeps_coming_back_the_same_writes_up() -> None:
    async def run_call(_use: dict[str, Any]) -> str:
        return '{"error": "busy", "request_id": "9f8e7d6c5b4a3210"}'

    def reply(n: int, kw: dict[str, Any]) -> Any:
        if kw.get("tool_choice") == {"type": "none"}:
            return _answer("It kept saying busy.")
        return _calls(_use("send", {"to": "a"}, f"t{n}"))

    model = _Model(reply)
    done = await _run(model, run_call)
    assert done.ended == "repeating" and done.text == "It kept saying busy."
    assert len(model.requests) == REPEAT_LIMIT + 1
    assert model.requests[-1]["messages"][-1]["content"][-1] == {"type": "text", "text": REPEATING_NOTE}


@pytest.mark.asyncio
async def test_writing_up_runs_no_call_even_one_the_provider_wrote_anyway() -> None:
    ran: list[str] = []

    async def run_call(use: dict[str, Any]) -> str:
        ran.append(str(use["id"]))
        return "busy"

    def reply(n: int, kw: dict[str, Any]) -> Any:
        if kw.get("tool_choice") == {"type": "none"}:
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="It kept saying busy."), _use("send", {}, "late")],
                stop_reason="tool_use",
            )
        return _calls(_use("send", {"to": "a"}, f"t{n}"))

    done = await _run(_Model(reply), run_call)
    assert done.ended == "repeating" and done.text == "It kept saying busy." and "late" not in ran


@pytest.mark.asyncio
async def test_a_long_result_comes_in_parts_without_running_the_tool_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A small model: one part is about 30,000 characters.
    monkeypatch.setattr(registry, "reading_limits_for", lambda _m="": feature_gate.reading_limits(50_000))
    long = "".join(f"line {i:06d}\n" for i in range(4_000))  # 48,000 characters
    ran: list[int] = []

    async def run_call(_use: dict[str, Any]) -> str:
        ran.append(1)
        return long

    def reply(n: int, kw: dict[str, Any]) -> Any:
        last = kw["messages"][-1]["content"]
        if n > 1 and "LAST PART" in str(last):
            return _answer("Read it all.")
        return _calls(_use("read_doc", {"id": "d"}, f"t{n}"))

    model = _Model(reply)
    seen: list[str] = []
    done = await _run(model, run_call, on_result=lambda _use, sent: seen.append(str(sent)))
    assert ran == [1] and done.text == "Read it all."
    # What the pass is told it read is each part as it was sent.
    assert "line 003999" not in seen[0] and "line 003999" in seen[-1]
    shown = "".join(str(r["messages"][-1]["content"][0]["content"]) for r in model.requests[1:])
    assert "line 000000" in shown and "line 003999" in shown


@pytest.mark.asyncio
async def test_reads_that_fill_the_model_write_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "reading_limits_for", lambda _m="": feature_gate.reading_limits(10_000))

    async def run_call(use: dict[str, Any]) -> str:
        return f"{use['input']['n']}:" + "z" * 8_000

    def reply(n: int, kw: dict[str, Any]) -> Any:
        if kw.get("tool_choice") == {"type": "none"}:
            return _answer("Here is what I read.")
        return _calls(_use("read", {"n": n}, f"t{n}"))

    model = _Model(reply)
    done = await _run(model, run_call)
    assert done.ended == "full" and done.text == "Here is what I read."
    assert model.requests[-1]["messages"][-1]["content"][-1] == {"type": "text", "text": FULL_NOTE}


@pytest.mark.asyncio
async def test_a_pass_that_has_had_enough_writes_up_or_stops() -> None:
    made: list[int] = []

    async def run_call(_use: dict[str, Any]) -> str:
        made.append(1)
        return "ok"

    def reply(n: int, kw: dict[str, Any]) -> Any:
        if kw.get("tool_choice") == {"type": "none"}:
            return _answer("Summary.")
        return _calls(_use("act", {"n": n}, f"t{n}"))

    model = _Model(reply)
    done = await _run(model, run_call, enough=lambda: "Write it up." if len(made) >= 2 else None)
    assert done.ended == "enough" and done.text == "Summary." and len(made) == 2
    assert model.requests[-1]["messages"][-1]["content"][-1] == {"type": "text", "text": "Write it up."}

    made.clear()
    model = _Model(reply)
    done = await _run(model, run_call, enough=lambda: "" if made else None)
    assert done.ended == "enough" and done.text == "" and len(model.requests) == 1


@pytest.mark.asyncio
async def test_each_response_is_recorded_and_seen_and_only_what_is_sent_goes(
    _usage: list[dict[str, Any]],
) -> None:
    seen: list[int] = []

    async def run_call(_use: dict[str, Any]) -> str:
        return "ok"

    model = _Model(lambda n, _kw: _calls(_use("a", {"n": n}, f"t{n}")) if n < 3 else _answer("done"))
    messages: list[dict[str, Any]] = [{"role": "user", "content": "old"}, {"role": "user", "content": "go"}]
    await rounds.run_rounds(
        model, model="m", max_tokens=10, system="s", tools=[], messages=messages, run_call=run_call,
        actor="admin_chat", max_rounds=8, session_id="c1", on_response=lambda _r, i: seen.append(i), sent=lambda m: m[1:],
    )
    assert seen == [1, 2, 3]
    assert [(u["actor"], u["iteration"], u["session_id"]) for u in _usage] == [("admin_chat", i, "c1") for i in (1, 2, 3)]
    assert all(r["messages"][0]["content"] == "go" for r in model.requests)
    assert messages[0]["content"] == "old" and messages[-1]["role"] == "assistant"


def _cut(*blocks: Any) -> Any:
    return SimpleNamespace(content=list(blocks), stop_reason="max_tokens")


@pytest.mark.asyncio
async def test_a_call_cut_off_at_the_length_limit_never_runs_and_the_pass_carries_on() -> None:
    ran: list[dict[str, Any]] = []

    async def run_call(use: dict[str, Any]) -> str:
        ran.append(use)
        return "ok"

    def reply(n: int, _kw: dict[str, Any]) -> Any:
        if n == 1:
            # A finished call, then the one it was still writing when cut.
            return _cut(_use("note", {"n": 1}, "t1"), _use("send_dm", {"text": "Hi Ana, the half"}, "t2"))
        if n == 2:
            return _calls(_use("send_dm", {"text": "Hi Ana."}, "t3"))
        return _answer("Sent.")

    model = _Model(reply)
    done = await _run(model, run_call)
    assert [u["id"] for u in ran] == ["t1", "t3"]
    assert done.text == "Sent." and done.ended == "done" and done.rounds == 3
    sent_back = model.requests[1]["messages"]
    assert sent_back[-2]["content"][-1] == {"type": "tool_use", "id": "t2", "name": "send_dm", "input": {}}
    assert {"type": "tool_result", "tool_use_id": "t2", "content": CUT_OFF_RESULT} in sent_back[-1]["content"]


@pytest.mark.asyncio
async def test_an_answer_cut_off_mid_text_goes_on_from_where_it_stopped() -> None:
    async def run_call(_use: dict[str, Any]) -> str:
        raise AssertionError("no call was made")

    def reply(n: int, _kw: dict[str, Any]) -> Any:
        if n == 1:
            return _cut(SimpleNamespace(type="text", text="**Quiet:** Nothing else "))
        return _answer("worth acting on.")

    model = _Model(reply)
    done = await _run(model, run_call)
    assert done.text == "**Quiet:** Nothing else worth acting on." and done.ended == "done"
    assert model.requests[1]["messages"][-1]["content"] == [{"type": "text", "text": CUT_OFF_NOTE}]


@pytest.mark.asyncio
async def test_cut_rounds_that_get_nowhere_end_the_pass() -> None:
    async def run_call(_use: dict[str, Any]) -> str:
        raise AssertionError("a cut call never runs")

    model = _Model(lambda n, _kw: _cut(_use("draft_artifact", {"body": "x" * 50}, f"t{n}")))
    done = await _run(model, run_call)
    assert done.ended == "cut" and len(model.requests) == MAX_EMPTY_CUT_ROUNDS + 1


@pytest.mark.asyncio
async def test_each_round_reads_the_history_before_it_from_the_cache() -> None:
    marked: list[list[str]] = []

    async def run_call(_use: dict[str, Any]) -> str:
        return "ok"

    def reply(n: int, kw: dict[str, Any]) -> Any:
        marked.append([
            b["tool_use_id"] for m in kw["messages"] if isinstance(m["content"], list)
            for b in m["content"] if b.get("type") == "tool_result" and b.get("cache_control")
        ])
        return _calls(_use("read", {"n": n}, f"t{n}")) if n < 4 else _answer("done")

    await _run(_Model(reply), run_call)
    # One breakpoint, always on the newest result: the round reads the rest.
    assert marked == [[], ["t1"], ["t2"], ["t3"]]


def test_a_long_run_of_one_character_is_keyed_in_one_pass() -> None:
    import time

    from openexecutive.orchestrator.reading import call_key

    started = time.monotonic()
    for text in ("-" * 600_000, "a_" * 300_000, "0-" * 300_000):
        call_key("browser_look", {}, '{"title": "' + text + '"}')
    assert time.monotonic() - started < 5


def test_ids_and_times_are_left_out_of_a_call_key() -> None:
    from openexecutive.orchestrator.reading import call_key

    one = call_key("t", {}, "busy; request 3f9a1c2e-4b5d-6e7f-8a9b-0c1d2e3f4a5b at 2026-11-18T01:02:03Z, 42s")
    two = call_key("t", {}, "busy; request 9e8d7c6b-5a4f-3e2d-1c0b-a9f8e7d6c5b4 at 2026-11-18T01:05:09Z, 7s")
    assert one == two
    assert call_key("t", {}, "20 of 40 done") != call_key("t", {}, "40 of 40 done")
