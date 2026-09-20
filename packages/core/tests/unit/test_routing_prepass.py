"""The routing pre-pass decides specialists before the full tool surface appears.

`consult_specialist` competes with ~58 other client tools on the main turn, and
a smaller model does not reach for it: measured on qwen3.8:27b, "What should I
be focusing on right now, and in what order?" consulted on 1/12 turns with the
full surface and 9-11/12 with this pre-pass plus the persona section. The pre-pass
offers that one tool and nothing else.

The invariant that keeps it safe is that it is purely additive: a pre-pass that
picks nobody must leave the message list untouched, so an action turn behaves
exactly as it did before.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-not-used")
os.environ.setdefault("EXEC_EMAIL_ADDRESS", "exec@example.test")

from openexecutive.orchestrator.executive import Executive  # noqa: E402


class _Block:
    """Minimal stand-in for an SDK tool_use / text content block."""

    def __init__(self, type: str, **kw: Any) -> None:
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class _Response:
    def __init__(self, content: list[_Block], stop_reason: str = "tool_use") -> None:
        self.content = content
        self.stop_reason = stop_reason


def _consult(block_id: str, specialist: str, query: str = "q") -> _Block:
    return _Block(
        "tool_use",
        id=block_id,
        name="consult_specialist",
        input={"specialist": specialist, "query": query},
    )


def _run_prepass(
    decision_content: list[_Block],
    messages: list[dict[str, Any]] | None = None,
    specialist_results: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str], Any]:
    """Drive _routing_prepass with the provider and route_parallel stubbed.

    Returns (messages, consulted, route_parallel_mock).
    """
    current_messages = (
        messages
        if messages is not None
        else [{"role": "user", "content": "what should I focus on?"}]
    )
    consulted: list[str] = []
    route_mock = AsyncMock(
        return_value=specialist_results
        if specialist_results is not None
        else ["analysis"] * sum(
            1
            for b in decision_content
            if b.type == "tool_use" and b.name == "consult_specialist"
        )
    )
    provider = AsyncMock()
    provider.messages_create = AsyncMock(return_value=_Response(decision_content))

    async def _drive() -> None:
        exec_ = Executive()
        agen = exec_._routing_prepass(
            [{"type": "text", "text": "persona"}],
            current_messages,
            model="test-model",
            episodic_context="",
            debug_collector=None,
            consulted_out=consulted,
            specialist_outputs_out=None,
            turn_id="t-test",
            conversation_context="ctx",
            specialists_consulted=[],
        )
        async for _ in agen:
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider",
            return_value=provider,
        ),
        patch(
            "openexecutive.orchestrator.executive.route_parallel", route_mock
        ),
        patch("openexecutive.orchestrator.executive.audit_log"),
    ):
        asyncio.run(_drive())
    return current_messages, consulted, route_mock


def test_prepass_offers_only_consult_specialist() -> None:
    """The whole point: one tool, so it cannot be lost among 58 others."""
    provider = AsyncMock()
    provider.messages_create = AsyncMock(return_value=_Response([]))

    async def _drive() -> None:
        exec_ = Executive()
        async for _ in exec_._routing_prepass(
            [{"type": "text", "text": "persona"}],
            [{"role": "user", "content": "hi"}],
            model="m",
            episodic_context="",
            debug_collector=None,
            consulted_out=None,
            specialist_outputs_out=None,
            turn_id=None,
            conversation_context="",
            specialists_consulted=[],
        ):
            pass

    with patch(
        "openexecutive.orchestrator.executive.get_provider", return_value=provider
    ):
        asyncio.run(_drive())

    tools = provider.messages_create.await_args.kwargs["tools"]
    assert [t["name"] for t in tools] == ["consult_specialist"]


def test_picks_are_dispatched_and_appended_as_a_well_formed_pair() -> None:
    messages, consulted, route_mock = _run_prepass(
        [_consult("tu_1", "cso"), _consult("tu_2", "cfo")],
        specialist_results=["strategy-says", "finance-says"],
    )
    assert consulted == ["cso", "cfo"]
    assert route_mock.await_count == 1

    # user turn, then the replayed decision, then its results
    assert len(messages) == 3
    assistant, results = messages[1], messages[2]
    assert assistant["role"] == "assistant"
    assert [b["id"] for b in assistant["content"]] == ["tu_1", "tu_2"]
    assert results["role"] == "user"
    assert [b["tool_use_id"] for b in results["content"]] == ["tu_1", "tu_2"]
    assert [b["content"] for b in results["content"]] == [
        "strategy-says",
        "finance-says",
    ]


def test_every_replayed_tool_use_has_exactly_one_result() -> None:
    """The API rejects a tool_use with no matching tool_result."""
    messages, _, _ = _run_prepass([_consult("a", "cso"), _consult("b", "cmo")])
    tool_use_ids = {b["id"] for b in messages[1]["content"]}
    result_ids = [b["tool_use_id"] for b in messages[2]["content"]]
    assert tool_use_ids == set(result_ids)
    assert len(result_ids) == len(set(result_ids))


def test_no_picks_leaves_the_conversation_untouched() -> None:
    """An action turn must fall through to the main loop exactly as before."""
    messages, consulted, route_mock = _run_prepass(
        [_Block("text", text="I'll schedule that.")]
    )
    assert messages == [{"role": "user", "content": "what should I focus on?"}]
    assert consulted == []
    assert route_mock.await_count == 0


def test_truncated_specialist_names_are_normalised_before_dispatch() -> None:
    """`cs`/`cf` are what the local model actually emits."""
    _, consulted, route_mock = _run_prepass(
        [_consult("tu_1", "cs"), _consult("tu_2", "cf")]
    )
    assert consulted == ["cso", "cfo"]
    assert [c["specialist"] for c in route_mock.await_args.args[0]] == ["cso", "cfo"]


def test_unresolvable_names_are_dropped_not_dispatched() -> None:
    messages, consulted, route_mock = _run_prepass(
        [_consult("tu_1", "cso"), _consult("tu_2", "zzz")],
        specialist_results=["strategy-says"],
    )
    assert consulted == ["cso"]
    assert [c["specialist"] for c in route_mock.await_args.args[0]] == ["cso"]
    assert [b["id"] for b in messages[1]["content"]] == ["tu_1"]


def test_all_names_unresolvable_is_the_same_as_no_picks() -> None:
    messages, consulted, route_mock = _run_prepass(
        [_consult("tu_1", "zzz"), _consult("tu_2", "nope")]
    )
    assert messages == [{"role": "user", "content": "what should I focus on?"}]
    assert consulted == []
    assert route_mock.await_count == 0


def test_non_consult_tool_uses_are_ignored() -> None:
    """The pre-pass offers one tool; anything else the model invents is noise."""
    messages, consulted, _ = _run_prepass(
        [
            _Block("tool_use", id="x", name="search_tools", input={"q": "a"}),
            _consult("tu_1", "cso"),
        ],
        specialist_results=["strategy-says"],
    )
    assert consulted == ["cso"]
    assert [b["id"] for b in messages[1]["content"]] == ["tu_1"]


def test_provider_failure_does_not_fail_the_turn() -> None:
    """The main loop still offers consult_specialist, so this costs quality, not the answer."""
    provider = AsyncMock()
    provider.messages_create = AsyncMock(side_effect=RuntimeError("backend down"))
    messages = [{"role": "user", "content": "hi"}]

    async def _drive() -> None:
        exec_ = Executive()
        async for _ in exec_._routing_prepass(
            [{"type": "text", "text": "p"}],
            messages,
            model="m",
            episodic_context="",
            debug_collector=None,
            consulted_out=None,
            specialist_outputs_out=None,
            turn_id=None,
            conversation_context="",
            specialists_consulted=[],
        ):
            pass

    with patch(
        "openexecutive.orchestrator.executive.get_provider", return_value=provider
    ):
        asyncio.run(_drive())  # must not raise
    assert messages == [{"role": "user", "content": "hi"}]


def test_fanout_cap_bounds_the_prepass() -> None:
    """Each pick carries its own RAG + memory prefetch, so the cap must bind here too.

    Without this the pre-pass would be a way to fan out past
    MAX_PARALLEL_SPECIALISTS, since it dispatches before the loop's own
    partition ever runs.
    """
    picks = [_consult(f"tu_{i}", s) for i, s in enumerate(["cso", "cfo", "cmo", "coo"])]
    provider = AsyncMock()
    provider.messages_create = AsyncMock(return_value=_Response(picks))
    route_mock = AsyncMock(return_value=["a", "b"])
    messages: list[dict[str, Any]] = [{"role": "user", "content": "q"}]
    consulted: list[str] = []

    async def _drive() -> None:
        exec_ = Executive()
        exec_._settings = exec_._settings.model_copy(
            update={"max_parallel_specialists": 2}
        )
        async for _ in exec_._routing_prepass(
            [{"type": "text", "text": "p"}],
            messages,
            model="m",
            episodic_context="",
            debug_collector=None,
            consulted_out=consulted,
            specialist_outputs_out=None,
            turn_id=None,
            conversation_context="",
            specialists_consulted=[],
        ):
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch("openexecutive.orchestrator.executive.route_parallel", route_mock),
        patch("openexecutive.orchestrator.executive.audit_log"),
    ):
        asyncio.run(_drive())

    assert consulted == ["cso", "cfo"]
    assert len(route_mock.await_args.args[0]) == 2
    # The over-cap picks are dropped outright, so they leave no dangling
    # tool_use behind in the replayed message.
    assert [b["id"] for b in messages[1]["content"]] == ["tu_0", "tu_1"]


def _drive_agent_loop(
    prepass_enabled: bool,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Run _stream_agent_loop with the pre-pass spied on.

    Returns (pre-pass call kwargs, the message lists handed to messages_stream).
    """
    calls: list[dict[str, Any]] = []

    async def _spy(
        self: Any, *args: Any, **kwargs: Any
    ) -> Any:  # pragma: no cover - generator body
        # args[1] is the loop's `current_messages`. Append a marker the way the
        # real pre-pass appends its tool_use/tool_result pair, so the caller can
        # prove the mutation reaches the provider rather than a private copy.
        calls.append(kwargs)
        args[1].append({"role": "user", "content": "PREPASS_MARKER"})
        if False:
            yield ""

    class _Stream:
        async def __aenter__(self) -> "_Stream":
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        def __aiter__(self) -> Any:
            async def _gen() -> Any:
                return
                yield

            return _gen()

        async def get_final_message(self) -> Any:
            msg = _Response([_Block("text", text="done")])
            msg.stop_reason = "end_turn"
            msg.usage = None
            return msg

    streamed: list[list[dict[str, Any]]] = []
    provider = AsyncMock()

    def _stream(**kw: Any) -> "_Stream":
        streamed.append(list(kw["messages"]))
        return _Stream()

    provider.messages_stream = _stream

    async def _drive() -> None:
        exec_ = Executive()
        exec_._settings = exec_._settings.model_copy(
            update={"routing_prepass_enabled": prepass_enabled}
        )
        async for _ in exec_._stream_agent_loop(
            [{"type": "text", "text": "p"}],
            [{"role": "user", "content": "what should I focus on?"}],
            model="m",
            search=None,
        ):
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch.object(Executive, "_routing_prepass", new=_spy),
        patch("openexecutive.orchestrator.executive._emit_cache_event"),
    ):
        asyncio.run(_drive())
    return calls, streamed


def test_agent_loop_runs_the_prepass_when_enabled() -> None:
    """Wiring test: nothing else in the suite proves the loop calls the pre-pass."""
    calls, streamed = _drive_agent_loop(prepass_enabled=True)
    assert len(calls) == 1
    # And what the pre-pass appended must reach the provider — the loop does
    # `current_messages = list(messages)`, so a pre-pass handed the wrong list
    # would run, look fine, and be invisible to the model.
    assert streamed and streamed[0][-1]["content"] == "PREPASS_MARKER"


def test_agent_loop_skips_the_prepass_when_disabled() -> None:
    calls, streamed = _drive_agent_loop(prepass_enabled=False)
    assert calls == []
    # Positive control: the loop still ran and streamed, so the empty `calls`
    # above means "pre-pass skipped", not "loop crashed before reaching it".
    assert len(streamed) == 1
    assert all(m["content"] != "PREPASS_MARKER" for m in streamed[0])


def test_uncapped_prepass_dispatches_every_pick() -> None:
    """Control for the cap test above: with the default inert cap all four run.

    Without this, `consulted == ["cso", "cfo"]` would also pass if the pre-pass
    silently truncated to two picks for some unrelated reason.
    """
    picks = [_consult(f"tu_{i}", s) for i, s in enumerate(["cso", "cfo", "cmo", "coo"])]
    _, consulted, route_mock = _run_prepass(picks, specialist_results=["a"] * 4)
    assert consulted == ["cso", "cfo", "cmo", "coo"]
    assert len(route_mock.await_args.args[0]) == 4


def test_prepass_populates_specialist_outputs_for_the_committee() -> None:
    """This branch feeds committee reviewer selection and was previously untested."""
    outputs: dict[str, str] = {}
    provider = AsyncMock()
    provider.messages_create = AsyncMock(
        return_value=_Response([_consult("tu_1", "cso"), _consult("tu_2", "cfo")])
    )
    route_mock = AsyncMock(return_value=["strategy-says", "finance-says"])

    async def _drive() -> None:
        exec_ = Executive()
        async for _ in exec_._routing_prepass(
            [{"type": "text", "text": "p"}],
            [{"role": "user", "content": "q"}],
            model="m",
            episodic_context="",
            debug_collector=None,
            consulted_out=None,
            specialist_outputs_out=outputs,
            turn_id=None,
            conversation_context="",
            specialists_consulted=[],
        ):
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch("openexecutive.orchestrator.executive.route_parallel", route_mock),
        patch("openexecutive.orchestrator.executive.audit_log"),
    ):
        asyncio.run(_drive())

    assert outputs == {"cso": "strategy-says", "cfo": "finance-says"}


def test_prepass_drains_specialist_debug_events() -> None:
    """route_parallel appends events but does not yield them; the caller must.

    The loop takes its own cursor AFTER the pre-pass, so an undrained event
    here is yielded by nobody and the live panel goes silent on what is now
    the primary routing path.
    """

    class _Collector:
        def __init__(self) -> None:
            self._events: list[dict[str, Any]] = []

        def emit(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
            evt = {"kind": kind, **payload}
            self._events.append(evt)
            return evt

        def to_sse_dict(self, evt: dict[str, Any]) -> dict[str, Any]:
            return {"sse": evt}

    collector = _Collector()

    async def _fake_route_parallel(calls: Any, **kw: Any) -> list[str]:
        # Mimic route_parallel: emit into the collector, yield nothing.
        for c in calls:
            kw["debug_collector"].emit("specialist_start", {"specialist": c["specialist"]})
            kw["debug_collector"].emit("specialist_done", {"specialist": c["specialist"]})
        return ["analysis"] * len(calls)

    provider = AsyncMock()
    provider.messages_create = AsyncMock(
        return_value=_Response([_consult("tu_1", "cso")])
    )
    yielded: list[Any] = []

    async def _drive() -> None:
        exec_ = Executive()
        async for item in exec_._routing_prepass(
            [{"type": "text", "text": "p"}],
            [{"role": "user", "content": "q"}],
            model="m",
            episodic_context="",
            debug_collector=collector,
            consulted_out=None,
            specialist_outputs_out=None,
            turn_id=None,
            conversation_context="",
            specialists_consulted=[],
        ):
            yielded.append(item)

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch(
            "openexecutive.orchestrator.executive.route_parallel", _fake_route_parallel
        ),
        patch("openexecutive.orchestrator.executive.audit_log"),
    ):
        asyncio.run(_drive())

    kinds = [i["sse"]["kind"] for i in yielded if isinstance(i, dict) and "sse" in i]
    assert "specialist_start" in kinds
    assert "specialist_done" in kinds


def test_routing_decision_reports_dropped_picks() -> None:
    """requested/skipped counts must expose the drop, not hide it behind the cap."""

    class _Collector:
        def __init__(self) -> None:
            self._events: list[dict[str, Any]] = []

        def emit(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
            evt = {"kind": kind, **payload}
            self._events.append(evt)
            return evt

        def to_sse_dict(self, evt: dict[str, Any]) -> dict[str, Any]:
            return {"sse": evt}

    collector = _Collector()
    provider = AsyncMock()
    # two good picks, one unresolvable -> requested 3, dispatched 2, skipped 1
    provider.messages_create = AsyncMock(
        return_value=_Response(
            [_consult("tu_1", "cso"), _consult("tu_2", "zzz"), _consult("tu_3", "cfo")]
        )
    )

    async def _drive() -> None:
        exec_ = Executive()
        async for _ in exec_._routing_prepass(
            [{"type": "text", "text": "p"}],
            [{"role": "user", "content": "q"}],
            model="m",
            episodic_context="",
            debug_collector=collector,
            consulted_out=None,
            specialist_outputs_out=None,
            turn_id=None,
            conversation_context="",
            specialists_consulted=[],
        ):
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch(
            "openexecutive.orchestrator.executive.route_parallel",
            AsyncMock(return_value=["a", "b"]),
        ),
        patch("openexecutive.orchestrator.executive.audit_log"),
    ):
        asyncio.run(_drive())

    decision = next(e for e in collector._events if e["kind"] == "routing_decision")
    assert decision["requested_count"] == 3
    assert decision["dispatched_count"] == 2
    assert decision["skipped_count"] == 1


def _reasoning_block() -> _Block:
    """The synthetic block the translator always appends LAST on the
    OpenAI-compatible path (providers/translator.py)."""
    return _Block(
        "openrouter_reasoning",
        reasoning_details=[{"type": "reasoning.text", "text": "thinking"}],
    )


def test_truncated_tool_use_is_dropped_behind_a_trailing_reasoning_block() -> None:
    """The guard must key on the last TOOL_USE, not the last content block.

    The translator appends `openrouter_reasoning` last, so on exactly the local
    reasoning models this pre-pass exists for, no tool_use is ever content[-1]
    and a positional check would never fire.
    """
    resp = _Response(
        [
            _consult("tu_1", "cso", query="real question"),
            _consult("tu_2", "cfo", query=""),
            _reasoning_block(),
        ],
        stop_reason="max_tokens",
    )
    provider = AsyncMock()
    provider.messages_create = AsyncMock(return_value=resp)
    route_mock = AsyncMock(return_value=["strategy-says"])
    consulted: list[str] = []

    async def _drive() -> None:
        exec_ = Executive()
        async for _ in exec_._routing_prepass(
            [{"type": "text", "text": "p"}],
            [{"role": "user", "content": "q"}],
            model="m",
            episodic_context="",
            debug_collector=None,
            consulted_out=consulted,
            specialist_outputs_out=None,
            turn_id=None,
            conversation_context="",
            specialists_consulted=[],
        ):
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch("openexecutive.orchestrator.executive.route_parallel", route_mock),
        patch("openexecutive.orchestrator.executive.audit_log"),
    ):
        asyncio.run(_drive())

    assert consulted == ["cso"]


def test_main_loop_records_the_canonical_specialist_name() -> None:
    """route_parallel rebinds its own local `calls`, so normalising only there
    would dispatch to the CSO while the loop recorded "cs" in consulted_out and
    as the audit actor — and committee reviewer selection and the department
    sync both drop an unrecognised key."""
    consulted: list[str] = []

    class _Stream:
        def __init__(self, msg: Any) -> None:
            self._msg = msg

        async def __aenter__(self) -> "_Stream":
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        def __aiter__(self) -> Any:
            async def _gen() -> Any:
                return
                yield

            return _gen()

        async def get_final_message(self) -> Any:
            return self._msg

    tool_msg = _Response([_consult("tu_1", "cs", query="q")], stop_reason="tool_use")
    tool_msg.usage = None
    done_msg = _Response([_Block("text", text="done")], stop_reason="end_turn")
    done_msg.usage = None
    scripted = [tool_msg, done_msg]

    provider = AsyncMock()
    provider.messages_stream = lambda **kw: _Stream(scripted.pop(0))

    async def _drive() -> None:
        exec_ = Executive()
        exec_._settings = exec_._settings.model_copy(
            update={"routing_prepass_enabled": False}
        )
        async for _ in exec_._stream_agent_loop(
            [{"type": "text", "text": "p"}],
            [{"role": "user", "content": "q"}],
            model="m",
            search=None,
            consulted_out=consulted,
        ):
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch(
            "openexecutive.orchestrator.executive.route_parallel",
            AsyncMock(return_value=["strategy-says"]),
        ),
        patch("openexecutive.orchestrator.executive._emit_cache_event"),
        patch("openexecutive.orchestrator.executive.audit_log"),
    ):
        asyncio.run(_drive())

    assert consulted == ["cso"], f"loop recorded the raw token: {consulted}"


def test_reasoning_blocks_are_replayed_into_the_assistant_turn() -> None:
    """A tool_use turn whose reasoning was dropped can be 400'd by a
    thinking-enabled model, and OpenRouter loses continuity without it. Every
    other block-by-block replay in the repo carries it; so must this one."""
    messages, _, _ = _run_prepass(
        [_consult("tu_1", "cso"), _reasoning_block()],
        specialist_results=["strategy-says"],
    )
    assistant = messages[1]
    kinds = [b["type"] for b in assistant["content"]]
    assert "openrouter_reasoning" in kinds, f"reasoning stripped from replay: {kinds}"
    # Reasoning leads, matching the loop's replay order.
    assert kinds[0] == "openrouter_reasoning"
    assert kinds[1:] == ["tool_use"]


def test_routing_decision_is_emitted_when_every_pick_is_dropped() -> None:
    """"Asked for specialists, reached nobody" is the looks-routed-but-isn't
    case — it must not be visible only in the audit log."""

    class _Collector:
        def __init__(self) -> None:
            self._events: list[dict[str, Any]] = []

        def emit(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
            evt = {"kind": kind, **payload}
            self._events.append(evt)
            return evt

        def to_sse_dict(self, evt: dict[str, Any]) -> dict[str, Any]:
            return {"sse": evt}

    collector = _Collector()
    provider = AsyncMock()
    provider.messages_create = AsyncMock(
        return_value=_Response([_consult("tu_1", "zzz"), _consult("tu_2", "nope")])
    )
    route_mock = AsyncMock(return_value=[])

    async def _drive() -> None:
        exec_ = Executive()
        async for _ in exec_._routing_prepass(
            [{"type": "text", "text": "p"}],
            [{"role": "user", "content": "q"}],
            model="m",
            episodic_context="",
            debug_collector=collector,
            consulted_out=None,
            specialist_outputs_out=None,
            turn_id=None,
            conversation_context="",
            specialists_consulted=[],
        ):
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch("openexecutive.orchestrator.executive.route_parallel", route_mock),
        patch("openexecutive.orchestrator.executive.audit_log"),
    ):
        asyncio.run(_drive())

    decision = next(e for e in collector._events if e["kind"] == "routing_decision")
    assert decision["requested_count"] == 2
    assert decision["dispatched_count"] == 0
    assert decision["skipped_count"] == 2
    assert route_mock.await_count == 0


def test_no_routing_decision_when_the_model_picked_nobody() -> None:
    """Control for the test above: an action turn must stay silent, not emit a
    zero-count routing event."""

    class _Collector:
        def __init__(self) -> None:
            self._events: list[dict[str, Any]] = []

        def emit(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
            evt = {"kind": kind, **payload}
            self._events.append(evt)
            return evt

        def to_sse_dict(self, evt: dict[str, Any]) -> dict[str, Any]:
            return {"sse": evt}

    collector = _Collector()
    provider = AsyncMock()
    provider.messages_create = AsyncMock(
        return_value=_Response([_Block("text", text="I'll schedule that.")])
    )

    async def _drive() -> None:
        exec_ = Executive()
        async for _ in exec_._routing_prepass(
            [{"type": "text", "text": "p"}],
            [{"role": "user", "content": "q"}],
            model="m",
            episodic_context="",
            debug_collector=collector,
            consulted_out=None,
            specialist_outputs_out=None,
            turn_id=None,
            conversation_context="",
            specialists_consulted=[],
        ):
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch("openexecutive.orchestrator.executive.audit_log"),
    ):
        asyncio.run(_drive())

    assert collector._events == []


class _RecordingCollector:
    """Minimal DebugCollector stand-in that records every emitted event."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def emit(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        evt = {"kind": kind, **payload}
        self._events.append(evt)
        return evt

    def to_sse_dict(self, evt: dict[str, Any]) -> dict[str, Any]:
        return {"sse": evt}


def _drive_loop_with_consults(
    scripted: list[Any], collector: _RecordingCollector
) -> None:
    """Run _stream_agent_loop (pre-pass off) over a scripted provider."""

    class _Stream:
        def __init__(self, msg: Any) -> None:
            self._msg = msg

        async def __aenter__(self) -> "_Stream":
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        def __aiter__(self) -> Any:
            async def _gen() -> Any:
                return
                yield

            return _gen()

        async def get_final_message(self) -> Any:
            return self._msg

    queue = list(scripted)
    provider = AsyncMock()
    provider.messages_stream = lambda **kw: _Stream(queue.pop(0))

    async def _drive() -> None:
        exec_ = Executive()
        exec_._settings = exec_._settings.model_copy(
            update={"routing_prepass_enabled": False}
        )
        async for _ in exec_._stream_agent_loop(
            [{"type": "text", "text": "p"}],
            [{"role": "user", "content": "q"}],
            model="m",
            search=None,
            debug_collector=collector,
        ):
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch(
            "openexecutive.orchestrator.executive.route_parallel",
            AsyncMock(side_effect=lambda calls, **kw: ["analysis"] * len(calls)),
        ),
        patch("openexecutive.orchestrator.executive._emit_cache_event"),
        patch("openexecutive.orchestrator.executive.audit_log"),
    ):
        asyncio.run(_drive())


def test_synthesis_start_re_emits_when_the_roster_grows() -> None:
    """A second consult round must update the panel, not be swallowed.

    Firing only once would report the first round's roster forever and never
    mention a specialist consulted later in the turn.
    """
    collector = _RecordingCollector()
    round1 = _Response([_consult("tu_1", "cso")], stop_reason="tool_use")
    round1.usage = None
    round2 = _Response([_consult("tu_2", "cfo")], stop_reason="tool_use")
    round2.usage = None
    done = _Response([_Block("text", text="final")], stop_reason="end_turn")
    done.usage = None

    _drive_loop_with_consults([round1, round2, done], collector)

    synth = [e for e in collector._events if e["kind"] == "synthesis_start"]
    assert len(synth) == 2, f"expected one emit per roster growth, got {len(synth)}"
    assert synth[0]["specialists_consulted"] == ["cso"]
    assert synth[1]["specialists_consulted"] == ["cso", "cfo"]
    assert synth[1]["specialist_count"] == 2


def test_synthesis_start_fires_at_iteration_one_from_the_prepass_roster() -> None:
    """The scenario the re-emit-on-growth logic exists for, end to end.

    Every other synthesis test disables the pre-pass, and the one wiring test
    stubs it with a spy that never touches specialists_consulted — so the
    pre-pass-seeded path was untested despite being the default.
    """
    collector = _RecordingCollector()

    class _Stream:
        async def __aenter__(self) -> "_Stream":
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        def __aiter__(self) -> Any:
            async def _gen() -> Any:
                return
                yield

            return _gen()

        async def get_final_message(self) -> Any:
            msg = _Response([_Block("text", text="final")], stop_reason="end_turn")
            msg.usage = None
            return msg

    provider = AsyncMock()
    provider.messages_stream = lambda **kw: _Stream()
    # The real _routing_prepass runs; only its provider call is scripted.
    provider.messages_create = AsyncMock(
        return_value=_Response([_consult("tu_1", "cso")])
    )

    async def _drive() -> None:
        exec_ = Executive()
        exec_._settings = exec_._settings.model_copy(
            update={"routing_prepass_enabled": True}
        )
        async for _ in exec_._stream_agent_loop(
            [{"type": "text", "text": "p"}],
            [{"role": "user", "content": "q"}],
            model="m",
            search=None,
            debug_collector=collector,
        ):
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch(
            "openexecutive.orchestrator.executive.route_parallel",
            AsyncMock(return_value=["strategy-says"]),
        ),
        patch("openexecutive.orchestrator.executive._emit_cache_event"),
        patch("openexecutive.orchestrator.executive.audit_log"),
    ):
        asyncio.run(_drive())

    synth = [e for e in collector._events if e["kind"] == "synthesis_start"]
    assert len(synth) == 1, f"expected exactly one emit, got {len(synth)}"
    assert synth[0]["specialists_consulted"] == ["cso"]


def test_synthesis_start_does_not_repeat_for_an_unchanged_roster() -> None:
    """Control for the test above: a non-specialist tool round must not
    re-announce synthesis with the same roster."""
    collector = _RecordingCollector()
    consult_round = _Response([_consult("tu_1", "cso")], stop_reason="tool_use")
    consult_round.usage = None
    skill_round = _Response(
        [_Block("tool_use", id="tu_2", name="list_people", input={})],
        stop_reason="tool_use",
    )
    skill_round.usage = None
    done = _Response([_Block("text", text="final")], stop_reason="end_turn")
    done.usage = None

    _drive_loop_with_consults([consult_round, skill_round, done], collector)

    synth = [e for e in collector._events if e["kind"] == "synthesis_start"]
    assert len(synth) == 1, f"roster did not grow; expected one emit, got {len(synth)}"
    assert synth[0]["specialists_consulted"] == ["cso"]


def test_main_loop_does_not_record_an_unresolvable_name_as_a_consult() -> None:
    """The loop cannot drop the tool_use (the API needs one result per
    tool_use, and the model needs the error to recover) — but an unresolvable
    name reached nobody, so it must not be filed as a consult. Otherwise the
    audit graph draws a specialist node for `zzz` and attributes later tool
    calls to it, which is the failure this whole change exists to remove."""
    consulted: list[str] = []
    outputs: dict[str, str] = {}
    audited: list[tuple[str, Any]] = []

    class _Stream:
        def __init__(self, msg: Any) -> None:
            self._msg = msg

        async def __aenter__(self) -> "_Stream":
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        def __aiter__(self) -> Any:
            async def _gen() -> Any:
                return
                yield

            return _gen()

        async def get_final_message(self) -> Any:
            return self._msg

    bad = _Response([_consult("tu_1", "zzz")], stop_reason="tool_use")
    bad.usage = None
    done = _Response([_Block("text", text="done")], stop_reason="end_turn")
    done.usage = None
    queue = [bad, done]
    provider = AsyncMock()
    provider.messages_stream = lambda **kw: _Stream(queue.pop(0))

    def _record(event_type: str, summary: str, **kw: Any) -> None:
        audited.append((event_type, kw.get("actor")))

    async def _drive() -> None:
        exec_ = Executive()
        exec_._settings = exec_._settings.model_copy(
            update={"routing_prepass_enabled": False}
        )
        async for _ in exec_._stream_agent_loop(
            [{"type": "text", "text": "p"}],
            [{"role": "user", "content": "q"}],
            model="m",
            search=None,
            consulted_out=consulted,
            specialist_outputs_out=outputs,
        ):
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch("openexecutive.orchestrator.executive._emit_cache_event"),
        patch("openexecutive.orchestrator.executive.audit_log", _record),
        patch("openexecutive.orchestrator.router.audit_log"),
    ):
        asyncio.run(_drive())

    assert consulted == [], f"unresolvable name recorded as consulted: {consulted}"
    assert outputs == {}
    assert [a for a in audited if a[0] == "specialist_consult"] == [], (
        f"filed a specialist_consult for a consult that never ran: {audited}"
    )


def test_main_loop_still_records_a_resolvable_consult() -> None:
    """Control for the test above: the filter must not swallow real consults."""
    consulted: list[str] = []
    audited: list[tuple[str, Any]] = []

    class _Stream:
        def __init__(self, msg: Any) -> None:
            self._msg = msg

        async def __aenter__(self) -> "_Stream":
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        def __aiter__(self) -> Any:
            async def _gen() -> Any:
                return
                yield

            return _gen()

        async def get_final_message(self) -> Any:
            return self._msg

    good = _Response([_consult("tu_1", "cso")], stop_reason="tool_use")
    good.usage = None
    done = _Response([_Block("text", text="done")], stop_reason="end_turn")
    done.usage = None
    queue = [good, done]
    provider = AsyncMock()
    provider.messages_stream = lambda **kw: _Stream(queue.pop(0))

    def _record(event_type: str, summary: str, **kw: Any) -> None:
        audited.append((event_type, kw.get("actor")))

    async def _drive() -> None:
        exec_ = Executive()
        exec_._settings = exec_._settings.model_copy(
            update={"routing_prepass_enabled": False}
        )
        async for _ in exec_._stream_agent_loop(
            [{"type": "text", "text": "p"}],
            [{"role": "user", "content": "q"}],
            model="m",
            search=None,
            consulted_out=consulted,
        ):
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch(
            "openexecutive.orchestrator.executive.route_parallel",
            AsyncMock(return_value=["strategy-says"]),
        ),
        patch("openexecutive.orchestrator.executive._emit_cache_event"),
        patch("openexecutive.orchestrator.executive.audit_log", _record),
    ):
        asyncio.run(_drive())

    assert consulted == ["cso"]
    assert ("specialist_consult", "cso") in audited


def test_loop_audits_a_corrected_specialist_name() -> None:
    """Normalising early makes route_to_specialist's own correction branch
    unreachable from the chat path, so the correction must be audited where it
    happens or it goes unrecorded entirely."""
    rows: list[tuple[str, Any]] = []

    def _record(event_type: str, summary: str, **kw: Any) -> None:
        rows.append((event_type, kw.get("details")))

    class _Stream:
        def __init__(self, msg: Any) -> None:
            self._msg = msg

        async def __aenter__(self) -> "_Stream":
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        def __aiter__(self) -> Any:
            async def _gen() -> Any:
                return
                yield

            return _gen()

        async def get_final_message(self) -> Any:
            return self._msg

    truncated = _Response([_consult("tu_1", "cs")], stop_reason="tool_use")
    truncated.usage = None
    done = _Response([_Block("text", text="done")], stop_reason="end_turn")
    done.usage = None
    queue = [truncated, done]
    provider = AsyncMock()
    provider.messages_stream = lambda **kw: _Stream(queue.pop(0))

    async def _drive() -> None:
        exec_ = Executive()
        exec_._settings = exec_._settings.model_copy(
            update={"routing_prepass_enabled": False}
        )
        async for _ in exec_._stream_agent_loop(
            [{"type": "text", "text": "p"}],
            [{"role": "user", "content": "q"}],
            model="m",
            search=None,
        ):
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch(
            "openexecutive.orchestrator.executive.route_parallel",
            AsyncMock(return_value=["strategy-says"]),
        ),
        patch("openexecutive.orchestrator.executive._emit_cache_event"),
        patch("openexecutive.orchestrator.executive.audit_log", _record),
    ):
        asyncio.run(_drive())

    assert ("routing_anomaly", {"requested": "cs", "resolved": "cso"}) in rows


def test_prepass_audits_a_corrected_specialist_name() -> None:
    """Same claim, pre-pass path — it previously audited drops but not
    corrections."""
    rows: list[tuple[str, Any]] = []

    def _record(event_type: str, summary: str, **kw: Any) -> None:
        rows.append((event_type, kw.get("details")))

    provider = AsyncMock()
    provider.messages_create = AsyncMock(
        return_value=_Response([_consult("tu_1", "cf")])
    )

    async def _drive() -> None:
        exec_ = Executive()
        async for _ in exec_._routing_prepass(
            [{"type": "text", "text": "p"}],
            [{"role": "user", "content": "q"}],
            model="m",
            episodic_context="",
            debug_collector=None,
            consulted_out=None,
            specialist_outputs_out=None,
            turn_id=None,
            conversation_context="",
            specialists_consulted=[],
        ):
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch(
            "openexecutive.orchestrator.executive.route_parallel",
            AsyncMock(return_value=["finance-says"]),
        ),
        patch("openexecutive.orchestrator.executive.audit_log", _record),
    ):
        asyncio.run(_drive())

    assert ("routing_anomaly", {"requested": "cf", "resolved": "cfo"}) in rows


def test_a_well_formed_name_produces_no_anomaly_row() -> None:
    """Control: the anomaly rows must mark real corrections, not every consult."""
    rows: list[tuple[str, Any]] = []

    def _record(event_type: str, summary: str, **kw: Any) -> None:
        rows.append((event_type, kw.get("details")))

    provider = AsyncMock()
    provider.messages_create = AsyncMock(
        return_value=_Response([_consult("tu_1", "cso")])
    )

    async def _drive() -> None:
        exec_ = Executive()
        async for _ in exec_._routing_prepass(
            [{"type": "text", "text": "p"}],
            [{"role": "user", "content": "q"}],
            model="m",
            episodic_context="",
            debug_collector=None,
            consulted_out=None,
            specialist_outputs_out=None,
            turn_id=None,
            conversation_context="",
            specialists_consulted=[],
        ):
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch(
            "openexecutive.orchestrator.executive.route_parallel",
            AsyncMock(return_value=["strategy-says"]),
        ),
        patch("openexecutive.orchestrator.executive.audit_log", _record),
    ):
        asyncio.run(_drive())

    assert [r for r in rows if r[0] == "routing_anomaly"] == []


def test_audit_row_shape_differs_only_by_phase_between_call_sites() -> None:
    """The shared helper exists so the two call sites cannot drift; nothing
    pinned its payload, so a schema change at one site would go unnoticed."""
    from openexecutive.orchestrator.executive import _audit_specialist_consults

    captured: list[dict[str, Any]] = []

    def _record(event_type: str, summary: str, **kw: Any) -> None:
        captured.append({"event_type": event_type, "summary": summary, **kw})

    calls = [{"specialist": "cso", "query": "the question"}]
    with patch("openexecutive.orchestrator.executive.audit_log", _record):
        for phase, iteration in (("prepass", 0), ("loop", 2)):
            _audit_specialist_consults(
                run_calls=calls,
                specialist_results=["the answer"],
                session_id="s-1",
                turn_id="t-1",
                iteration=iteration,
                phase=phase,
                duration_ms=42,
                conversation_context="CTX",
                system_blocks=[],
            )

    prepass, loop = captured
    for row in (prepass, loop):
        assert row["event_type"] == "specialist_consult"
        assert row["actor"] == "cso"
        assert row["details"]["duration_ms"] == 42
        assert row["details"]["context_preview"] == "CTX"
        assert row["full"]["query"] == "the question"
        assert row["full"]["response"] == "the answer"
    assert prepass["details"]["phase"] == "prepass"
    assert prepass["details"]["iteration"] == 0
    assert loop["details"]["phase"] == "loop"
    assert loop["details"]["iteration"] == 2


def test_prepass_audits_a_dropped_name_as_routing_anomaly() -> None:
    """The headline "malformed names stop being invisible" claim — unasserted
    until now, because every other test patches audit_log with a bare mock."""
    captured: list[tuple[str, Any]] = []

    def _record(event_type: str, summary: str, **kw: Any) -> None:
        captured.append((event_type, kw.get("details")))

    provider = AsyncMock()
    provider.messages_create = AsyncMock(
        return_value=_Response([_consult("tu_1", "zzz")])
    )

    async def _drive() -> None:
        exec_ = Executive()
        async for _ in exec_._routing_prepass(
            [{"type": "text", "text": "p"}],
            [{"role": "user", "content": "q"}],
            model="m",
            episodic_context="",
            debug_collector=None,
            consulted_out=None,
            specialist_outputs_out=None,
            turn_id=None,
            conversation_context="",
            specialists_consulted=[],
        ):
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch("openexecutive.orchestrator.executive.audit_log", _record),
    ):
        asyncio.run(_drive())

    assert ("routing_anomaly", {"requested": "zzz", "resolved": None}) in captured
    assert [c for c in captured if c[0] == "specialist_consult"] == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [({"specialist": "cso"}, {"specialist": "cso"}), (None, {}), (["cs"], {}), ("x", {})],
)
def test_tool_input_coerces_a_non_dict_to_empty(value: Any, expected: Any) -> None:
    """Only Anthropic validates tool arguments; a local backend can hand back
    null or a list, on which .get() raises."""
    from openexecutive.orchestrator.executive import _tool_input

    assert _tool_input({"input": value}) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("cso", "cso"), ("cs", "cso"), ("CSO", "cso"), ("zzz", "zzz"), (None, "None")],
)
def test_canonical_specialist_contract(raw: Any, expected: str) -> None:
    """Returns the registry key, or the name unchanged so the unresolvable case
    still reaches route_to_specialist for its error string."""
    from openexecutive.orchestrator.executive import _canonical_specialist

    assert _canonical_specialist(raw) == expected


def test_prepass_records_a_cache_event_for_its_own_call() -> None:
    """/audit/usage aggregates cache_event only — an unrecorded call under-reports cost."""
    provider = AsyncMock()
    provider.messages_create = AsyncMock(return_value=_Response([]))

    async def _drive() -> None:
        exec_ = Executive()
        async for _ in exec_._routing_prepass(
            [{"type": "text", "text": "p"}],
            [{"role": "user", "content": "q"}],
            model="m",
            episodic_context="",
            debug_collector=None,
            consulted_out=None,
            specialist_outputs_out=None,
            turn_id="t-1",
            conversation_context="",
            specialists_consulted=[],
        ):
            pass

    with (
        patch(
            "openexecutive.orchestrator.executive.get_provider", return_value=provider
        ),
        patch("openexecutive.orchestrator.executive._emit_cache_event") as cache_evt,
    ):
        asyncio.run(_drive())

    assert cache_evt.call_count == 1
    assert cache_evt.call_args.kwargs["actor"] == "routing_prepass"
    assert cache_evt.call_args.kwargs["iteration"] == 0


@pytest.mark.parametrize("enabled", [True, False])
def test_settings_flag_gates_the_prepass(enabled: bool) -> None:
    from openexecutive.config import Settings

    settings = Settings(
        ANTHROPIC_API_KEY="sk-test",
        EXEC_EMAIL_ADDRESS="exec@example.test",
        ROUTING_PREPASS_ENABLED=enabled,
    )
    assert settings.routing_prepass_enabled is enabled


def test_prepass_defaults_on() -> None:
    from openexecutive.config import Settings

    settings = Settings(
        ANTHROPIC_API_KEY="sk-test", EXEC_EMAIL_ADDRESS="exec@example.test"
    )
    assert settings.routing_prepass_enabled is True
