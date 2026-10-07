"""Tests for the generic OpenAI-compatible provider used by local backends.

OpenRouter's request/response/stream behavior is pinned in
``test_openrouter_provider_lifecycle.py`` (OpenRouter now subclasses this
provider). This file covers the behavior that's specific to the local /
self-hosted path: optional auth, verbatim model passthrough, and the
Anthropic-only feature stripping that a plain OpenAI-compatible server needs.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from openexecutive.providers import openai_compatible
from openexecutive.providers.feature_gate import FeatureSpec
from openexecutive.providers.openai_compatible import OpenAICompatibleProvider

_LOCAL_SPEC = FeatureSpec(
    supports_cache_control=False,
    supports_thinking=False,
    supports_web_search=False,
    supports_tool_use=True,
)


def _local_provider(
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="http://localhost:11434/v1",
        api_key=api_key,
        spec_lookup={"llama3.3": _LOCAL_SPEC},
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        top_p=top_p,
    )


def _ok_response() -> MagicMock:
    fake = MagicMock()
    fake.json.return_value = {
        "id": "x",
        "choices": [
            {
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
    }
    fake.raise_for_status = MagicMock()
    return fake


def _run_create(provider: OpenAICompatibleProvider, **kwargs: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def _fake_post(url: str, **post_kwargs: Any) -> Any:
        captured["url"] = url
        captured["headers"] = post_kwargs.get("headers", {})
        captured["json"] = post_kwargs.get("json", {})
        return _ok_response()

    provider._client.post = AsyncMock(side_effect=_fake_post)  # type: ignore[method-assign]
    asyncio.run(
        provider.messages_create(
            model=kwargs.pop("model", "llama3.3"),
            max_tokens=kwargs.pop("max_tokens", 8),
            messages=kwargs.pop(
                "messages", [{"role": "user", "content": "hi"}]
            ),
            **kwargs,
        )
    )
    return captured


def test_no_auth_header_when_api_key_absent() -> None:
    """Local servers (Ollama, LM Studio) need no auth — we must NOT send a
    bogus ``Authorization: Bearer None`` that a strict server could reject."""
    captured = _run_create(_local_provider(api_key=None))
    assert "Authorization" not in captured["headers"]


def test_auth_header_present_when_api_key_given() -> None:
    """vLLM or a gateway in front of a local server may require a token."""
    captured = _run_create(_local_provider(api_key="vllm-secret"))
    assert captured["headers"]["Authorization"] == "Bearer vllm-secret"


def test_unknown_model_passes_through_verbatim() -> None:
    """Local model names aren't translated — they're sent as-is so they match
    what the server actually serves."""
    captured = _run_create(_local_provider(), model="llama3.3")
    assert captured["json"]["model"] == "llama3.3"


def test_anthropic_only_fields_stripped_for_local_model() -> None:
    """thinking / output_config / cache_control have no OpenAI-format
    equivalent and would 400 a plain server — the feature gate drops them."""
    captured = _run_create(
        _local_provider(),
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        system=[
            {"type": "text", "text": "P", "cache_control": {"type": "ephemeral"}}
        ],
    )
    body = captured["json"]
    assert "thinking" not in body
    assert "output_config" not in body
    # no LOCAL_REASONING_EFFORT configured: the server decides whether to think
    assert "reasoning" not in body
    assert "reasoning_effort" not in body
    # system flattened into a plain string message — no cache_control survives.
    assert isinstance(body["messages"][0]["content"], str)


def test_sampling_not_sent_when_unset() -> None:
    """Unset means the field is absent. Worth pinning because absent is NOT
    neutral on Ollama's /v1 — it substitutes temperature=1.0 and top_p=1.0,
    overriding the Modelfile — so "we send nothing" must be a deliberate
    state, not an accident."""
    captured = _run_create(_local_provider())
    assert "temperature" not in captured["json"]
    assert "top_p" not in captured["json"]


def test_sampling_sent_when_configured() -> None:
    captured = _run_create(_local_provider(temperature=0.7, top_p=0.8))
    assert captured["json"]["temperature"] == 0.7
    assert captured["json"]["top_p"] == 0.8


def test_explicit_caller_temperature_wins_over_the_configured_default() -> None:
    """`_extend_body` uses setdefault. integrations/response_gate.py passes
    temperature=0 and must stay deterministic even with LOCAL_TEMPERATURE set
    — a plain assignment here would silently make the gate stochastic."""
    captured = _run_create(_local_provider(temperature=0.7, top_p=0.8), temperature=0)
    assert captured["json"]["temperature"] == 0
    # top_p, which the caller did not set, still takes the configured default.
    assert captured["json"]["top_p"] == 0.8


def test_sampling_sent_on_stream() -> None:
    """The Executive streams, so the streaming body must carry them too."""
    stream = _local_provider(temperature=0.7, top_p=0.8).messages_stream(
        model="llama3.3",
        max_tokens=8,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert stream._body["temperature"] == 0.7  # type: ignore[attr-defined]
    assert stream._body["top_p"] == 0.8  # type: ignore[attr-defined]


def test_sampling_suppressed_once_reasoning_is_turned_on() -> None:
    """A strict reasoning backend (OpenAI's o-series, and hosted gateways are
    documented as reachable through LOCAL_BASE_URL) requires temperature and
    top_p to be ABSENT, so sending them 400s every call. They are also the
    wrong numbers there: the defaults are Qwen's non-thinking preset."""
    captured = _run_create(_local_provider(temperature=0.7, top_p=0.8, reasoning_effort="high"))
    assert captured["json"]["reasoning_effort"] == "high"
    assert "temperature" not in captured["json"]
    assert "top_p" not in captured["json"]


def test_sampling_still_sent_with_reasoning_explicitly_off() -> None:
    """`none` is the deployment's own setting and the condition the preset was
    chosen for, so it must NOT suppress them — that is the Ollama Modelfile
    override this pair exists to stop."""
    captured = _run_create(_local_provider(temperature=0.7, top_p=0.8, reasoning_effort="none"))
    assert captured["json"]["temperature"] == 0.7
    assert captured["json"]["top_p"] == 0.8


def test_sampling_suppressed_on_stream_too_when_reasoning_is_on() -> None:
    stream = _local_provider(temperature=0.7, top_p=0.8, reasoning_effort="low").messages_stream(
        model="llama3.3",
        max_tokens=8,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert "temperature" not in stream._body  # type: ignore[attr-defined]
    assert "top_p" not in stream._body  # type: ignore[attr-defined]


def test_an_explicit_caller_temperature_survives_reasoning_being_on() -> None:
    """Only the CONFIGURED defaults are suppressed, never a value the caller
    asked for. integrations/response_gate.py passes temperature=0 because the
    outbound gate must be deterministic; stripping it on a reasoning backend
    would make the gate stochastic silently, which is worse than the 400 that
    backend will raise and which an operator can actually diagnose."""
    captured = _run_create(
        _local_provider(temperature=0.7, top_p=0.8, reasoning_effort="high"), temperature=0
    )
    assert captured["json"]["temperature"] == 0
    assert "top_p" not in captured["json"]


def test_no_top_k_is_sent() -> None:
    """Negative control for a setting we deliberately did NOT add: top_k is
    not in the OpenAI schema and Ollama's /v1 silently drops it (verified
    against the live runner), so a LOCAL_TOP_K would be inert and
    misleading. If someone adds one, this fails and sends them to read why."""
    captured = _run_create(_local_provider(temperature=0.7, top_p=0.8))
    assert "top_k" not in captured["json"]


def test_reasoning_effort_sent_flat_on_create() -> None:
    """Ollama's /v1 endpoint reads the flat OpenAI field, not a nested object."""
    captured = _run_create(_local_provider(reasoning_effort="none"))
    assert captured["json"]["reasoning_effort"] == "none"
    assert "reasoning" not in captured["json"]


def _effort_provider(effort: str | None) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="https://api.fireworks.ai/inference/v1",
        spec_lookup={"llama3.3": _LOCAL_SPEC},
        reasoning_effort=effort,
    )


def test_reasoning_effort_omitted_by_default() -> None:
    """Most OpenAI-compatible servers don't know the field; unset = not sent."""
    captured = _run_create(_local_provider())
    assert "reasoning_effort" not in captured["json"]


def test_reasoning_effort_sent_when_configured() -> None:
    """Thinking-only models (GLM on Fireworks) otherwise burn the whole
    max_tokens budget reasoning and return no tool call."""
    captured = _run_create(_effort_provider("low"))
    assert captured["json"]["reasoning_effort"] == "low"


def test_reasoning_effort_sent_on_stream() -> None:
    provider = _effort_provider("low")
    stream = provider.messages_stream(
        model="llama3.3",
        max_tokens=8,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert stream._body["reasoning_effort"] == "low"  # type: ignore[attr-defined]
    assert stream._body["stream"] is True  # type: ignore[attr-defined]


def test_stream_options_sent_on_stream() -> None:
    """A plain OpenAI-compatible backend omits the usage block entirely on a
    streamed response without this opt-in — confirmed out of band against
    Ollama 0.34.0. This test pins only that we send the flag."""
    stream = _local_provider().messages_stream(
        model="llama3.3",
        max_tokens=8,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert stream._body["stream_options"] == {"include_usage": True}  # type: ignore[attr-defined]


def test_stream_options_absent_on_non_streaming_call() -> None:
    """``stream_options`` is only valid alongside ``stream: true`` — an
    OpenAI-spec server 400s it otherwise. This pins the flag to
    ``messages_stream`` and keeps it out of the shared body builder."""
    captured = _run_create(_local_provider())
    assert "stream_options" not in captured["json"]


def test_reasoning_effort_logged_once_per_slug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The effort in play is what decides whether a thinking-only model
    answers at all, so it's logged, but once per model, not per call."""
    monkeypatch.setattr(openai_compatible, "_effort_announced", set())
    fake_logger = MagicMock()
    monkeypatch.setattr(openai_compatible, "logger", fake_logger)
    provider = _effort_provider("low")
    _run_create(provider)
    _run_create(provider)
    assert fake_logger.info.call_count == 1
    assert fake_logger.info.call_args.args[1:] == ("low", "llama3.3")
