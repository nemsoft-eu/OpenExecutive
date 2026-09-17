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

from openexecutive.providers.feature_gate import FeatureSpec
from openexecutive.providers.openai_compatible import OpenAICompatibleProvider

_LOCAL_SPEC = FeatureSpec(
    supports_cache_control=False,
    supports_thinking=False,
    supports_web_search=False,
    supports_tool_use=True,
)


def _local_provider(
    api_key: str | None = None, reasoning_effort: str | None = None
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="http://localhost:11434/v1",
        api_key=api_key,
        spec_lookup={"llama3.3": _LOCAL_SPEC},
        reasoning_effort=reasoning_effort,
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
    # system flattened into a plain string message — no cache_control survives.
    assert isinstance(body["messages"][0]["content"], str)


def test_no_reasoning_effort_field_by_default() -> None:
    """Unset keeps today's wire format: the server decides whether to think."""
    captured = _run_create(_local_provider(), thinking={"type": "adaptive"})
    assert "reasoning_effort" not in captured["json"]
    assert "reasoning" not in captured["json"]


def test_reasoning_effort_sent_flat_on_create() -> None:
    """Ollama's /v1 endpoint reads the flat OpenAI field, not a nested object."""
    captured = _run_create(_local_provider(reasoning_effort="none"))
    assert captured["json"]["reasoning_effort"] == "none"
    assert "reasoning" not in captured["json"]


def test_reasoning_effort_sent_on_stream() -> None:
    """The Executive streams — the streaming body must carry the field too."""
    stream = _local_provider(reasoning_effort="low").messages_stream(
        model="llama3.3",
        max_tokens=8,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert stream._body["reasoning_effort"] == "low"  # type: ignore[attr-defined]
    assert stream._body["stream"] is True  # type: ignore[attr-defined]


def test_nested_reasoning_wins_over_flat_effort() -> None:
    """A reasoning-capable spec keeps Anthropic thinking, which the translator
    turns into the nested ``reasoning`` object — the flat field must not be
    added alongside it."""
    provider = OpenAICompatibleProvider(
        base_url="http://localhost:11434/v1",
        spec_lookup={"reasoner": FeatureSpec(supports_cache_control=False)},
        reasoning_effort="none",
    )
    captured = _run_create(
        provider,
        model="reasoner",
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
    )
    assert captured["json"]["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in captured["json"]


def test_openrouter_provider_never_sends_flat_effort() -> None:
    """Negative control: the OpenRouter subclass is built without the option,
    so its bodies keep OpenRouter's own format only."""
    from openexecutive.providers.openrouter_provider import OpenRouterProvider

    provider = OpenRouterProvider(api_key="sk-or-v1-test")
    captured = _run_create(provider, model="openai/gpt-5")
    assert "reasoning_effort" not in captured["json"]
