"""Shared setup for the web-search tests (SearXNG client tool and its loops).

One place for the env keys the search config reads and for routing a slug to
the local backend, so the modules that exercise search cannot drift apart on
which keys they clear.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

SEARCH_ENV_KEYS = (
    "ENABLE_WEB_SEARCH",
    "WEB_SEARCH_MAX_USES",
    "WEB_SEARCH_ALLOWED_DOMAINS",
    "WEB_SEARCH_BLOCKED_DOMAINS",
    "SEARXNG_URL",
    "SEARXNG_TIMEOUT_S",
    "SEARXNG_MAX_RESULTS",
    "LOCAL_MODELS",
    "LOCAL_MODELS_ENABLED",
    "LOCAL_BASE_URL",
    "OPENROUTER_ENABLED",
    "OPENROUTER_API_KEY",
)


def clear_search_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from search on, nothing else configured."""
    for key in SEARCH_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "true")


def use_local_model(
    monkeypatch: pytest.MonkeyPatch,
    slug: str = "qwen-local",
    *,
    searxng: bool = True,
) -> None:
    """Route ``slug`` to the local backend, with or without SearXNG.

    LOCAL_BASE_URL is mandatory once local routing is on, so it is always set.
    """
    monkeypatch.setenv("LOCAL_MODELS_ENABLED", "true")
    monkeypatch.setenv("LOCAL_MODELS", slug)
    monkeypatch.setenv("LOCAL_BASE_URL", "http://localhost:11434/v1")
    if searxng:
        monkeypatch.setenv("SEARXNG_URL", "http://searxng:8080")
    else:
        monkeypatch.delenv("SEARXNG_URL", raising=False)


def tool_use_block(name: str, block_id: str = "tu_1", **inp: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=inp)


def msg(*blocks: Any, stop: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), stop_reason=stop)


def stream_cm(final_msg: Any) -> MagicMock:
    """Async context manager standing in for a provider's messages_stream()."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=cm)
    cm.__aexit__ = AsyncMock(return_value=None)

    async def _aiter() -> Any:
        if False:  # pragma: no cover
            yield None

    cm.__aiter__ = lambda self=cm: _aiter()
    cm.get_final_message = AsyncMock(return_value=final_msg)
    return cm
