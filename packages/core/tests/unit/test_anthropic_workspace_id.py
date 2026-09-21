"""Organisation-scoped Anthropic keys need a workspace header (#128).

A key issued inside a workspace carries its own scope. A key issued at the
organisation level does not, and Anthropic rejects every call made with one
unless the request sends `anthropic-workspace-id`. Before this setting existed
the failure surfaced as a generic "internal error" in the UI, because the 400
happens after the SSE stream has already returned 200.

The header is set on the client rather than per request: `create` and `stream`
both need it, and every Claude call in the app resolves through this one
provider.
"""
from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-not-used")

import pytest  # noqa: E402

from openexecutive.providers import registry as registry_mod  # noqa: E402
from openexecutive.providers.anthropic_provider import (  # noqa: E402
    AnthropicProvider,
    configured_async_client,
)


@pytest.fixture(autouse=True)
def _reset_singleton() -> Any:
    registry_mod._reset_for_tests()
    yield
    registry_mod._reset_for_tests()


def _headers(provider: AnthropicProvider) -> dict[str, str]:
    """The default headers the SDK client will send on every request."""
    return dict(provider._client.default_headers)


def test_workspace_id_is_sent_as_a_default_header() -> None:
    provider = AnthropicProvider(api_key="sk-test", workspace_id="wrkspc_123")

    assert _headers(provider)["anthropic-workspace-id"] == "wrkspc_123"


def test_no_workspace_id_sends_no_header() -> None:
    """A workspace-scoped key must keep working untouched — sending an empty
    or absent workspace id is not the same as sending none."""
    provider = AnthropicProvider(api_key="sk-test")

    assert "anthropic-workspace-id" not in _headers(provider)


def test_empty_workspace_id_sends_no_header() -> None:
    """`ANTHROPIC_WORKSPACE_ID=` in a .env file arrives as an empty string, not
    None. Forwarding that would turn a working install into a 400."""
    provider = AnthropicProvider(api_key="sk-test", workspace_id="")

    assert "anthropic-workspace-id" not in _headers(provider)


def test_the_header_is_purely_additive() -> None:
    """`default_headers` must not displace what the SDK sets for itself.

    Auth and the API version ride in the same mapping, so a construction that
    replaced it rather than adding to it would break every call — and would
    look identical to a working one from the workspace assertions above.
    """
    plain = AnthropicProvider(api_key="sk-test")
    with_ws = AnthropicProvider(api_key="sk-test", workspace_id="wrkspc_123")

    ours = _headers(with_ws)
    theirs = _headers(plain)

    assert set(ours) - set(theirs) == {"anthropic-workspace-id"}
    assert {k: v for k, v in ours.items() if k != "anthropic-workspace-id"} == theirs, (
        "every header the SDK sets must survive unchanged in value, not just in name"
    )
    assert with_ws._client.auth_headers == plain._client.auth_headers


def test_workspace_id_survives_alongside_a_timeout() -> None:
    provider = AnthropicProvider(api_key="sk-test", timeout=30.0, workspace_id="wrkspc_123")

    assert _headers(provider)["anthropic-workspace-id"] == "wrkspc_123"
    assert provider._client.timeout == 30.0


def test_registry_passes_the_configured_workspace_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards the wiring end to end, not just the constructor.

    `get_settings()` builds a fresh `Settings` on every call, so this goes
    through the environment exactly as a real `.env` does — patching the
    settings object would prove nothing about a deployment.
    """
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_from_env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    provider = registry_mod._anthropic()

    assert _headers(provider)["anthropic-workspace-id"] == "wrkspc_from_env"


def test_registry_sends_no_header_when_the_var_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    provider = registry_mod._anthropic()

    assert "anthropic-workspace-id" not in _headers(provider)


def test_settings_reads_the_var_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The alias has to match what operators put in `.env`."""
    from openexecutive.config import get_settings

    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_alias_check")

    assert get_settings().anthropic_workspace_id == "wrkspc_alias_check"


def test_settings_default_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from openexecutive.config import get_settings

    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)

    assert get_settings().anthropic_workspace_id is None


# ── what a .env can actually hand us ──────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        "# leave blank for a workspace-scoped key",  # dotenv reads an inline
                                                     # comment after `KEY=` as
                                                     # the value (verified)
        "   ",
        "",
    ],
)
def test_a_blank_or_comment_value_reads_as_unset(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure mode this setting must not create.

    Sending a comment string as the workspace id is a legal HTTP header, so it
    reaches Anthropic and 400s every call — handing the exact symptom of #128
    to someone whose key never needed a workspace at all.
    """
    from openexecutive.config import get_settings

    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", raw)

    assert get_settings().anthropic_workspace_id is None


@pytest.mark.parametrize(
    "raw",
    ["wrkspc_1 trailing", "wrkspc\r\nx-injected: 1", '"wrkspc_1"', "wrkspc_é"],
)
def test_an_unusable_value_fails_at_boot(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail where the operator can act on it.

    Each of these is either an illegal header value — which httpx/h11 reject as
    a *connection* error two silent retries into the first Claude call — or a
    quoted/padded copy-paste that Anthropic would simply reject. Neither points
    at the .env line that caused it, so the settings load is the place to stop.
    """
    from openexecutive.config import get_settings

    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", raw)

    with pytest.raises(Exception, match="ANTHROPIC_WORKSPACE_ID"):
        get_settings()


def test_surrounding_whitespace_is_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    from openexecutive.config import get_settings

    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "  wrkspc_123  ")

    assert get_settings().anthropic_workspace_id == "wrkspc_123"


# ── callers outside the request path ──────────────────────────────────────


def test_configured_client_carries_the_workspace_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """The eval runner and its judges build their own client rather than going
    through the registry. Without this factory they would 400 on every judge
    call for an organisation-scoped key, with chat working fine — the same bug
    surviving in the one place nobody looks."""
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_evals")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    client = configured_async_client()

    assert dict(client.default_headers)["anthropic-workspace-id"] == "wrkspc_evals"


def test_configured_client_sends_no_header_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    client = configured_async_client()

    assert "anthropic-workspace-id" not in dict(client.default_headers)


def test_configured_client_keeps_an_explicit_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """`executive_quality_judge` passes its own key through."""
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_evals")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")

    client = configured_async_client(api_key="sk-explicit")

    assert client.api_key == "sk-explicit"
    assert dict(client.default_headers)["anthropic-workspace-id"] == "wrkspc_evals"


def test_provider_does_not_resolve_a_workspace_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider stays explicit. Only the registry decides what it gets, so
    a stray env var cannot change what a directly-constructed provider sends."""
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_ambient")

    provider = AnthropicProvider(api_key="sk-test")

    assert "anthropic-workspace-id" not in _headers(provider)
