"""Source tokens for progress updates (integrations/progress_sources.py)."""
from __future__ import annotations

import hashlib
import hmac

import pytest

from openexecutive.integrations import progress_sources as ps

TOKEN = "s3cr3t-token-for-the-marketing-planner-0123456789"
DIGEST = hashlib.sha256(TOKEN.encode()).hexdigest()


@pytest.fixture(autouse=True)
def no_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ps.ENV, raising=False)
    # forget the logged malformed value so test order can't hide a log line
    monkeypatch.setattr(ps, "_logged_bad_raw", None)


def test_parse_reads_each_entry() -> None:
    other = "b" * 64
    assert ps.parse_sources(f" marketing-planner:{DIGEST} , ops-bot:{other.upper()} ") == {
        "marketing-planner": DIGEST,
        "ops-bot": other,
    }
    assert ps.parse_sources("") == {}
    assert ps.parse_sources("   ") == {}


@pytest.mark.parametrize(
    "raw",
    [
        "marketing-planner",  # no hash
        f"Marketing:{DIGEST}",  # upper-case id
        f"-planner:{DIGEST}",  # leading hyphen
        f"m:{DIGEST}",  # too short
        "marketing-planner:abc123",  # not a sha256
        f"marketing planner:{DIGEST}",  # space in the id
        " , ",  # set but lists nothing
    ],
)
def test_parse_refuses_a_malformed_value(raw: str) -> None:
    with pytest.raises(ps.SourcesError):
        ps.parse_sources(raw)


def test_parse_refuses_a_repeated_id() -> None:
    with pytest.raises(ps.SourcesError, match="listed twice"):
        ps.parse_sources(f"marketing-planner:{DIGEST},marketing-planner:{'c' * 64}")


def test_a_token_resolves_to_its_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ps.ENV, f"ops-bot:{'c' * 64},marketing-planner:{DIGEST}")
    assert ps.enabled() is True
    assert ps.source_for(TOKEN) == "marketing-planner"


@pytest.mark.parametrize("token", ["", None, "not-the-token", TOKEN + "x", "x" * 10_000])
def test_an_unknown_token_resolves_to_nothing(monkeypatch: pytest.MonkeyPatch, token: str | None) -> None:
    monkeypatch.setenv(ps.ENV, f"marketing-planner:{DIGEST}")
    assert ps.source_for(token) is None


def test_off_with_no_sources() -> None:
    assert ps.enabled() is False
    assert ps.source_for(TOKEN) is None


def test_a_malformed_setting_fails_closed(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    # attach caplog to the module logger directly: the app's logging config
    # (run by other tests on the same worker) sets propagate = False on the
    # openexecutive tree, so the root-level capture would see nothing
    ps.logger.addHandler(caplog.handler)
    try:
        monkeypatch.setenv(ps.ENV, f"marketing-planner:{DIGEST},broken")
        assert ps.enabled() is False
        assert ps.source_for(TOKEN) is None
    finally:
        ps.logger.removeHandler(caplog.handler)
    assert "PROGRESS_SOURCES can't be used" in caplog.text


def test_a_malformed_setting_is_logged_once_per_value(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # the route and the MCP tool call this before any auth, so a repeated
    # request must not repeat the log line. propagate is off as the app's
    # logging config sets it, so the root capture can't count a record twice
    monkeypatch.setattr(ps.logger, "propagate", False)
    ps.logger.addHandler(caplog.handler)
    try:
        monkeypatch.setenv(ps.ENV, f"marketing-planner:{DIGEST},broken")
        for _ in range(3):
            assert ps.enabled() is False
            assert ps.source_for(TOKEN) is None
        assert len(caplog.records) == 1
        assert DIGEST not in caplog.text

        monkeypatch.setenv(ps.ENV, "still-broken")
        assert ps.enabled() is False
        assert ps.enabled() is False
        assert len(caplog.records) == 2

        monkeypatch.setenv(ps.ENV, f"marketing-planner:{DIGEST}")
        assert ps.enabled() is True
        assert ps.source_for(TOKEN) == "marketing-planner"
        assert len(caplog.records) == 2
    finally:
        ps.logger.removeHandler(caplog.handler)


def test_every_entry_is_compared_in_constant_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ps.ENV, f"marketing-planner:{DIGEST},ops-bot:{'c' * 64},web:{'d' * 64}")
    calls: list[tuple[str, str]] = []
    real = hmac.compare_digest

    def spy(a: str, b: str) -> bool:
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(ps.hmac, "compare_digest", spy)
    # the match is the first entry, and the rest are still compared
    assert ps.source_for(TOKEN) == "marketing-planner"
    assert len(calls) == 3
    assert all(a == DIGEST for a, _ in calls)


def test_make_token_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    token, entry = ps.make_token("marketing-planner")
    assert entry.startswith("marketing-planner:")
    assert token not in entry
    assert len(token) >= 43
    monkeypatch.setenv(ps.ENV, entry)
    assert ps.source_for(token) == "marketing-planner"
    with pytest.raises(ps.SourcesError):
        ps.make_token("Not Valid")
