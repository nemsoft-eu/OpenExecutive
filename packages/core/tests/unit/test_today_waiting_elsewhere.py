"""Home's Needs you rows for what waits on other pages (GET /today
``waiting_elsewhere``): the Watch list's suggestions, so Home never calls
the day quiet while they wait."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openexecutive.api.routes import today as today_route


def _suggestion(target: str, created_at: str, routed_to: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(target=target, created_at=created_at, route_to_person_id=routed_to)


def _stub(monkeypatch: pytest.MonkeyPatch, items: list[SimpleNamespace] | Exception) -> None:
    def pending(*a: object, **k: object) -> list[SimpleNamespace]:
        if isinstance(items, Exception):
            raise items
        return items

    monkeypatch.setattr("openexecutive.monitoring.store.list_pending_suggestions", pending)


def test_no_suggestions_means_no_row(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, [])
    assert today_route._waiting_elsewhere() == []


def test_suggestions_make_one_row_naming_the_first_two(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(
        monkeypatch,
        [
            _suggestion("https://www.example.com/legal", "2026-01-02T10:00:00+00:00"),
            _suggestion("https://news.example.org/", "2026-01-04T10:00:00+00:00"),
            _suggestion("INITECH", "2026-01-03T10:00:00+00:00"),
            _suggestion("acme pricing", "2026-01-01T10:00:00+00:00"),
        ],
    )
    [row] = today_route._waiting_elsewhere()
    assert row.key == "watch-suggestions"
    assert row.title == "4 things I'd like to watch"
    assert row.detail == "example.com, news.example.org and 2 more"
    assert row.count == 4
    assert row.href == "/watchlist"
    assert row.action == "Look"
    assert row.at == "2026-01-04T10:00:00+00:00"


def test_one_suggestion_reads_singular(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, [_suggestion("INITECH", "2026-01-02T10:00:00+00:00")])
    [row] = today_route._waiting_elsewhere()
    assert row.title == "1 thing I'd like to watch"
    assert row.detail == "INITECH"


def test_two_suggestions_are_both_named(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, [_suggestion("A", "x"), _suggestion("B", "y")])
    [row] = today_route._waiting_elsewhere()
    assert row.detail == "A and B"


def test_a_store_failure_leaves_home_working(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, RuntimeError("db gone"))
    assert today_route._waiting_elsewhere() == []


def test_a_malformed_url_target_reads_as_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, [_suggestion("https://[news] acme pricing", "x")])
    [row] = today_route._waiting_elsewhere()
    assert row.detail == "https://[news] acme pricing"


def test_repeated_hosts_are_named_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, [_suggestion(f"https://acme.com/{p}", "x") for p in "abc"])
    [row] = today_route._waiting_elsewhere()
    assert row.title == "3 things I'd like to watch"
    assert row.detail == "acme.com"
    _stub(monkeypatch, [_suggestion(t, "x") for t in ("https://a.com/1", "https://a.com/2", "B", "C")])
    [row] = today_route._waiting_elsewhere()
    assert row.detail == "a.com, B and 1 more"


def test_suggestions_routed_to_a_head_are_theirs(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, [_suggestion("A", "x", routed_to=7), _suggestion("B", "y")])
    [row] = today_route._waiting_elsewhere()
    assert (row.count, row.detail) == (1, "B")
    _stub(monkeypatch, [_suggestion("A", "x", routed_to=7)])
    assert today_route._waiting_elsewhere() == []




def test_broken_connections_are_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from openexecutive.briefing import needs_reconnecting

    seen: dict[str, object] = {}

    def mailbox_row(email: str | None, href: str) -> dict[str, object] | None:
        seen["mailbox"] = (email, href)
        return {"key": "reconnect:mailbox", "title": "t", "count": 1, "href": href, "action": "Reconnect"}

    monkeypatch.setattr(needs_reconnecting, "mailbox_row", mailbox_row)
    monkeypatch.setattr(
        needs_reconnecting, "account_rows",
        lambda href: [{"key": "reconnect:google_workspace", "title": "g", "count": 1, "href": href, "action": "Reconnect"}],
    )
    monkeypatch.setattr(
        "openexecutive.people.store.get_person", lambda pid: SimpleNamespace(email="me@co.example"),
    )
    rows = today_route._needs_reconnecting(4, accounts=True)
    assert [r.key for r in rows] == ["reconnect:mailbox", "reconnect:google_workspace"]
    assert seen["mailbox"] == ("me@co.example", "/delegate#as-you")
    assert rows[1].href == "/settings/status"
    # A teammate sees their own mailbox, never the Executive's accounts.
    assert [r.key for r in today_route._needs_reconnecting(4, accounts=False)] == ["reconnect:mailbox"]


def test_long_text_is_cut_at_a_word() -> None:
    cut = today_route._cut_at_word("Check the Google Play review status for Open Executive", 30)
    assert cut == "Check the Google Play review…"
    assert today_route._cut_at_word("  short   one ", 30) == "short one"


def test_bottom_line_counts_only_what_needs_action(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime, timedelta

    past = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    future = (datetime.now(UTC) + timedelta(days=2)).isoformat()
    monkeypatch.setattr(
        "openexecutive.attunement.open_loops.list_open_loops",
        lambda person_id=None, limit=50: [SimpleNamespace(due_at=past), SimpleNamespace(due_at=future)],
    )
    dept = SimpleNamespace(at_risk_count=0, off_track_count=0, unchecked_count=4)
    people = [
        SimpleNamespace(id=1, status="needs_reply"),  # the reader
        SimpleNamespace(id=2, status="awaiting"),
        SimpleNamespace(id=3, status="clear"),
    ]
    response = SimpleNamespace(departments=[dept], people=people)
    counts = today_route._to_act_on(response, 1)  # type: ignore[arg-type]
    assert [(c.text, c.href) for c in counts] == [
        ("1 task overdue", "/tasks"),
        ("4 goals never checked", "/goals"),
        ("1 person to follow up", "/people"),
    ]
    quiet = SimpleNamespace(departments=[SimpleNamespace(at_risk_count=0, off_track_count=0, unchecked_count=0)], people=[])
    monkeypatch.setattr("openexecutive.attunement.open_loops.list_open_loops", lambda person_id=None, limit=50: [])
    assert today_route._to_act_on(quiet, 1) == []  # type: ignore[arg-type]
