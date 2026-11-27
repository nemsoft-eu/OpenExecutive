"""Home's Needs you rows for connections that stopped working
(``briefing.needs_reconnecting``, GET /today ``waiting_elsewhere``)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from openexecutive.briefing import needs_reconnecting

ME = "olivia@co.example"


@pytest.fixture
def mailbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, object]:
    from openexecutive.delegation import gmail, sign_in_notice

    state: dict[str, object] = {"broken": None}
    monkeypatch.setattr(sign_in_notice, "broken_since", lambda email: state["broken"] if email == ME else None)
    monkeypatch.setattr(gmail, "credential_path", lambda email: tmp_path / f"{email}.json")
    state["file"] = tmp_path / f"{ME}.json"
    return state


def test_a_working_mailbox_has_no_row(mailbox: dict[str, object]) -> None:
    Path(str(mailbox["file"])).write_text("{}")
    assert needs_reconnecting.mailbox_row(ME, "/delegate#as-you") is None


def test_a_broken_mailbox_is_one_row_with_reconnect(mailbox: dict[str, object]) -> None:
    Path(str(mailbox["file"])).write_text("{}")
    mailbox["broken"] = "2026-11-15T20:00:00+00:00"
    row = needs_reconnecting.mailbox_row(ME.upper(), "/delegate#as-you")
    assert row == {
        "key": "reconnect:mailbox",
        "title": "Your mailbox needs reconnecting",
        "detail": needs_reconnecting.MAILBOX_DETAIL,
        "count": 1,
        "href": "/delegate#as-you",
        "action": "Reconnect",
        "at": "2026-11-15T20:00:00+00:00",
    }


def test_no_saved_sign_in_means_nothing_to_reconnect(mailbox: dict[str, object]) -> None:
    mailbox["broken"] = "2026-11-15T20:00:00+00:00"
    assert needs_reconnecting.mailbox_row(ME, "/delegate") is None
    assert needs_reconnecting.mailbox_row(None, "/delegate") is None


def test_the_executives_dead_accounts_each_get_a_row() -> None:
    gateway = SimpleNamespace(needs_reconnect={"google_workspace", "some_other_server"})
    rows = needs_reconnecting.account_rows("/settings/integrations", gateway)
    assert [(r["key"], r["title"], r["action"]) for r in rows] == [
        ("reconnect:google_workspace", "My Google account needs reconnecting", "Reconnect"),
    ]
    assert needs_reconnecting.account_rows("/x", SimpleNamespace(needs_reconnect=set())) == []
