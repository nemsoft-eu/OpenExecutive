"""The person hears once when their own mailbox stops taking its sign-in
(delegation/sign_in_notice.py)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openexecutive.delegation import sign_in_notice

OWNER = "olivia@co.example"


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, str]]:
    from openexecutive import audit
    from openexecutive.people import store as people_store
    from openexecutive.scheduler import runner

    out: list[tuple[Any, str]] = []
    person = SimpleNamespace(id=7, archived=False, is_principal=True)
    monkeypatch.setattr(people_store, "find_person_by_email", lambda email: person if email == OWNER else None)

    async def deliver(who: Any, text: str, *, label: str = "") -> Any:
        out.append((who, text))
        return SimpleNamespace(ok=True, reason="", channel="email")

    monkeypatch.setattr(runner, "deliver_to_person", deliver)
    monkeypatch.setattr(audit, "log_event", lambda *a, **k: None)
    return out


async def test_a_break_is_told_once_and_again_after_a_reconnect(tmp_path: Path, sent: list[tuple[Any, str]]) -> None:
    db = tmp_path / "e.db"
    assert await sign_in_notice.note(OWNER, "needs_reconnect", db_path=db)
    assert len(sent) == 1 and "Delegate → As you" in sent[0][1]
    # Every later check during the same break tells nothing.
    assert not await sign_in_notice.note(OWNER, "needs_reconnect", db_path=db)
    assert not await sign_in_notice.note(OWNER, "error", db_path=db)
    # Connected again, then broken again: told again.
    assert not await sign_in_notice.note(OWNER, "connected", db_path=db)
    assert await sign_in_notice.note(OWNER, "needs_reconnect", db_path=db)
    assert len(sent) == 2


async def test_a_notice_that_cant_go_is_tried_on_the_next_check(
    tmp_path: Path, sent: list[tuple[Any, str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.scheduler import runner

    db = tmp_path / "e.db"

    async def nowhere(who: Any, text: str, *, label: str = "") -> Any:
        return SimpleNamespace(ok=False, reason="no channel", channel="")

    monkeypatch.setattr(runner, "deliver_to_person", nowhere)
    assert await sign_in_notice.note(OWNER, "needs_reconnect", db_path=db)
    # Unclaimed, so the next check tries again.
    assert await sign_in_notice.note(OWNER, "needs_reconnect", db_path=db)


async def test_someone_not_in_people_is_told_nothing(tmp_path: Path, sent: list[tuple[Any, str]]) -> None:
    await sign_in_notice.note("stranger@x.example", "needs_reconnect", db_path=tmp_path / "e.db")
    assert sent == []


async def test_the_break_is_readable_until_it_is_settled(tmp_path: Path, sent: list[tuple[Any, str]]) -> None:
    db = tmp_path / "e.db"
    assert sign_in_notice.broken_since(OWNER, db_path=db) is None
    await sign_in_notice.note(OWNER, "needs_reconnect", db_path=db)
    assert sign_in_notice.broken_since(OWNER, db_path=db) is not None
    # A passing failure keeps it; disconnecting on purpose ends it.
    await sign_in_notice.note(OWNER, "error", db_path=db)
    assert sign_in_notice.broken_since(OWNER, db_path=db) is not None
    await sign_in_notice.note(OWNER, "not_configured", db_path=db)
    assert sign_in_notice.broken_since(OWNER, db_path=db) is None


async def test_a_sign_in_that_opens_another_mailbox_is_a_break_too(
    tmp_path: Path, sent: list[tuple[Any, str]],
) -> None:
    # The web and the phone show "mismatch" as Needs reconnecting, so Home
    # and the one notice do too.
    db = tmp_path / "e.db"
    assert await sign_in_notice.note(OWNER, "mismatch", db_path=db)
    assert len(sent) == 1
    assert sign_in_notice.broken_since(OWNER, db_path=db) is not None
    await sign_in_notice.note(OWNER, "connected", db_path=db)
    assert sign_in_notice.broken_since(OWNER, db_path=db) is None
