"""The progress_updates table (integrations/progress_store.py)."""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from openexecutive.integrations import progress_store as store
from openexecutive.integrations.progress_guard import CleanUpdate
from openexecutive.memory import episodic

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
CAPS = {"max_per_day": 30, "relay_max_per_hour": 2, "relay_max_per_day": 8}

BASE = CleanUpdate(
    topic="GTM plan",
    topic_key="gtm plan",
    status="on_track",
    summary="Positioning draft done.",
    ask=None,
    link=None,
    idempotency_key=None,
)


@pytest.fixture(autouse=True)
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "episodic.db"
    monkeypatch.setattr(episodic, "DB_PATH", path)
    return path


def update(n: int, **changes: object) -> CleanUpdate:
    return replace(BASE, summary=f"Positioning draft {n} done.", **changes)  # type: ignore[arg-type]


def admit(u: CleanUpdate, *, at: datetime = NOW, source: str = "marketing-planner", **caps: int) -> store.Admission:
    return store.admit(source, u, now=at, **{**CAPS, **caps})


def test_a_new_update_is_stored_and_claims_a_relay(db: Path) -> None:
    got = admit(BASE)
    assert got == store.Admission(got.id, "stored", True)
    rows = sqlite3.connect(db).execute(
        "SELECT source, topic_key, relay_claimed, relay_state, content_hash FROM progress_updates"
    ).fetchall()
    assert rows == [("marketing-planner", "gtm plan", 1, "pending", store.content_hash(BASE))]


def test_duplicate_by_idempotency_key_returns_the_first_id() -> None:
    first = admit(update(1, idempotency_key="key-00000001"))
    # different content, same key: still the first report, even days later
    again = admit(update(2, idempotency_key="key-00000001"), at=NOW + timedelta(days=3))
    assert again == store.Admission(first.id, "duplicate", False)
    # the same key from another source is that source's own
    assert admit(update(2, idempotency_key="key-00000001"), source="ops-bot").outcome == "stored"


def test_duplicate_by_content_within_a_day() -> None:
    first = admit(BASE)
    assert admit(BASE, at=NOW + timedelta(hours=23)) == store.Admission(first.id, "duplicate", False)
    # a new key does not make the same content new
    assert admit(replace(BASE, idempotency_key="key-00000002"), at=NOW + timedelta(hours=1)).outcome == "duplicate"
    # after the window it is a fresh report
    later = admit(BASE, at=NOW + timedelta(hours=25))
    assert later.outcome == "stored" and later.id != first.id


def test_the_row_past_the_daily_cap_is_refused(db: Path) -> None:
    for n in range(3):
        assert admit(update(n), max_per_day=3).outcome == "stored"
    assert admit(update(3), max_per_day=3) == store.Admission(None, "over_limit", False)
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM progress_updates").fetchone() == (3,)
    # another source has its own cap, and the window rolls
    assert admit(update(3), max_per_day=3, source="ops-bot").outcome == "stored"
    assert admit(update(3), max_per_day=3, at=NOW + timedelta(days=1, minutes=1)).outcome == "stored"


def test_no_relay_is_claimed_past_the_relay_caps() -> None:
    claims = [admit(update(n), at=NOW + timedelta(minutes=n)).relay_claimed for n in range(4)]
    assert claims == [True, True, False, False]
    # an hour later the hourly cap has room again
    assert admit(update(10), at=NOW + timedelta(minutes=61)).relay_claimed is True
    # the daily cap holds across hours
    claims = [
        admit(update(20 + n), at=NOW + timedelta(hours=2 + n), relay_max_per_day=4).relay_claimed
        for n in range(3)
    ]
    assert claims == [True, False, False]


def test_a_concurrent_duplicate_key_returns_the_first_id(db: Path) -> None:
    store._connect(None).close()
    barrier = threading.Barrier(8)
    results: list[store.Admission] = []
    lock = threading.Lock()

    def submit(n: int) -> None:
        barrier.wait()
        got = admit(update(n, idempotency_key="retry-00000001"))
        with lock:
            results.append(got)

    threads = [threading.Thread(target=submit, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stored = [r for r in results if r.outcome == "stored"]
    assert len(stored) == 1
    assert {r.id for r in results} == {stored[0].id}
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM progress_updates").fetchone() == (1,)


def test_concurrent_submits_cannot_pass_a_cap_together(db: Path) -> None:
    store._connect(None).close()
    barrier = threading.Barrier(10)
    results: list[store.Admission] = []
    lock = threading.Lock()

    def submit(n: int) -> None:
        barrier.wait()
        got = admit(update(n), max_per_day=4)
        with lock:
            results.append(got)

    threads = [threading.Thread(target=submit, args=(n,)) for n in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(r.outcome for r in results).count("stored") == 4
    assert sum(r.relay_claimed for r in results) == 2


def test_a_key_conflict_at_insert_rereads_the_first_row(monkeypatch: pytest.MonkeyPatch) -> None:
    first = admit(update(1, idempotency_key="race-00000001"))
    # the lookup misses (as if the other writer had not committed yet) and
    # the insert hits the unique index
    monkeypatch.setattr(store, "_first_duplicate", lambda *a, **k: None)
    assert admit(update(2, idempotency_key="race-00000001")) == store.Admission(first.id, "duplicate", False)


def test_store_unavailable_on_a_broken_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    broken = tmp_path / "a-directory"
    broken.mkdir()
    monkeypatch.setattr(episodic, "DB_PATH", broken)
    with pytest.raises(store.StoreUnavailable):
        admit(BASE)
    # reads and the relay state never raise
    assert store.latest_per_topic() == []
    store.set_relay_state(1, "sent")


def test_store_unavailable_on_a_corrupt_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"not a database at all, just bytes" * 100)
    monkeypatch.setattr(episodic, "DB_PATH", corrupt)
    with pytest.raises(store.StoreUnavailable):
        admit(BASE)


def test_old_rows_are_pruned_by_age_only(db: Path) -> None:
    old = admit(update(0), at=NOW - timedelta(days=91))
    young = [admit(update(n), at=NOW - timedelta(days=89, minutes=n), max_per_day=1000) for n in range(1, 60)]
    admit(update(100), max_per_day=1000)
    ids = {r[0] for r in sqlite3.connect(db).execute("SELECT id FROM progress_updates")}
    assert old.id not in ids
    assert {y.id for y in young} <= ids
    assert len(ids) == 60


def test_set_relay_state(db: Path) -> None:
    got = admit(BASE)
    assert got.id is not None
    store.set_relay_state(got.id, "queued", now=NOW)
    row = sqlite3.connect(db).execute("SELECT relay_state, relayed_at FROM progress_updates").fetchone()
    assert row == ("queued", NOW.isoformat())
    with pytest.raises(ValueError):
        store.set_relay_state(got.id, "held")


def test_still_pending_matches_id_source_and_content(db: Path) -> None:
    got = admit(BASE)
    assert got.id is not None
    digest = store.content_hash(BASE)
    assert store.still_pending(got.id, "marketing-planner", digest) is True
    # the same id in another company's table: a different source or content
    assert store.still_pending(got.id, "ops-bot", digest) is False
    assert store.still_pending(got.id, "marketing-planner", store.content_hash(update(2))) is False
    assert store.still_pending(got.id + 1, "marketing-planner", digest) is False
    # relayed already
    store.set_relay_state(got.id, "sent", now=NOW)
    assert store.still_pending(got.id, "marketing-planner", digest) is False


def test_still_pending_is_false_on_a_broken_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    broken = tmp_path / "a-directory"
    broken.mkdir()
    monkeypatch.setattr(episodic, "DB_PATH", broken)
    assert store.still_pending(1, "marketing-planner", store.content_hash(BASE)) is False


def test_latest_per_topic() -> None:
    admit(update(1), at=NOW - timedelta(days=40))  # too old
    admit(replace(BASE, topic="Launch event", topic_key="launch event"), at=NOW - timedelta(days=2))
    admit(update(2), at=NOW - timedelta(days=1))
    newest = admit(update(3, status="blocked", ask="Approve the venue"), at=NOW - timedelta(hours=1))
    admit(update(4), at=NOW - timedelta(hours=2), source="ops-bot")
    got = store.latest_per_topic(now=NOW)
    assert [(u.source, u.topic, u.summary) for u in got] == [
        ("marketing-planner", "GTM plan", "Positioning draft 3 done."),
        ("ops-bot", "GTM plan", "Positioning draft 4 done."),
        ("marketing-planner", "Launch event", "Positioning draft done."),
    ]
    assert got[0].id == newest.id and got[0].status == "blocked" and got[0].ask == "Approve the venue"
    assert len(store.latest_per_topic(limit=2, now=NOW)) == 2
    assert [u.topic for u in store.latest_per_topic(max_age_days=1, now=NOW)] == ["GTM plan", "GTM plan"]
