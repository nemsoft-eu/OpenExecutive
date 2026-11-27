"""The local half every document sync shares (``knowledge.sync_source``).

Each sync's own tests cover it through that sync; these pin the shared
behaviour itself, on a made-up source, so a change here is seen once.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest

from openexecutive.knowledge import confluence_sync, drive_sync, notion_sync, onedrive_sync
from openexecutive.knowledge.sync_source import SyncSource


class FakeStore:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, dict[str, Any]]] = []

    def delete_documents(self, collection: str, where: dict[str, Any]) -> None:
        self.deleted.append((collection, where))


def _safe(value: str) -> str | None:
    """Lower-case letters and digits only; upper case is the same id spelled
    another way, anything else is unsafe."""
    raw = str(value or "")
    return raw.lower() if re.fullmatch(r"[A-Za-z0-9]{1,20}", raw) else None


@pytest.fixture
def source(tmp_path: Path) -> SyncSource:
    docs = tmp_path / "docs" / "wiki"
    docs.mkdir(parents=True)
    return SyncSource(
        "wiki",
        noun="page",
        id_field="wiki_page_id",
        id_comment=re.compile(r"<!--\s*wiki_page_id:\s*([A-Za-z0-9]{1,20})\s*-->"),
        collections=("wiki_docs", "old_docs"),
        safe_id=_safe,
        index_id=_safe,
        state_path=lambda: tmp_path / "wiki_sync_state.json",
        docs_dir=lambda: docs,
        heartbeat_intent="Wiki sync.",
        lingering="pages stay until reconciliation runs",
    )


def _write(source: SyncSource, doc_id: str, name: str) -> Path:
    path = source.docs_dir() / name
    path.write_text(f"<!-- wiki_page_id: {doc_id} -->\n\n# T\n\nbody\n", encoding="utf-8")
    return path


def test_names_follow_the_key(source: SyncSource) -> None:
    assert source.name == "wiki_sync"
    assert source.state_file == "wiki_sync_state.json"
    assert source.heartbeat_kind == "wiki_sync_scan"
    assert source.records == "pages"


@pytest.mark.parametrize("module", [drive_sync, onedrive_sync, confluence_sync, notion_sync])
def test_each_sync_keeps_its_heartbeat_kind_and_state_file(module: Any) -> None:
    """The scheduler finds a tick by its kind and a reset finds the state by
    its file name: both must stay what they were before the syncs shared
    this."""
    assert module.SOURCE.heartbeat_kind == module.HEARTBEAT_KIND
    assert module.SOURCE.state_file == f"{module.SOURCE.key}_sync_state.json"
    assert module.SOURCE.name == f"{module.SOURCE.key}_sync"


def test_state_survives_a_round_trip_and_a_broken_file(source: SyncSource) -> None:
    assert source.load_state() == {"pages": {}}
    source.save_state({"pages": {"a1": {"filename": "wiki-a1.md"}}, "last_run": "x"})
    assert source.load_state()["pages"]["a1"] == {"filename": "wiki-a1.md"}
    source.state_path().write_text("{not json", encoding="utf-8")
    assert source.load_state() == {"pages": {}}
    source.state_path().write_text('{"pages": []}', encoding="utf-8")
    assert source.load_state() == {"pages": {}}


def test_reset_drops_the_state_file(source: SyncSource, tmp_path: Path) -> None:
    source.save_state({"pages": {}})
    source.reset_local_state()
    assert not source.state_path().exists()
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "wiki_sync_state.json").write_text("{}", encoding="utf-8")
    source.reset_local_state(profile_path=other / "profile.yaml")
    assert not (other / "wiki_sync_state.json").exists()


def test_safe_filename_takes_only_this_sources_own_files(source: SyncSource) -> None:
    assert source.safe_filename("wiki-a1-title.md") == "wiki-a1-title.md"
    assert source.safe_filename("../../wiki-a1.md") == "wiki-a1.md"
    assert source.safe_filename("notes.md") is None
    assert source.safe_filename("wiki-a1.txt") is None
    assert source.safe_filename("") is None


def test_purge_removes_file_chunks_and_record(source: SyncSource) -> None:
    kept = _write(source, "b2", "wiki-b2.md")
    gone = _write(source, "a1", "wiki-a1.md")
    renamed = _write(source, "a1", "wiki-a1-old-title.md")
    state = {"pages": {"a1": {"filename": "wiki-a1.md"}, "b2": {"filename": "wiki-b2.md"}}}
    store = FakeStore()

    assert source.purge("a1", store, state) is True  # type: ignore[arg-type]

    assert not gone.exists() and not renamed.exists() and kept.exists()
    assert state["pages"] == {"b2": {"filename": "wiki-b2.md"}}
    assert store.deleted == [
        ("wiki_docs", {"wiki_page_id": "a1"}),
        ("old_docs", {"wiki_page_id": "a1"}),
    ]


def test_purge_refuses_an_unsafe_id(source: SyncSource) -> None:
    state = {"pages": {"../x": {"filename": "wiki-x.md"}}}
    store = FakeStore()
    assert source.purge("../x", store, state) is False  # type: ignore[arg-type]
    assert store.deleted == []
    assert "../x" in state["pages"]


def test_purge_finds_a_record_under_the_id_as_it_was_spelled(source: SyncSource) -> None:
    path = _write(source, "a1", "wiki-a1.md")
    state = {"pages": {"A1": {"filename": "wiki-a1.md"}}}
    assert source.purge("A1", FakeStore(), state) is True  # type: ignore[arg-type]
    assert state["pages"] == {}
    assert not path.exists()


def test_reconcile_purges_what_is_no_longer_listed_and_orphan_files(source: SyncSource) -> None:
    listed = _write(source, "a1", "wiki-a1.md")
    unlisted = _write(source, "b2", "wiki-b2.md")
    orphan = _write(source, "c3", "wiki-c3.md")
    state = {"pages": {"a1": {"filename": "wiki-a1.md"}, "b2": {"filename": "wiki-b2.md"}}}

    purged = source.reconcile({"a1"}, FakeStore(), state)  # type: ignore[arg-type]

    assert purged == 2
    assert listed.exists() and not unlisted.exists() and not orphan.exists()
    assert set(state["pages"]) == {"a1"}


def test_reconcile_counts_a_record_spelled_differently_once(source: SyncSource) -> None:
    _write(source, "b2", "wiki-b2.md")
    state = {"pages": {"B2": {"filename": "wiki-b2.md"}}}
    assert source.reconcile(set(), FakeStore(), state) == 1  # type: ignore[arg-type]
    assert state["pages"] == {}


def test_reconcile_with_only_leaves_other_records_and_orphans(source: SyncSource) -> None:
    orphan = _write(source, "c3", "wiki-c3.md")
    state = {
        "pages": {
            "a1": {"space": "OPS", "filename": ""},
            "b2": {"space": "HR", "filename": ""},
        }
    }
    purged = source.reconcile(
        set(), FakeStore(), state, only=lambda rec: rec.get("space") == "OPS"  # type: ignore[arg-type]
    )
    assert purged == 1
    assert set(state["pages"]) == {"b2"}
    assert orphan.exists()


def test_purge_all_empties_files_records_and_the_collection(source: SyncSource) -> None:
    _write(source, "a1", "wiki-a1.md")
    junk = source.docs_dir() / "wiki-no-header.md"
    junk.write_text("no id here\n", encoding="utf-8")
    other = source.docs_dir() / "notes.md"
    other.write_text("not ours\n", encoding="utf-8")
    source.save_state({"pages": {"a1": {"filename": "wiki-a1.md"}, "b2": {"filename": ""}}})
    wiped: list[bool] = []

    purged = source.purge_all(FakeStore(), None, lambda: wiped.append(True))  # type: ignore[arg-type]

    assert purged == 2
    assert wiped == [True]
    assert [p.name for p in source.docs_dir().iterdir()] == ["notes.md"]
    assert source.load_state()["pages"] == {}


def test_write_and_index_leaves_nothing_for_empty_text(source: SyncSource) -> None:
    old = _write(source, "a1", "wiki-a1-old.md")
    store = FakeStore()
    chunks = source.write_and_index(
        "a1", "Title", "wiki-a1-title.md", "  \n", store,  # type: ignore[arg-type]
        domain="general", max_chars=1000, metadata={},
    )
    assert chunks == 0
    assert not old.exists()
    assert not (source.docs_dir() / "wiki-a1-title.md").exists()
    assert store.deleted[0] == ("wiki_docs", {"wiki_page_id": "a1"})


def test_write_and_index_writes_the_file_and_indexes_it(
    source: SyncSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_ingest(body: str, store: Any, **kwargs: Any) -> int:
        seen.update(kwargs, body=body)
        return 3

    monkeypatch.setattr("openexecutive.knowledge.sync_source.ingest_text_sync", fake_ingest)
    chunks = source.write_and_index(
        "a1", "Title", "wiki-a1-title.md", "x" * 50, FakeStore(),  # type: ignore[arg-type]
        domain="general", max_chars=40, metadata={"url": "https://example.com/a1"},
    )
    assert chunks == 3
    written = (source.docs_dir() / "wiki-a1-title.md").read_text(encoding="utf-8")
    assert written.startswith("<!-- wiki_page_id: a1 -->\n\n# Title\n\n")
    assert len(seen["body"]) == 40
    assert seen["source_name"] == "wiki/wiki-a1-title.md"
    assert seen["collection"] == "wiki_docs"
    assert seen["extra_metadata"] == {
        "wiki_page_id": "a1",
        "type": "wiki",
        "url": "https://example.com/a1",
    }
    assert source.build_file_index() == {"a1": [source.docs_dir() / "wiki-a1-title.md"]}


def test_skips_count_up_until_a_reconcile_resets_them(source: SyncSource) -> None:
    state: dict[str, Any] = {}
    source.note_reconcile_skip(state, "listing incomplete")
    source.note_reconcile_skip(state, "listing incomplete")
    assert state["reconcile_skips"] == 2


def test_partial_failure_names_the_documents(source: SyncSource) -> None:
    assert source.partial_failure(1).startswith("1 page could not be synced.")
    assert source.partial_failure(2).startswith("2 pages could not be synced.")


async def test_run_skips_when_a_tick_is_already_running(source: SyncSource) -> None:
    lock = asyncio.Lock()
    calls: list[int] = []

    async def tick() -> dict[str, int]:
        calls.append(1)
        return {"seen": 1}

    async with lock:
        assert await source.run(lock, tick) == {"busy": 1}
    assert calls == []
    assert source.last_finished_at is None
    assert await source.run(lock, tick) == {"seen": 1}
    assert source.last_finished_at is not None


async def test_run_records_why_a_tick_failed_and_still_marks_it_finished(
    source: SyncSource,
) -> None:
    async def tick() -> dict[str, int]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await source.run(asyncio.Lock(), tick)
    assert source.load_state()["last_error"].startswith("The last sync failed unexpectedly.")
    assert source.last_finished_at is not None
