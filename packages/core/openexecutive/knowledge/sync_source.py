"""What every document sync keeps on this machine, and how it is kept.

Drive, OneDrive, Confluence and Notion each read a different service, but
what they do locally is one thing: a state file of what is on record, one
Markdown file per document under ``docs/<source>/`` whose first line names the
document it came from, the document's chunks in the source's own collection,
a purge when the document is no longer shared, one tick at a time, and a
heartbeat that chains the next tick. Each sync used to carry its own copy of
all of it, so a fix to how a revoked document is purged had to be made four
times.

``SyncSource`` is that local half, once. A sync module keeps what only it
knows: how to list and read its service, what makes a document current, and
what it records about one.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from openexecutive.knowledge.loader import ingest_text_sync
from openexecutive.knowledge.store import ChromaDBStore

HEARTBEAT_CHANNEL = "__internal__"


class SyncSource:
    """One synced source's local files, records, chunks and heartbeat."""

    def __init__(
        self,
        key: str,
        *,
        noun: str,
        id_field: str,
        id_comment: re.Pattern[str],
        collections: tuple[str, ...],
        safe_id: Callable[[str], str | None],
        state_path: Callable[[], Path],
        docs_dir: Callable[[], Path],
        heartbeat_intent: str,
        lingering: str,
        index_id: Callable[[str], str | None] | None = None,
        head_chars: int = 2000,
    ) -> None:
        # "drive", "onedrive", "confluence", "notion": the folder under
        # docs/, the file prefix, the chunk ``type`` and the log prefix.
        self.key = key
        self.name = f"{key}_sync"
        self.state_file = f"{key}_sync_state.json"
        self.heartbeat_kind = f"{key}_sync_scan"
        self.heartbeat_intent = heartbeat_intent
        # "file" or "page": what one document is called, and (plural) the
        # state key its records live under.
        self.noun = noun
        self.records = f"{noun}s"
        # The chunk metadata field, and the comment on a file's first line,
        # that name the document.
        self.id_field = id_field
        self.id_comment = id_comment
        # Where the chunks are; the first is where new ones go, the rest are
        # places an older version left some.
        self.collections = collections
        # The id as it is recorded, or None for one that must never reach a
        # path or a query.
        self.safe_id = safe_id
        self.index_id = index_id or (lambda found: found)
        self.head_chars = head_chars
        # Called on every use, so a test (or a company switch) that moves the
        # company folder is honoured.
        self.state_path = state_path
        self.docs_dir = docs_dir
        # What a skipped reconciliation leaves behind, for the warning.
        self.lingering = lingering
        # When the last tick in this process ended, failed or not.
        # ``last_run`` in the state is stamped only on success and with the
        # tick's start, so a cooldown keyed on it alone lets a failing or
        # slow source be re-run back to back.
        self.last_finished_at: datetime | None = None
        self.logger = logging.getLogger(f"openexecutive.knowledge.{self.name}")

    # -- state file ---------------------------------------------------------

    def load_state(self) -> dict[str, Any]:
        path = self.state_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        if not isinstance(data.get(self.records), dict):
            data[self.records] = {}
        return data

    def save_state(self, state: dict[str, Any]) -> None:
        path = self.state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)

    def reset_local_state(self, *, profile_path: Path | None = None) -> None:
        """Drop the records so the next tick re-ingests. Call it whenever the
        source's collection is wiped (fixture load/reset, client-slot
        rebuild), or leftover records skip every unchanged document as
        already current."""
        path = (
            Path(profile_path).parent / self.state_file
            if profile_path is not None
            else self.state_path()
        )
        path.unlink(missing_ok=True)

    def record(self, state: dict[str, Any], doc_id: str) -> dict[str, Any]:
        records = state.get(self.records)
        raw = records.get(doc_id) if isinstance(records, dict) else None
        return raw if isinstance(raw, dict) else {}

    def _records(self, state: dict[str, Any]) -> dict[str, Any]:
        records = state.setdefault(self.records, {})
        if not isinstance(records, dict):
            records = state[self.records] = {}
        return records

    def record_error(self, message: str) -> None:
        """Keep a short, user-facing reason the last tick failed (never a trace)."""
        try:
            state = self.load_state()
            state["last_error"] = message
            self.save_state(state)
        except Exception:
            self.logger.exception("%s: could not record the last error", self.name)

    def partial_failure(self, failed: int) -> str:
        return (
            f"{failed} {self.noun}{'' if failed == 1 else 's'} could not be synced. "
            "Check the server logs."
        )

    # -- files on disk ------------------------------------------------------

    def safe_filename(self, name: str) -> str | None:
        """``name`` when it is one of this source's own files, by its bare
        name: a recorded filename never reaches a path as it was stored."""
        candidate = Path(str(name)).name
        if candidate.startswith(f"{self.key}-") and candidate.endswith(".md"):
            return candidate
        return None

    def build_file_index(self) -> dict[str, list[Path]]:
        """Each synced document's id → its file(s) on disk, in one directory
        pass. Reading every file's head is the cost of finding a file whose
        record is gone; building it once keeps a purge of many documents from
        rescanning the folder for each."""
        index: dict[str, list[Path]] = {}
        for path in self.docs_dir().glob(f"{self.key}-*.md"):
            try:
                head = path.read_text(encoding="utf-8", errors="replace")[: self.head_chars]
            except OSError:
                continue
            found = self.id_comment.search(head)
            doc_id = self.index_id(found.group(1)) if found else None
            if doc_id:
                index.setdefault(doc_id, []).append(path)
        return index

    def write_and_index(
        self,
        doc_id: str,
        title: str,
        filename: str,
        text: str,
        store: ChromaDBStore,
        *,
        domain: str,
        max_chars: int,
        metadata: dict[str, Any],
    ) -> int:
        """Replace one document's file and chunks with ``text``. Empty text
        leaves neither (a document with nothing readable in it). Returns the
        chunk count."""
        for collection in self.collections:
            store.delete_documents(collection, {self.id_field: doc_id})
        for path in self.build_file_index().get(doc_id, []):
            path.unlink(missing_ok=True)
        if not text.strip():
            return 0
        header = f"<!-- {self.id_field}: {doc_id} -->\n\n# {title}\n\n"
        body = (header + text).strip()[:max_chars]
        (self.docs_dir() / filename).write_text(body + "\n", encoding="utf-8")
        return ingest_text_sync(
            body,
            store,
            source_name=f"{self.key}/{filename}",
            domain=domain,
            collection=self.collections[0],
            extra_metadata={self.id_field: doc_id, "type": self.key, **metadata},
        )

    # -- purging ------------------------------------------------------------

    def purge(
        self,
        doc_id: str,
        store: ChromaDBStore,
        state: dict[str, Any] | None = None,
        file_index: dict[str, list[Path]] | None = None,
    ) -> bool:
        """Remove one synced document's file, chunks and record. False for an
        id that isn't safe to act on."""
        safe = self.safe_id(doc_id)
        if not safe:
            self.logger.warning(
                "%s: refuse to purge unsafe %s id %r", self.name, self.noun, doc_id
            )
            return False
        records = self._records(state) if state is not None else {}
        # A record written under the id as the service spelled it is still
        # this document's.
        meta = records.get(safe)
        if meta is None:
            meta = records.get(doc_id)
        recorded = meta.get("filename") if isinstance(meta, dict) else None
        filename = self.safe_filename(str(recorded or ""))
        for collection in self.collections:
            store.delete_documents(collection, {self.id_field: safe})
        if filename:
            (self.docs_dir() / filename).unlink(missing_ok=True)
        index = file_index if file_index is not None else self.build_file_index()
        for path in index.get(safe, []):
            path.unlink(missing_ok=True)
        records.pop(safe, None)
        records.pop(doc_id, None)
        self.logger.info("%s: purged %s %s", self.name, self.noun, safe)
        return True

    def reconcile(
        self,
        visible_ids: set[str],
        store: ChromaDBStore,
        state: dict[str, Any],
        *,
        only: Callable[[Any], bool] | None = None,
    ) -> int:
        """Purge what is on record but no longer listed, and any file on disk
        whose record is gone. With ``only``, just the records it accepts are
        considered (a part of the listing that completed), and files without
        a record are left for a full reconcile. Blocking file and Chroma
        work: run it via ``asyncio.to_thread`` on the event loop."""
        file_index = self.build_file_index()
        records = self._records(state)
        gone = {
            doc_id
            for doc_id, rec in records.items()
            if (self.safe_id(str(doc_id)) or doc_id) not in visible_ids
            and (only is None or only(rec))
        }
        if only is None:
            known = {self.safe_id(str(doc_id)) or doc_id for doc_id in records}
            gone |= {f for f in file_index if f not in visible_ids and f not in known}
        return sum(
            self.purge(str(doc_id), store, state, file_index=file_index)
            for doc_id in sorted(gone, key=str)
        )

    def purge_all(
        self, store: ChromaDBStore, state: dict[str, Any] | None, wipe: Callable[[], Any]
    ) -> int:
        """Remove every synced document (files, chunks, records). ``wipe``
        empties the source's collection."""
        current = state if state is not None else self.load_state()
        file_index = self.build_file_index()
        purged = sum(
            self.purge(str(doc_id), store, current, file_index=file_index)
            for doc_id in set(current[self.records]) | set(file_index)
        )
        for path in self.docs_dir().glob(f"{self.key}-*.md"):
            path.unlink(missing_ok=True)
        wipe()
        current[self.records] = {}
        self.save_state(current)
        return purged

    def note_reconcile_skip(self, state: dict[str, Any], cause: str) -> None:
        """Count and log one more tick in a row without reconciliation.
        Reconciliation is what removes a document that is no longer shared,
        so the length of the streak is what matters, not each skip."""
        raw = state.get("reconcile_skips")
        skips = (raw if isinstance(raw, int) else 0) + 1
        state["reconcile_skips"] = skips
        self.logger.warning(
            "%s: skipping reconciliation — %s (%d consecutive skip(s); %s)",
            self.name, cause, skips, self.lingering,
        )

    # -- one tick at a time -------------------------------------------------

    async def run(
        self, lock: asyncio.Lock, tick: Callable[[], Awaitable[dict[str, int]]]
    ) -> dict[str, int]:
        """Run ``tick`` under ``lock``, which the scheduler heartbeat and a
        "Sync now" share, so a manual run never interleaves with a scheduled
        one. A caller that finds it held skips (``busy``) rather than queue
        behind a slow network phase."""
        if lock.locked():
            self.logger.info("%s: a tick is already running — skipping", self.name)
            return {"busy": 1}
        async with lock:
            try:
                return await tick()
            except Exception:
                self.record_error("The last sync failed unexpectedly. Check the server logs.")
                raise
            finally:
                self.last_finished_at = datetime.now(UTC)

    # -- heartbeat ----------------------------------------------------------

    def heartbeat_pending(self, db_path: Path | None = None) -> bool:
        from openexecutive.memory.episodic import _get_conn, _resolve_db_path

        resolved = _resolve_db_path(db_path)
        if not resolved.exists():
            return False
        with _get_conn(resolved) as conn:
            row = conn.execute(
                "SELECT 1 FROM scheduled_actions "
                "WHERE kind = ? AND status IN ('pending', 'running') LIMIT 1",
                (self.heartbeat_kind,),
            ).fetchone()
        return row is not None

    def enqueue(self, run_at: datetime, db_path: Path | None) -> int | None:
        from openexecutive.memory.episodic import insert_scheduled_action

        try:
            action_id = insert_scheduled_action(
                run_at=run_at.isoformat(),
                channel=HEARTBEAT_CHANNEL,
                channel_ref=self.name,
                intent_text=self.heartbeat_intent,
                kind=self.heartbeat_kind,
                db_path=db_path,
            )
        except Exception:
            self.logger.exception("%s: failed to enqueue the heartbeat", self.name)
            return None
        self.logger.info("%s: next scan at %s (id=%d)", self.name, run_at.isoformat(), action_id)
        return action_id

    def bootstrap(self, db_path: Path | None = None) -> int | None:
        """Queue the first tick on boot, unless one is already waiting."""
        if self.heartbeat_pending(db_path):
            return None
        return self.enqueue(datetime.now(UTC) + timedelta(minutes=1), db_path)

    def enqueue_next(
        self, minutes: int, *, after: datetime | None = None, db_path: Path | None = None
    ) -> int | None:
        base = (after or datetime.now(UTC)).astimezone(UTC)
        return self.enqueue(base + timedelta(minutes=minutes), db_path)


__all__ = ["HEARTBEAT_CHANNEL", "SyncSource"]
