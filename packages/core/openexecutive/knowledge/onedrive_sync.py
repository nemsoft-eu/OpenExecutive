"""Incremental OneDrive folder → isolated collection sync.

The Microsoft analogue of ``knowledge.drive_sync``, and built the same way:
opt-in (``ONEDRIVE_SYNC_ENABLED``), it reads the folders in
``ONEDRIVE_SYNC_FOLDERS`` (and their subfolders, to the same depth as Drive)
with GETs only (``knowledge.onedrive_client``). A file whose content tag
(``cTag``) or modified time changed is turned into text, written under
``<company>/docs/onedrive/`` and re-indexed into the ONEDRIVE Chroma
collection, keyed by ``onedrive_key`` (``<drive id>:<item id>``). A file no
longer listed is purged.

It reads as the Executive's own Microsoft 365 sign-in
(``knowledge.onedrive_account``), so what it can see is whatever that account
can; the folder list is the scope. The collection is separate from COMPANY for
the reason DRIVE is: a shared folder is multi-writer, so its files are
unvetted next to curated uploads. The retriever labels them, ranks them below
company docs, and shows each chunk's key and sync time so the Executive can
open the live file with its own OneDrive tools when the latest version matters.

Office files are read with the knowledge loader's extractors. Formats it
can't read (PowerPoint, legacy .doc / .xls, OpenDocument, RTF) are fetched as
Graph's PDF rendering of the file instead.

Locking, heartbeat and the network-then-local-write split are drive_sync's.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from openexecutive.config import Settings, get_settings

# The text extraction, size caps and tree limits are Drive's, so the two
# syncs read a file the same way and stay in step when one is tuned.
from openexecutive.knowledge.drive_sync import (
    _EXTRACT_TIMEOUT_S,
    _MAX_DEPTH,
    _MAX_FILE_BYTES,
    _MAX_FILE_CHARS,
    _MAX_VISIBLE_ITEMS,
    _REQUEST_PAUSE_S,
    _extract,
    _https_or_none,
    _safe_name,
    _Unreadable,
)
from openexecutive.knowledge.isolated import ParserBusy
from openexecutive.knowledge.notion_sync import infer_domain
from openexecutive.knowledge.onedrive_client import (
    OneDriveClient,
    OneDriveFileTooLarge,
    OneDriveItem,
    item_key,
    parse_item_key,
)
from openexecutive.knowledge.store import ChromaDBStore
from openexecutive.knowledge.sync_source import SyncSource

logger = logging.getLogger(__name__)

HEARTBEAT_KIND = "onedrive_sync_scan"
HEARTBEAT_INTENT = "OneDrive folder sync — incremental file ingest into isolated collection."

# Read as they are.
_DIRECT: dict[str, str] = {
    ".pdf": ".pdf",
    ".docx": ".docx",
    ".xlsx": ".xlsx",
    ".xlsm": ".xlsx",
    ".txt": ".txt",
    ".md": ".md",
    ".markdown": ".md",
    ".csv": ".csv",
}
# Fetched as Graph's PDF rendering (``/content?format=pdf``).
_AS_PDF: frozenset[str] = frozenset({
    ".pptx", ".ppt", ".pps", ".ppsx", ".doc", ".xls", ".rtf", ".odt", ".odp", ".ods",
})

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_KEY_COMMENT = re.compile(r"<!--\s*onedrive_key:\s*([A-Za-z0-9!_-]{1,256}:[A-Za-z0-9!_-]{1,256})\s*-->")


def _suffix(name: str) -> str:
    return PurePosixPath(name.lower()).suffix


def is_supported(item: OneDriveItem) -> bool:
    suffix = _suffix(item.name)
    return suffix in _DIRECT or suffix in _AS_PDF


def _state_path() -> Path:
    return get_settings().company_profile_path.parent / "onedrive_sync_state.json"


def _docs_dir() -> Path:
    path = get_settings().company_profile_path.parent / "docs" / "onedrive"
    path.mkdir(parents=True, exist_ok=True)
    return path


SOURCE = SyncSource(
    "onedrive",
    noun="file",
    id_field="onedrive_key",
    id_comment=_KEY_COMMENT,
    collections=(ChromaDBStore.ONEDRIVE_COLLECTION,),
    safe_id=lambda key: key if parse_item_key(key) is not None else None,
    state_path=lambda: _state_path(),
    docs_dir=lambda: _docs_dir(),
    heartbeat_intent=HEARTBEAT_INTENT,
    lingering="files removed from the folders stay indexed until reconciliation runs",
)
load_state = SOURCE.load_state
save_state = SOURCE.save_state
reset_local_state = SOURCE.reset_local_state
purge_file = SOURCE.purge
reconcile_missing_files = SOURCE.reconcile


def slugify(name: str, key: str) -> str:
    slug = _SLUG_RE.sub("-", name.lower()).strip("-")[:60] or "file"
    item_id = key.split(":", 1)[-1]
    short = re.sub(r"[^a-z0-9]", "", item_id.lower())[-10:] or "item"
    return f"onedrive-{short}-{slug}.md"


def _is_current(recorded: dict[str, Any], item: OneDriveItem) -> bool:
    return bool(recorded) and (
        recorded.get("modified") == item.modified and recorded.get("ctag", "") == item.ctag
    )


async def _sleep() -> None:
    await asyncio.sleep(_REQUEST_PAUSE_S)


# ---------------------------------------------------------------------------
# Network phase
# ---------------------------------------------------------------------------


@dataclass
class _TickFetch:
    visible_keys: set[str] = field(default_factory=set)
    incomplete: str | None = None
    fetched: list[tuple[OneDriveItem, str]] = field(default_factory=list)


async def _list_tree(
    client: OneDriveClient, roots: list[tuple[str, str]], fetch: _TickFetch
) -> list[OneDriveItem]:
    """Every file under ``roots`` to ``_MAX_DEPTH`` levels, each once."""
    files: dict[str, OneDriveItem] = {}
    seen: set[str] = set()
    queue: list[tuple[str, str, int]] = [(d, i, 0) for d, i in roots]
    listed = 0
    while queue:
        drive_id, folder_id, depth = queue.pop(0)
        key = item_key(drive_id, folder_id)
        if key in seen:
            continue
        seen.add(key)
        try:
            await _sleep()
            items, truncated = await client.list_folder(
                drive_id, folder_id, max_items=_MAX_VISIBLE_ITEMS - listed + 1
            )
        except Exception as exc:
            logger.warning("onedrive_sync: listing folder %s failed: %s", key, type(exc).__name__)
            fetch.incomplete = fetch.incomplete or f"folder {key} could not be listed"
            continue
        listed += len(items)
        if truncated or listed > _MAX_VISIBLE_ITEMS:
            fetch.incomplete = fetch.incomplete or (
                f"more than {_MAX_VISIBLE_ITEMS} items across the synced folders"
            )
        for item in items:
            if item.is_folder:
                if depth < _MAX_DEPTH:
                    queue.append((item.drive_id, item.id, depth + 1))
                else:
                    logger.info(
                        "onedrive_sync: folder %s is deeper than %d levels — not synced",
                        item.key, _MAX_DEPTH,
                    )
            else:
                files.setdefault(item.key, item)
        if listed > _MAX_VISIBLE_ITEMS:
            break
    return list(files.values())


async def _file_text(client: OneDriveClient, item: OneDriveItem) -> str:
    suffix = _suffix(item.name)
    as_pdf = suffix in _AS_PDF
    read_as = ".pdf" if as_pdf else _DIRECT[suffix]
    # A conversion's size isn't known until it arrives; a stored file's is.
    if not as_pdf and item.size is not None and item.size > _MAX_FILE_BYTES:
        raise OneDriveFileTooLarge(item.key)
    data = await client.download(item.drive_id, item.id, max_bytes=_MAX_FILE_BYTES, as_pdf=as_pdf)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_extract, data, read_as), timeout=_EXTRACT_TIMEOUT_S
        )
    except (_Unreadable, ParserBusy):
        raise
    except TimeoutError as exc:
        raise _Unreadable(f"text extraction took over {_EXTRACT_TIMEOUT_S:.0f}s") from exc
    except Exception as exc:
        raise _Unreadable(f"text extraction failed: {type(exc).__name__}") from exc


async def _fetch_tick(
    *,
    client: OneDriveClient,
    settings: Settings,
    state: dict[str, Any],
    reconcile_only: bool,
    stats: dict[str, int],
) -> _TickFetch:
    fetch = _TickFetch()
    items = await _list_tree(client, settings.onedrive_sync_folder_list, fetch)
    stats["seen"] = len(items)

    dirty: list[OneDriveItem] = []
    for item in items:
        fetch.visible_keys.add(item.key)
        if not is_supported(item) or _is_current(SOURCE.record(state, item.key), item):
            stats["skipped"] += 1
            continue
        dirty.append(item)
    if reconcile_only:
        return fetch

    cap = max(0, settings.onedrive_max_files_per_scan)
    dirty.sort(key=lambda i: i.modified, reverse=True)
    if len(dirty) > cap:
        stats["capped"] = len(dirty) - cap
        logger.warning(
            "onedrive_sync: %d file(s) changed, cap is %d — the rest retry next tick",
            len(dirty), cap,
        )
    for item in dirty[:cap]:
        try:
            await _sleep()
            text = await _file_text(client, item)
        except OneDriveFileTooLarge:
            logger.info("onedrive_sync: %s is over %d bytes — not synced", item.key, _MAX_FILE_BYTES)
            text = ""
        except _Unreadable as exc:
            logger.warning("onedrive_sync: %s is unreadable (%s) — not synced", item.key, exc)
            text = ""
        except Exception as exc:
            stats["failed"] += 1
            # The type only: a download error can name a pre-authenticated URL.
            logger.warning("onedrive_sync: failed to fetch %s (%s)", item.key, type(exc).__name__)
            continue
        fetch.fetched.append((item, text))
    return fetch


# ---------------------------------------------------------------------------
# Local write phase
# ---------------------------------------------------------------------------


def _ingest_file_sync(
    item: OneDriveItem, text: str, store: ChromaDBStore, synced_at: str
) -> int:
    name = _safe_name(item.name)
    return SOURCE.write_and_index(
        item.key,
        name,
        slugify(name, item.key),
        text,
        store,
        domain=infer_domain(name),
        max_chars=_MAX_FILE_CHARS,
        metadata={"name": name, "url": item.link, "synced_at": synced_at},
    )


async def _apply_tick(
    *,
    store: ChromaDBStore,
    fetch: _TickFetch,
    now: datetime,
    reconcile_only: bool,
    stats: dict[str, int],
) -> None:
    """Local write phase — the caller must hold ``_FIXTURE_OP_LOCK``."""
    state = load_state()
    known = state["files"]
    if fetch.incomplete:
        SOURCE.note_reconcile_skip(state, f"listing incomplete ({fetch.incomplete})")
    elif not fetch.visible_keys and known:
        SOURCE.note_reconcile_skip(
            state,
            f"the folders listed 0 files while {len(known)} are on record "
            "(refusing a mass purge on a blank listing)",
        )
    else:
        stats["purged"] = await asyncio.to_thread(
            reconcile_missing_files, fetch.visible_keys, store, state
        )
        state["reconcile_skips"] = 0

    if not reconcile_only:
        synced_at = now.isoformat()
        for item, text in fetch.fetched:
            try:
                chunks = await asyncio.to_thread(_ingest_file_sync, item, text, store, synced_at)
            except Exception:
                stats["failed"] += 1
                logger.exception("onedrive_sync: failed to index file %s", item.key)
                continue
            name = _safe_name(item.name)
            state["files"][item.key] = {
                "modified": item.modified,
                "ctag": item.ctag,
                "name": name,
                "filename": slugify(name, item.key) if chunks else "",
                "url": item.link,
                "synced_at": synced_at,
            }
            stats["updated"] += 1
            logger.info("onedrive_sync: indexed %s (%d chunks)", item.key, chunks)

    state["last_run"] = now.isoformat()
    if stats["failed"]:
        state["last_error"] = SOURCE.partial_failure(stats["failed"])
    else:
        state.pop("last_error", None)
    save_state(state)


_RUN_LOCK = asyncio.Lock()


def is_syncing() -> bool:
    return _RUN_LOCK.locked()


def last_finished_at() -> datetime | None:
    return SOURCE.last_finished_at


async def run_onedrive_sync(
    *,
    store: ChromaDBStore | None = None,
    client: OneDriveClient | None = None,
    now: datetime | None = None,
    reconcile_only: bool = False,
) -> dict[str, int]:
    """One sync tick. Returns counts: seen / updated / skipped / failed / purged / capped
    (plus ``busy`` when another tick is already running in this process)."""
    return await SOURCE.run(
        _RUN_LOCK,
        lambda: _run_locked(store=store, client=client, now=now, reconcile_only=reconcile_only),
    )


async def _run_locked(
    *,
    store: ChromaDBStore | None,
    client: OneDriveClient | None,
    now: datetime | None,
    reconcile_only: bool,
) -> dict[str, int]:
    from openexecutive.knowledge.onedrive_account import (
        OneDriveAuthTransient,
        OneDriveCredentialMissing,
        onedrive_token_provider,
    )

    settings = get_settings()
    stats = {"seen": 0, "updated": 0, "skipped": 0, "failed": 0, "purged": 0, "capped": 0}
    if not settings.onedrive_sync_enabled:
        return stats

    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK
    from openexecutive.clients.slots import get_active_client

    if _FIXTURE_OP_LOCK.locked():
        logger.info("onedrive_sync: fixture/rotation in progress — skipping this tick")
        return stats
    if store is None:
        store = ChromaDBStore(persist_directory=settings.vector_store_path)

    generation = get_active_client(settings)
    http: httpx.AsyncClient | None = None
    if client is None:
        try:
            token = onedrive_token_provider(settings)
        except OneDriveCredentialMissing as exc:
            logger.warning("onedrive_sync: %s", exc)
            SOURCE.record_error(
                "Could not sign in to OneDrive: the Microsoft 365 launcher isn't installed. "
                "See docs/onedrive_sync_setup.md."
            )
            stats["failed"] += 1
            return stats
        http = httpx.AsyncClient(timeout=60.0)
        client = OneDriveClient(http, token)
    try:
        try:
            # One token up front: a refused sign-in stops the tick here with a
            # clear reason, rather than as every folder failing to list.
            await client._token()
        except OneDriveCredentialMissing:
            SOURCE.record_error(
                "Microsoft refused the Executive's sign-in for OneDrive. Sign in to "
                "Microsoft 365 again and allow access to files."
            )
            stats["failed"] += 1
            return stats
        except OneDriveAuthTransient:
            SOURCE.record_error("Could not get a Microsoft token; the next sync retries.")
            stats["failed"] += 1
            return stats
        fetch = await _fetch_tick(
            client=client,
            settings=settings,
            state=load_state(),
            reconcile_only=reconcile_only,
            stats=stats,
        )
        async with _FIXTURE_OP_LOCK:
            if get_active_client(settings) != generation:
                logger.warning(
                    "onedrive_sync: active client changed while fetching — discarding this tick"
                )
                return stats
            await _apply_tick(
                store=store,
                fetch=fetch,
                now=now or datetime.now(UTC),
                reconcile_only=reconcile_only,
                stats=stats,
            )
    finally:
        if http is not None:
            await http.aclose()

    logger.info("onedrive_sync: %s", stats)
    return stats


def purge_all_synced(store: ChromaDBStore, state: dict[str, Any] | None = None) -> int:
    """Remove every locally synced OneDrive file (text files, chunks, state)."""
    return SOURCE.purge_all(store, state, store.delete_onedrive_docs)


def list_synced_files() -> dict[str, Any]:
    """What the Knowledge page shows for OneDrive: one row per file on record."""
    state = load_state()
    last_run = state.get("last_run") if isinstance(state.get("last_run"), str) else None
    files: list[dict[str, Any]] = []
    for key, rec in state["files"].items():
        if parse_item_key(str(key)) is None or not isinstance(rec, dict):
            continue
        filename = rec.get("filename") if isinstance(rec.get("filename"), str) else ""
        files.append(
            {
                "id": key,
                "name": str(rec.get("name") or filename or key),
                "url": _https_or_none(rec.get("url")),
                "modified_at": rec.get("modified") if isinstance(rec.get("modified"), str) else None,
                "synced_at": rec.get("synced_at")
                if isinstance(rec.get("synced_at"), str)
                else last_run,
                "indexed": bool(filename),
            }
        )
    files.sort(key=lambda f: f["name"].lower())
    error = state.get("last_error")
    return {
        "last_run": last_run,
        "last_error": error if isinstance(error, str) else None,
        "files": files,
    }


def read_synced_file(key: str) -> dict[str, Any] | None:
    """The stored text of one synced file, or None when unknown or empty.
    The path comes from the state record for a checked key, never the caller."""
    if parse_item_key(key) is None:
        return None
    rec = load_state()["files"].get(key)
    if not isinstance(rec, dict):
        return None
    filename = SOURCE.safe_filename(str(rec.get("filename") or ""))
    if not filename:
        return None
    try:
        text = (_docs_dir() / filename).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    text = _KEY_COMMENT.sub("", text, count=1).lstrip()
    return {
        "id": key,
        "name": str(rec.get("name") or filename),
        "url": _https_or_none(rec.get("url")),
        "content": text,
    }


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


def bootstrap_onedrive_sync_scan(db_path: Path | None = None) -> int | None:
    return SOURCE.bootstrap(db_path)


def enqueue_next_onedrive_sync_scan(
    *, after: datetime | None = None, db_path: Path | None = None
) -> int | None:
    return SOURCE.enqueue_next(
        get_settings().onedrive_sync_interval_minutes, after=after, db_path=db_path
    )
