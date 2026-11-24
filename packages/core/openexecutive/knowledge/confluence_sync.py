"""Incremental Confluence space → isolated collection sync.

Opt-in (``CONFLUENCE_SYNC_ENABLED``). Each tick lists every page in the spaces
named in ``CONFLUENCE_SYNC_SPACE_KEYS`` (``knowledge.confluence_client``), with
its version, ancestors and read restrictions. A page whose version changed is
fetched, its storage body turned into Markdown
(``knowledge.confluence_storage``), written under ``<company>/docs/confluence/``
and re-indexed into the CONFLUENCE Chroma collection, keyed by
``confluence_page_id``. A page no longer listed is purged.

The token acts as one Confluence user, and anyone who can talk to the
Executive can retrieve what it syncs. So the space list is required, and by
default (``CONFLUENCE_SYNC_SKIP_RESTRICTED``) a page with a read restriction,
its own or a parent page's, is left out and purged if it was synced before.
When the listing does not say whether a page is restricted, it is treated as
restricted.

That collection is separate from COMPANY for the reason NOTION and DRIVE are:
a wiki is multi-writer, so its pages are unvetted next to curated uploads. The
retriever labels them as such and ranks them below company docs.

Structure mirrors ``knowledge.drive_sync``: a network phase with no lock, then
a local write phase under ``_FIXTURE_OP_LOCK``; heartbeat bootstrap on boot,
one tick per run, chain the next.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from openexecutive.config import Settings, get_settings
from openexecutive.knowledge.confluence_client import (
    ConfluenceClient,
    ConfluenceHostNotPublic,
    ConfluencePage,
    ConfluenceResponseTooLarge,
    auth_headers,
    sanitize_page_id,
    ssl_verify,
)
from openexecutive.knowledge.confluence_storage import storage_to_markdown
from openexecutive.knowledge.notion_sync import infer_domain
from openexecutive.knowledge.store import ChromaDBStore
from openexecutive.knowledge.sync_source import SyncSource

logger = logging.getLogger(__name__)

HEARTBEAT_KIND = "confluence_sync_scan"
HEARTBEAT_INTENT = "Confluence space sync — incremental page ingest into isolated collection."

_MAX_PAGE_CHARS = 200_000
_MAX_VISIBLE_PAGES = 5000  # pages listed per tick, all spaces together
_REQUEST_PAUSE_S = 0.2
# Storage XHTML handed to the converter. Well past what 200k characters of
# Markdown need, and short enough that converting it stays quick.
_MAX_STORAGE_CHARS = 1_000_000
# How long one page's conversion may run, off the event loop, before the page
# is recorded as unreadable (not retried until its version changes).
_CONVERT_TIMEOUT_S = 60.0

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_PAGE_ID_COMMENT = re.compile(r"<!--\s*confluence_page_id:\s*([0-9]{1,20})\s*-->")


def _state_path() -> Path:
    return get_settings().company_profile_path.parent / "confluence_sync_state.json"


def _docs_dir() -> Path:
    path = get_settings().company_profile_path.parent / "docs" / "confluence"
    path.mkdir(parents=True, exist_ok=True)
    return path


SOURCE = SyncSource(
    "confluence",
    noun="page",
    id_field="confluence_page_id",
    id_comment=_PAGE_ID_COMMENT,
    collections=(ChromaDBStore.CONFLUENCE_COLLECTION,),
    safe_id=sanitize_page_id,
    state_path=lambda: _state_path(),
    docs_dir=lambda: _docs_dir(),
    heartbeat_intent=HEARTBEAT_INTENT,
    lingering="pages removed or restricted stay indexed until reconciliation runs",
)
load_state = SOURCE.load_state
save_state = SOURCE.save_state
reset_local_state = SOURCE.reset_local_state
purge_page = SOURCE.purge


def _safe_title(title: str) -> str:
    cleaned = title
    while True:  # until stable: one pass can leave a new marker behind
        stripped = cleaned.replace("<!--", "").replace("-->", "")
        if stripped == cleaned:
            break
        cleaned = stripped
    return " ".join(cleaned.split())[:200] or "Untitled"


def slugify(title: str, page_id: str) -> str:
    slug = _SLUG_RE.sub("-", title.lower()).strip("-")[:60] or "page"
    return f"confluence-{page_id}-{slug}.md"


def _is_current(recorded: dict[str, Any], page: ConfluencePage) -> bool:
    return bool(recorded) and recorded.get("version") == page.version


async def _sleep() -> None:
    await asyncio.sleep(_REQUEST_PAUSE_S)


def _restricted_ids(
    pages: dict[str, ConfluencePage], complete_spaces: Collection[str]
) -> tuple[set[str], set[str]]:
    """Pages to leave out, as ``(restricted, undecided)``.

    Restricted: a read restriction on the page or on any parent page, or no
    way to tell (restrictions missing from the listing, or a parent page its
    fully listed space did not return). These are purged.

    Undecided: a parent page is missing from a space whose listing was cut
    short or failed, so it may simply be past the cut. These are not synced
    this tick, but not purged either, or a page under an old parent would
    flap in and out of the knowledge base on every tick."""
    restricted: set[str] = set()
    undecided: set[str] = set()
    for page in pages.values():
        if page.restricted is not False:
            restricted.add(page.id)
            continue
        for ancestor_id in page.ancestors:
            ancestor = pages.get(ancestor_id)
            if ancestor is None and page.space not in complete_spaces:
                undecided.add(page.id)
                break
            if ancestor is None or ancestor.restricted is not False:
                restricted.add(page.id)
                break
    return restricted, undecided


# ---------------------------------------------------------------------------
# Network phase
# ---------------------------------------------------------------------------


@dataclass
class _TickFetch:
    """Everything the network phase of a tick learned, ready to apply locally."""

    # Every page in the synced spaces that may be synced (restricted ones
    # left out), changed or not.
    visible_ids: set[str] = field(default_factory=set)
    # Pages the listing returned, restricted ones included.
    listed: int = 0
    # Listed pages left out as restricted. Positive information: they are
    # purged even when the rest of the listing is incomplete.
    restricted_ids: set[str] = field(default_factory=set)
    # Spaces listed in full, and how many pages each returned. Reconcile can
    # run for these even when another space failed.
    complete_spaces: dict[str, int] = field(default_factory=dict)
    # Why the listing is not the whole picture (reconcile must not run).
    incomplete: str | None = None
    # A short, user-facing reason for the knowledge page, when listing failed.
    error: str | None = None
    # (page, markdown) per page fetched; "" when it had no readable text.
    fetched: list[tuple[ConfluencePage, str]] = field(default_factory=list)


def _listing_error(space: str, exc: Exception) -> str:
    if isinstance(exc, ConfluenceHostNotPublic):
        return (
            "CONFLUENCE_URL points at a private or local address, which "
            "CONFLUENCE_SYNC_PUBLIC_HOSTS_ONLY refuses."
        )
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in (401, 403):
            return (
                f"Confluence refused the sync's sign-in (HTTP {code}). "
                "Check the Confluence token."
            )
        if code == 404:
            return f"Confluence space {space} was not found, or the token cannot see it."
        return f"Confluence returned HTTP {code} while listing space {space}."
    return f"Could not reach Confluence while listing space {space}."


async def _list_spaces(
    client: ConfluenceClient, settings: Settings, fetch: _TickFetch
) -> dict[str, ConfluencePage]:
    pages: dict[str, ConfluencePage] = {}
    for space in settings.confluence_sync_space_key_list:
        remaining = _MAX_VISIBLE_PAGES - len(pages)
        if remaining <= 0:
            fetch.incomplete = fetch.incomplete or (
                f"more than {_MAX_VISIBLE_PAGES} pages across the synced spaces"
            )
            break
        try:
            await _sleep()
            listed, truncated = await client.list_space(
                space,
                max_items=remaining,
                with_restrictions=settings.confluence_sync_skip_restricted,
            )
        except Exception as exc:
            logger.warning("confluence_sync: listing space %s failed: %s", space, exc)
            fetch.incomplete = fetch.incomplete or f"space {space} could not be listed"
            fetch.error = fetch.error or _listing_error(space, exc)
            continue
        if truncated:
            fetch.incomplete = fetch.incomplete or (
                f"space {space} could not be listed in full"
            )
        else:
            fetch.complete_spaces[space] = len(listed)
        for page in listed:
            pages.setdefault(page.id, page)
    return pages


async def _fetch_tick(
    *,
    client: ConfluenceClient,
    settings: Settings,
    state: dict[str, Any],
    reconcile_only: bool,
    stats: dict[str, int],
) -> _TickFetch:
    """List the synced spaces and fetch the bodies of changed pages. Reads
    ``state`` only to decide what is current; nothing is written here."""
    fetch = _TickFetch()
    pages = await _list_spaces(client, settings, fetch)
    stats["seen"] = fetch.listed = len(pages)
    restricted: set[str] = set()
    undecided: set[str] = set()
    if settings.confluence_sync_skip_restricted:
        restricted, undecided = _restricted_ids(pages, fetch.complete_spaces)
    stats["restricted"] = len(restricted)
    fetch.restricted_ids = restricted

    dirty: list[ConfluencePage] = []
    for page in pages.values():
        if page.id in restricted or page.id in undecided:
            continue
        fetch.visible_ids.add(page.id)
        if _is_current(SOURCE.record(state, page.id), page):
            stats["skipped"] += 1
            continue
        dirty.append(page)
    if reconcile_only:
        return fetch

    cap = max(0, settings.confluence_max_pages_per_scan)
    dirty.sort(key=lambda p: p.modified, reverse=True)
    if len(dirty) > cap:
        stats["capped"] = len(dirty) - cap
        logger.warning(
            "confluence_sync: %d page(s) changed, cap is %d — the rest retry next tick",
            len(dirty), cap,
        )
    for page in dirty[:cap]:
        try:
            await _sleep()
            seen, storage = await client.page_body(page)
        except ConfluenceResponseTooLarge:
            logger.info("confluence_sync: page %s is too large — not synced", page.id)
            fetch.fetched.append((page, ""))
            continue
        except Exception:
            stats["failed"] += 1
            logger.exception("confluence_sync: failed to fetch page %s", page.id)
            continue
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(storage_to_markdown, storage[:_MAX_STORAGE_CHARS]),
                timeout=_CONVERT_TIMEOUT_S,
            )
        except TimeoutError:
            logger.warning(
                "confluence_sync: page %s took over %.0fs to convert — not synced",
                page.id, _CONVERT_TIMEOUT_S,
            )
            text = ""
        except Exception:
            stats["failed"] += 1
            logger.exception("confluence_sync: failed to convert page %s", page.id)
            continue
        fetch.fetched.append((seen, text))
    return fetch


# ---------------------------------------------------------------------------
# Local write phase
# ---------------------------------------------------------------------------


def reconcile_missing_pages(
    visible_ids: set[str],
    store: ChromaDBStore,
    state: dict[str, Any],
    *,
    spaces: set[str] | None = None,
) -> int:
    """Purge pages (and orphan text files) no longer listed, or now
    restricted. With ``spaces``, only pages on record in those spaces are
    considered (the spaces whose listing completed); orphan files are then
    left for a full reconcile. Blocking I/O: run it via ``asyncio.to_thread``."""
    if spaces is None:
        return SOURCE.reconcile(visible_ids, store, state)
    return SOURCE.reconcile(
        visible_ids,
        store,
        state,
        only=lambda rec: isinstance(rec, dict) and rec.get("space") in spaces,
    )


def purge_restricted_pages(
    restricted_ids: set[str], store: ChromaDBStore, state: dict[str, Any]
) -> int:
    """Purge pages on record that the listing returned as restricted. Runs
    on every tick, complete listing or not: a page known to be restricted
    must leave the knowledge base now. Blocking I/O, like reconcile."""
    targets = sorted(p for p in state.get("pages", {}) if p in restricted_ids)
    if not targets:
        return 0
    file_index = SOURCE.build_file_index()
    return sum(purge_page(pid, store, state, file_index=file_index) for pid in targets)


def _ingest_page_sync(
    page: ConfluencePage, text: str, store: ChromaDBStore, synced_at: str
) -> int:
    """Write the page's Markdown under docs/confluence/ and re-index its
    chunks. Empty text leaves no chunks (a page with nothing readable)."""
    title = _safe_title(page.title)
    return SOURCE.write_and_index(
        page.id,
        title,
        slugify(title, page.id),
        text,
        store,
        domain=infer_domain(title),
        max_chars=_MAX_PAGE_CHARS,
        metadata={"title": title, "space": page.space, "url": page.url, "synced_at": synced_at},
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
    # Reload rather than reuse the pre-fetch snapshot: a fixture load or reset
    # may have replaced the state file while the network phase ran.
    state = load_state()
    known = state["pages"]
    stats["purged"] = await asyncio.to_thread(
        purge_restricted_pages, fetch.restricted_ids, store, state
    )
    if fetch.incomplete:
        SOURCE.note_reconcile_skip(state, f"listing incomplete ({fetch.incomplete})")
        # Spaces that did list in full still reconcile, unless one came back
        # empty with pages on record (the same blank-listing caution).
        on_record = {
            rec.get("space") for rec in known.values() if isinstance(rec, dict)
        }
        complete = {
            space for space, count in fetch.complete_spaces.items()
            if count or space not in on_record
        }
        if complete:
            stats["purged"] += await asyncio.to_thread(
                reconcile_missing_pages, fetch.visible_ids, store, state, spaces=complete
            )
    elif not fetch.listed and known:
        # Only an empty listing is suspect. Pages listed but all restricted
        # are purged: what became restricted must leave the knowledge base.
        SOURCE.note_reconcile_skip(
            state,
            f"the spaces listed 0 pages while {len(known)} are on record "
            "(refusing a mass purge on a blank listing)",
        )
    else:
        stats["purged"] += await asyncio.to_thread(
            reconcile_missing_pages, fetch.visible_ids, store, state
        )
        state["reconcile_skips"] = 0

    if not reconcile_only:
        synced_at = now.isoformat()
        for page, text in fetch.fetched:
            try:
                chunks = await asyncio.to_thread(_ingest_page_sync, page, text, store, synced_at)
            except Exception:
                stats["failed"] += 1
                logger.exception("confluence_sync: failed to index page %s", page.id)
                continue
            title = _safe_title(page.title)
            state["pages"][page.id] = {
                "version": page.version,
                "modified": page.modified,
                "title": title,
                "space": page.space,
                "filename": slugify(title, page.id) if chunks else "",
                "url": page.url,
                "synced_at": synced_at,
            }
            stats["updated"] += 1
            logger.info("confluence_sync: indexed %s (%d chunks)", page.id, chunks)

    state["last_run"] = now.isoformat()
    if fetch.error:
        state["last_error"] = fetch.error
    elif stats["failed"]:
        state["last_error"] = SOURCE.partial_failure(stats["failed"])
    else:
        state.pop("last_error", None)
    save_state(state)


# One tick at a time per process, shared by the scheduler heartbeat and any
# manual run, like drive_sync. A caller that finds it held skips (``busy``).
_RUN_LOCK = asyncio.Lock()


def is_syncing() -> bool:
    return _RUN_LOCK.locked()


def last_finished_at() -> datetime | None:
    return SOURCE.last_finished_at


def build_client(settings: Settings) -> tuple[ConfluenceClient, httpx.AsyncClient]:
    """A client for the configured site, and the HTTP client to close after."""
    http = httpx.AsyncClient(
        timeout=60.0,
        verify=ssl_verify(settings.confluence_ssl_verify),
        follow_redirects=False,
        # The public-hosts check pins the address it checked, and a proxy
        # tunnel would check the certificate against that bare address
        # instead of the site's name, so those requests go direct.
        trust_env=not settings.confluence_sync_public_hosts_only,
    )
    client = ConfluenceClient(
        http,
        str(settings.confluence_url),
        auth_headers(settings),
        public_hosts_only=settings.confluence_sync_public_hosts_only,
    )
    return client, http


async def run_confluence_sync(
    *,
    store: ChromaDBStore | None = None,
    client: ConfluenceClient | None = None,
    now: datetime | None = None,
    reconcile_only: bool = False,
) -> dict[str, int]:
    """One sync tick. Returns counts: seen / updated / skipped / restricted /
    failed / purged / capped (plus ``busy`` when a tick is already running)."""
    return await SOURCE.run(
        _RUN_LOCK,
        lambda: _run_confluence_sync_locked(
            store=store, client=client, now=now, reconcile_only=reconcile_only
        ),
    )


async def _run_confluence_sync_locked(
    *,
    store: ChromaDBStore | None,
    client: ConfluenceClient | None,
    now: datetime | None,
    reconcile_only: bool,
) -> dict[str, int]:
    settings = get_settings()
    stats = {
        "seen": 0, "updated": 0, "skipped": 0, "restricted": 0,
        "failed": 0, "purged": 0, "capped": 0,
    }
    if not settings.confluence_sync_enabled:
        return stats

    # Same locking as drive_sync: skip while a fixture load / client rotation
    # runs, fetch unlocked, lock only the local writes.
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK
    from openexecutive.clients.slots import get_active_client

    if _FIXTURE_OP_LOCK.locked():
        logger.info("confluence_sync: fixture/rotation in progress — skipping this tick")
        return stats
    if store is None:
        store = ChromaDBStore(persist_directory=settings.vector_store_path)

    generation = get_active_client(settings)
    http: httpx.AsyncClient | None = None
    if client is None:
        try:
            client, http = build_client(settings)
        except Exception:
            logger.exception("confluence_sync: cannot set up the Confluence client")
            SOURCE.record_error(
                "Could not set up the Confluence connection. Check CONFLUENCE_SSL_VERIFY."
            )
            stats["failed"] += 1
            return stats
    try:
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
                    "confluence_sync: active client changed while fetching — discarding this tick"
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

    logger.info("confluence_sync: %s", stats)
    return stats


def purge_all_synced(store: ChromaDBStore, state: dict[str, Any] | None = None) -> int:
    """Remove every locally synced Confluence page (text files, chunks, state)."""
    return SOURCE.purge_all(store, state, store.delete_confluence_docs)


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


def bootstrap_confluence_sync_scan(db_path: Path | None = None) -> int | None:
    return SOURCE.bootstrap(db_path)


def enqueue_next_confluence_sync_scan(
    *, after: datetime | None = None, db_path: Path | None = None
) -> int | None:
    return SOURCE.enqueue_next(
        get_settings().confluence_sync_interval_minutes, after=after, db_path=db_path
    )
