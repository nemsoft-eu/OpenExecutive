"""Who may post a progress update: one token per source.

``PROGRESS_SOURCES`` lists ``source-id:sha256-hex[,…]``, as
``scripts/make-source-token.py`` prints it. The API keeps only the hash of
each token, so a leaked API environment cannot mint a caller, and the source
an update is filed under comes from the token, never from the payload.
With no source configured (or a value that cannot be read) the feature is
off: :func:`enabled` is false and :func:`source_for` accepts nothing.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets

logger = logging.getLogger(__name__)

ENV = "PROGRESS_SOURCES"
# a source id is part of every relay's header line and of the audit actor,
# so it stays short, lowercase and free of anything a line could hide behind
SOURCE_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{1,39}")
_HASH_RE = re.compile(r"[0-9a-f]{64}")
# a presented token longer than this is not one of ours; hashing it anyway
# would let a caller make the API hash megabytes per request
_MAX_TOKEN_CHARS = 256
TOKEN_BYTES = 32
# the last malformed value already logged: configured() runs on every request,
# before any auth, so logging each time would let anyone flood the log
_logged_bad_raw: str | None = None


class SourcesError(ValueError):
    """``PROGRESS_SOURCES`` is set but can't be used."""


def token_hash(token: str) -> str:
    """The sha256 hex of ``token``, as ``PROGRESS_SOURCES`` stores it."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def make_token(source_id: str) -> tuple[str, str]:
    """A new random token for ``source_id`` and its ``PROGRESS_SOURCES``
    entry. Raises :class:`SourcesError` on a malformed id."""
    if not SOURCE_ID_RE.fullmatch(source_id):
        raise SourcesError("a source id is 2-40 of a-z, 0-9 and -, starting with a letter or digit")
    token = secrets.token_urlsafe(TOKEN_BYTES)
    return token, f"{source_id}:{token_hash(token)}"


def parse_sources(raw: str) -> dict[str, str]:
    """``source-id:sha256-hex[,…]`` as a dict of source id to hash. Empty is
    no sources; a malformed entry, an id listed twice, or a value that lists
    none (" , ") raises :class:`SourcesError`."""
    sources: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        source_id, sep, digest = (part.strip() for part in entry.partition(":"))
        digest = digest.lower()
        if not sep or not SOURCE_ID_RE.fullmatch(source_id) or not _HASH_RE.fullmatch(digest):
            raise SourcesError(
                "each source is <source id>:<token sha256>, as scripts/make-source-token.py prints it"
            )
        if source_id in sources:
            raise SourcesError(f"source {source_id} is listed twice")
        sources[source_id] = digest
    if raw.strip() and not sources:
        raise SourcesError("it is set but lists no source")
    return sources


def configured() -> dict[str, str]:
    """The configured sources; empty, with an error logged once until the
    value changes, when the setting can't be read (fail closed: no token is
    accepted)."""
    global _logged_bad_raw
    from openexecutive.config import get_settings

    raw = get_settings().progress_sources
    try:
        return parse_sources(raw)
    except SourcesError as exc:
        # the line carries the error only, never the raw value: it holds
        # token hashes
        if raw != _logged_bad_raw:
            _logged_bad_raw = raw
            logger.error("%s can't be used, progress updates are off: %s", ENV, exc)
        return {}


def enabled() -> bool:
    """Whether at least one source is configured."""
    return bool(configured())


def source_for(token: str | None) -> str | None:
    """The source id ``token`` belongs to, or None. Every configured hash is
    compared in constant time, and all of them are compared whatever
    matched, so the time taken says nothing about which entry was close."""
    if not isinstance(token, str) or not token or len(token) > _MAX_TOKEN_CHARS:
        return None
    presented = token_hash(token)
    found: str | None = None
    for source_id, digest in configured().items():
        if hmac.compare_digest(presented, digest):
            found = source_id
    return found
