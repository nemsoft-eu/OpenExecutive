"""``POST /progress-updates``: a source posts one progress update over HTTP.

The route authenticates the source by its own token (``x-source-token``,
``integrations/progress_sources.py``), so it is exempt from the shared-secret
gate (``api.main._UNAUTHENTICATED_PATHS``) the way the webhooks are. It never
reads the caller: with signed callers on, a request with no assertion is a
service, and the source an update is filed under comes from the token alone.

The order is fixed so that nothing about the body is looked at for a caller
who may not send one: feature off (no sources, or a client slot or demo
fixture live) → 404, token missing or unknown → 401, then the body (a
non-numeric Content-Length → 400, over 8 KB → 413), parsing (422) and
``integrations.progress_updates.submit``, which holds every other rule.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError

from openexecutive.integrations import progress_sources
from openexecutive.integrations import progress_updates as pu

logger = logging.getLogger(__name__)

router = APIRouter()

PATH = "/progress-updates"
TOKEN_HEADER = progress_sources.TOKEN_HEADER
MAX_BODY_BYTES = 8 * 1024


class ProgressRequest(BaseModel):
    """The body as sent. Only the shape is checked here; the guard
    (``integrations/progress_guard.py``) checks every limit and pattern.
    There is no ``source`` field: one is a 422, not an attribution."""

    model_config = ConfigDict(extra="forbid", strict=True)

    topic: str
    status: str
    summary: str
    ask: str | None = None
    link: str | None = None
    idempotency_key: str | None = None


class ProgressResponse(BaseModel):
    """What became of the update. ``relay`` is ``accepted`` or
    ``duplicate``; whether a DM went out is never returned, so a source
    cannot learn a founder's quiet hours or settings."""

    id: int
    stored: bool
    relay: pu.Relay


def _error(status_code: int, detail: object) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status_code)


def _rejected(field: str, reason: str) -> JSONResponse:
    return _error(422, {"field": field, "reason": reason})


async def _read_capped(request: Request) -> bytes | None:
    # a copy of integrations/slack_events.py's: a chunked request carries no
    # Content-Length, so the cap is enforced on what is actually read
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BODY_BYTES:
            return None
    return bytes(body)


@router.post(PATH, status_code=201, response_model=ProgressResponse)
async def post_progress_update(request: Request) -> JSONResponse:
    """Submit one progress update as the source the token names. 201 when
    stored, 200 for a repeat of an update already stored."""
    if not pu.accepting():
        return _error(404, "Not Found")
    tokens = request.headers.getlist(TOKEN_HEADER)
    source = progress_sources.source_for(tokens[0]) if len(tokens) == 1 else None
    if source is None:
        return _error(401, "a valid x-source-token header is required")

    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            too_big = int(declared) > MAX_BODY_BYTES
        except ValueError:
            return _error(400, "invalid Content-Length")
        if too_big:
            return _error(413, f"the body must be at most {MAX_BODY_BYTES} bytes")
    body = await _read_capped(request)
    if body is None:
        return _error(413, f"the body must be at most {MAX_BODY_BYTES} bytes")

    try:
        parsed = ProgressRequest.model_validate_json(body)
    except ValidationError as exc:
        # the first error's location and message only: pydantic's own detail
        # carries the input, which must not be echoed back
        first = exc.errors(include_url=False, include_input=False, include_context=False)[0]
        field = ".".join(str(part) for part in first["loc"]) or "payload"
        return _rejected(field, first["msg"])

    try:
        result = await pu.submit(source, parsed.model_dump(exclude_none=True))
    except pu.Rejected as exc:
        return _rejected(exc.field, exc.reason)
    except pu.Paused:
        # a slot or fixture went live after the check above: still "off"
        return _error(404, "Not Found")
    except pu.OverLimit:
        return _error(429, "this source is at its daily limit of progress updates")
    except pu.StoreUnavailable:
        logger.exception("progress update from %s not stored", source)
        return _error(503, "progress updates can't be stored right now")
    response = ProgressResponse(id=result.id, stored=result.stored, relay=result.relay)
    return JSONResponse(response.model_dump(), status_code=201 if result.stored else 200)
