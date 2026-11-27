"""Chat tool that reads a document file — including a scanned PDF.

Attachments sent in chat, Discord, Telegram, Slack, Google Chat and inbound
email are read for the Executive before the turn starts. ``read_document``
covers the rest: a file another tool saved to disk (a Gmail attachment from
``get_gmail_attachment_content``, a Drive file from
``get_drive_file_download_url``) or a company document by name. A PDF with no
text layer is converted by ``knowledge.pdf_reader`` — Claude when the
deployment reaches it directly, local OCR otherwise — instead of coming back
empty.

Reads are confined to the tool download folders (``tool_catalog.
resolve_readable_file``, the same check workflow steps use) and the
company documents folder. The tool is withheld on turns private to the
principal (``schedule_tools.PRIVATE_TURN_WITHHELD_TOOLS``): those come from a
contact's email, whose own attachments the poller has already read, and must
not steer the Executive into the company's documents. The schema is static,
so the cached tool prefix never moves.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PAGES_RE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+))?\s*$")
# What the "from character" and "cut off" notes on a part need.
_NOTES_ROOM = 200

READ_DOCUMENT_TOOL: dict[str, Any] = {
    "name": "read_document",
    "description": (
        "Read the full text of a PDF, Word, Excel, CSV, Markdown, text, calendar "
        "invite (.ics) or saved email (.eml) file. "
        "Scanned or image-only PDFs are converted to text, so use this whenever "
        "a PDF came back empty, 'could not extract any text', or 'may be "
        "scanned/image-only'. Pass `path` for a file another tool saved to disk "
        "— e.g. the 'Saved to:' path from get_gmail_attachment_content, or from "
        "get_drive_file_download_url for a Drive PDF whose content could not be "
        "read — or `filename` for a company document, or `file` for any file by "
        "its handle (mail:<message_id>#<n>, drive:<file_id>, file:<name>). For a long PDF, pass "
        "`pages` (e.g. '1-20') to read part of it. A long file of any kind comes back in parts: "
        "pass the `start` its cut-off note gives to read on."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path of a file another tool downloaded.",
            },
            "filename": {
                "type": "string",
                "description": "Name of a company document, e.g. 'board-deck-q3.pdf'.",
            },
            "file": {
                "type": "string",
                "description": "A file's handle: mail:<message_id>#<n>, drive:<file_id> or file:<name>.",
            },
            "pages": {
                "type": "string",
                "description": "PDF page range to read, e.g. '5' or '1-20'. Default: all.",
            },
            "start": {
                "type": "integer",
                "description": "Character to start from, to read on where the last read was cut off. Default: 0.",
            },
        },
    },
}

DOCUMENT_TOOLS: list[dict[str, Any]] = [READ_DOCUMENT_TOOL]


def _err(message: str) -> str:
    return json.dumps({"error": message})


def _company_docs_dir() -> Path:
    from openexecutive.config import get_settings

    return (get_settings().company_profile_path.parent / "docs").resolve()


def _parse_pages(raw: str) -> tuple[int, int] | str:
    """1-based inclusive ``pages`` → 0-based ``(start, end)``, or an error."""
    m = _PAGES_RE.match(raw)
    if not m:
        return "pages must look like '5' or '1-20'"
    first = int(m.group(1))
    last = int(m.group(2) or first)
    if first < 1 or last < first:
        return "pages must look like '5' or '1-20'"
    return first - 1, last


async def _read_pages(path: Path, first: int, last: int) -> tuple[Any, str]:
    """Read pages ``[first, last)`` of a PDF. Returns ``(result, range_label)``."""
    from openexecutive.knowledge.pdf_reader import page_range, read_pdf_text

    data = await asyncio.to_thread(path.read_bytes)
    total, part = await page_range(data, first, last)
    if part is None:
        raise ValueError(f"that PDF has only {total} pages")
    last = min(last, total)
    result = await read_pdf_text(part, filename=path.name)
    return result, f"pages {first + 1}-{last} of {total}"


async def handle_read_document(tool_input: dict[str, Any]) -> str:
    from openexecutive.delegation.lockdown import outside_reach_refusal

    if (refused := outside_reach_refusal("read_document")) is not None:
        return refused
    from openexecutive.workflows.tool_catalog import resolve_readable_file

    raw_path = str(tool_input.get("path") or "").strip()
    filename = str(tool_input.get("filename") or "").strip()
    if tool_input.get("file") and not raw_path and not filename:
        return await _read_handle(tool_input)
    if not raw_path and not filename:
        return _err("pass `path` (a downloaded file), `filename` (a company document) or `file` (a handle)")

    docs_dir = _company_docs_dir()
    if filename and not raw_path:
        safe = Path(filename).name
        if not safe or safe.startswith(".") or safe != filename:
            return _err("filename must be a bare document name")
        raw_path = str(docs_dir / safe)

    resolved = resolve_readable_file(raw_path, extra_dirs=[docs_dir])
    if isinstance(resolved, str):
        return _err(resolved)

    return await _read_path(resolved, tool_input)


async def _read_handle(tool_input: dict[str, Any]) -> str:
    """``read_document`` on a file named by its handle (``file_refs``),
    written to a temporary folder of its own for the reader and removed."""
    import tempfile

    from openexecutive.orchestrator import file_refs

    try:
        files = await file_refs.resolve([tool_input.get("file")], max_bytes=file_refs.READ_MAX_BYTES)
    except file_refs.RefError as exc:
        return _err(str(exc))
    if not files.items:
        return _err("that file is too large to read")
    item = files.items[0]
    with tempfile.TemporaryDirectory(prefix="oe-read-") as folder:
        path = Path(folder) / (Path(item.name).name or "file")
        await asyncio.to_thread(path.write_bytes, item.data)
        return await _read_path(path, tool_input)


async def _read_path(resolved: Path, tool_input: dict[str, Any]) -> str:
    from openexecutive.knowledge.isolated import ParserBusy
    from openexecutive.knowledge.loader import read_document_text
    from openexecutive.providers.registry import reading_limits_for

    pages_raw = str(tool_input.get("pages") or "").strip()
    start = tool_input.get("start") or 0
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        return _err("start must be a whole number, 0 or more")
    scope = ""
    try:
        if pages_raw and resolved.suffix.lower() == ".pdf":
            bounds = _parse_pages(pages_raw)
            if isinstance(bounds, str):
                return _err(bounds)
            result, scope = await _read_pages(resolved, *bounds)
        else:
            result = await read_document_text(resolved)
    except ValueError as exc:
        return _err(str(exc))
    except ParserBusy:
        return _err("the document reader is busy right now; try again in a minute")
    except Exception as exc:
        logger.warning("read_document: extraction failed (%s)", type(exc).__name__)
        return _err("could not read that file")

    if not result.text.strip():
        return _err(result.note or "no text could be read from that file")

    header = [f"[{resolved.name}"]
    if scope:
        header.append(f", {scope}")
    if result.converted:
        header.append(", converted from scanned pages")
    if result.note:
        header.append(f" — {result.note}")
    header.append("]")

    # A long file comes back in parts, whatever its kind: the note names
    # where to read on, so the rest is one more call (not a Python job). A
    # part leaves room under the turn's per-result cap for the header and the
    # notes, so a part never cuts the note off. The part is what one read of
    # the turn's model holds (reading.turn_limits; the default model's
    # outside a turn), counting text written about one token a character
    # three times its length.
    total = len(result.text)
    if start >= total:
        return _err(f"that file has only {total} characters")
    head = "".join(header)
    from openexecutive.orchestrator.reading import text_weight, turn_limits

    per_read = (turn_limits() or reading_limits_for()).per_read
    weight = text_weight(result.text[start:])
    end = start + max(1, int(per_read / weight) - len(head) - _NOTES_ROOM)
    text = result.text[start:end]
    if start:
        text = f"…[from character {start} of {total}]\n" + text
    if end < total:
        # ``start`` counts into the text of the pages read, so the same pages
        # go with it.
        again = f"pages='{pages_raw}' and start={end}" if scope else f"start={end}"
        text += f"\n…[cut off at character {end} of {total} — call again with {again} to read on]"
    return head + "\n" + text


DOCUMENT_TOOL_HANDLERS: dict[str, Callable[[dict[str, Any]], Awaitable[str]]] = {
    "read_document": handle_read_document,
}
