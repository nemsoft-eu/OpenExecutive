"""orchestrator.document_tools — the Executive's `read_document` tool."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from openexecutive.knowledge import pdf_reader
from openexecutive.orchestrator.document_tools import (
    READ_DOCUMENT_TOOL,
    handle_read_document,
)


@pytest.fixture()
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """(download dir, company docs dir), both empty."""
    downloads = tmp_path / "attachments"
    downloads.mkdir()
    company = tmp_path / "company"
    (company / "docs").mkdir(parents=True)
    monkeypatch.setenv("WORKFLOW_FILE_DIRS", str(downloads))
    monkeypatch.setenv("COMPANY_PROFILE_PATH", str(company / "profile.yaml"))
    return downloads, company / "docs"


@pytest.fixture()
def fake_pdf(monkeypatch: pytest.MonkeyPatch) -> list[bytes | Path]:
    """Stub the converter: every PDF reads as scanned pages converted to
    'PAGE TEXT (<n> bytes)'. Returns what it was handed: a whole file comes
    as its path (the reader's child process opens it), a page range as the
    sliced bytes."""
    seen: list[bytes | Path] = []

    async def fake_read(data: bytes | Path, *, filename: str = "", inbound: bool = False) -> pdf_reader.PdfReadResult:
        seen.append(data)
        size = len(data) if isinstance(data, bytes) else data.stat().st_size
        return pdf_reader.PdfReadResult(f"PAGE TEXT ({size} bytes)", "ocr", 1)

    monkeypatch.setattr(pdf_reader, "read_pdf_text", fake_read)
    return seen


def _parts_of(per_read: int):
    from openexecutive.providers.feature_gate import ReadingLimits

    return lambda _model="": ReadingLimits(per_read=per_read, per_turn=per_read * 2)


def _error(out: str) -> str:
    return json.loads(out)["error"]


async def test_reads_a_downloaded_scanned_pdf(dirs, fake_pdf) -> None:
    downloads, _ = dirs
    (downloads / "appraisal.pdf").write_bytes(b"%PDF-scan")

    out = await handle_read_document({"path": str(downloads / "appraisal.pdf")})

    assert out == "[appraisal.pdf, converted from scanned pages]\nPAGE TEXT (9 bytes)"
    assert fake_pdf == [downloads / "appraisal.pdf"]


async def test_reads_a_company_document_by_name(dirs) -> None:
    _, docs = dirs
    (docs / "plan.md").write_text("# Plan\nHire two engineers.")

    out = await handle_read_document({"filename": "plan.md"})

    assert out == "[plan.md]\n# Plan\nHire two engineers."


async def test_refuses_files_outside_the_allowed_folders(dirs, tmp_path: Path) -> None:
    downloads, _ = dirs
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")
    link = downloads / "innocent.txt"
    link.symlink_to(secret)

    for path in (str(secret), str(downloads / ".." / "secret.txt"), str(link)):
        out = await handle_read_document({"path": path})
        assert "outside the folders" in _error(out) and "TOP SECRET" not in out


async def test_a_company_filename_cannot_climb_out_of_the_docs_folder(dirs, tmp_path) -> None:
    (tmp_path / "secret.txt").write_text("TOP SECRET")

    for name in ("../../secret.txt", "../secret.txt", ".env"):
        out = await handle_read_document({"filename": name})
        assert "bare document name" in _error(out) and "TOP SECRET" not in out


async def test_needs_a_path_or_a_filename(dirs) -> None:
    assert "pass `path`" in _error(await handle_read_document({}))


async def test_reads_a_page_range_of_a_pdf(dirs, fake_pdf) -> None:
    import io

    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for _ in range(30):
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    downloads, _ = dirs
    (downloads / "deck.pdf").write_bytes(buf.getvalue())

    out = await handle_read_document({"path": str(downloads / "deck.pdf"), "pages": "11-20"})

    assert out.startswith("[deck.pdf, pages 11-20 of 30, converted from scanned pages]")
    (part,) = fake_pdf
    assert len(PdfReader(io.BytesIO(part)).pages) == 10

    past_end = await handle_read_document({"path": str(downloads / "deck.pdf"), "pages": "31"})
    assert _error(past_end) == "that PDF has only 30 pages"
    bad = await handle_read_document({"path": str(downloads / "deck.pdf"), "pages": "ten"})
    assert "pages must look like" in _error(bad)


async def test_an_unreadable_pdf_says_why(dirs, monkeypatch) -> None:
    async def fake_read(data: bytes, *, filename: str = "", inbound: bool = False) -> pdf_reader.PdfReadResult:
        return pdf_reader.PdfReadResult("", "none", 2, "no text could be read: OCR is off")

    monkeypatch.setattr(pdf_reader, "read_pdf_text", fake_read)
    downloads, _ = dirs
    (downloads / "scan.pdf").write_bytes(b"%PDF")

    out = await handle_read_document({"path": str(downloads / "scan.pdf")})

    assert _error(out) == "no text could be read: OCR is off"


async def test_long_text_is_read_in_parts(dirs, monkeypatch) -> None:
    monkeypatch.setattr("openexecutive.providers.registry.reading_limits_for", _parts_of(1000))
    downloads, _ = dirs
    (downloads / "long.txt").write_text("x" * 5000)

    out = await handle_read_document({"path": str(downloads / "long.txt")})

    # The part leaves room for the note, so the turn's own cap never cuts it.
    assert len(out) <= 1000
    assert "call again with start=" in out


async def test_a_long_word_file_is_read_on_in_parts(dirs, monkeypatch) -> None:
    # A long file of any kind, not only a PDF, reads on from where the last
    # part was cut off, so its end never needs a Python job.
    import re

    monkeypatch.setattr("openexecutive.providers.registry.reading_limits_for", _parts_of(1000))
    downloads, _ = dirs
    text = "".join(f"line {i:04d}\n" for i in range(500))
    (downloads / "psa.txt").write_text(text)

    read, start = "", 0
    for _ in range(20):
        out = await handle_read_document({"path": str(downloads / "psa.txt"), "start": start})
        body = out.split("\n", 1)[1]
        body = re.sub(r"^…\[from character \d+ of \d+\]\n", "", body)
        cut = re.search(r"\n…\[cut off at character (\d+) of \d+ — call again with start=(\d+) to read on\]$", body)
        if cut is None:
            read += body
            break
        read += body[: cut.start()]
        start = int(cut.group(2))
    assert read == text

    past = await handle_read_document({"path": str(downloads / "psa.txt"), "start": len(text)})
    assert _error(past) == f"that file has only {len(text)} characters"
    bad = await handle_read_document({"path": str(downloads / "psa.txt"), "start": -1})
    assert "start must be" in _error(bad)


def test_is_registered_on_the_executive_and_withheld_on_private_turns() -> None:
    from openexecutive.orchestrator import executive
    from openexecutive.orchestrator.schedule_tools import (
        PRIVATE_TURN_WITHHELD_TOOLS,
        private_turn_withholds,
    )

    assert READ_DOCUMENT_TOOL in executive._ALL_SKILL_TOOLS
    assert executive._ALL_SKILL_HANDLERS["read_document"] is handle_read_document
    assert "read_document" in PRIVATE_TURN_WITHHELD_TOOLS
    assert private_turn_withholds("read_document", {"path": "/x"})


async def test_a_long_page_range_reads_on_with_the_same_pages(dirs, monkeypatch) -> None:
    # ``start`` counts into the pages read, so the note names them again.
    import io

    from pypdf import PdfWriter

    async def fake_read(data: bytes | Path, *, filename: str = "", inbound: bool = False) -> pdf_reader.PdfReadResult:
        return pdf_reader.PdfReadResult("y" * 5000, "text", 0)

    monkeypatch.setattr(pdf_reader, "read_pdf_text", fake_read)
    monkeypatch.setattr("openexecutive.providers.registry.reading_limits_for", _parts_of(1000))
    writer = PdfWriter()
    for _ in range(30):
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    downloads, _ = dirs
    (downloads / "deck.pdf").write_bytes(buf.getvalue())

    out = await handle_read_document({"path": str(downloads / "deck.pdf"), "pages": "11-20"})

    assert "call again with pages='11-20' and start=" in out


async def test_a_part_is_what_the_turns_own_model_holds(dirs, monkeypatch) -> None:
    """Inside a turn the part follows that turn's model, not the default one."""
    from openexecutive.orchestrator.reading import turn_limits, with_turn_limits
    from openexecutive.providers.feature_gate import ReadingLimits

    monkeypatch.setattr("openexecutive.providers.registry.reading_limits_for", _parts_of(100_000))
    downloads, _ = dirs
    (downloads / "long.txt").write_text("x" * 5000)

    out = await with_turn_limits(
        ReadingLimits(per_read=1000, per_turn=2000), handle_read_document({"path": str(downloads / "long.txt")})
    )

    assert len(out) <= 1000 and "call again with start=" in out
    assert turn_limits() is None  # bound for that call only


async def test_a_part_of_text_written_a_token_a_character_is_a_third_as_long(dirs, monkeypatch) -> None:
    monkeypatch.setattr("openexecutive.providers.registry.reading_limits_for", _parts_of(3000))
    downloads, _ = dirs
    (downloads / "jp.txt").write_text("契約" * 5000)

    out = await handle_read_document({"path": str(downloads / "jp.txt")})

    assert len(out) <= 1000 and "call again with start=" in out


async def test_a_busy_reader_says_to_try_again(dirs, monkeypatch) -> None:
    from openexecutive.knowledge.isolated import ParserBusy

    async def busy(_path):
        raise ParserBusy("busy")

    monkeypatch.setattr("openexecutive.knowledge.loader.read_document_text", busy)
    downloads, _ = dirs
    (downloads / "scan.pdf").write_bytes(b"%PDF")

    out = await handle_read_document({"path": str(downloads / "scan.pdf")})

    assert "busy" in _error(out)
