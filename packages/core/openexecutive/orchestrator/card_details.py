"""What an Approve card shows of the step it holds, in plain words.

A held step keeps the exact call (``payload["input"]``) so Approve does just
that. The card shows the person what it does instead: a plain label for each
field, a file's kind and size in place of its bytes, a link in place of a
Drive id, and none of the plumbing (the Executive's own account, a checksum,
a mail thread id). Every place a card is drawn reads this: the web card on
Today and in the chat (``department_gate.hold``, ``api.routes.decisions``) and
the card in a chat app (``integrations.channel_cards``).
"""
from __future__ import annotations

import contextlib
import json
import re
from datetime import datetime, tzinfo
from typing import Any

from openexecutive.memory import workspace_settings

# A served payload keeps a field's value up to this long; longer (a file's
# bytes) is left out of what the page downloads. Approve reads the stored row.
SERVED_MAX = 20_000

# Who a Google Workspace call runs as: always the Executive's own account,
# which the gateway checks (``_check_acting_account``). For any other app it
# may pick the mailbox, so it is shown.
_GOOGLE_ACCOUNT = frozenset({"user_google_email", "user_email", "account", "account_id", "acting_account"})
# Plumbing: checksums, a file's type (shown with its size), and the ids a
# reply or a draft threads on (``_REFS``, shown only when nothing else is).
_HIDDEN = frozenset({
    "base64_sha256", "sha256", "content_sha256", "checksum",
    "content_mime_type", "mime_type", "mimetype", "content_type",
    "thread_id", "message_id", "in_reply_to", "references", "draft_id", "idempotency_key",
    "request_id", "supports_all_drives",
    # How sure the model was, and whether the owner is invited: not what it does.
    "confidence", "include_principal",
})
# An event's times, shown as one "When" in the company's time zone.
_STARTS = ("start", "start_time")
_ENDS = ("end", "end_time")
# Who a meeting is with, by their People entry.
_PEOPLE_IDS = frozenset({"attendee_person_ids", "person_ids", "participant_person_ids"})
# One person, by their People entry: the field's label says who they are
# to the step. Any other ``*person_id`` reads "Who".
_PERSON_LABELS: dict[str, str] = {
    "owner_person_id": "Owner", "assigned_to_person_id": "Assigned to", "for_person_id": "For",
    "reports_to_person_id": "Reports to",
}
_REFS = frozenset({"thread_id", "message_id", "in_reply_to", "references", "draft_id"})
# Shown only when not the default: an event on another calendar, a file in
# a shared drive.
_SHOWN_UNLESS_DEFAULT: dict[str, frozenset[str]] = {
    "calendar_id": frozenset({"primary"}), "drive_id": frozenset({"root", "my-drive"}),
}

_LABELS: dict[str, str] = {
    "to": "To", "cc": "Cc", "bcc": "Bcc", "subject": "Subject",
    "body": "Message", "text": "Message", "message": "Message", "html_body": "Message",
    "file_name": "File name", "filename": "File name", "name": "Name", "title": "Title",
    "folder_id": "Folder", "parent_folder_id": "Folder", "parent_id": "Folder", "parents": "Folder",
    "file_id": "File", "document_id": "Doc", "spreadsheet_id": "Sheet",
    "presentation_id": "Slides",
    "start": "Start", "end": "End", "start_time": "Start", "end_time": "End",
    "attendees": "Guests", "description": "Description", "location": "Where",
    "attachments": "Attachments", "content": "Content",
}

# A Google Sheets or Docs call's own fields; another app's "range" or
# "start_index" may mean something else (a quarter, a page offset).
_GOOGLE_LABELS: dict[str, str] = {
    "range_name": "Cells", "range": "Cells", "a1_range": "Cells", "values": "Values",
    "find_text": "Find", "replace_text": "Replace with",
    "start_index": "From character", "end_index": "To character", "text": "Text",
}

_FOLDER_KEYS = frozenset({"folder_id", "parent_folder_id", "parent_id", "parents"})
_FILE_KEYS = frozenset({"file_id", "document_id", "spreadsheet_id", "presentation_id"})
# What a Drive link opens, for its text when the file's name isn't known.
_OPENS: dict[str, str] = {
    "spreadsheet_id": "Open the sheet", "document_id": "Open the doc",
    "presentation_id": "Open the slides", "file_id": "Open the file",
}
# A sheet's cells ("Sheet1!A1:F13", "'Q4 Plan'!A:G", "B2").
_RANGE_KEYS = frozenset({"range_name", "range", "a1_range"})
_CELLS = re.compile(r"^(?P<c1>[A-Z]{1,3})?(?P<r1>\d+)?(?::(?P<c2>[A-Z]{1,3})?(?P<r2>\d+)?)?$")

_KINDS: dict[str, str] = {
    "pdf": "PDF", "doc": "Word", "docx": "Word", "xls": "Excel", "xlsx": "Excel", "csv": "CSV",
    "ppt": "PowerPoint", "pptx": "PowerPoint", "png": "Image", "jpg": "Image", "jpeg": "Image",
    "gif": "Image", "webp": "Image", "txt": "Text", "zip": "Zip", "mp3": "Audio", "m4a": "Audio",
    "wav": "Audio", "mp4": "Video", "mov": "Video",
}
_MIME_KINDS: dict[str, str] = {
    "application/pdf": "PDF", "text/csv": "CSV", "text/plain": "Text", "application/zip": "Zip",
}

_BASE64 = re.compile(r"[A-Za-z0-9+/_\-\r\n]+={0,2}")
_DRIVE_ID = re.compile(r"^[A-Za-z0-9_\-]{10,}$")


def _is_bytes(key: str, value: Any) -> bool:
    """Whether ``value`` is a file's bytes (base64), not words to read."""
    # Only what reads as base64 all through: words under a key like "data"
    # are shown, since Approve sends them.
    if not isinstance(value, str) or len(value) < 64 or not _BASE64.fullmatch(value):
        return False
    if "base64" in key.lower() or key.lower() in {"data", "file_data", "file_content", "bytes", "blob"}:
        return True
    return len(value) > 1000


def _size(value: str) -> str:
    raw = len(value.replace("\n", "").replace("\r", "")) * 3 // 4
    if raw >= 1024 * 1024:
        return f"{raw / (1024 * 1024):.1f} MB"
    if raw >= 1024:
        return f"{round(raw / 1024)} KB"
    return f"{raw} bytes"


def _kind(tool_input: dict[str, Any]) -> str:
    for key in ("content_mime_type", "mime_type", "mimetype", "content_type"):
        mime = tool_input.get(key)
        if isinstance(mime, str) and mime:
            mime = mime.split(";")[0].strip().lower()
            if mime in _MIME_KINDS:
                return _MIME_KINDS[mime]
            if mime.startswith("image/"):
                return "Image"
            if "spreadsheet" in mime or "excel" in mime:
                return "Excel"
            if "word" in mime:
                return "Word"
    for key in ("file_name", "filename", "name", "title"):
        name = tool_input.get(key)
        if isinstance(name, str) and "." in name:
            ext = name.rsplit(".", 1)[-1].lower()
            if ext in _KINDS:
                return _KINDS[ext]
    return ""


def _label(key: str) -> str:
    norm = key.strip().lower()
    if norm in _LABELS:
        return _LABELS[norm]
    words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", key).replace("_", " ").replace("-", " ").split()
    return " ".join(words).capitalize() or key


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _word(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return "" if value is None else str(value)


def _lines(value: Any, indent: str = "") -> list[str]:
    """A value with parts (a record, a list of edits) as indented
    "Name: value" lines, never brackets."""
    if _scalar(value):
        return [f"{indent}{_flat(_word(value))}"]
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            if item is None or item == "" or item == [] or item == {}:
                # Often the point of the call (clear a field, replace with
                # nothing): Approve sends it, so the card says so.
                out.append(f"{indent}{_label(str(key))}: (empty)")
                continue
            if _scalar(item) or (isinstance(item, list) and all(_scalar(v) for v in item)):
                out.append(f"{indent}{_label(str(key))}: {_flat(_text(item))}")
            else:
                out.append(f"{indent}{_label(str(key))}:")
                out.extend(_lines(item, indent + "  "))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            sub = _lines(item, indent + "  ")
            if sub:
                out.append(f"{indent}- {sub[0][len(indent) + 2:]}")
                out.extend(sub[1:])
            else:
                out.append(f"{indent}- (empty)")
        return out
    return [f"{indent}{value}"]


def _flat(text: str) -> str:
    """A value inside a record on one line: a line break in it would read
    as another field or item."""
    return text.replace("\r\n", " ⏎ ").replace("\n", " ⏎ ").replace("\r", " ⏎ ")


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if _scalar(value):
        return _word(value)
    if isinstance(value, list) and all(_scalar(v) for v in value):
        return ", ".join(_word(v) for v in value)
    return "\n".join(_lines(value))


def _grid(value: Any) -> list[list[str]] | None:
    """Rows of cells (a sheet's values), as text, or None when ``value``
    isn't rows. A JSON string of rows counts: some apps take it that way."""
    if isinstance(value, str) and value.lstrip().startswith("[["):
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            return None
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(row, list) and all(_scalar(c) for c in row) for row in value):
        return None
    # A cell shows what the sheet will hold: True is TRUE there, not "Yes".
    return [["TRUE" if c is True else "FALSE" if c is False else _word(c) for c in row] for row in value]


def _rows_text(rows: list[list[str]]) -> str:
    """The rows as plain lines, for where a table can't be drawn (a chat
    app): every cell, so Approve does what the card shows."""
    count = f"{len(rows)} row{'' if len(rows) == 1 else 's'}"
    return "\n".join([count, *(" | ".join(map(_cell, row)) for row in rows)])


def _cell(text: str) -> str:
    """One cell on one line: a "|" or a line break inside it would read as
    another cell or row."""
    return _flat(text.replace("|", "\\|"))


def cells(text: str) -> str:
    """A sheet range in words, with the range as written after it:
    "Sheet1!A1:F13" is "Sheet1, rows 1 to 13, columns A to F (A1:F13)";
    "A2:C" is "rows 2 to the end, columns A to C (A2:C)". A range it can't
    read whole (a named range, "A:F10") is shown as written."""
    raw = text.strip()
    tab, _, area = raw.rpartition("!")
    tab = tab.strip()
    if len(tab) >= 2 and tab[0] == tab[-1] == "'":
        tab = tab[1:-1].replace("''", "'")
    area = area.strip()
    found = _CELLS.fullmatch(area) if area else None
    if found is None:
        return raw
    c1, r1, c2, r2 = found["c1"], found["r1"], found["c2"], found["r2"]
    ranged = ":" in area
    words = ""
    if not ranged:
        words = f"cell {c1}{r1}" if c1 and r1 else ""
    elif c1 and r1 and c2 and r2:
        rows = f"row {r1}" if r1 == r2 else f"rows {r1} to {r2}"
        cols = f"column {c1}" if c1 == c2 else f"columns {c1} to {c2}"
        words = f"{rows}, {cols}"
    elif c1 and r1 and c2 and not r2:
        cols = f"column {c1}" if c1 == c2 else f"columns {c1} to {c2}"
        words = f"rows {r1} to the end, {cols}"
    elif c1 and c2 and not r1 and not r2:
        words = f"all rows, {'column ' + c1 if c1 == c2 else f'columns {c1} to {c2}'}"
    elif r1 and r2 and not c1 and not c2:
        words = f"{'row ' + r1 if r1 == r2 else f'rows {r1} to {r2}'}, all columns"
    if not words:
        return raw
    shown = f"{tab}, {words} ({area})" if tab else f"{words} ({area})"
    return shown[:1].upper() + shown[1:]


def _drive_link(key: str, value: Any) -> str | None:
    ident = value[0] if isinstance(value, list) and len(value) == 1 else value
    if not isinstance(ident, str) or not _DRIVE_ID.match(ident):
        return None
    if key in _FOLDER_KEYS:
        return f"https://drive.google.com/drive/folders/{ident}"
    return f"https://drive.google.com/open?id={ident}"


def file_name(file_id: str) -> str:
    """A Drive file's name, from what Drive's own metadata called it, or ""
    when nothing has. Never raises."""
    if not file_id:
        return ""
    try:
        from openexecutive.memory import drive_reads

        return drive_reads.name_of(file_id)
    except Exception:
        return ""


def _moment(value: Any, zone: tzinfo) -> datetime | None:
    if not isinstance(value, str) or "T" not in value:
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    # A time with no offset is already the company's local time.
    return dt.astimezone(zone) if dt.tzinfo is not None else dt


def _clock(dt: datetime) -> str:
    return f"{dt.hour % 12 or 12}:{dt.minute:02d}"


def when(start: Any, end: Any = None, zone: tzinfo | None = None) -> str | None:
    """``Tue Nov 18, 11:00 to 11:30 AM`` in the company's time zone, or
    None when ``start`` isn't a date and time."""
    zone = zone or workspace_settings.get_user_timezone()
    first = _moment(start, zone)
    if first is None:
        return None
    day = f"{first:%a} {first:%b} {first.day}"
    last = _moment(end, zone)
    if last is None or last <= first:
        return f"{day}, {_clock(first)} {first:%p}"
    if last.date() != first.date():
        return f"{day}, {_clock(first)} {first:%p} to {last:%a} {last:%b} {last.day}, {_clock(last)} {last:%p}"
    if f"{first:%p}" == f"{last:%p}":
        return f"{day}, {_clock(first)} to {_clock(last)} {last:%p}"
    return f"{day}, {_clock(first)} {first:%p} to {_clock(last)} {last:%p}"


def _names(value: Any) -> str | None:
    """The People names for a list of person ids, or None."""
    ids = value if isinstance(value, list) else [value]
    try:
        from openexecutive.people import registry

        names = []
        for raw in ids:
            person = registry.get_person(int(raw), include_contacts=True)
            if person is None:
                return None
            names.append(person.full_name or person.email or "someone")
    except Exception:
        return None
    return ", ".join(names) if names else None


_STYLES = (("bold", "bold"), ("italic", "italic"), ("underline", "underlined"))


def doc_change(tool: str, tool_input: dict[str, Any]) -> str:
    """What a Doc edit does, in a line ("Replace characters 10 to 24 with
    the text below", "Make characters 1 to 12 bold"), or ""."""
    if not tool.endswith("modify_doc_text"):
        return ""
    start, end = tool_input.get("start_index"), tool_input.get("end_index")
    span = (
        f"characters {start} to {end}" if start is not None and end is not None
        else f"character {start}" if start is not None
        else f"up to character {end}" if end is not None else "the doc"
    )
    text = tool_input.get("text")
    if isinstance(text, str) and text:
        if end is not None:
            return f"Replace {span} with the text below"
        return f"Add the text below at {span}" if start is not None else "Add the text below"
    looks = [word for key, word in _STYLES if tool_input.get(key) is True]
    looks += [f"not {word}" for key, word in _STYLES if tool_input.get(key) is False]
    if tool_input.get("font_size"):
        looks.append(f"size {tool_input['font_size']}")
    if tool_input.get("font_family"):
        looks.append(f"in {tool_input['font_family']}")
    if not looks:
        return ""
    return f"Make {span} " + (", ".join(looks[:-1]) + " and " + looks[-1] if len(looks) > 1 else looks[0])


def plain(tool: str, tool_input: Any) -> list[dict[str, Any]]:
    """Each field of the call worth showing, as ``{"field", "value"}`` in
    plain words: never a file's bytes, an internal name or the plumbing."""
    if not isinstance(tool_input, dict):
        return []
    google = isinstance(tool, str) and tool.startswith("google_workspace__")
    # One of our own tools, not a connected app's ("server__tool").
    own = isinstance(tool, str) and "__" not in tool
    out: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()

    def label(key: str) -> str:
        own = _GOOGLE_LABELS.get(key.strip().lower()) if google else None
        return own or _label(key)

    start_key = next((k for k in _STARTS if k in tool_input), None)
    end_key = next((k for k in _ENDS if k in tool_input), None)
    shown_when = when(tool_input.get(start_key), tool_input.get(end_key)) if start_key else None
    for key, value in tool_input.items():
        key = str(key)
        norm = key.strip().lower()
        if shown_when is not None and norm in (start_key, end_key):
            if norm == start_key:
                out.append({"field": "When", "value": shown_when})
                seen.add("When")
            continue
        if norm in _PEOPLE_IDS and value not in (None, "", [], {}):
            names = _names(value)
            if names is not None:
                out.append({"field": "With", "value": names})
                seen.add("With")
                continue
        if own:
            # Our own tools name people, departments and goals by their
            # number: the card shows the name, and never a bare number.
            named = _own_ref(tool, norm, value)
            if named is not None:
                if named:
                    field = named[0] if named[0] not in seen else f"{named[0]} ({key.replace('_', ' ')})"
                    out.append({"field": field, "value": named[1]})
                    seen.add(field)
                elif value not in (None, "", 0, [], {}):
                    # Shown only when the call names nothing else.
                    refs.append({"field": label(key), "value": _text(value)})
                continue
        if norm in _REFS and value not in (None, "", [], {}):
            refs.append({"field": label(key), "value": _text(value)})
            continue
        if google and norm == "replace_text" and value == "":
            out.append({"field": label(key), "value": "(nothing: it deletes what it finds)"})
            continue
        if (google and norm in _GOOGLE_ACCOUNT) or norm in _HIDDEN or value is None or value == "" or value == [] or value == {}:
            continue
        if norm in _SHOWN_UNLESS_DEFAULT and str(value).strip().lower() in _SHOWN_UNLESS_DEFAULT[norm]:
            continue
        extra: dict[str, Any] = {}
        # A JSON string of rows only counts for a sheet's values: anywhere
        # else (a mail body) it is sent as written, so it shows as written.
        rows = _grid(value) if isinstance(value, list) or (google and norm == "values") else None
        if _is_bytes(key, value):
            kind = _kind(tool_input)
            field, text = "File", f"{kind + ' file' if kind else 'A file'}, {_size(value)}"
        elif google and (norm in _FOLDER_KEYS or norm in _FILE_KEYS) and _drive_link(norm, value):
            # The file by its name, linked; a bare link only says "open it".
            link = str(_drive_link(norm, value))
            ident = value[0] if isinstance(value, list) else value
            name = file_name(str(ident))
            field, text = label(key), name or _OPENS.get(norm, "Open the folder")
            extra = {"href": link}
        elif google and norm in _RANGE_KEYS and isinstance(value, str):
            field, text = label(key), cells(value)
        elif rows is not None:
            # A sheet's rows: a table on the page, every cell as a line where
            # a table can't be drawn.
            field, text = label(key), _rows_text(rows)
            # The first row is a header only when the rows start at row 1.
            area = next((tool_input[k] for k in _RANGE_KEYS if isinstance(tool_input.get(k), str)), "")
            start = _CELLS.fullmatch(str(area).rpartition("!")[2].strip())
            header = start is not None and start["r1"] == "1"
            extra = {"table": rows, "header": header}
        else:
            # Words are shown whole: Approve does exactly what the card shows.
            field, text = label(key), _text(value)
        if field in seen:
            field = f"{field} ({key.replace('_', ' ')})"
        seen.add(field)
        out.append({"field": field, "value": text, **extra})
    change = doc_change(tool, tool_input) if google else ""
    if change:
        # The line says which characters: the bare positions go.
        out = [d for d in out if d["field"] not in ("From character", "To character")]
        out.insert(0, {"field": "Change", "value": change})
    # A call that names only what it acts on (send this draft, delete this
    # message) still shows that, never an empty card.
    return out or refs


def _own_ref(tool: str, norm: str, value: Any) -> tuple[str, str] | tuple[()] | None:
    """For one of our own tools' reference fields, the (label, name) the
    card shows: a person, a department or a goal by name. ``()`` when the
    field is a record's number only the system reads (a task's, a card's): it
    shows only on a card that names nothing else. None for any other field."""
    from openexecutive.orchestrator import take_the_lead

    target = take_the_lead._TARGET_KEYS.get(tool)
    if target is not None and target[1] != "person" and norm == target[0]:
        # A chat app's own id for who it goes to: their name when they're
        # on People, else the id, the only way to tell who it reaches.
        if value in (None, ""):
            return ()
        person = take_the_lead._target_person(tool, {target[0]: value})
        return ("To", (person.full_name or person.email or str(value)) if person is not None else str(value))
    if norm.endswith("person_ids"):
        ids = value if isinstance(value, list) else [value]
        if not [i for i in ids if i not in (None, "")]:
            return ()
        # Each by name; one People doesn't have still counts, so the card
        # shows everyone Approve reaches.
        each = [_names(i) or ("someone not on People" if _record_number(i) else _text(i)) for i in ids]
        return ("With", ", ".join(each))
    if norm.endswith("person_id"):
        if value in (None, "", 0):
            return ()
        label = "To" if tool == "message_person" and norm == "person_id" else _PERSON_LABELS.get(norm, "Who")
        if tool == "assign_open_loop" and norm == "person_id":
            label = "Assigned to"
        names = _names(value)
        if names:
            return (label, names)
        # A number People doesn't have; anything else is shown as it is.
        return (label, "Someone not on People" if _record_number(value) else _text(value))
    if norm == "department_slug":
        return ("Department", _department(value)) if value else ()
    if norm in ("goal_id", "for_goal_id"):
        goal = _goal(value)
        return ("Goal", goal) if goal else ()
    if (norm.endswith("_id") or norm.endswith("_ids")) and _record_number(value):
        return ()
    return None


def _record_number(value: Any) -> bool:
    """Whether a value is one of our own records' numbers (a task's, a
    card's): never a yes/no or an account a step sets, which show."""
    values = value if isinstance(value, list) else [value]
    return bool(values) and all(isinstance(v, int) and not isinstance(v, bool) for v in values)


def _department(slug: Any) -> str:
    """A department's name, or its slug in words when it isn't found."""
    try:
        from openexecutive.departments.store import get_department

        dept = get_department(str(slug))
        if dept is not None and dept.config.title:
            return dept.config.title
    except Exception:
        pass
    return str(slug).replace("_", " ").replace("-", " ").strip().capitalize()


def _goal(goal_id: Any) -> str:
    """A goal's key result, or "" when there's none by that number."""
    try:
        from openexecutive.departments.store import get_goal

        goal = get_goal(int(goal_id)) if int(goal_id) else None
    except Exception:
        return ""
    return goal.key_result if goal is not None else ""


def held_steps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Every call a held card runs on Approve: its ``steps`` when one card
    holds several (levels.hold_many), else its one ``input``."""
    steps = payload.get("steps")
    if isinstance(steps, list) and len(steps) > 1 and all(isinstance(s, dict) for s in steps):
        return steps
    tool_input = payload.get("input")
    return [tool_input] if isinstance(tool_input, dict) else []


def plain_steps(tool: str, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``plain`` for each call one card holds, numbered when there are
    several, so the card shows everything Approve does."""
    if len(steps) == 1:
        return plain(tool, steps[0])
    return [
        {**detail, "field": f"{n}. {detail['field']}"}
        for n, step in enumerate(steps, 1)
        for detail in plain(tool, step)
    ]


def _file_of(tool_input: dict[str, Any]) -> str:
    for key in ("spreadsheet_id", "document_id", "presentation_id", "file_id"):
        value = tool_input.get(key)
        if isinstance(value, str) and _DRIVE_ID.match(value):
            return value
    return ""


def _named(payload: dict[str, Any], tool_input: dict[str, Any]) -> dict[str, str]:
    """The card's title and summary with the file's name in place of "a
    spreadsheet" ("Update Launch tracker (Google Sheets)"), when the step
    names a Drive file only by its id and its name is known; {} otherwise.
    Only a title the card would have made from the call itself is
    rewritten."""
    tool = str(payload.get("tool") or "")
    if not payload.get("mcp") or not tool.startswith("google_workspace__") or payload.get("steps"):
        return {}
    from openexecutive.orchestrator import take_the_lead

    if take_the_lead._first_text(tool_input, take_the_lead._MCP_NAME_KEYS):
        return {}
    name = file_name(_file_of(tool_input))
    if not name:
        return {}
    named_input = {**tool_input, "title": name}
    out: dict[str, str] = {}
    for key in ("headline", "summary"):
        shown = payload.get(key)
        if not isinstance(shown, str) or not shown:
            continue
        for private in (True, False):
            try:
                if shown == take_the_lead.title(tool, tool_input, mcp=True, private=private):
                    out[key] = take_the_lead.title(tool, named_input, mcp=True, private=private)
                    break
                for quote in (True, False):
                    if shown == take_the_lead.summarize(tool, tool_input, mcp=True, quote=quote):
                        out[key] = take_the_lead.summarize(tool, named_input, mcp=True, quote=quote)
                        break
            except Exception:
                continue
            if key in out:
                break
    return out


def _why(payload: dict[str, Any], steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For a step that waited because it's about money, the words that
    matched, so the card shows why ("…Growth $200 a month…")."""
    if payload.get("kind") != "money":
        return []
    from openexecutive.orchestrator import take_the_lead

    text = "\n".join(t for step in steps for t in take_the_lead._strings(step))
    quote = take_the_lead.money_quote(text)
    return [{"field": "The amount it saw", "value": f"“{quote}”"}] if quote else []


def served(payload: dict[str, Any]) -> dict[str, Any]:
    """A held step's payload as a page reads it: its details in plain words
    (cards held before this read raw), its title naming the file it works
    on, and no field long enough to be a file's bytes. Approve never reads
    this copy."""
    tool_input = payload.get("input")
    if not isinstance(tool_input, dict):
        return payload
    out = dict(payload)
    steps = held_steps(payload)
    out["details"] = plain_steps(str(payload.get("tool") or ""), steps) + _why(payload, steps)
    with contextlib.suppress(Exception):
        out.update(_named(payload, tool_input))

    def short(step: dict[str, Any]) -> dict[str, Any]:
        return {k: ("" if isinstance(v, str) and len(v) > SERVED_MAX else v) for k, v in step.items()}

    out["input"] = short(tool_input)
    if isinstance(payload.get("steps"), list):
        out["steps"] = [short(s) if isinstance(s, dict) else s for s in payload["steps"]]
    return out


def _first_name(full: str) -> str:
    return (full or "").strip().split(" ")[0] if (full or "").strip() else ""


def top(payload: dict[str, Any], *, approver_id: int | None, viewer_id: int | None) -> dict[str, str]:
    """The lines every held step's card opens with, the same on every app
    (web, phone, chat apps):

    - ``waiting_on``: whose OK it waits for, when that isn't the reader
      ("Alex"); empty when it's theirs or the reader isn't known ("Needs
      your OK").
    - ``started``: who started it: "You asked", "Alex asked", or "I started
      this" when no one on the team asked (Take the lead's own work).
    - ``waits``: why it waits, in a sentence.
    - ``title``: the card's title, naming the file it works on (``served``).
    - ``because``: why it waits in a few words ("it's about money"), for a
      one-line row (Home's Waiting on list); may be empty.
    """
    from openexecutive.people.store import get_person

    def name(person_id: object) -> str:
        if not isinstance(person_id, int):
            return ""
        with contextlib.suppress(Exception):
            person = get_person(person_id)
            if person is not None:
                return _first_name(person.full_name)
        return ""

    asker_id = payload.get("requested_by_person_id") or payload.get("asked_by_person_id")
    if isinstance(asker_id, int) and asker_id == viewer_id:
        started = "You asked"
    elif isinstance(asker_id, int) or payload.get("requested_by"):
        who = name(asker_id) or _first_name(str(payload.get("requested_by") or ""))
        started = f"{who} asked" if who else "Someone on the team asked"
    elif payload.get("source") == "chat":
        # Asked in a conversation by someone it can't name.
        started = "Someone asked"
    else:
        started = "I started this"
    # Unknown reader (a build with no one signed in): say it plainly, as
    # before, rather than name someone who may be the reader.
    waiting_on = name(approver_id) if None not in (approver_id, viewer_id) and approver_id != viewer_id else ""
    reason = " ".join(str(payload.get("reason") or "").split()).removesuffix(".")
    because = "it's a new kind of step" if payload.get("kind") == "training" else reason
    if payload.get("kind") == "training":
        waits = "It's a new kind of step for me, so it waits for an OK once."
    elif reason:
        waits = f"It waits because {reason}."
    else:
        waits = "It waits for an OK."
    if waiting_on:
        waits += f" It's on {waiting_on}'s list."
    title = ""
    with contextlib.suppress(Exception):
        title = str(served(payload).get("headline") or "")
    return {"waiting_on": waiting_on, "started": started, "waits": waits, "title": title, "because": because}


def held_top(decision_id: int, viewer_id: int | None) -> dict[str, str] | None:
    """``top`` for a held step's card by its decision, or None when it
    can't be read."""
    from openexecutive.memory.decision_ledger import get_decision_instance

    try:
        instance = get_decision_instance(decision_id)
        payload = json.loads(instance.proposed_payload_json or "") if instance is not None else None
    except Exception:
        return None
    if not isinstance(payload, dict) or instance is None:
        return None
    return top(payload, approver_id=instance.approver_person_id, viewer_id=viewer_id)
