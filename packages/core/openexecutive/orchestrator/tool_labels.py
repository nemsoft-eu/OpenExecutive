"""Plain-language names for connected (MCP) tools, shared by the chat's chips
and its progress line.

A connected tool reaches the Executive as `call_tool` wrapping a raw name such
as `google_workspace__search_drive_files` or `microsoft_365__get-mail-message`.
Those names mean nothing to the person reading the chat, so both
`action_chips` (the finished chip, past tense) and `activity_labels` (the
in-flight line, present tense) read their wording from here. Keeping the two
tenses in one table is what keeps "Searching Drive…" and "Searched Drive"
saying the same thing.

A tool missing from `_LABELS` is not shown raw: `labels_for` turns its name
into words ("list_task_lists" on Google -> "Listed task lists · Google").
When a connected server gains a tool people will see often, add a row here.

`detail_for` names what one call looked at (the search words, the file or
email it opened). The chat lists these when a chip is tapped, so a chip that
collapses several runs still says what each one did.
"""
from __future__ import annotations

import json
import re
from typing import Any

from openexecutive.memory.drive_reads import parse_content_result

# Whose mailbox a mail read is in, said one way everywhere: "my email" is the
# Executive's own mailbox (the connected Gmail or Outlook tools below), "your
# email" is the person's own (Act as me's reads, `activity_labels`). Every
# search, read and attachment label for either takes its words from here, so
# the two can never trade places.
MY_MAIL = "my"
YOUR_MAIL = "your"
_MAIL_ACTIONS: dict[str, tuple[str, str]] = {
    "search": ("Searched {} email", "Searching {} email"),
    "read": ("Read {} email", "Reading {} email"),
    "attachment": ("Opened an attachment in {} email", "Opening an attachment in {} email"),
    "attachments": ("Checked attachments in {} email", "Checking attachments in {} email"),
}


def mail_labels(whose: str, action: str) -> tuple[str, str]:
    """(done, doing) for a mail search, read or attachment in `whose` mailbox
    (`MY_MAIL` or `YOUR_MAIL`)."""
    done, doing = _MAIL_ACTIONS[action]
    return done.format(whose), doing.format(whose)


_MY_SEARCH = mail_labels(MY_MAIL, "search")
_MY_READ = mail_labels(MY_MAIL, "read")
_MY_ATTACHMENT = mail_labels(MY_MAIL, "attachment")

# Underlying tool name (without its `server__` prefix, hyphens folded to
# underscores) -> (done, doing). `doing` carries no ellipsis; the progress line
# adds it. Google Workspace and Microsoft 365 names share the table: the two
# servers do not reuse a name for different things.
_LABELS: dict[str, tuple[str, str]] = {
    # Mail. These connected tools are the Executive's own mailbox: "my email".
    "search_gmail_messages": _MY_SEARCH,
    "list_mail_messages": _MY_SEARCH,
    "list_mail_folder_messages": _MY_SEARCH,
    "get_gmail_message_content": _MY_READ,
    "get_gmail_messages_content_batch": _MY_READ,
    "get_gmail_thread_content": _MY_READ,
    "get_gmail_threads_content_batch": _MY_READ,
    "get_mail_message": _MY_READ,
    "get_gmail_attachment_content": _MY_ATTACHMENT,
    "get_mail_attachment": _MY_ATTACHMENT,
    "list_mail_attachments": mail_labels(MY_MAIL, "attachments"),
    "draft_gmail_message": ("Drafted an email", "Drafting an email"),
    "create_draft_email": ("Drafted an email", "Drafting an email"),
    "create_reply_draft": ("Drafted a reply", "Drafting a reply"),
    "create_reply_all_draft": ("Drafted a reply", "Drafting a reply"),
    "create_forward_draft": ("Drafted a forward", "Drafting a forward"),
    "send_gmail_message": ("Sent an email", "Sending an email"),
    "send_mail": ("Sent an email", "Sending an email"),
    "send_draft_message": ("Sent an email", "Sending an email"),
    "reply_mail_message": ("Sent a reply", "Sending a reply"),
    "reply_all_mail_message": ("Sent a reply", "Sending a reply"),
    "forward_mail_message": ("Forwarded an email", "Forwarding an email"),
    "move_mail_message": ("Filed an email", "Filing an email"),
    "update_mail_message": ("Updated an email", "Updating an email"),
    "delete_mail_message": ("Deleted an email", "Deleting an email"),
    "modify_gmail_message_labels": ("Labeled an email", "Labeling an email"),
    "list_gmail_labels": ("Checked my email folders", "Checking my email folders"),
    # Files
    "search_drive_files": ("Searched Drive", "Searching Drive"),
    "list_drive_items": ("Searched Drive", "Searching Drive"),
    "get_drive_file_content": ("Read a Drive file", "Reading a Drive file"),
    "create_drive_file": ("Saved a file to Drive", "Saving a file to Drive"),
    "search_onedrive_files": ("Searched OneDrive", "Searching OneDrive"),
    "list_drives": ("Checked OneDrive", "Checking OneDrive"),
    "list_folder_files": ("Searched OneDrive", "Searching OneDrive"),
    "get_drive_item": ("Read a OneDrive file", "Reading a OneDrive file"),
    "download_onedrive_file_content": ("Read a OneDrive file", "Reading a OneDrive file"),
    "upload_file_content": ("Saved a file to OneDrive", "Saving a file to OneDrive"),
    "copy_drive_item": ("Copied a file", "Copying a file"),
    "move_rename_onedrive_item": ("Moved a file", "Moving a file"),
    "share_drive_item": ("Shared a file", "Sharing a file"),
    # Docs and sheets
    "search_docs": ("Searched my docs", "Searching my docs"),
    "get_doc_content": ("Read a doc", "Reading a doc"),
    "create_doc": ("Created a doc", "Creating a doc"),
    "modify_doc_text": ("Edited a doc", "Editing a doc"),
    "read_sheet_values": ("Read a spreadsheet", "Reading a spreadsheet"),
    "get_spreadsheet_info": ("Read a spreadsheet", "Reading a spreadsheet"),
    "list_spreadsheets": ("Searched my spreadsheets", "Searching my spreadsheets"),
    "modify_sheet_values": ("Updated a spreadsheet", "Updating a spreadsheet"),
    "create_spreadsheet": ("Created a spreadsheet", "Creating a spreadsheet"),
    # Calendar
    "list_calendars": ("Checked my calendar", "Checking my calendar"),
    "get_events": ("Checked my calendar", "Checking my calendar"),
    "list_calendar_events": ("Checked my calendar", "Checking my calendar"),
    "get_calendar_event": ("Checked my calendar", "Checking my calendar"),
    "get_calendar_view": ("Checked my calendar", "Checking my calendar"),
    "query_freebusy": ("Checked availability", "Checking availability"),
    "find_meeting_times": ("Checked availability", "Checking availability"),
    "manage_event": ("Updated my calendar", "Updating my calendar"),
    "create_calendar_event": ("Updated my calendar", "Updating my calendar"),
    "update_calendar_event": ("Updated my calendar", "Updating my calendar"),
    "update_specific_calendar_event": ("Updated my calendar", "Updating my calendar"),
    "delete_calendar_event": ("Removed a calendar event", "Removing a calendar event"),
    "accept_calendar_event": ("Answered an invite", "Answering an invite"),
    "decline_calendar_event": ("Answered an invite", "Answering an invite"),
    "tentatively_accept_calendar_event": ("Answered an invite", "Answering an invite"),
    "forward_calendar_event": ("Forwarded an invite", "Forwarding an invite"),
}

# Server prefix -> the name people know it by, for the fallback's suffix.
_SERVICES: dict[str, str] = {
    "google_workspace": "Google",
    "microsoft_365": "Microsoft 365",
    "oe": "Open Executive",
}

# First word of an unlisted tool -> (done, doing) verb for the fallback.
_VERBS: dict[str, tuple[str, str]] = {
    "search": ("Searched", "Searching"),
    "find": ("Looked up", "Looking up"),
    "query": ("Looked up", "Looking up"),
    "get": ("Read", "Reading"),
    "read": ("Read", "Reading"),
    "fetch": ("Read", "Reading"),
    "download": ("Read", "Reading"),
    "list": ("Listed", "Listing"),
    "create": ("Created", "Creating"),
    "add": ("Added", "Adding"),
    "update": ("Updated", "Updating"),
    "modify": ("Updated", "Updating"),
    "edit": ("Edited", "Editing"),
    "set": ("Updated", "Updating"),
    "delete": ("Deleted", "Deleting"),
    "remove": ("Removed", "Removing"),
    "send": ("Sent", "Sending"),
    "reply": ("Replied to", "Replying to"),
    "forward": ("Forwarded", "Forwarding"),
    "draft": ("Drafted", "Drafting"),
    "move": ("Moved", "Moving"),
    "share": ("Shared", "Sharing"),
    "upload": ("Uploaded", "Uploading"),
    "copy": ("Copied", "Copying"),
}

# A mail search that looks only at sent mail says so: "my email" would hide
# that it is checking what was sent.
_MAIL_SEARCHES = frozenset({
    "search_gmail_messages", "list_mail_messages", "list_mail_folder_messages",
})
_SENT_LABELS = ("Checked email I sent", "Checking email I sent")
# Gmail's sent operators, and Microsoft 365's Sent Items folder by name.
_SENT_RE = re.compile(r"(?<![\w-])(?:in:sent|label:sent|from:me)(?=[\s)]|$)|^sent ?items$", re.IGNORECASE)
# A query that also reaches other mail ("in:sent OR in:inbox", "{…}", a
# "NOT", another folder or label, or excluding sent) stays
# "my email": a wrong "email I sent" is worse than the plain line.
_ANY_OF_RE = re.compile(
    r"\sOR\s|\bNOT\b|[{}]|(?<![\w-])(?:in|label):(?!sent(?=[\s)]|$))|(?<![\w])-(?:in|label):sent",
    re.IGNORECASE,
)

GENERIC_DONE = "Used a connected tool"
GENERIC_DOING = "Using a connected tool"

# Tool names and argument values are model- or server-authored text on their
# way to the DOM: keep them to one short, plain line.
_WORDS_MAX = 40
_DETAIL_MAX = 60
_UNSAFE = re.compile(r"[^\w .,:;'&@()/#+-]", re.UNICODE)

# Argument keys that name what a call looked for or at, most telling first.
_DETAIL_KEYS = (
    "query", "q", "search", "search_query", "subject", "title",
    "file_name", "filename", "name", "summary",
)
_SUBJECT_RE = re.compile(r"^Subject:[ \t]*(.+)$", re.MULTILINE)
# The tools whose result is one email, the only ones a subject is read from.
_MAIL_READS = frozenset({
    "get_gmail_message_content",
    "get_gmail_thread_content",
    "get_mail_message",
})
# Larger results are not parsed just to find a subject.
_JSON_MAX = 200_000


def _split(raw: str) -> tuple[str, str]:
    """`server__tool` -> (server, normalized tool)."""
    server, sep, tool = raw.partition("__")
    if not sep:
        server, tool = "", raw
    return server, tool.replace("-", "_").lower()


def _clean(text: str, limit: int) -> str:
    text = _UNSAFE.sub("", " ".join(text.split())).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _searches_sent(arguments: Any) -> bool:
    """Whether a mail search's arguments limit it to sent mail."""
    if not isinstance(arguments, dict):
        return False
    texts = [value for value in arguments.values() if isinstance(value, str)]
    if any(_ANY_OF_RE.search(text) for text in texts):
        return False
    # A folder argument that names another folder ("Inbox") reaches other mail.
    for key, value in arguments.items():
        if (
            isinstance(value, str) and "folder" in key.lower() and value.strip()
            and not _SENT_RE.search(value.strip())
        ):
            return False
    return any(_SENT_RE.search(text.strip()) for text in texts)


def labels_for(raw_name: Any, arguments: Any = None) -> tuple[str, str]:
    """(done, doing) phrases for an MCP tool name. Never returns the raw name.

    `arguments` matters only for a tool whose action depends on them, such
    as Google's `manage_event`, which also deletes: a deletion must not read
    as "Updated my calendar". A mail search limited to sent mail reads as
    "Checking email I sent".
    """
    if not isinstance(raw_name, str) or not raw_name.strip():
        return GENERIC_DONE, GENERIC_DOING
    server, tool = _split(raw_name.strip())
    if (
        tool == "manage_event"
        and isinstance(arguments, dict)
        and str(arguments.get("action", "")).lower() == "delete"
    ):
        return _LABELS["delete_calendar_event"]
    if tool in _MAIL_SEARCHES and _searches_sent(arguments):
        return _SENT_LABELS
    if tool in _LABELS:
        return _LABELS[tool]
    words = [w for w in tool.split("_") if w]
    if not words:
        return GENERIC_DONE, GENERIC_DOING
    verb = _VERBS.get(words[0])
    rest = " ".join(words[1:] if verb else words)
    rest = _clean(rest, _WORDS_MAX)
    service = _SERVICES.get(server) or _clean(server.replace("_", " ").title(), 24)
    suffix = f" · {service}" if service else ""
    if verb and rest:
        return f"{verb[0]} {rest}{suffix}", f"{verb[1]} {rest}{suffix}"
    if rest:
        return f"Used {rest}{suffix}", f"Using {rest}{suffix}"
    return GENERIC_DONE, GENERIC_DOING


def detail_for(raw_name: Any, arguments: Any, result: str) -> str | None:
    """What one call looked at, in a few words, or None when nothing says."""
    _, tool = _split(raw_name) if isinstance(raw_name, str) else ("", "")
    if tool == "get_drive_file_content":
        opened = parse_content_result(result) if isinstance(result, str) else None
        if opened and opened.get("name"):
            return _clean(opened["name"], _DETAIL_MAX) or None
    if isinstance(arguments, dict):
        for key in _DETAIL_KEYS:
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                cleaned = _clean(value, _DETAIL_MAX)
                if cleaned:
                    return f'"{cleaned}"' if key in ("query", "q", "search", "search_query") else cleaned
    if tool in _MAIL_READS and isinstance(result, str) and result:
        # Only an email's own header block names it: a "Subject:" line in a
        # body or forwarded text must not become the chip's target.
        header = result[:4000].split("\n\n", 1)[0]
        m = _SUBJECT_RE.search(header)
        if m:
            return _clean(m.group(1), _DETAIL_MAX) or None
        parsed = None
        if len(result) <= _JSON_MAX:
            try:
                parsed = json.loads(result)
            except (ValueError, TypeError):
                parsed = None
        if isinstance(parsed, dict):
            value = parsed.get("subject")
            if isinstance(value, str) and value.strip():
                return _clean(value, _DETAIL_MAX) or None
    return None
