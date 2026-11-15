"""The content guard for a progress update: validate raw text, then clean it.

An update is posted by another agent and relayed verbatim to the founders'
chat apps, so what the guard rejects stays out of the store as well as out of
Slack. It rejects instead of redacting: the caller gets the field and the
reason and can rephrase.

The order matters. Every check runs on text nothing has cleaned yet, so
cleaning cannot hide what the guard must reject:

1. Any control, format or invisible character (``NFKC(raw) != plain(raw)``).
   A CRLF line break is the one thing turned into ``\\n`` first; it hides
   nothing, while a lone ``\\r`` is still rejected.
2. Line checks on the raw text: any line break other than ``\\n`` (U+2028
   and U+2029, which a chat client may render as one), a line break in a
   one-line field, more than ``SUMMARY_MAX_LINES`` lines in the summary.
3. The deny patterns, on the NFKC text (a full-width ＜ is a <).
4. Cleaning (runs of spaces and tabs collapsed, each line trimmed, line breaks
   kept) and then the length limits on the cleaned form.

This is a tripwire, not the control. It cannot catch an amount or a price
(amounts are accepted by design: whether a price is published is the
declared content contract's job), people's names, health or HR detail,
indirect identifiers, or anything paraphrased.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from openexecutive.utils.prompt_blocks import plain

STATUSES = ("on_track", "at_risk", "blocked", "done", "info")

TOPIC_MIN, TOPIC_MAX = 3, 80
SUMMARY_MIN, SUMMARY_MAX = 10, 800
SUMMARY_MAX_LINES = 8
ASK_MAX = 300

FIELDS = ("topic", "status", "summary", "ask", "link", "idempotency_key")

# a caller's own retry key: long enough to be unique, short and plain enough
# to index and log
IDEMPOTENCY_KEY_RE = re.compile(r"[A-Za-z0-9_-]{8,64}")

# an email address
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
# a phone number: international (+32 470 12 34 56), national with a trunk
# zero (0470 12 34 56, (020) 123 4567) or North American (555-123-4567);
# a match counts only with PHONE_MIN_DIGITS digits, so dates, times and
# amounts do not trip it
PHONE_RE = re.compile(
    r"(?<![\w+])(?:\+\d{1,3}|\(0?\d{1,4}\)|0\d{0,4})(?:[ .\-/]?\(?\d{1,4}\)?){2,6}(?![\w:])"
    r"|(?<!\w)\d{3}[ .-]\d{3}[ .-]\d{4}(?![\w:])"
)
PHONE_MIN_DIGITS = 9
# an IBAN: country code, check digits and 11-30 more characters, in groups
# of four or not. The shape can run on into the next upper-case word, so a
# match counts when some prefix of it of an IBAN's length holds
# IBAN_MIN_DIGITS digits and a valid mod-97 checksum; capitals in prose
# ("FY26 PLAN") do not carry that many digits
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,40}")
IBAN_MIN_DIGITS = 10
# a credential: well-known token prefixes (Anthropic/OpenAI, Slack, GitHub,
# GitLab, AWS, Google), a JWT, a PEM private key, or a secret-named setting
# with a value
CREDENTIAL_RE = re.compile(
    r"\bsk-(?:ant-)?[A-Za-z0-9_-]{16,}"
    r"|\bxox[abposr]-[A-Za-z0-9-]{10,}"
    r"|\bxapp-[A-Za-z0-9-]{10,}"
    r"|\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"
    r"|\bgithub_pat_[A-Za-z0-9_]{20,}"
    r"|\bglpat-[A-Za-z0-9_-]{16,}"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|\bAIza[0-9A-Za-z_-]{35}"
    r"|\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\."
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|\b(?:password|passwd|api[_ -]?key|access[_ -]?key|secret[_ -]?key|client[_ -]?secret"
    r"|auth[_ -]?token|access[_ -]?token)\s*[:=]\s*\S+",
    re.IGNORECASE,
)
# a run of 24 or more letters and digits mixing upper case, lower case and
# digits, the shape of a random token (this feature's own source tokens
# mostly included); a commit hash is lower-case hex, and a hyphenated slug
# breaks into short runs, so neither trips it
OPAQUE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*\d)"
    r"[A-Za-z0-9]{24,}(?![A-Za-z0-9])"
)
# < and >: no markup, and no line that could pass for the relay's header or
# its "> " quote marker
ANGLE_RE = re.compile(r"[<>]")
# a URL in free text (a scheme, www., or a bare domain under a common TLD);
# a link goes only in the link field, where its host is checked
URL_RE = re.compile(
    r"\b[a-z][a-z0-9+.-]*://|\bwww\."
    r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|io|ai|co|eu|be|nl|de|fr|uk|dev|app|me|info|biz)\b",
    re.IGNORECASE,
)

_SPACES_RE = re.compile(r"[ \t]+")
# the line breaks str.splitlines() honours that the control-character check
# lets through (they are Zl and Zp, not Cc); every other one is a control
_UNICODE_LINE_BREAK_RE = re.compile("[\u2028\u2029]")


class Rejected(ValueError):
    """An update the guard refuses. ``field`` and ``reason`` go back to the
    caller as they are, so they never quote the text."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


@dataclass(frozen=True)
class CleanUpdate:
    """A validated, cleaned update. ``topic_key`` is the topic case-folded,
    what "latest per topic" groups on."""

    topic: str
    topic_key: str
    status: str
    summary: str
    ask: str | None
    link: str | None
    idempotency_key: str | None


def _iban_checksum_ok(compact: str) -> bool:
    if sum(ch.isdigit() for ch in compact) < IBAN_MIN_DIGITS:
        return False
    rearranged = compact[4:] + compact[:4]
    digits = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(digits) % 97 == 1


def _has_phone(text: str) -> bool:
    return any(sum(ch.isdigit() for ch in m.group(0)) >= PHONE_MIN_DIGITS for m in PHONE_RE.finditer(text))


def _has_iban(text: str) -> bool:
    for m in IBAN_RE.finditer(text):
        compact = m.group(0).replace(" ", "")
        if any(_iban_checksum_ok(compact[:n]) for n in range(15, min(len(compact), 34) + 1)):
            return True
    return False


# each deny check with the reason the caller gets back
_DENY: tuple[tuple[Callable[[str], bool], str], ...] = (
    (lambda t: EMAIL_RE.search(t) is not None, "contains an email address"),
    (_has_phone, "contains a phone number"),
    (_has_iban, "contains a bank account number (IBAN)"),
    (lambda t: CREDENTIAL_RE.search(t) is not None, "contains something shaped like a credential"),
    (lambda t: OPAQUE_TOKEN_RE.search(t) is not None, "contains something shaped like a credential"),
    (lambda t: ANGLE_RE.search(t) is not None, "contains < or >"),
    (lambda t: URL_RE.search(t) is not None, "contains a URL; put one link in the link field"),
)


def _visible_only(field: str, raw: str) -> str:
    """The NFKC form of ``raw``; raises when ``raw`` holds a control, format
    or invisible character, which ``plain`` would drop and so hide."""
    folded = unicodedata.normalize("NFKC", raw)
    if folded != plain(raw):
        raise Rejected(field, "contains a control or invisible character")
    return folded


def _free_text(
    payload: Mapping[str, Any],
    field: str,
    *,
    required: bool,
    one_line: bool,
    min_len: int,
    max_len: int,
) -> str | None:
    raw = payload.get(field)
    if raw is None:
        if required:
            raise Rejected(field, "is required")
        return None
    if not isinstance(raw, str):
        raise Rejected(field, "must be a string")
    # CRLF from curl or a Windows client hides nothing; a lone \r stays and
    # is rejected below as a control character
    raw = raw.replace("\r\n", "\n")
    folded = _visible_only(field, raw)
    if _UNICODE_LINE_BREAK_RE.search(raw):
        raise Rejected(field, "contains a line or paragraph separator")
    if one_line and "\n" in raw:
        raise Rejected(field, "must be one line")
    if not one_line and len(raw.strip("\n").split("\n")) > SUMMARY_MAX_LINES:
        raise Rejected(field, f"has more than {SUMMARY_MAX_LINES} lines")
    for found, reason in _DENY:
        if found(folded):
            raise Rejected(field, reason)
    cleaned = "\n".join(_SPACES_RE.sub(" ", line).strip() for line in folded.split("\n")).strip("\n")
    if not cleaned and not required:
        return None
    if len(cleaned) < min_len:
        raise Rejected(field, f"must be at least {min_len} characters")
    if len(cleaned) > max_len:
        raise Rejected(field, f"must be at most {max_len} characters")
    return cleaned


def _required_text(payload: Mapping[str, Any], field: str, *, one_line: bool, min_len: int, max_len: int) -> str:
    value = _free_text(payload, field, required=True, one_line=one_line, min_len=min_len, max_len=max_len)
    if value is None:
        raise Rejected(field, "is required")
    return value


def _link(payload: Mapping[str, Any], allowed_hosts: list[str]) -> str | None:
    raw = payload.get("link")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise Rejected("link", "must be a string")
    _visible_only("link", raw)
    link = raw.strip()
    if not link:
        return None
    if any(ch.isspace() for ch in link) or ANGLE_RE.search(link):
        raise Rejected("link", "must be a single URL")
    try:
        parts = urlsplit(link)
        port = parts.port
    except ValueError:
        raise Rejected("link", "is not a valid URL") from None
    if parts.scheme.lower() != "https":
        raise Rejected("link", "must be an https URL")
    if parts.username is not None or parts.password is not None or port is not None:
        raise Rejected("link", "must not carry credentials or a port")
    host = (parts.hostname or "").lower()
    if not host or host not in {h.strip().lower() for h in allowed_hosts}:
        raise Rejected("link", "points at a host that is not allowed")
    return link


def check(payload: Mapping[str, Any], *, link_hosts: list[str] | None = None) -> CleanUpdate:
    """``payload`` validated and cleaned, or :class:`Rejected` naming the
    first field that fails. ``link_hosts`` defaults to
    ``PROGRESS_LINK_HOSTS``."""
    if not isinstance(payload, Mapping):
        raise Rejected("payload", "must be an object")
    unknown = sorted(str(k) for k in payload if k not in FIELDS)
    if unknown:
        raise Rejected(unknown[0], "is not a field of a progress update")
    topic = _required_text(payload, "topic", one_line=True, min_len=TOPIC_MIN, max_len=TOPIC_MAX)
    status = payload.get("status")
    if status not in STATUSES:
        raise Rejected("status", f"must be one of {', '.join(STATUSES)}")
    summary = _required_text(payload, "summary", one_line=False, min_len=SUMMARY_MIN, max_len=SUMMARY_MAX)
    ask = _free_text(payload, "ask", required=False, one_line=True, min_len=1, max_len=ASK_MAX)
    if link_hosts is None:
        from openexecutive.config import get_settings

        link_hosts = get_settings().progress_link_hosts
    link = _link(payload, link_hosts)
    key = payload.get("idempotency_key")
    if key is not None and (not isinstance(key, str) or not IDEMPOTENCY_KEY_RE.fullmatch(key)):
        raise Rejected("idempotency_key", "must be 8-64 of A-Z, a-z, 0-9, _ and -")
    return CleanUpdate(
        topic=topic,
        topic_key=" ".join(topic.split()).casefold(),
        status=str(status),
        summary=summary,
        ask=ask,
        link=link,
        idempotency_key=key,
    )
