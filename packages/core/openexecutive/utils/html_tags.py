"""Tag stripping and visible-text reduction, shared by everything that reduces
someone else's HTML to text.

``monitoring.sources.page_watch`` reads pages a watched site's server shapes,
``orchestrator.web_fetch`` reads pages the model asked for, and
``delegation.gmail`` reads mail anyone can send. All run on the event loop, so
all drop tags with this forward scan rather than ``re.sub(r"<[^>]+>", ...)``:
on text with many "<" and no ">", that pattern runs from each "<" to the end of
the text, fails, and starts over at the next "<" — quadratic, about 40 minutes
of CPU for a 2 MB page.
"""
from __future__ import annotations

import html
import re
import string

# <script>/<style>/<noscript> blocks: drop content, not just tags, so inline
# JS/CSS never counts as visible text. Removed by a linear scan
# (_strip_blocks), not a lazy `.*?` regex: on a page full of unterminated
# openers such a regex is quadratic, and a page the server sends is the
# attacker's to shape.
_DROP_BLOCK_TAGS = ("script", "style", "noscript")
_WS_RE = re.compile(r"\s+")
# ASCII-only lowercasing for the block scan. str.lower() can lengthen a
# string ("İ" becomes two code points), and the scan finds positions in the
# lowered copy to slice the original, so the two must stay the same length.
# Tag names are ASCII, so nothing else needs folding.
_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def strip_tags(text: str, *, repl: str = "") -> str:
    """``text`` with each tag — a "<" through the next ">", with something
    between — replaced by ``repl``, in one forward pass.

    Gives the same result as ``re.sub(r"<[^>]+>", repl, text)``: ``<>`` is not
    a tag, and a "<" that nothing after it closes stays as text.
    """
    out: list[str] = []
    at = 0
    while (lt := text.find("<", at)) != -1:
        gt = text.find(">", lt + 1)
        if gt == -1:
            break  # nothing after this closes a tag
        out.append(text[at:gt + 1] if gt == lt + 1 else text[at:lt] + repl)
        at = gt + 1
    out.append(text[at:])
    return "".join(out)


def visible_text(markup: str) -> str:
    """The visible text of an HTML page, whitespace collapsed to single spaces.

    Drops comments and <script>/<style>/<noscript> blocks, removes the
    remaining tags, unescapes entities, and collapses all whitespace — so a
    re-minified page that reads the same gives the same text. Not truncated:
    callers bound their input and their output.
    """
    markup = _strip_blocks(markup)
    markup = strip_tags(markup, repl=" ")
    return _WS_RE.sub(" ", html.unescape(markup)).strip()


def _strip_blocks(markup: str) -> str:
    """Drop HTML comments and <script>/<style>/<noscript> blocks (content
    included) in one pass that is linear in the page length. An unterminated
    comment or block runs to the end of the page: a truncated script is not
    visible text either."""
    lower = markup.translate(_ASCII_LOWER)
    n = len(markup)
    out: list[str] = []
    i = 0
    while i < n:
        j = lower.find("<", i)
        if j == -1:
            out.append(markup[i:])
            break
        out.append(markup[i:j])
        if lower.startswith("<!--", j):
            end = lower.find("-->", j + 4)
            i = n if end == -1 else end + 3
            out.append(" ")
            continue
        tag = next(
            (
                t for t in _DROP_BLOCK_TAGS
                if lower.startswith("<" + t, j)
                and (j + 1 + len(t) >= n or not (lower[j + 1 + len(t)].isalnum() or lower[j + 1 + len(t)] == "_"))
            ),
            None,
        )
        if tag is None:
            out.append("<")
            i = j + 1
            continue
        close = lower.find("</" + tag, j)
        if close == -1:
            i = n
        else:
            gt = lower.find(">", close)
            i = n if gt == -1 else gt + 1
        out.append(" ")
    return "".join(out)
