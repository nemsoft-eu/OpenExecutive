"""The shared tag strip (utils.html_tags): the regex it replaced, in one pass,
and the visible-text reduction built on it."""
from __future__ import annotations

import itertools
import re

import pytest

from openexecutive.utils.html_tags import strip_tags, visible_text

# What page_watch and delegation.gmail stripped tags with before: quadratic on
# text with many "<" and no ">".
_OLD_TAG_RE = re.compile(r"<[^>]+>")

# Every string of up to seven characters over "<", ">", a letter and a newline
# (21,845 of them), and a few shapes real markup has.
_SAMPLES = [
    "".join(chars) for n in range(8) for chars in itertools.product("<>a\n", repeat=n)
] + [
    '<a href="https://x.example/" title="<">click</a> here',
    "if x<y then <b>bold</b>",
    "a <> b <<c>> d",
    "<!-- note -->text<br/>more",
    "<p>1 < 2</p> and 3 < 4",
]


@pytest.mark.parametrize("repl", ["", " "])
def test_strip_tags_matches_the_regex_it_replaced(repl: str) -> None:
    for text in _SAMPLES:
        assert strip_tags(text, repl=repl) == _OLD_TAG_RE.sub(repl, text), repr(text)


@pytest.mark.parametrize(
    ("markup", "expected"),
    [
        ("<p>Hello &amp;\n\n <b>world</b></p>", "Hello & world"),
        ("a<script>x()</script>b<style>p{}</style>c<noscript>n</noscript>d", "a b c d"),
        ("a<!-- hidden -->b", "a b"),
        ("<scripts>kept</scripts>", "kept"),  # not a <script> block
        ("before<script>never closed", "before"),
        ("before<!-- never closed", "before"),
        # "İ".lower() is two code points; the scan must not drift past it.
        ("İİİ<SCRIPT>leak()</SCRIPT>after", "İİİ after"),
    ],
)
def test_visible_text(markup: str, expected: str) -> None:
    assert visible_text(markup) == expected
