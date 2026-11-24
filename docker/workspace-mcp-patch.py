"""Patch the pinned workspace-mcp so modify_doc_text appends with
end_of_segment=true and no start_index, as its own description tells the
model to.

Run at image build against the installed package (docker/Dockerfile). Each
replacement must match exactly once, so a bump of the pin that changes this
code fails the build instead of shipping unpatched.
"""
from __future__ import annotations

import sys
from pathlib import Path

# modify_doc_text's description says "to append text to the end of the
# document without calculating indices, set end_of_segment=true", but the code
# then refuses any call without start_index unless a heading is given, so every
# append failed with "'start_index' is required unless 'heading' is provided".
# With end_of_segment the index is never used for the insert, so any value
# lets it through; formatting and replacing with end_of_segment are still
# refused further down, as before.
APPEND_OLD = '''    if start_index is None:
        return "Error: 'start_index' is required unless 'heading' is provided."'''

APPEND_NEW = '''    if start_index is None and end_of_segment:
        # Open Executive: appending needs no index.
        start_index = 0
    if start_index is None:
        return "Error: 'start_index' is required unless 'heading' is provided."'''

PATCHES = ((APPEND_OLD, APPEND_NEW),)


def patch(path: Path) -> None:
    text = path.read_text()
    for old, new in PATCHES:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"workspace-mcp-patch: expected 1 match in {path}, found {count}")
        text = text.replace(old, new)
    path.write_text(text)


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
