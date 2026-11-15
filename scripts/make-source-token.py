#!/usr/bin/env python3
"""Make a token for an agent or service that posts progress updates.

Each source (for example the marketing planning sessions) gets its own
token. The source sends it in the x-source-token header; the API keeps only
its sha256 in PROGRESS_SOURCES, and files every update under the source the
token belongs to:

    uv run python scripts/make-source-token.py marketing-planner

- The token goes to that source only, once. Keep it as secret as
  BACKEND_SHARED_SECRET; it is not printed again and can't be recovered from
  the hash.
- The PROGRESS_SOURCES entry goes on the API. Several sources are
  comma-separated; restart the API after a change.

To rotate or revoke a source, replace or drop its entry and restart the API.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "packages" / "core"))

from openexecutive.integrations.progress_sources import (  # noqa: E402
    SOURCE_ID_RE,
    make_token,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "source",
        help="the source id: 2-40 of a-z, 0-9 and -, starting with a letter or digit",
    )
    args = parser.parse_args()
    if not SOURCE_ID_RE.fullmatch(args.source):
        print("the source id must be 2-40 of a-z, 0-9 and -, starting with a letter or digit", file=sys.stderr)
        return 2
    token, entry = make_token(args.source)
    print(f"# The source ({args.source}) only, sent as the x-source-token header. Secret:")
    print(token)
    print("# The API only; add it to PROGRESS_SOURCES (comma-separated):")
    print(f"PROGRESS_SOURCES={entry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
