#!/usr/bin/env python3
"""Assert that the running API serves the version in this checkout.

Reads the ``version`` from ``GET /health`` and compares it to
``packages/core/pyproject.toml``. Nothing else: no image inspection, no
container names, no credential, no second endpoint.

An earlier revision asserted that ``/health``'s ``version`` "is a static string
that does not move between commits" and built a build-timestamp heuristic on
that. **The claim was false** — the version is a release-please literal (see
``release-please-config.json``), so it moves with every release. It read
``0.1.0`` on the deployment that had been stale for 17 days while the checkout
read ``0.5.2``.

**What this does not catch.** The version moves per *release*, not per commit,
so every commit inside one release window reports the same number. On an
install following unreleased ``main`` this is a floor on staleness, not a proof
of freshness, which is why a passing run prints where HEAD sits relative to its
version's tag. It checks the API only; ``docs/deployment.md`` step 4 has the
behaviour checks that go with it and the UI's separate gap.

Usage: no arguments for ``http://localhost:8000``; ``--expect X.Y.Z`` for an
install running pinned images, where the checkout's own version means nothing.

Exit codes: ``0`` the deployment reports the expected version, ``1`` only for
an observed mismatch, ``2`` inconclusive — unreachable, any status but 200, a
redirect, an unparseable or oversized answer, an unreadable checkout, or any
unexpected error. An inconclusive check must never read as a pass, and a
tooling failure must never read as a mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    tomllib = None  # type: ignore[assignment]

# The checkout this script belongs to, from the script's own location so the
# answer does not depend on the caller's working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "packages" / "core" / "pyproject.toml"

# `/health` is the only endpoint probed, on purpose: it is outside the API's
# shared-secret gate (`api/main.py`'s `_UNAUTHENTICATED_PATHS`) and makes no
# outbound call, so this script needs no credential and can never leak one.
HEALTH_PATH = "/health"
# One byte past the cap is read so an oversized body is rejected rather than
# truncated into something that still parses.
MAX_BODY_BYTES = 64 * 1024
DEFAULT_TIMEOUT_S = 10.0
GIT_TIMEOUT_S = 5.0


class Inconclusive(RuntimeError):
    """The check could not be completed, which is not the same as a failure."""


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """A redirect is inconclusive: another origin answered for the named one."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        raise Inconclusive(
            f"{req.full_url} answered HTTP {code} redirecting to {newurl}; this check "
            "does not follow redirects. Point --url at the API itself."
        )


def say(*lines: str) -> None:
    """Print verdict lines, surviving a reader that has already gone away.

    Every verdict goes through here. `| head -0` closes the pipe, and an
    unhandled `BrokenPipeError` — raised at the `print` under `-u`, or at
    interpreter shutdown when buffered — would replace the exit code with
    CPython's 120 or a traceback's 1. The verdict lives in the exit code, so
    the remaining output goes to /dev/null instead.
    """
    try:
        for line in lines:
            print(line)
        sys.stdout.flush()
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())


def checkout_version() -> str:
    """The version release-please wrote into this checkout's pyproject.toml."""
    if tomllib is None:
        raise Inconclusive("tomllib is unavailable; Python 3.11+ is required")
    try:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Inconclusive(f"could not read {PYPROJECT}: {exc}") from exc
    except (ValueError, tomllib.TOMLDecodeError) as exc:
        # ValueError also covers UnicodeDecodeError from read_text, which fires
        # before tomllib ever sees the bytes.
        raise Inconclusive(f"could not parse {PYPROJECT}: {exc}") from exc
    project = data.get("project")
    version = project.get("version") if isinstance(project, dict) else None
    if not isinstance(version, str) or not version.strip():
        raise Inconclusive(f"no usable [project].version in {PYPROJECT}")
    # Stripped on both sides of the comparison, so padding cannot produce an
    # exit 1 that reads as a real mismatch.
    return version.strip()


def deployed_version(url: str, timeout: float) -> str:
    """The version the running API reports, or Inconclusive if it won't say."""
    endpoint = f"{url.rstrip('/')}{HEALTH_PATH}"
    scheme = urllib.parse.urlsplit(endpoint).scheme
    if scheme not in ("http", "https"):
        raise Inconclusive(
            f"{endpoint} is not an HTTP(S) URL (scheme {scheme or 'missing'!r}); "
            "this check talks to a running deployment, not a local file"
        )
    request = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
    # A cached answer verifies nothing, and the remote case goes through a
    # reverse proxy that could serve a pre-deploy body.
    request.add_header("Cache-Control", "no-cache, no-store")
    # An empty ProxyHandler, not urllib's default one: the default reads
    # `http_proxy` from the environment, which would let the environment rather
    # than --url decide which host answers for the verdict.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _RefuseRedirects)
    try:
        with opener.open(request, timeout=timeout) as response:
            status, body = response.status, response.read(MAX_BODY_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise Inconclusive(f"{endpoint} answered HTTP {exc.code}") from exc
    except Inconclusive:
        raise
    except Exception as exc:
        # urlopen's failure surface is wide — URLError, TimeoutError, an
        # HTTPException or a reset while reading. None is a version mismatch.
        raise Inconclusive(f"{endpoint} could not be read: {exc!r}") from exc
    if status != 200:
        # 203 is what a transforming proxy returns when it has rewritten the
        # payload, and 206 means the body is a fragment.
        raise Inconclusive(f"{endpoint} answered HTTP {status}, not 200")
    if len(body) > MAX_BODY_BYTES:
        raise Inconclusive(f"{endpoint} returned more than {MAX_BODY_BYTES} bytes")
    try:
        payload = json.loads(body)
    except ValueError as exc:
        raise Inconclusive(f"{endpoint} did not return JSON: {exc}") from exc
    version = payload.get("version") if isinstance(payload, dict) else None
    if not isinstance(version, str) or not version.strip():
        raise Inconclusive(f"{endpoint} reported no usable version ({version!r})")
    return version.strip()


def tag_distance(version: str) -> str:
    """Where HEAD sits relative to ``v{version}``.

    Advisory: it sizes the blind spot between releases and never affects the
    exit code, so a failure reports the cause it actually saw rather than
    asserting one.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-list", "--left-right", "--count",
             f"v{version}...HEAD"],
            capture_output=True, text=True, check=False, timeout=GIT_TIMEOUT_S,
        )
        if out.returncode != 0:
            first = out.stderr.strip().splitlines()
            raise ValueError(first[0][:80] if first else "git failed")
        behind, ahead = (int(n) for n in out.stdout.split())
    except FileNotFoundError:
        return f"distance from v{version}: unknown (no git on PATH)"
    except Exception as exc:
        return f"distance from v{version}: unknown ({exc})"
    if ahead and behind:
        return (
            f"HEAD has diverged from v{version}: {ahead} commit(s) only it has, "
            f"{behind} of that release it is missing"
        )
    if ahead:
        return (
            f"HEAD is {ahead} commit(s) past v{version}; they share this version, so "
            "this check cannot see them"
        )
    if behind:
        return f"HEAD is {behind} commit(s) behind v{version}"
    return f"HEAD is the v{version} release commit, so the version pins it exactly"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Assert the running API serves this checkout's version."
    )
    ap.add_argument(
        "--url", default="http://localhost:8000", help="API base URL (default: %(default)s)"
    )
    ap.add_argument(
        "--expect",
        default=None,
        help="version to require; defaults to this checkout's packages/core/pyproject.toml",
    )
    ap.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="HTTP timeout in seconds"
    )
    args = ap.parse_args()

    if args.timeout <= 0:
        raise Inconclusive(f"--timeout must be positive, got {args.timeout}")
    pinned = args.expect is not None
    if pinned and not args.expect.strip():
        raise Inconclusive("--expect was given an empty value")
    expected = args.expect.strip() if pinned else checkout_version()
    running = deployed_version(args.url, args.timeout)

    if running != expected:
        source = "the version requested" if pinned else "this checkout"
        # `docker compose` spelled out in full: there is no standalone
        # `compose` executable, and from the repo root `docker compose` finds
        # no file, this repo's being docker/docker-compose.yml — which is why
        # the checkout path names the Makefile target that carries -f already.
        fix = (
            "    Pull the tags you intend and recreate: `docker compose pull`, then "
            "`docker compose up -d`."
            if pinned
            else "    Rebuild from an up-to-date checkout: `git pull`, then `make docker`"
            " (force a clean build with `docker compose --env-file .env"
            " -f docker/docker-compose.yml build --no-cache` if a cached layer is"
            " suspected)."
        )
        say(
            "DEPLOY VERIFICATION FAILED",
            f"  - {args.url.rstrip('/')}{HEALTH_PATH} reports version {running}, but "
            f"{source} is {expected}. The deployment is not serving that code.",
            fix,
        )
        return 1

    # Computed before anything is printed, so a slow git cannot land between an
    # OK line and the exit code.
    note = (
        f"--expect was given, so this compares against {expected}, not this checkout"
        if pinned
        else tag_distance(expected)
    )
    say(
        f"OK: {args.url.rstrip('/')}{HEALTH_PATH} reports version {running}, matching "
        f"{'the version requested' if pinned else 'this checkout'}",
        f"  note: {note}",
        "  this is a floor on staleness, not a proof of freshness, and it covers the",
        "  API only — see docs/deployment.md step 4.",
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Inconclusive as exc:
        say(f"DEPLOY VERIFICATION INCONCLUSIVE: {exc}")
        sys.exit(2)
    except Exception as exc:
        # Exit 1 is reserved for an observed version mismatch; anything else
        # that goes wrong is a tooling failure, not a verdict about the code.
        say(f"DEPLOY VERIFICATION INCONCLUSIVE: unexpected {type(exc).__name__}: {exc}")
        sys.exit(2)
