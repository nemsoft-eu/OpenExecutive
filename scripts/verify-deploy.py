#!/usr/bin/env python3
"""Assert that the running deployment serves the version in this checkout.

A green ``/health`` alone does not prove an upgrade took: it answers 200 with
the configured company name whatever code is behind it. What it *does* carry is
a ``version``, and that is the comparison worth making. On 2026-10-08 the
running API image turned out to have been built from source 17 days and 253
commits behind the checkout it was supposedly built from: the containers had
been *recreated* the evening before, which rebuilt nothing, and every health
check was green throughout. The standing briefs were silently running
two-week-old code.

``/health``'s ``version`` would have narrowed that on day one. It reported
``0.1.0`` while the checkout read ``0.5.2`` — the pre-upstream-sync literal
against the post-sync one. An earlier revision of this script claimed the
opposite, that ``version`` "is a static string that does not move between
commits", and built a build-timestamp heuristic on top of that claim. **The
claim was false.** ``version`` is a release-please-managed literal
(``packages/core/openexecutive/api/models.py``, ``api/main.py``,
``packages/core/pyproject.toml``, ``uv.lock`` and ``packages/ui/package.json``
all carry it, per ``release-please-config.json``), so it moves with every
release.

The heuristic that replaced it was unsound as well: an image's *build time*
records when it was assembled, never what source went into it, so an operator
who rebuilds an old checkout today got ``OK`` from a stale deployment; and it
reached for images and containers by name, which are a Compose convention
rather than a fact, so its podman-compose-shaped defaults resolved nothing
under ``docker compose -f docker/docker-compose.yml``. Reading the version over
HTTP dissolves both: there is no image to inspect, no container to name, and
nothing inferred from a timestamp.

**What this does not catch.** The version moves per *release*, not per commit.
Every commit inside one release window reports the same version, so this cannot
distinguish them — on an install that follows unreleased ``main`` it is a
weaker signal, blind to everything merged since the last release. A passing run
prints how HEAD sits relative to its version's tag — ahead, behind, diverged,
exactly on it, or why git could not say — so the size of that blind spot is
visible rather than implied. It is a floor on staleness, not a proof of
freshness: pair it with the behaviour checks in ``docs/deployment.md`` step 4,
or with a grep for the expected code inside the running container, which is
exact but needs a shell in it.

Usage::

    python3 scripts/verify-deploy.py
    python3 scripts/verify-deploy.py --url https://api.example.internal
    python3 scripts/verify-deploy.py --path /version   # needs the shared secret
    python3 scripts/verify-deploy.py --expect 0.5.1    # an install on pinned images

Run it on the host in the reference topology: there only the UI origin is
public and the API is reached through the UI's ``/api/backend/*`` proxy, which
is gated on a verified session, so a bare request there answers 401 rather than
a version. ``--url`` is for an install that exposes the API on its own
hostname.

``--expect`` is the right flag whenever the checkout is not what was deployed —
notably the published-image upgrade path, where the operator pins an image tag
and the checkout's own version means nothing.

``/health`` is outside the shared-secret gate (``api/main.py``'s
``_UNAUTHENTICATED_PATHS``), so the default sends no credential at all — the
header is attached only for a path that needs it, such as ``/version``. The
secret is read from ``$BACKEND_SHARED_SECRET`` in the environment and never
accepted as an argument, because a command line is visible in a process
listing.

Three deliberate restrictions keep that header, and the verdict, on the host
the operator actually named. Only ``http``/``https`` are accepted; a redirect
is refused rather than followed, because urllib copies request headers onto the
redirected request; and the opener is built with an **empty** ``ProxyHandler``,
because urllib's default one reads ``http_proxy``/``https_proxy`` from the
environment — which would let the environment rather than ``--url`` decide
which host answers, and hand it the credential in cleartext on the way. An
earlier revision claimed this property while leaving the proxy hole open; it
was reported and is closed here.

``--path /version`` has one wrinkle worth knowing: unless
``UPDATE_CHECK_ENABLED=false``, that route also asks GitHub for the latest
release (5 s timeout). ``/health`` has no outbound dependency, which is another
reason it is the default.

Exit codes:

* ``0`` — the deployment reports the expected version;
* ``1`` — it reports a different one. Reserved for that observed mismatch and
  nothing else;
* ``2`` — inconclusive: the API was unreachable, answered anything other than
  200 (including another 2xx, or a redirect), returned something unparseable or
  implausibly large, reported two conflicting versions, or the checkout's
  version could not be read. An inconclusive check must never read as a pass —
  that is the same trap as the stale deployment one level up, so every
  unexpected exception lands here too.

There is no fourth code. A closed stdout would otherwise surface as CPython's
exit 120 at shutdown, which would hide a mismatch from a caller branching on
1, so stdout is flushed before exiting.
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

# The checkout this script belongs to, found from the script's own location so
# the answer does not depend on the caller's working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "packages" / "core" / "pyproject.toml"

# `/health` carries `version`, `/version` carries `current`, and no endpoint
# carries both. That is a fact about today's API rather than a guarantee, so
# both keys are accepted and a body carrying two different values is reported
# as inconclusive rather than silently resolved by this tuple's order.
VERSION_KEYS = ("version", "current")

# A version document is a few hundred bytes. One byte over the cap is read so
# an oversized body can be rejected outright: truncating it could leave
# parseable JSON and a confident wrong answer.
MAX_BODY_BYTES = 64 * 1024

ALLOWED_SCHEMES = ("http", "https")

# Paths the API serves without the shared secret (`api/main.py`'s
# `_UNAUTHENTICATED_PATHS`). The credential is withheld on these so the
# documented default invocation sends none at all.
UNAUTHENTICATED_PATHS = frozenset({"/health"})

DEFAULT_TIMEOUT_S = 10.0
# The advisory git call is not allowed to hang the verification step.
GIT_TIMEOUT_S = 5.0


class Inconclusive(RuntimeError):
    """The check could not be completed, which is not the same as a failure."""


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Treat a redirect as inconclusive instead of following it.

    Two reasons, both load-bearing. The verdict would be attributed to the URL
    the operator typed while a different origin actually answered it; and
    urllib copies the request headers onto the redirected request, so an
    ``x-api-key`` would be handed to whatever host the redirect names.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        raise Inconclusive(
            f"{req.full_url} answered HTTP {code} redirecting to {newurl}; this check "
            "does not follow redirects. Point --url at the API itself."
        )


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
    if not isinstance(project, dict):
        raise Inconclusive(f"no [project] table in {PYPROJECT}")
    version = project.get("version")
    if not isinstance(version, str) or not version.strip():
        raise Inconclusive(f"no usable [project].version in {PYPROJECT}")
    # Stripped, so surrounding whitespace on either side of the comparison
    # cannot produce an exit 1 that reads as a real mismatch.
    return version.strip()


def deployed_version(url: str, path: str, timeout: float) -> str:
    """The version the running API reports, or Inconclusive if it won't say."""
    endpoint = f"{url.rstrip('/')}{path}"
    scheme = urllib.parse.urlsplit(endpoint).scheme
    if scheme not in ALLOWED_SCHEMES:
        raise Inconclusive(
            f"{endpoint} is not an HTTP(S) URL (scheme {scheme or 'missing'!r}); "
            "this check talks to a running deployment, not a local file"
        )
    request = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
    # A cached answer verifies nothing — the documented remote invocation goes
    # through a reverse proxy that could serve a pre-deploy body.
    request.add_header("Cache-Control", "no-cache, no-store")
    request.add_header("Pragma", "no-cache")
    secret = os.environ.get("BACKEND_SHARED_SECRET")
    sent_secret = bool(secret) and path not in UNAUTHENTICATED_PATHS
    if secret and sent_secret:
        request.add_header("x-api-key", secret)
    # `ProxyHandler({})` rather than the default one: urllib's default reads
    # `http_proxy`/`https_proxy` from the environment, which would let the
    # environment — not --url — choose which host answers, and would send
    # `x-api-key` to it in cleartext. With no proxy and no redirect, the
    # request reaches the host named in --url or it fails.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _RefuseRedirects)
    try:
        with opener.open(request, timeout=timeout) as response:
            status = response.status
            # One byte past the cap, so an over-long body is rejected rather
            # than truncated into something that still parses.
            body = response.read(MAX_BODY_BYTES + 1)
    except urllib.error.HTTPError as exc:
        hint = ""
        if exc.code in (401, 403):
            hint = (
                " — the shared secret in the environment was rejected"
                if sent_secret
                else " — authenticated; export BACKEND_SHARED_SECRET and retry"
            )
        raise Inconclusive(f"{endpoint} answered HTTP {exc.code}{hint}") from exc
    except Inconclusive:
        raise
    except Exception as exc:
        # urlopen's failure surface is wide — URLError, TimeoutError, an
        # http.client.HTTPException raised while reading the body, an SSL or
        # reset error mid-stream. None of them is a version mismatch.
        raise Inconclusive(f"{endpoint} could not be read: {exc!r}") from exc
    if status != 200:
        # A 2xx that is not 200 is not the answer this asks for: 203 is what a
        # transforming proxy returns when it has rewritten the payload, and 206
        # means the body is a fragment.
        raise Inconclusive(f"{endpoint} answered HTTP {status}, not 200")
    if len(body) > MAX_BODY_BYTES:
        raise Inconclusive(
            f"{endpoint} returned more than {MAX_BODY_BYTES} bytes; that is not a "
            "version document"
        )
    try:
        payload = json.loads(body)
    except ValueError as exc:
        raise Inconclusive(f"{endpoint} did not return JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise Inconclusive(f"{endpoint} returned {type(payload).__name__}, not an object")
    present = [key for key in VERSION_KEYS if key in payload]
    usable = {
        key: payload[key].strip()
        for key in present
        if isinstance(payload[key], str) and payload[key].strip()
    }
    if len(set(usable.values())) > 1:
        # Rather than let VERSION_KEYS' order pick a winner silently.
        raise Inconclusive(
            f"{endpoint} reports two different versions: "
            + ", ".join(f"{k}={v}" for k, v in sorted(usable.items()))
        )
    for key in VERSION_KEYS:
        if key in usable:
            return usable[key]
    if present:
        raise Inconclusive(
            f"{endpoint} returned {present[0]} but not as a non-empty string "
            f"({payload[present[0]]!r})"
        )
    raise Inconclusive(
        f"{endpoint} carries neither {' nor '.join(VERSION_KEYS)} "
        f"(keys: {', '.join(sorted(payload)) or 'none'})"
    )


def tag_distance(version: str) -> str:
    """How far HEAD sits from ``v{version}``, as an operator caveat.

    Advisory only — it quantifies the blind spot this check cannot see into and
    never affects the verdict, so every failure degrades to a note that names
    its own cause rather than inventing one.
    """
    argv = [
        "git",
        "-C",
        str(REPO_ROOT),
        "rev-list",
        "--left-right",
        "--count",
        f"v{version}...HEAD",
    ]
    try:
        out = subprocess.run(
            argv, capture_output=True, text=True, check=False, timeout=GIT_TIMEOUT_S
        )
    except FileNotFoundError:
        return "commits since that release: unknown (no git on PATH)"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"commits since that release: unknown ({type(exc).__name__})"
    if out.returncode != 0:
        # Both "not a git repository" and "unknown revision v<version>" land
        # here, and a shallow clone has no tags at all, so say which happened
        # rather than asserting a cause.
        reason = out.stderr.strip().splitlines()
        return f"commits since that release: unknown ({reason[0][:80] if reason else 'git failed'})"
    try:
        behind_s, ahead_s = out.stdout.split()
        behind, ahead = int(behind_s), int(ahead_s)
    except ValueError:
        return "commits since that release: unknown (unreadable git output)"
    if ahead and behind:
        # Checked before the plain-ahead case: "N past the tag" would be false
        # here and would understate the gap, because the checkout is also
        # *missing* commits the release contains.
        return (
            f"HEAD has diverged from v{version}: {ahead} commit(s) only it has, "
            f"{behind} commit(s) of that release it is missing"
        )
    if ahead:
        return (
            f"HEAD is {ahead} commit(s) past v{version}; those commits share this "
            f"version, so this check cannot see them"
        )
    if behind:
        return (
            f"HEAD is {behind} commit(s) behind v{version}, which is odd for a "
            f"checkout declaring that version"
        )
    return f"HEAD is the v{version} release commit, so the version pins it exactly"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Assert the running deployment serves this checkout's version."
    )
    ap.add_argument(
        "--url", default="http://localhost:8000", help="API base URL (default: %(default)s)"
    )
    ap.add_argument(
        "--path",
        default="/health",
        help="endpoint reporting the version: /health (default, unauthenticated) or /version",
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
    # A leading slash is easy to drop, and without it the base URL and the path
    # fuse into a bad port rather than a request.
    path = "/" + args.path.lstrip("/")
    running = deployed_version(args.url, path, args.timeout)

    if running != expected:
        print("DEPLOY VERIFICATION FAILED")
        source = "the version requested" if pinned else "this checkout"
        print(
            f"  - {args.url.rstrip('/')}{path} reports version {running}, but {source} "
            f"is {expected}. The deployment is not serving that code."
        )
        # Spelled out rather than abbreviated to `compose …`: there is no
        # standalone `compose` executable, and from the repo root
        # `docker compose` finds no file at all because this repo's lives at
        # docker/docker-compose.yml, which is why the checkout path points at
        # the Makefile target that already carries -f and --env-file.
        if pinned:
            print(
                "    Pull the tags you intend and recreate: `docker compose pull`, "
                "then `docker compose up -d`."
            )
        else:
            print(
                "    Rebuild from an up-to-date checkout: `git pull`, then `make docker`."
            )
            print(
                "    If a cached layer is suspected, force a clean build first: "
                "`docker compose --env-file .env -f docker/docker-compose.yml "
                "build --no-cache`."
            )
        return 1

    # Computed before anything is printed, so a slow or broken git cannot land
    # between an OK line and the exit code.
    note = (
        f"--expect was given, so this compares against {expected}, not this checkout"
        if pinned
        else tag_distance(expected)
    )
    matched = "the version requested" if pinned else "this checkout"
    print(f"OK: {args.url.rstrip('/')}{path} reports version {running}, matching {matched}")
    print(f"  note: {note}")
    print("  this is a floor on staleness, not a proof of freshness — also verify the")
    print("  behaviour you upgraded for (docs/deployment.md step 4).")
    return 0


def _exit(code: int) -> None:
    """Exit with ``code``, surviving a closed stdout.

    Prints are block-buffered when stdout is a pipe, so a reader that has gone
    away (``| head -1``) raises ``BrokenPipeError`` during interpreter
    shutdown, which CPython reports as exit 120 — a fourth exit code this
    contract does not define, and one that hides a real mismatch from a caller
    branching on 1. Flushing here, with the remaining output sent to
    ``/dev/null``, keeps the verdict in the exit code.
    """
    try:
        sys.stdout.flush()
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    sys.exit(code)


if __name__ == "__main__":
    try:
        _exit(main())
    except Inconclusive as exc:
        print(f"DEPLOY VERIFICATION INCONCLUSIVE: {exc}")
        _exit(2)
    except Exception as exc:
        # Exit 1 is reserved for an observed version mismatch. Anything else
        # that goes wrong is a tooling failure, and reporting it as "not
        # serving this code" would be its own false verdict.
        print(f"DEPLOY VERIFICATION INCONCLUSIVE: unexpected {type(exc).__name__}: {exc}")
        _exit(2)
