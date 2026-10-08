#!/usr/bin/env python3
"""Assert that the running containers are actually serving the code in this checkout.

A green ``/health`` does not prove that. It returns 200 with the configured
company name whatever code is behind it, and the ``version`` it reports is a
static string that does not move between commits — so it passes identically
for a current deployment and a stale one. On 2026-10-08 the running API image
turned out to have been built on 2026-09-21, 17 days and 253 commits behind
the checkout it was supposedly built from: the containers had been *recreated*
the evening before, which rebuilt nothing, and every health check was green
throughout. The standing briefs were silently running two-week-old code.

The two failures this catches are mirror images of each other, and both exit 0
under ``podman-compose``:

* built but not recreated — a new image exists while the container still runs
  the old one (compare the container's image id against the tag's);
* recreated but not built — the container runs the tagged image, but that
  image predates the commit it is meant to contain (compare timestamps).

Usage::

    python3 scripts/verify-deploy.py                  # podman, default names
    python3 scripts/verify-deploy.py --engine docker
    python3 scripts/verify-deploy.py --service api --image localhost/openexecutive_api:latest

Exits non-zero with the reason on stdout when the deployment is not serving
HEAD. Intended as the last step of an upgrade, after the containers are up.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime, timedelta

# A build legitimately finishes after the commit it contains, never before.
# Allow a little slack for clock skew between the git author date and the
# container engine's clock; anything beyond this is a real staleness signal,
# not jitter.
SKEW_ALLOWANCE = timedelta(minutes=10)


def _run(args: list[str]) -> str:
    out = subprocess.run(args, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:3])}… failed: {out.stderr.strip()[:200]}")
    return out.stdout.strip()


def _parse_ts(raw: str) -> datetime:
    """Parse a container-engine timestamp, which comes in several shapes.

    podman prints ``2026-09-21 14:17:09.98 +0000 UTC``; docker prints RFC3339.
    Both are normalised to an aware UTC datetime so the comparison below
    cannot silently compare naive to aware and raise.
    """
    cleaned = raw.strip().removesuffix(" UTC").strip()
    for candidate in (cleaned, cleaned.replace(" ", "T", 1)):
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    raise RuntimeError(f"could not parse a timestamp from {raw!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="podman", help="podman (default) or docker")
    ap.add_argument("--service", default="api", help="compose service name, e.g. api")
    ap.add_argument("--project", default="openexecutive", help="compose project name")
    ap.add_argument("--image", default=None, help="image tag; defaults to <project>_<service>:latest")
    ap.add_argument("--container", default=None, help="container name; defaults to <project>_<service>_1")
    args = ap.parse_args()

    image = args.image or f"localhost/{args.project}_{args.service}:latest"
    container = args.container or f"{args.project}_{args.service}_1"

    head_sha = _run(["git", "rev-parse", "--short=12", "HEAD"])
    head_date = _parse_ts(_run(["git", "log", "-1", "--format=%cI", "HEAD"]))

    image_id = _run([args.engine, "image", "inspect", image, "--format", "{{.Id}}"])
    image_built = _parse_ts(
        _run([args.engine, "image", "inspect", image, "--format", "{{.Created}}"])
    )
    # `.ImageID` is what the container is actually running, which is NOT
    # necessarily what the tag points at now.
    running_id = _run(
        [args.engine, "inspect", container, "--format", "{{.Image}}"]
    )

    problems: list[str] = []

    if not running_id.startswith(image_id[: len(running_id)]) and not image_id.startswith(
        running_id.removeprefix("sha256:")
    ):
        problems.append(
            f"{container} runs image {running_id[:19]} but {image} is now "
            f"{image_id[:12]} — built without recreating. Re-run `up -d --force-recreate`."
        )

    if image_built + SKEW_ALLOWANCE < head_date:
        behind = head_date - image_built
        problems.append(
            f"{image} was built {image_built.isoformat()}, which predates HEAD "
            f"({head_sha}, {head_date.isoformat()}) by {behind}. The deployment is "
            f"serving older code than this checkout. Rebuild with `build --no-cache`."
        )

    if problems:
        print("DEPLOY VERIFICATION FAILED")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(
        f"OK: {container} runs {image_id[:12]} built {image_built.isoformat()}, "
        f"at or after HEAD {head_sha} ({head_date.isoformat()})"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        # An engine or git failure must not read as a pass.
        print(f"DEPLOY VERIFICATION INCONCLUSIVE: {exc}")
        sys.exit(2)
