"""`scripts/verify-deploy.py`'s exit contract, exercised rather than asserted.

The script's whole value is the contract: 0 when the deployment reports the
expected version, 1 *only* for that observed mismatch, 2 for everything that
went wrong. A tooling failure reported as exit 1 is a false "not serving this
code" verdict, and an inconclusive check reported as 0 is the stale deployment
the script exists to catch, one level up.

Review found four defects in that contract which all had the same cause: every
branch was reachable only by hand-driving a real deployment, so nobody ran most
of them. Those branches are covered here against a local `http.server`, which
is a real HTTP round trip rather than a stubbed response — the point is that a
status code, a body shape and a header actually travel.

The script lives outside the package (operators run it as `python3
scripts/verify-deploy.py` from a checkout with no venv), so it is loaded by
path. `main()` is called directly, which means the exit codes under test are
its return value; the `_exit` wrapper is covered separately.
"""
from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

#  …/packages/core/tests/unit/this_file  ->  repo root is four levels up.
SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "verify-deploy.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("verify_deploy_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def vd() -> Any:
    return _load_module()


class _Probe:
    """A one-route HTTP server that answers whatever the test wants."""

    def __init__(self) -> None:
        self.status = 200
        self.body = b'{"version": "9.9.9"}'
        self.redirect_to: str | None = None
        self.content_length: int | None = None
        self.seen_headers: list[dict[str, str]] = []
        probe = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
                probe.seen_headers.append({k.lower(): v for k, v in self.headers.items()})
                if probe.redirect_to is not None:
                    self.send_response(302)
                    self.send_header("Location", probe.redirect_to)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(probe.status)
                self.send_header("Content-Type", "application/json")
                declared = (
                    probe.content_length
                    if probe.content_length is not None
                    else len(probe.body)
                )
                self.send_header("Content-Length", str(declared))
                self.end_headers()
                self.wfile.write(probe.body)

            def log_message(self, *args: Any) -> None:
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def probe() -> Iterator[_Probe]:
    p = _Probe()
    try:
        yield p
    finally:
        p.close()


def _run(vd: Any, argv: list[str]) -> int:
    """`main()` with `argv`, translating an Inconclusive into its exit code."""
    old = sys.argv
    sys.argv = ["verify-deploy.py", *argv]
    try:
        return vd.main()
    except vd.Inconclusive:
        return 2
    finally:
        sys.argv = old


# --------------------------------------------------------------------------- #
# The three verdicts
# --------------------------------------------------------------------------- #


def test_matching_version_exits_zero(vd: Any, probe: _Probe) -> None:
    probe.body = b'{"version": "1.2.3"}'
    assert _run(vd, ["--url", probe.url, "--expect", "1.2.3"]) == 0


def test_different_version_exits_one(vd: Any, probe: _Probe) -> None:
    probe.body = b'{"version": "1.2.3"}'
    assert _run(vd, ["--url", probe.url, "--expect", "9.9.9"]) == 1


def test_unreachable_api_is_inconclusive(vd: Any) -> None:
    # A port nothing is listening on: bind, read the number, release it.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    assert _run(vd, ["--url", f"http://127.0.0.1:{dead}", "--timeout", "3"]) == 2


# --------------------------------------------------------------------------- #
# Exit 1 is reserved for an observed mismatch
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("status", [201, 202, 203, 206, 226, 299])
def test_non_200_success_codes_are_inconclusive(vd: Any, probe: _Probe, status: int) -> None:
    """A 2xx that is not 200 must not pass.

    203 is what a transforming proxy returns when it has rewritten the body,
    and 206 means the body is a fragment — either way the number is not the
    deployment's answer. Every one of these exited 0 before review.
    """
    probe.status = status
    probe.body = b'{"version": "1.2.3"}'
    assert _run(vd, ["--url", probe.url, "--expect", "1.2.3"]) == 2


def test_redirect_is_inconclusive_not_followed(vd: Any, probe: _Probe) -> None:
    probe.redirect_to = "http://127.0.0.1:1/health"
    assert _run(vd, ["--url", probe.url, "--expect", "1.2.3"]) == 2


def test_non_http_scheme_is_inconclusive(vd: Any, tmp_path: Path) -> None:
    doc = tmp_path / "fake.json"
    doc.write_text(json.dumps({"version": "1.2.3"}), encoding="utf-8")
    assert _run(vd, ["--url", "file://", "--path", str(doc), "--expect", "1.2.3"]) == 2


def test_oversized_body_is_inconclusive(vd: Any, probe: _Probe) -> None:
    """Truncating at the cap could leave parseable JSON and a confident answer."""
    padding = b" " * (vd.MAX_BODY_BYTES + 1024)
    probe.body = b'{"version": "1.2.3"}' + padding
    assert _run(vd, ["--url", probe.url, "--expect", "1.2.3"]) == 2


def test_conflicting_version_keys_are_inconclusive(vd: Any, probe: _Probe) -> None:
    probe.body = b'{"version": "1.1.1", "current": "2.2.2"}'
    assert _run(vd, ["--url", probe.url, "--expect", "1.1.1"]) == 2


def test_version_key_of_wrong_type_is_inconclusive(vd: Any, probe: _Probe) -> None:
    probe.body = b'{"version": 1}'
    assert _run(vd, ["--url", probe.url, "--expect", "1"]) == 2


def test_no_version_key_is_inconclusive(vd: Any, probe: _Probe) -> None:
    probe.body = b'{"status": "ok"}'
    assert _run(vd, ["--url", probe.url, "--expect", "1.2.3"]) == 2


def test_non_object_body_is_inconclusive(vd: Any, probe: _Probe) -> None:
    probe.body = b'["1.2.3"]'
    assert _run(vd, ["--url", probe.url, "--expect", "1.2.3"]) == 2


def test_unparseable_body_is_inconclusive(vd: Any, probe: _Probe) -> None:
    probe.body = b"<html>not json</html>"
    assert _run(vd, ["--url", probe.url, "--expect", "1.2.3"]) == 2


def test_bad_timeout_is_inconclusive(vd: Any, probe: _Probe) -> None:
    assert _run(vd, ["--url", probe.url, "--expect", "1.2.3", "--timeout", "-1"]) == 2


def test_empty_expect_is_inconclusive(vd: Any, probe: _Probe) -> None:
    assert _run(vd, ["--url", probe.url, "--expect", "  "]) == 2


def test_whitespace_only_difference_is_not_a_mismatch(vd: Any, probe: _Probe) -> None:
    """Both sides are stripped, so padding cannot fake a version mismatch."""
    probe.body = b'{"version": " 1.2.3\\n"}'
    assert _run(vd, ["--url", probe.url, "--expect", "1.2.3"]) == 0


def test_path_without_leading_slash_still_works(vd: Any, probe: _Probe) -> None:
    probe.body = b'{"version": "1.2.3"}'
    assert _run(vd, ["--url", probe.url, "--path", "health", "--expect", "1.2.3"]) == 0


# --------------------------------------------------------------------------- #
# The credential never travels further than --url
# --------------------------------------------------------------------------- #


def test_health_is_probed_without_the_shared_secret(
    vd: Any, probe: _Probe, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BACKEND_SHARED_SECRET", "probe-value-not-a-real-secret")
    probe.body = b'{"version": "1.2.3"}'
    assert _run(vd, ["--url", probe.url, "--path", "/health", "--expect", "1.2.3"]) == 0
    assert "x-api-key" not in probe.seen_headers[0]


def test_authenticated_path_carries_the_shared_secret(
    vd: Any, probe: _Probe, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BACKEND_SHARED_SECRET", "probe-value-not-a-real-secret")
    probe.body = b'{"current": "1.2.3"}'
    assert _run(vd, ["--url", probe.url, "--path", "/version", "--expect", "1.2.3"]) == 0
    assert probe.seen_headers[0]["x-api-key"] == "probe-value-not-a-real-secret"


def test_http_proxy_in_the_environment_is_ignored(
    vd: Any, probe: _Probe, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The environment must not get to choose which host answers.

    urllib's default ProxyHandler reads `http_proxy`, which would both
    misattribute the verdict and hand `x-api-key` to a host the operator never
    named. The opener uses an empty ProxyHandler, so the probe — not the
    unreachable proxy — answers.
    """
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    probe.body = b'{"version": "1.2.3"}'
    assert _run(vd, ["--url", probe.url, "--expect", "1.2.3"]) == 0
    assert probe.seen_headers, "the probe, not the proxy, must have been contacted"


def test_cache_busting_headers_are_sent(vd: Any, probe: _Probe) -> None:
    probe.body = b'{"version": "1.2.3"}'
    assert _run(vd, ["--url", probe.url, "--expect", "1.2.3"]) == 0
    assert probe.seen_headers[0]["cache-control"] == "no-cache, no-store"


# --------------------------------------------------------------------------- #
# The checkout side
# --------------------------------------------------------------------------- #


def test_checkout_version_reads_pyproject(vd: Any) -> None:
    """Against the real file, so a rename of [project].version fails here."""
    assert vd.checkout_version() == vd.PYPROJECT.read_text(encoding="utf-8").split(
        'version = "', 1
    )[1].split('"', 1)[0]


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(None, id="missing-file"),
        pytest.param('[project]\nversion = "', id="malformed-toml"),
        pytest.param("project = 'not-a-table'\n", id="project-not-a-table"),
        pytest.param("[project]\nversion = 1\n", id="version-not-a-string"),
        pytest.param('[project]\nversion = ""\n', id="version-empty"),
        pytest.param("[project]\nname = 'x'\n", id="version-absent"),
    ],
)
def test_unreadable_checkout_version_is_inconclusive(
    vd: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str | None
) -> None:
    target = tmp_path / "pyproject.toml"
    if content is not None:
        target.write_text(content, encoding="utf-8")
    monkeypatch.setattr(vd, "PYPROJECT", target)
    with pytest.raises(vd.Inconclusive):
        vd.checkout_version()


def test_non_utf8_pyproject_is_inconclusive(
    vd: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """read_text decodes before tomllib sees the bytes, so this is a ValueError."""
    target = tmp_path / "pyproject.toml"
    target.write_bytes(b'\xff\xfe[project]\nversion = "1.2.3"\n')
    monkeypatch.setattr(vd, "PYPROJECT", target)
    with pytest.raises(vd.Inconclusive):
        vd.checkout_version()


def test_checkout_version_is_stripped(
    vd: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "pyproject.toml"
    target.write_text('[project]\nversion = " 1.2.3 "\n', encoding="utf-8")
    monkeypatch.setattr(vd, "PYPROJECT", target)
    assert vd.checkout_version() == "1.2.3"


# --------------------------------------------------------------------------- #
# The advisory tag note: never affects the verdict, never invents a cause
# --------------------------------------------------------------------------- #


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.test",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.test",
            "PATH": "/usr/bin:/bin",
            "HOME": str(repo),
        },
    )


def _commit(repo: Path, name: str) -> None:
    (repo / name).write_text(name, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-m", name)


@pytest.fixture
def tagged_repo(vd: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "a")
    _git(repo, "tag", "v1.0.0")
    monkeypatch.setattr(vd, "REPO_ROOT", repo)
    return repo


def test_tag_note_exact_release_commit(vd: Any, tagged_repo: Path) -> None:
    assert "pins it exactly" in vd.tag_distance("1.0.0")


def test_tag_note_ahead(vd: Any, tagged_repo: Path) -> None:
    _commit(tagged_repo, "b")
    note = vd.tag_distance("1.0.0")
    assert "1 commit(s) past v1.0.0" in note


def test_tag_note_behind(vd: Any, tagged_repo: Path) -> None:
    _commit(tagged_repo, "b")
    _git(tagged_repo, "tag", "-f", "v1.0.0")
    _git(tagged_repo, "checkout", "-q", "HEAD~1")
    assert "behind v1.0.0" in vd.tag_distance("1.0.0")


def test_tag_note_diverged_says_diverged(vd: Any, tagged_repo: Path) -> None:
    """A diverged checkout is missing commits the release has.

    Reporting it as "N past the tag" understates the gap, and was the message
    before review because the ahead branch was tested first.
    """
    _commit(tagged_repo, "release-only")
    _git(tagged_repo, "tag", "-f", "v1.0.0")
    _git(tagged_repo, "checkout", "-q", "-b", "side", "HEAD~1")
    _commit(tagged_repo, "mine-1")
    _commit(tagged_repo, "mine-2")
    note = vd.tag_distance("1.0.0")
    assert "diverged" in note
    assert "past v1.0.0" not in note


def test_tag_note_missing_tag_names_its_cause(vd: Any, tagged_repo: Path) -> None:
    note = vd.tag_distance("9.9.9")
    assert note.startswith("commits since that release: unknown (")
    assert "no matching tag" not in note, "must not assert a cause it did not observe"


def test_tag_note_outside_a_repo_names_its_cause(
    vd: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vd, "REPO_ROOT", tmp_path)
    assert "unknown (" in vd.tag_distance("1.0.0")


def test_tag_note_never_raises_on_a_missing_git(
    vd: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: Any, **_k: Any) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(vd.subprocess, "run", boom)
    assert "no git on PATH" in vd.tag_distance("1.0.0")


# --------------------------------------------------------------------------- #
# The wrapper: a closed stdout must not invent a fourth exit code
# --------------------------------------------------------------------------- #


def test_broken_stdout_preserves_the_mismatch_exit_code(probe: _Probe) -> None:
    """`| head -1` on a mismatch used to surface as CPython's exit 120.

    `PIPESTATUS[0]`, not the pipeline's own status — a trailing `head` exits 0
    and would report a failed check as a passing one, which is the same class
    of mistake this script exists to prevent.
    """
    probe.body = b'{"version": "1.2.3"}'
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"python3 {SCRIPT} --url {probe.url} --expect 9.9.9 | head -1; "
            "exit ${PIPESTATUS[0]}",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
    assert "BrokenPipeError" not in proc.stderr
