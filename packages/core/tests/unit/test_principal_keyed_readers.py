"""A frozen allowlist of the brief-path reads that resolve the OWNER.

``people.store.find_principal_person()`` returns the lowest-id principal. A
reader that calls it takes no recipient, so whatever it returns belongs to one
specific person — and the standing briefs now fan out to every active
principal. Hand such a read's output to an audience of more than one and one
founder's day lands in the other's DM.

That has happened three times on this branch, each time a different reader and
each time caught by a reviewer rather than by the build:

* ``open_loops.principal_due_soon`` and ``top_three.build_top_three`` ran on
  the mode alone, so the solo brief's commitments and calendar titles fanned
  out to co-principals.
* ``live_signals._conversations`` and ``top_three.read_todays_calendar`` were
  gated on "may this READER see private data" without also asking whether the
  data was theirs, so a co-principal running the brief from their own DM
  received the lowest-id founder's chat titles and calendar.

The durable fix is to make these reads take a ``Person`` — then a new one
cannot compile without saying whose data it wants — and that is a real
refactor. Until it lands, this test is the thing that notices: the set of call
sites inside the brief path is pinned, so a newly added owner-resolving reader
fails CI instead of leaking silently.

**What a failure means, and what to do about it.** You added (or moved) a call
to ``find_principal_person`` in a module the standing briefs read from. The
test is not telling you the call is wrong — it is asking you to decide, for
this one call, which of these holds:

1. The value is only used for data everyone on the roster may see (a "is there
   a principal at all" check, a name for a shared line). Add the site to
   ``_ALLOWED`` with a comment saying so.
2. The value is used for data private to that person. Then gate the read on
   the roster the way ``workflows/morning_brief.py`` does — ``private_ok and
   len(active_principals()) <= 1``, both halves, since neither implies the
   other — or give the reader a ``Person`` parameter and pass the recipient.
   Then add the site to ``_ALLOWED``.

Updating the allowlist without making that call is the one thing this test
exists to stop, so say which case it is in the comment.

``find_principal_person`` is deliberately NOT made to raise inside a brief
context: it has around forty call sites, many legitimately non-private and
reachable from inside a brief run (``today._build_today``, ``content_trust``,
``decisions``' "is there a principal at all" checks). That would be a landmine,
not a tripwire.

The repo already parses source for parity assertions this way — see
``tests/unit/test_delegation_reply_send.py`` and the UI's
``packages/ui/scripts/*.test.mjs``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import openexecutive

_ROOT = Path(openexecutive.__file__).parent

# The modules the standing briefs and their blocks read from. Narrow on
# purpose: a `find_principal_person` call in, say, `api/routes/decisions.py` is
# not a brief-path read and is none of this test's business.
_SCOPE: tuple[str, ...] = (
    "briefing",
    "attunement/open_loops.py",
    "memory/history_brief.py",
    "workflows/morning_brief.py",
    "workflows/end_of_day_digest.py",
    "workflows/weekly_review.py",
)

# Every owner-resolving call the brief path is allowed to make, as
# `<path under openexecutive/>::<enclosing function>`. Each is private to the
# lowest-id principal and each is gated on a one-principal roster by its
# caller (`morning_brief.own_private_ok`, `weekly_review`'s commitments step).
_ALLOWED: frozenset[str] = frozenset({
    # The principal's own chat titles, for the brief's CONVERSATIONS block.
    "briefing/live_signals.py::_conversations",
    # Today's events on the principal's own calendar (their address is the
    # `calendar_id`), for TOP THREE TODAY and the brief's CALENDAR block.
    "briefing/top_three.py::read_todays_calendar",
    # The owner's Always-in-the-loop notes. Gated on `PRINCIPAL_DELIVERY`
    # directly, which the scheduler sets only for a single recipient.
    "memory/history_brief.py::owner_keeping_notes",
    # The principal's own dated commitments, for solo's DUE THIS WEEK.
    "attunement/open_loops.py::principal_due_soon",
})

_READER = "find_principal_person"


def _scoped_files() -> list[Path]:
    out: list[Path] = []
    for entry in _SCOPE:
        target = _ROOT / entry
        out.extend(sorted(target.rglob("*.py")) if target.is_dir() else [target])
    return out


class _Sites(ast.NodeVisitor):
    """Collect ``<enclosing function>`` for every ``find_principal_person``
    call, plus any spelling of the call that would hide one.

    A function STACK rather than a walk per function, so each call is
    attributed to its NEAREST enclosing `def` and a nested one is not also
    charged to its parent. A call outside any function — a module-level
    constant, a class attribute, a decorator argument — is attributed to
    ``<module>``, which `_ALLOWED` does not contain, so it fails rather than
    passing unseen.
    """

    def __init__(self) -> None:
        self.stack: list[str] = []
        self.sites: set[str] = set()
        # Spellings that defeat matching on the call's own name. Reported as
        # findings rather than silently missed, since each would let an
        # owner-resolving read into the brief path with this test still green.
        self.evasions: set[str] = set()

    def _where(self) -> str:
        return self.stack[-1] if self.stack else "<module>"

    def _visit_function(self, node: ast.AST) -> None:
        self.stack.append(getattr(node, "name", "<anonymous>"))
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function
    visit_Lambda = _visit_function

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == _READER and alias.asname:
                self.evasions.add(f"imported as `{alias.asname}` — rename it back")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = (
            func.id if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute)
            else ""
        )
        if name == _READER:
            self.sites.add(self._where())
        elif name == "getattr" and any(
            isinstance(a, ast.Constant) and a.value == _READER for a in node.args
        ):
            self.evasions.add(f"reached through getattr in {self._where()}")
        self.generic_visit(node)


def _scan() -> tuple[set[str], set[str]]:
    """``(sites, evasions)`` over the scoped modules. Sites are
    ``<path>::<enclosing function>``; imports and mentions in prose don't
    count, since only a call resolves an owner."""
    sites: set[str] = set()
    evasions: set[str] = set()
    for path in _scoped_files():
        rel = path.relative_to(_ROOT).as_posix()
        visitor = _Sites()
        visitor.visit(ast.parse(path.read_text(), filename=str(path)))
        sites |= {f"{rel}::{where}" for where in visitor.sites}
        evasions |= {f"{rel}: {e}" for e in visitor.evasions}
    return sites, evasions


def _call_sites() -> set[str]:
    return _scan()[0]


def test_no_owner_resolving_read_is_spelled_so_this_test_cannot_see_it() -> None:
    """The allowlist below matches on the call's own name, so an alias or a
    `getattr` would slip a reader past it. Both are banned outright rather
    than chased: there is no reason to spell this call indirectly."""
    assert _scan()[1] == set()


def test_the_brief_paths_owner_resolving_reads_are_the_frozen_set() -> None:
    found = _call_sites()
    new = found - _ALLOWED
    gone = _ALLOWED - found
    assert not new, (
        f"New owner-resolving read(s) in the brief path: {sorted(new)}. "
        f"{_READER}() returns the LOWEST-ID principal and takes no recipient, "
        "while the standing briefs fan out to every active principal — so this "
        "read is one founder's data being assembled for an audience that may "
        "not be them. Read this module's docstring, decide whether the value is "
        "shared or private, gate it if private, then add the site to _ALLOWED "
        "with a comment saying which."
    )
    assert not gone, (
        f"Allowlisted read(s) no longer there: {sorted(gone)}. If the reader now "
        "takes a Person (which is the fix this list is waiting for), drop the "
        "entry — and drop this whole test once _ALLOWED is empty."
    )


def test_the_scope_this_guards_actually_exists() -> None:
    """A renamed module would otherwise silently empty the scope and make the
    assertion above pass over nothing."""
    for entry in _SCOPE:
        assert (_ROOT / entry).exists(), f"{entry} moved — update _SCOPE"
    # Negative control: the pinned set is non-empty and really comes from
    # parsing, not from the literal above.
    assert _call_sites()
