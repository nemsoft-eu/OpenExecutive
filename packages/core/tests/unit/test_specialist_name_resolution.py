"""Tolerate the specialist names a small model actually emits.

Measured on qwen3.8:27b: the `specialist` enum value arrives truncated to two
characters (`cs`, `cf`, `cm`) or re-cased (`csO`). One sampled turn emitted
`['cs', 'cf', 'cm']` — three consults, none of which reached a specialist,
because the old code answered any unrecognised name with a plain
"Unknown specialist" string. The turn looked routed and was not.

These tests pin the two tolerances and, just as importantly, the refusal to
guess when a prefix is ambiguous.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-not-used")

from openexecutive.orchestrator.router import (  # noqa: E402
    SPECIALIST_REGISTRY,
    resolve_specialist_name,
    route_parallel,
    route_to_specialist,
)


@pytest.mark.parametrize("name", sorted(SPECIALIST_REGISTRY))
def test_every_registry_key_resolves_to_itself(name: str) -> None:
    assert resolve_specialist_name(name) == name


@pytest.mark.parametrize(
    ("emitted", "expected"),
    [
        ("cs", "cso"),  # observed truncation
        ("cf", "cfo"),  # observed truncation
        ("cm", "cmo"),  # observed truncation
        ("csO", "cso"),  # observed re-casing
        ("CFO", "cfo"),
        ("  cso  ", "cso"),
        ("bo", "board_comms"),
        ("tr", "triage"),
    ],
)
def test_malformed_names_resolve(emitted: str, expected: str) -> None:
    assert resolve_specialist_name(emitted) == expected


@pytest.mark.parametrize("ambiguous", ["c", "t"])
def test_ambiguous_prefixes_refuse_to_guess(ambiguous: str) -> None:
    """`c` spans cso/cfo/chro/coo/cmo/cpo — routing to any one is worse than not routing."""
    assert len({k for k in SPECIALIST_REGISTRY if k.startswith(ambiguous)}) > 1
    assert resolve_specialist_name(ambiguous) is None


@pytest.mark.parametrize("bad", ["", "   ", "chief_financial_officer", "zzz", "ceo"])
def test_unknown_names_stay_unknown(bad: str) -> None:
    assert resolve_specialist_name(bad) is None


@pytest.mark.parametrize("bad", [None, 3, 3.5, True, ["cso"], {"specialist": "cso"}])
def test_non_string_names_are_rejected_not_raised(bad: object) -> None:
    """A local OpenAI-compatible backend passes tool args through json.loads
    unvalidated, so `specialist` can arrive as null, a number or a list. Before
    the isinstance guard this raised out of route_parallel's asyncio.gather and
    failed the whole turn instead of returning a recoverable tool_result."""
    assert resolve_specialist_name(bad) is None  # type: ignore[arg-type]


def test_non_string_name_returns_an_error_string_not_an_exception() -> None:
    with patch("openexecutive.orchestrator.router.audit_log"):
        result = asyncio.run(route_to_specialist(None, query="q"))  # type: ignore[arg-type]
    assert "Unknown specialist" in result


@pytest.mark.parametrize("bad", [["cso"], {"specialist": "cso"}, None, 3])
def test_route_parallel_survives_a_non_string_name(bad: object) -> None:
    """The orchestrator reaches specialists through route_parallel, not
    route_to_specialist, and retrieval runs BEFORE dispatch — an unhashable
    name reached DOMAIN_ALIASES.get() and raised TypeError out of the gather,
    failing the whole turn before route_to_specialist's guard could answer."""
    with patch("openexecutive.orchestrator.router.audit_log"):
        results = asyncio.run(
            route_parallel([{"specialist": bad, "query": "q"}])  # type: ignore[list-item]
        )
    assert len(results) == 1
    assert "Unknown specialist" in results[0]


def test_no_registry_key_prefixes_another() -> None:
    """Prefix resolution silently makes the longer key unreachable if one key
    prefixes another (e.g. adding `cs` beside `cso`). Nothing else would fail,
    so pin the invariant the docstring relies on."""
    keys = sorted(SPECIALIST_REGISTRY)
    offenders = [
        (a, b) for a in keys for b in keys if a != b and b.startswith(a)
    ]
    assert offenders == [], f"key prefixes another, breaking prefix resolution: {offenders}"


def test_route_to_specialist_dispatches_a_truncated_name() -> None:
    """`cs` must reach the real CSO agent, not an error string."""
    cso_mock = AsyncMock(return_value="cso-analysis")
    with patch.object(SPECIALIST_REGISTRY["cso"], "analyze", cso_mock):
        result = asyncio.run(route_to_specialist("cs", query="what now?"))
    assert result == "cso-analysis"
    assert cso_mock.await_count == 1


def test_route_to_specialist_audits_a_normalised_name() -> None:
    with (
        patch.object(SPECIALIST_REGISTRY["cso"], "analyze", AsyncMock(return_value="x")),
        patch("openexecutive.orchestrator.router.audit_log") as audit,
    ):
        asyncio.run(route_to_specialist("cs", query="q"))
    details = [c.kwargs["details"] for c in audit.call_args_list]
    assert {"requested": "cs", "resolved": "cso"} in details


def test_unresolved_name_is_audited_and_lists_valid_names() -> None:
    """The silent no-op is the bug: an unresolved consult must leave a trace."""
    with patch("openexecutive.orchestrator.router.audit_log") as audit:
        result = asyncio.run(route_to_specialist("zzz", query="q"))
    assert "Unknown specialist: zzz" in result
    # The model gets told what it may say instead of failing the same way twice.
    assert "cso" in result and "cfo" in result
    assert audit.call_count == 1
    assert audit.call_args.kwargs["details"]["resolved"] is None


def test_route_parallel_normalises_before_retrieval_and_prefetch() -> None:
    """Normalising only inside route_to_specialist is too late.

    `cs` would still dispatch to the CSO agent, but retrieve() would be called
    with specialist_name="cs" — whose DOMAIN_ALIASES lookup misses and silently
    degrades to UNFILTERED retrieval — the department prefetch would be skipped
    (slug_for_specialist -> None), and consulted_out would be keyed "cs", which
    committee reviewer selection and the Honcho dept sync both drop. That makes
    a truncated name WORSE than the plain error string it used to produce.
    """
    seen: list[str] = []

    def _fake_retrieve(*, query: str, specialist_name: str = "") -> str:
        seen.append(specialist_name)
        return ""

    with (
        patch.object(SPECIALIST_REGISTRY["cso"], "analyze", AsyncMock(return_value="x")),
        patch("openexecutive.knowledge.retriever.retrieve", _fake_retrieve),
        patch("openexecutive.knowledge.retriever.retrieve_failures", _fake_retrieve),
        patch(
            "openexecutive.departments.registry.slug_for_specialist",
            lambda name: seen.append(name) or None,
        ),
        patch("openexecutive.orchestrator.router.audit_log"),
    ):
        asyncio.run(route_parallel([{"specialist": "cs", "query": "q"}]))

    assert seen, "retrieval/prefetch never ran — test would pass vacuously"
    assert "cs" not in seen, f"raw truncated name leaked to a consumer: {seen}"
    assert set(seen) == {"cso"}


def test_unresolved_name_never_reaches_an_agent() -> None:
    cso_mock = AsyncMock(return_value="should-not-run")
    with (
        patch.object(SPECIALIST_REGISTRY["cso"], "analyze", cso_mock),
        patch("openexecutive.orchestrator.router.audit_log"),
    ):
        asyncio.run(route_to_specialist("c", query="q"))
    assert cso_mock.await_count == 0
