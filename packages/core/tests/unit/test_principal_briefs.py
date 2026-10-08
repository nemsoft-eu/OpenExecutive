"""Tests for the morning_brief / end_of_day_digest workflows and the
scheduler-side seeding + chain-next plumbing.

The workflow LLM calls are stubbed — we verify the registration,
input model shape, scheduler seeding idempotency, time-of-day parsing,
and the chain-on-fire behaviour. Actual LLM-rendered content is out
of scope for unit tests.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from openexecutive.memory import episodic
from openexecutive.scheduler import runner
from openexecutive.workflows import WORKFLOW_REGISTRY


def _setup_isolated_db(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(episodic, "DB_PATH", db)
    episodic.initialize_db(db)


@pytest.fixture(autouse=True)
def _no_default_audit_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """The brief handler audits every outcome; keep those rows out of the
    default ./episodic_memory.db, where they leak into other modules."""
    monkeypatch.setattr("openexecutive.audit.log_event", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# Workflow registration
# ---------------------------------------------------------------------------

def test_morning_brief_registered() -> None:
    assert "morning_brief" in WORKFLOW_REGISTRY
    wf = WORKFLOW_REGISTRY["morning_brief"]
    assert wf.title
    assert wf.description
    assert len(wf.steps()) >= 1
    meta = wf.meta()
    # Input model has at least the period_label field, all optional.
    assert "period_label" in meta.input_schema.get("properties", {})


def test_end_of_day_digest_registered() -> None:
    assert "end_of_day_digest" in WORKFLOW_REGISTRY
    wf = WORKFLOW_REGISTRY["end_of_day_digest"]
    assert wf.title
    meta = wf.meta()
    assert "period_label" in meta.input_schema.get("properties", {})


# ---------------------------------------------------------------------------
# Time-of-day parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("08:00", (8, 0)),
        ("18:30", (18, 30)),
        ("00:00", (0, 0)),
        ("23:59", (23, 59)),
        ("", (8, 0)),  # falls back to default
        ("not-a-time", (8, 0)),
        ("25:00", (8, 0)),  # out of range
        ("12:60", (8, 0)),  # minutes out of range
    ],
)
def test_parse_hhmm(raw: str, expected: tuple[int, int]) -> None:
    assert runner._parse_hhmm(raw, "08:00") == expected


def test_next_occurrence_picks_today_when_target_is_later() -> None:
    base = datetime(2026, 5, 26, 7, 0, tzinfo=UTC)
    result = runner._next_occurrence(base, 8, 0)
    assert result == datetime(2026, 5, 26, 8, 0, tzinfo=UTC)


def test_next_occurrence_advances_to_tomorrow_when_target_passed() -> None:
    base = datetime(2026, 5, 26, 9, 0, tzinfo=UTC)
    result = runner._next_occurrence(base, 8, 0)
    assert result == datetime(2026, 5, 27, 8, 0, tzinfo=UTC)


def test_next_occurrence_advances_when_equal_to_now() -> None:
    """Exactly at target time should advance to tomorrow — strictly future."""
    base = datetime(2026, 5, 26, 8, 0, tzinfo=UTC)
    result = runner._next_occurrence(base, 8, 0)
    assert result == datetime(2026, 5, 27, 8, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Scheduler seeding idempotency
# ---------------------------------------------------------------------------

def test_seed_principal_briefs_inserts_all_recurring_rows_on_fresh_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shift 5 added `executive_reflection` to the seed loop alongside
    the morning and EoD brief. Fresh DB → 3 rows."""
    _setup_isolated_db(tmp_path / "briefs.db", monkeypatch)

    inserted = runner.seed_principal_briefs()
    assert inserted == 3

    actions = episodic.list_scheduled_actions(status="pending", limit=10)
    kinds = {a.kind for a in actions}
    assert "principal_brief_morning" in kinds
    assert "principal_brief_eod" in kinds
    assert "executive_reflection" in kinds


def test_seed_principal_briefs_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_isolated_db(tmp_path / "briefs2.db", monkeypatch)
    runner.seed_principal_briefs()

    # Second call must not duplicate.
    inserted = runner.seed_principal_briefs()
    assert inserted == 0
    pending = [
        a for a in episodic.list_scheduled_actions(status="pending", limit=10)
        if a.kind in (
            "principal_brief_morning",
            "principal_brief_eod",
            "executive_reflection",
        )
    ]
    # Shift 5 added executive_reflection; the idempotency contract now
    # covers all three.
    assert len(pending) == 3


def test_seed_uses_env_var_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When PRINCIPAL_BRIEF_MORNING_TIME is set, the seeded row uses that
    HH:MM rather than the 08:00 default."""
    _setup_isolated_db(tmp_path / "briefs3.db", monkeypatch)
    monkeypatch.setenv("PRINCIPAL_BRIEF_MORNING_TIME", "06:30")
    monkeypatch.setenv("PRINCIPAL_BRIEF_EOD_TIME", "17:15")

    runner.seed_principal_briefs()
    actions = {
        a.kind: a for a in episodic.list_scheduled_actions(status="pending", limit=10)
    }
    morning = actions["principal_brief_morning"]
    eod = actions["principal_brief_eod"]
    # Parse run_at and check HH:MM matches the env vars.
    morning_dt = datetime.fromisoformat(morning.run_at)
    eod_dt = datetime.fromisoformat(eod.run_at)
    assert (morning_dt.hour, morning_dt.minute) == (6, 30)
    assert (eod_dt.hour, eod_dt.minute) == (17, 15)


# ---------------------------------------------------------------------------
# Chain-next on fire
# ---------------------------------------------------------------------------

def test_enqueue_next_principal_brief_advances_24h(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a brief fires at noon, the next occurrence must be tomorrow at
    the configured time-of-day — not 24h literal from now."""
    _setup_isolated_db(tmp_path / "briefs4.db", monkeypatch)
    monkeypatch.setenv("PRINCIPAL_BRIEF_MORNING_TIME", "08:00")

    # Fire-time is "after" — well past today's 08:00.
    after = datetime(2026, 5, 26, 12, 0, tzinfo=UTC)
    aid = runner._enqueue_next_principal_brief("principal_brief_morning", after=after)
    assert aid is not None

    row = episodic.get_scheduled_action(aid)
    assert row is not None
    next_dt = datetime.fromisoformat(row.run_at)
    # Strictly after the fire-time AND at 08:00 UTC.
    assert next_dt > after
    assert next_dt.hour == 8 and next_dt.minute == 0


def test_has_pending_brief_detects_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_isolated_db(tmp_path / "briefs5.db", monkeypatch)
    assert runner._has_pending_brief("principal_brief_morning") is False
    runner.seed_principal_briefs()
    assert runner._has_pending_brief("principal_brief_morning") is True
    assert runner._has_pending_brief("principal_brief_eod") is True


# ---------------------------------------------------------------------------
# Delivered-brief state: the runner records the fingerprint only on delivery
# ---------------------------------------------------------------------------


def _fake_brief_workflow(fingerprint: str, artifact: str = "BRIEF"):
    from openexecutive.workflows.base import WorkflowEvent
    from openexecutive.workflows.morning_brief import MorningBriefInput, MorningBriefWorkflow

    class _Fake(MorningBriefWorkflow):
        async def run(self, inputs, store):  # type: ignore[override]
            yield WorkflowEvent(type="result", data={"brief_fingerprint": fingerprint, "suppressed": False})
            yield WorkflowEvent(type="artifact", content=artifact)
            yield WorkflowEvent(type="done")

        def input_model(self):  # type: ignore[override]
            return MorningBriefInput

    return _Fake()


def _run_brief(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    deliver_ok: bool = True,
    workflow: object | None = None,
    deliver: object | None = None,
    deliver_person: object | None = None,
    principals: int = 1,
) -> list[tuple[int, str]]:
    """Run one morning brief against isolated DBs; returns the recorded sends
    as ``(person_id, text)``, in the order the fan-out made them.

    The seam patched here is ``deliver_to_person`` — the per-recipient one —
    so ``deliver_to_each_principal`` and its ``active_principals()`` lookup
    run for real and the number of sends is the behaviour under test.
    ``principals`` seeds that many principal rows; ``deliver`` is a
    text-only stub shared by every recipient, ``deliver_person`` one that
    sees the Person (for asymmetric outcomes).
    """
    import asyncio

    from openexecutive.briefing import narrative_cache
    from openexecutive.people import registry as people_registry
    from openexecutive.people import store as people_store
    from openexecutive.workflows import persistence as wf_persistence

    db = tmp_path / "brief.db"
    _setup_isolated_db(db, monkeypatch)
    monkeypatch.setattr(narrative_cache, "DB_PATH", tmp_path / "cache.db")
    monkeypatch.setattr(wf_persistence, "DB_PATH", db)
    monkeypatch.setattr(people_store, "DB_PATH", db)
    people_store.initialize_db(db)
    people_registry.invalidate()
    for n in range(principals):
        people_store.upsert_person(
            full_name=f"Founder {n}", is_principal=True, slack_user_id=f"U{n}"
        )
    monkeypatch.setitem(
        WORKFLOW_REGISTRY, "morning_brief", workflow or _fake_brief_workflow("fp-123")
    )

    async def _deliver(text: str, **_kw: object) -> runner.PrincipalDelivery:
        if deliver_ok:
            return runner.PrincipalDelivery(True, "discord_dm → 1", "delivered", "discord_dm")
        return runner.PrincipalDelivery(False, "delivery failed", "send_failed")

    sends: list[tuple[int, str]] = []
    per_person = deliver_person
    by_text = deliver or _deliver

    async def _recording(person, text: str, **kw: object) -> runner.PrincipalDelivery:  # type: ignore[no-untyped-def]
        sends.append((person.id, text))
        if per_person is not None:
            return await per_person(person, text, **kw)  # type: ignore[operator]
        return await by_text(text, **kw)  # type: ignore[operator]

    monkeypatch.setattr(runner, "deliver_to_person", _recording)
    monkeypatch.setattr(runner, "_enqueue_next_principal_brief", lambda kind, after: None)

    class _Store:
        def __init__(self, **kw): ...

    import openexecutive.knowledge.store as kstore

    monkeypatch.setattr(kstore, "ChromaDBStore", _Store)

    action_id = episodic.insert_scheduled_action(
        run_at=datetime.now(UTC).isoformat(), channel="__internal__", channel_ref="principal",
        intent_text="brief", kind="principal_brief_morning",
    )
    action = episodic.get_scheduled_action(action_id)
    assert action is not None
    asyncio.run(runner._run_principal_brief(action, datetime.now(UTC)))
    return sends


def test_run_principal_brief_records_fingerprint_after_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.briefing import brief_state

    _run_brief(tmp_path, monkeypatch, deliver_ok=True)
    last = brief_state.last_delivered("principal_brief_morning")
    assert last is not None and last.input_hash == "fp-123" and last.narrative_text == "BRIEF"
    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None
    assert (outcome.kind, outcome.reason, outcome.channel) == (
        "principal_brief_morning", "delivered", "discord_dm",
    )


def test_run_principal_brief_does_not_record_on_delivery_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.briefing import brief_state

    _run_brief(tmp_path, monkeypatch, deliver_ok=False)
    assert brief_state.last_delivered("principal_brief_morning") is None
    # The failure is still recorded, for the Briefing's "not sent" notice.
    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None and (outcome.reason, outcome.channel) == ("send_failed", None)


def _failing_brief_workflow(event: str):  # type: ignore[no-untyped-def]
    from openexecutive.workflows.base import WorkflowEvent
    from openexecutive.workflows.morning_brief import MorningBriefInput, MorningBriefWorkflow

    class _Fails(MorningBriefWorkflow):
        async def run(self, inputs, store):  # type: ignore[override]
            if event == "error":
                yield WorkflowEvent(type="error", message="model unavailable")
            yield WorkflowEvent(type="done")  # "empty": no artifact at all

        def input_model(self):  # type: ignore[override]
            return MorningBriefInput

    return _Fails()


@pytest.mark.parametrize("event", ["error", "empty"])
def test_a_brief_that_couldnt_be_written_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, event: str
) -> None:
    from openexecutive.briefing import brief_state

    sends: list[str] = []

    async def _deliver(text: str, **_kw: object) -> runner.PrincipalDelivery:
        sends.append(text)
        return runner.PrincipalDelivery(True, "email → x", "delivered", "email")

    _run_brief(tmp_path, monkeypatch, workflow=_failing_brief_workflow(event), deliver=_deliver)
    assert sends == []
    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None and outcome.reason == "not_written"


def test_a_send_that_crashes_is_recorded_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.briefing import brief_state

    async def _crash(text: str, **_kw: object) -> runner.PrincipalDelivery:
        raise RuntimeError("people store locked")

    _run_brief(tmp_path, monkeypatch, deliver=_crash)
    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None and outcome.reason == "send_failed"


# ---------------------------------------------------------------------------
# Delivery: preferred channel honoured, email via the MCP gateway, and no
# synthesis when nothing can reach the principal
# ---------------------------------------------------------------------------


class _Sent:
    """Records every outbound send the delivery path makes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []  # type: ignore[type-arg]


class _Gateway:
    def __init__(self, sent: _Sent, result: str = "Email sent! Message ID: m-1") -> None:
        self._sent = sent
        self._result = result

    async def call_tool(self, tool_input: dict) -> str:  # type: ignore[type-arg]
        self._sent.calls.append(("gmail", tool_input))
        return self._result


@pytest.fixture
def sent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Sent:
    """Isolated people DB, recording chat send handlers, no MCP gateway."""
    import json

    from openexecutive.orchestrator import mcp_gateway, schedule_tools
    from openexecutive.people import registry as people_registry
    from openexecutive.people import store as people_store

    db = tmp_path / "people.db"
    _setup_isolated_db(db, monkeypatch)
    monkeypatch.setattr(people_store, "DB_PATH", db)
    people_store.initialize_db(db)
    people_registry.invalidate()
    monkeypatch.setattr(mcp_gateway, "_active_gateway", None)

    rec = _Sent()

    def _handler(name: str):  # type: ignore[no-untyped-def]
        async def _send(args: dict) -> str:  # type: ignore[type-arg]
            rec.calls.append((name, args))
            return json.dumps({"status": "sent"})
        return _send

    monkeypatch.setattr(schedule_tools, "handle_send_slack_dm", _handler("slack"))
    monkeypatch.setattr(schedule_tools, "handle_send_discord_dm", _handler("discord"))
    monkeypatch.setattr(schedule_tools, "handle_send_telegram_message", _handler("telegram"))
    return rec


def _principal(**fields: object) -> int:
    from openexecutive.people import store as people_store

    return people_store.upsert_person(full_name="Owner", is_principal=True, **fields)  # type: ignore[arg-type]


def _with_gateway(monkeypatch: pytest.MonkeyPatch, sent: _Sent, **kw: str) -> None:
    """An MCP gateway running the Google Workspace server, so email is ready."""
    from openexecutive.orchestrator import mcp_gateway

    monkeypatch.setattr(mcp_gateway, "_active_gateway", _Gateway(sent, **kw))
    monkeypatch.setattr(mcp_gateway, "configured_server_names", lambda _path: ["google_workspace"])


def _deliver(text: str = "BRIEF", label: str = "Morning Brief") -> tuple[bool, str]:
    """The channel-order tests below seed exactly one principal, so the
    fan-out's single result is that person's. Driving them through
    ``deliver_to_each_principal`` rather than ``deliver_to_person`` keeps
    them on the seam production actually uses."""
    result = _deliver_result(text, label=label)
    return result.ok, result.detail


def test_preferred_slack_is_honoured(sent: _Sent) -> None:
    """`preferred_channel` is "slack", not "slack_dm" — it used to match nothing."""
    _principal(
        preferred_channel="slack", slack_user_id="U1", discord_user_id="D1",
        telegram_chat_id="42",
    )
    ok, detail = _deliver()
    assert ok and detail == "slack_dm → U1"
    assert [name for name, _ in sent.calls] == ["slack"]


def test_preferred_discord_goes_ahead_of_slack(sent: _Sent) -> None:
    _principal(preferred_channel="discord", slack_user_id="U1", discord_user_id="D1")
    ok, _ = _deliver()
    assert ok
    assert [name for name, _ in sent.calls] == ["discord"]


def test_preferred_email_sends_through_the_gateway(
    sent: _Sent, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.config import get_settings

    _principal(preferred_channel="email", email="owner@example.com", slack_user_id="U1")
    _with_gateway(monkeypatch, sent)

    ok, detail = _deliver("**the** brief", label="Morning Brief")

    assert ok and detail == "email → owner@example.com"
    ((name, call),) = sent.calls
    assert name == "gmail"
    assert call["name"] == "google_workspace__send_gmail_message"
    args = call["arguments"]
    assert args["user_google_email"] == get_settings().exec_email_address
    assert args["to"] == "owner@example.com"
    assert args["subject"] == runner._email_subject("Morning Brief")
    # Formatted, not the raw Markdown.
    assert args["body_format"] == "html"
    assert "<strong>the</strong> brief" in args["body"] and "**" not in args["body"]


def test_email_error_falls_back_to_chat(sent: _Sent, monkeypatch: pytest.MonkeyPatch) -> None:
    _principal(preferred_channel="email", email="owner@example.com", slack_user_id="U1")
    _with_gateway(monkeypatch, sent, result='{"error": "recipient not on the roster"}')

    ok, detail = _deliver()

    assert ok and detail == "slack_dm → U1"
    assert [name for name, _ in sent.calls] == ["gmail", "slack"]


def test_any_with_only_an_email_uses_email(sent: _Sent, monkeypatch: pytest.MonkeyPatch) -> None:
    _principal(email="owner@example.com")  # preferred_channel defaults to "any"
    _with_gateway(monkeypatch, sent)

    ok, detail = _deliver()

    assert ok and detail == "email → owner@example.com"


def test_any_with_a_chat_channel_never_emails(
    sent: _Sent, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An owner linked by email at setup (preference "any") who also has Slack
    must not get the briefs by email too while Slack works."""
    _principal(email="owner@example.com", slack_user_id="U1")
    _with_gateway(monkeypatch, sent)

    ok, detail = _deliver()

    assert ok and detail == "slack_dm → U1"
    assert [name for name, _ in sent.calls] == ["slack"]


def test_email_is_the_backup_when_the_preferred_chat_is_not_connected(
    sent: _Sent, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _principal(preferred_channel="slack", email="owner@example.com")  # no Slack id
    _with_gateway(monkeypatch, sent)

    ok, detail = _deliver()

    assert ok and detail == "email → owner@example.com"
    assert [name for name, _ in sent.calls] == ["gmail"]


def test_email_is_the_backup_when_every_chat_send_fails(
    sent: _Sent, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from openexecutive.orchestrator import schedule_tools

    _principal(preferred_channel="slack", slack_user_id="U1", email="owner@example.com")
    _with_gateway(monkeypatch, sent)

    async def _slack_down(args: dict) -> str:  # type: ignore[type-arg]
        sent.calls.append(("slack", args))
        return json.dumps({"error": "token revoked"})

    monkeypatch.setattr(schedule_tools, "handle_send_slack_dm", _slack_down)

    ok, detail = _deliver()

    assert ok and detail == "email → owner@example.com"
    assert [name for name, _ in sent.calls] == ["slack", "gmail"]


def _deliver_result(text: str = "BRIEF", label: str = "Update") -> runner.PrincipalDelivery:
    import asyncio

    results = asyncio.run(runner.deliver_to_each_principal(text, label=label))
    assert len(results) == 1, f"expected one principal, got {len(results)}"
    return results[0][1]


def test_no_owner_sends_nothing_at_all(sent: _Sent) -> None:
    """With no principal there is no per-recipient result to carry a reason;
    the empty fan-out is the signal, which ``_run_principal_brief`` records
    as ``no_owner`` (see test_no_principal_at_all_is_recorded_as_no_owner)."""
    import asyncio

    assert asyncio.run(runner.deliver_to_each_principal("BRIEF")) == []
    assert sent.calls == []


def test_nothing_connected_is_its_own_reason(sent: _Sent) -> None:
    _principal(email="owner@example.com")  # no gateway, no chat
    result = _deliver_result()
    assert (result.ok, result.reason) == (False, "no_channel")


def test_a_failed_send_is_its_own_reason(sent: _Sent, monkeypatch: pytest.MonkeyPatch) -> None:
    from openexecutive.orchestrator import schedule_tools

    async def _broken(args: dict) -> str:  # type: ignore[type-arg]
        raise RuntimeError("telegram down")

    _principal(telegram_chat_id="42")
    monkeypatch.setattr(schedule_tools, "handle_send_telegram_message", _broken)
    result = _deliver_result()
    assert (result.ok, result.reason, result.channel) == (False, "send_failed", None)


def test_the_email_subject_carries_the_users_date(monkeypatch: pytest.MonkeyPatch) -> None:
    from zoneinfo import ZoneInfo

    from openexecutive.memory import workspace_settings

    monkeypatch.setattr(
        workspace_settings, "get_user_timezone", lambda *_a: ZoneInfo("America/Los_Angeles")
    )
    # 02:00 UTC on Saturday is still Friday evening in California.
    evening = datetime(2026, 9, 26, 2, 0, tzinfo=UTC)
    assert runner._email_subject("End-of-Day Digest", evening) == "End-of-Day Digest — Fri 25 Sep"


def test_email_without_a_gateway_is_not_delivered(sent: _Sent) -> None:
    _principal(preferred_channel="email", email="owner@example.com")

    ok, detail = _deliver()

    assert not ok and "no deliverable channel" in detail
    assert sent.calls == []


def test_client_digest_still_delivers(sent: _Sent, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from openexecutive.clients import rotation

    _principal(preferred_channel="email", email="owner@example.com")
    _with_gateway(monkeypatch, sent)

    async def _rotate(_settings: object) -> dict:  # type: ignore[type-arg]
        return {"ran": True, "rotated": ["acme"], "failed": {}, "digest": "# Across your clients"}

    monkeypatch.setattr(rotation, "run_client_rotation", _rotate)
    monkeypatch.setattr(rotation, "seed_client_rotation", lambda: None)
    action_id = episodic.insert_scheduled_action(
        run_at=datetime.now(UTC).isoformat(), channel="__internal__",
        channel_ref="client_rotation", intent_text="rotate", kind="client_rotation",
    )
    action = episodic.get_scheduled_action(action_id)
    assert action is not None

    asyncio.run(runner._execute_action(action, None))

    ((_, call),) = sent.calls
    assert call["arguments"]["subject"].startswith("Across your clients — ")
    assert "<h1>Across your clients</h1>" in call["arguments"]["body"]


def test_brief_with_no_deliverable_channel_is_still_generated_and_stored(
    sent: _Sent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A web-only principal still gets the brief run and artifact (read on the
    Artifacts page); it is just not delivered, so the window does not move,
    and the next occurrence is chained."""
    import asyncio

    from openexecutive.alerts import review
    from openexecutive.briefing import brief_state, narrative_cache
    from openexecutive.workflows import persistence as wf_persistence
    from openexecutive.workflows.base import WorkflowEvent
    from openexecutive.workflows.morning_brief import MorningBriefInput, MorningBriefWorkflow

    _principal(email="owner@example.com")  # "any", an email, but no gateway
    monkeypatch.setattr(narrative_cache, "DB_PATH", tmp_path / "cache.db")
    monkeypatch.setattr(wf_persistence, "DB_PATH", episodic.DB_PATH)
    wf_persistence.initialize_runs_db(episodic.DB_PATH)
    ran: list[str] = []

    class _Brief(MorningBriefWorkflow):
        async def run(self, inputs, store):  # type: ignore[override]
            ran.append("brief")
            yield WorkflowEvent(type="result", data={"brief_fingerprint": "fp", "suppressed": False})
            yield WorkflowEvent(type="artifact", content="BRIEF")

        def input_model(self):  # type: ignore[override]
            return MorningBriefInput

    async def _review(**_kw: object) -> None:
        ran.append("review")

    class _NoStore:
        def __init__(self, **_kw: object) -> None: ...

    monkeypatch.setitem(WORKFLOW_REGISTRY, "morning_brief", _Brief())
    monkeypatch.setattr(review, "run_alert_review", _review)
    monkeypatch.setattr("openexecutive.knowledge.store.ChromaDBStore", _NoStore)
    chained: list[str] = []
    monkeypatch.setattr(
        runner, "_enqueue_next_principal_brief", lambda kind, after: chained.append(kind),
    )
    action_id = episodic.insert_scheduled_action(
        run_at=datetime.now(UTC).isoformat(), channel="__internal__", channel_ref="principal",
        intent_text="brief", kind="principal_brief_morning",
    )
    action = episodic.get_scheduled_action(action_id)
    assert action is not None

    asyncio.run(runner._run_principal_brief(action, datetime.now(UTC)))

    assert ran == ["review", "brief"]
    assert sent.calls == []  # nothing delivered
    (run,) = wf_persistence.list_runs(workflow_name="morning_brief")
    assert run["status"] == "done"
    stored = wf_persistence.get_run(run["run_id"])
    assert stored is not None and stored["artifact"] == "BRIEF"
    # Only a delivered brief advances the window.
    assert brief_state.last_delivered("principal_brief_morning") is None
    # Why it wasn't sent is kept for the Briefing's notice.
    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None and outcome.reason == "no_channel"
    row = episodic.get_scheduled_action(action_id)
    assert row is not None and row.status == "done"
    assert chained == ["principal_brief_morning"]


def test_a_private_brief_is_delivered_whole_but_kept_out_of_run_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scheduler runs the brief for the principal alone (PRINCIPAL_DELIVERY),
    so it may read their private mail; a brief that did keeps its text out of
    the shared run history, while the principal still gets all of it."""
    from openexecutive.workflows import persistence as wf_persistence
    from openexecutive.workflows.base import WorkflowEvent
    from openexecutive.workflows.morning_brief import (
        PRINCIPAL_DELIVERY,
        MorningBriefInput,
        MorningBriefWorkflow,
    )

    seen_flag: list[bool] = []

    class _Private(MorningBriefWorkflow):
        async def run(self, inputs, store):  # type: ignore[override]
            seen_flag.append(PRINCIPAL_DELIVERY.get())
            yield WorkflowEvent(type="result", data={
                "brief_fingerprint": "fp", "suppressed": False, "private_to_principal": True,
            })
            yield WorkflowEvent(type="artifact", content="PRIVATE BRIEF")
            yield WorkflowEvent(type="done")

        def input_model(self):  # type: ignore[override]
            return MorningBriefInput

    delivered: list[str] = []

    async def _deliver(text: str, **_kw: object) -> runner.PrincipalDelivery:
        delivered.append(text)
        return runner.PrincipalDelivery(True, "discord_dm → 1", "delivered", "discord_dm")

    _run_brief(tmp_path, monkeypatch, workflow=_Private(), deliver=_deliver)

    assert seen_flag == [True]
    assert PRINCIPAL_DELIVERY.get() is False  # reset after the run
    assert delivered == ["PRIVATE BRIEF"]
    runs = wf_persistence.list_runs(workflow_name="morning_brief")
    stored = wf_persistence.get_run(runs[0]["run_id"])
    assert stored is not None and stored["artifact"] == wf_persistence.PRIVATE_RUN_ARTIFACT


# ---------------------------------------------------------------------------
# Co-principals: both founders get the brief, and the private-content flag
# drops away as soon as there is more than one recipient
# ---------------------------------------------------------------------------


def test_each_principal_is_dmed_on_their_own_channel(sent: _Sent) -> None:
    """The reason this fan-out exists: with two founders the brief used to
    reach only the lowest-id row. Each result is reported separately, and each
    send goes to that person's OWN id, never twice to the first one's."""
    import asyncio

    from openexecutive.people import store as people_store

    people_store.upsert_person(full_name="Maarten", is_principal=True, slack_user_id="UMAARTEN")
    people_store.upsert_person(full_name="Nick", is_principal=True, slack_user_id="UNICK")

    results = asyncio.run(runner.deliver_to_each_principal("BRIEF", label="Morning Brief"))

    assert [p.full_name for p, _ in results] == ["Maarten", "Nick"]
    assert all(d.ok and d.channel == "slack_dm" for _, d in results)
    assert [args["user_id"] for name, args in sent.calls if name == "slack"] == [
        "UMAARTEN", "UNICK",
    ]


def test_a_founder_offboarded_mid_fan_out_is_not_sent_to(
    sent: _Sent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fan-out awaits one network send per recipient, so the roster read
    before the loop is already stale by the time a later founder is reached.
    Archiving Nick while Maarten's Slack call is in flight must stop Nick's
    send. The Slack handler refuses an archived id of its own accord now, so
    this is about the fan-out's own manners — Nick is dropped quietly here
    rather than sent to and refused — and about a SHARED run, where no
    private-egress gate applies at all.

    Nick is DROPPED from the results rather than carrying a reason, which is
    what the audience cap means: he is not a recipient of this run. See
    `test_an_offboarded_founder_does_not_poison_the_runs_aggregate` for why
    a per-recipient reason here cannot work."""
    import asyncio

    from openexecutive.orchestrator import schedule_tools
    from openexecutive.people import store as people_store

    people_store.upsert_person(full_name="Maarten", is_principal=True, slack_user_id="UMAARTEN")
    nick = people_store.upsert_person(
        full_name="Nick", is_principal=True, slack_user_id="UNICK"
    )

    import json

    async def _send_then_offboard_nick(args: dict) -> str:  # type: ignore[type-arg]
        sent.calls.append(("slack", args))
        # Nick is archived DURING Maarten's send, which is exactly the window
        # the pre-loop snapshot cannot see.
        if args["user_id"] == "UMAARTEN":
            people_store.archive_person(nick)
        return json.dumps({"status": "sent"})

    monkeypatch.setattr(schedule_tools, "handle_send_slack_dm", _send_then_offboard_nick)

    results = asyncio.run(runner.deliver_to_each_principal("BRIEF", label="Morning Brief"))

    assert [args["user_id"] for _, args in sent.calls] == ["UMAARTEN"]  # never UNICK
    assert [(p.full_name, d.reason) for p, d in results] == [("Maarten", "delivered")]


def test_a_reassigned_slack_id_is_read_fresh_at_send_time(
    sent: _Sent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half: the row survives but its channel id changed inside the
    window. The send must use the id the roster holds now, not the snapshot's,
    or the brief goes to whoever was given the old handle."""
    import asyncio
    import json

    from openexecutive.orchestrator import schedule_tools
    from openexecutive.people import store as people_store

    people_store.upsert_person(full_name="Maarten", is_principal=True, slack_user_id="UMAARTEN")
    nick = people_store.upsert_person(
        full_name="Nick", is_principal=True, slack_user_id="UOLD"
    )

    async def _send_then_rotate_nicks_id(args: dict) -> str:  # type: ignore[type-arg]
        sent.calls.append(("slack", args))
        if args["user_id"] == "UMAARTEN":
            people_store.upsert_person(
                full_name="Nick", is_principal=True, slack_user_id="UNEW", person_id=nick
            )
        return json.dumps({"status": "sent"})

    monkeypatch.setattr(schedule_tools, "handle_send_slack_dm", _send_then_rotate_nicks_id)

    asyncio.run(runner.deliver_to_each_principal("BRIEF", label="Morning Brief"))

    assert [args["user_id"] for _, args in sent.calls] == ["UMAARTEN", "UNEW"]


def test_a_founder_demoted_mid_fan_out_is_not_sent_to(
    sent: _Sent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Being archived is not the only way to stop being a recipient. An
    onboarding re-run rewrites `is_principal`, and a DEMOTED founder's row
    stays right there, unarchived — so a revalidation that tests `archived`
    alone sends them the standing brief, and the channel handlers' roster
    checks wave it through because an ordinary team member is on the roster.

    `active_principals()` filters on `is_principal` as well as `archived` (and
    on `kind = 'team'`), so the fix is to ask it rather than to restate it."""
    import asyncio
    import json

    from openexecutive.orchestrator import schedule_tools
    from openexecutive.people import store as people_store

    people_store.upsert_person(full_name="Maarten", is_principal=True, slack_user_id="UMAARTEN")
    nick = people_store.upsert_person(
        full_name="Nick", is_principal=True, slack_user_id="UNICK"
    )

    async def _send_then_demote_nick(args: dict) -> str:  # type: ignore[type-arg]
        sent.calls.append(("slack", args))
        if args["user_id"] == "UMAARTEN":
            people_store.upsert_person(
                full_name="Nick", is_principal=False, slack_user_id="UNICK", person_id=nick
            )
        return json.dumps({"status": "sent"})

    monkeypatch.setattr(schedule_tools, "handle_send_slack_dm", _send_then_demote_nick)

    results = asyncio.run(runner.deliver_to_each_principal("BRIEF", label="Morning Brief"))

    assert [args["user_id"] for _, args in sent.calls] == ["UMAARTEN"]  # never UNICK
    assert [(p.full_name, d.reason) for p, d in results] == [("Maarten", "delivered")]
    # Demoted, not offboarded: the row is still on the roster and unarchived,
    # which is exactly what an `archived`-only check cannot see.
    fresh = people_store.get_person(nick)
    assert fresh is not None and not fresh.archived and not fresh.is_principal


def test_an_archived_principal_gets_no_brief(sent: _Sent) -> None:
    """Off-boarding a founder has to stop the standing report too — otherwise
    it keeps DMing someone who no longer runs the company."""
    import asyncio

    from openexecutive.people import store as people_store

    people_store.upsert_person(full_name="Stays", is_principal=True, slack_user_id="USTAYS")
    gone = people_store.upsert_person(full_name="Gone", is_principal=True, slack_user_id="UGONE")
    people_store.archive_person(gone)

    results = asyncio.run(runner.deliver_to_each_principal("BRIEF"))

    assert [p.full_name for p, _ in results] == ["Stays"]
    assert [args["user_id"] for _, args in sent.calls] == ["USTAYS"]


def test_two_principals_make_the_brief_shared_not_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PRINCIPAL_DELIVERY unlocks ONE person's mail, drafts and calendar (every
    private reader resolves the owner through ``find_principal_person``), so
    fanning a private run out would put Maarten's inbox in Nick's DM. With
    co-principals the brief must be generated shared."""
    from openexecutive.workflows.morning_brief import PRINCIPAL_DELIVERY

    seen: list[bool] = []
    workflow = _flag_observing_workflow(seen)

    _run_brief(tmp_path, monkeypatch, workflow=workflow, principals=2)
    assert seen == [False]

    seen.clear()
    solo = tmp_path / "solo"
    solo.mkdir()
    _run_brief(solo, monkeypatch, workflow=workflow, principals=1)
    assert seen == [True]
    assert PRINCIPAL_DELIVERY.get() is False  # reset after the run


def _flag_observing_workflow(seen: list[bool]):  # type: ignore[no-untyped-def]
    from openexecutive.workflows.base import WorkflowEvent
    from openexecutive.workflows.morning_brief import (
        PRINCIPAL_DELIVERY,
        MorningBriefInput,
        MorningBriefWorkflow,
    )

    class _Observes(MorningBriefWorkflow):
        async def run(self, inputs, store):  # type: ignore[override]
            seen.append(PRINCIPAL_DELIVERY.get())
            yield WorkflowEvent(type="result", data={"brief_fingerprint": "fp-123"})
            yield WorkflowEvent(type="artifact", content="BRIEF")

        def input_model(self):  # type: ignore[override]
            return MorningBriefInput

    return _Observes()


# ---------------------------------------------------------------------------
# The private brief's guarantee lives at the EGRESS
#
# Five review rounds found variants of "the recipient row went stale across an
# await", each answered by moving the caller's snapshot closer to the send. A
# snapshot cannot be await-safe at any granularity, so `_run_principal_brief`
# now raises `people_tools.restrict_to_principal()` around the fan-out and
# every channel leg re-reads `is_principal` immediately before its own call.
# The fan-out's own revalidation (`runner._live_principal`) is a courtesy
# pre-filter from here on, so these tests stub it out on purpose: what is
# under test is what happens when it does NOT catch the recipient.
# ---------------------------------------------------------------------------


@pytest.fixture
def slack_reached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Isolated people DB plus the REAL `handle_send_slack_dm`, Slack API
    stubbed. Yields the list of user ids that reached `chat_postMessage` —
    empty means the handler refused before the network call.

    Deliberately NOT the `sent` fixture: that replaces the handler wholesale,
    which is exactly the seam an egress test must not stub.
    """
    from openexecutive.orchestrator import mcp_gateway, schedule_tools
    from openexecutive.people import registry as people_registry
    from openexecutive.people import store as people_store

    db = tmp_path / "people.db"
    _setup_isolated_db(db, monkeypatch)
    monkeypatch.setattr(people_store, "DB_PATH", db)
    people_store.initialize_db(db)
    people_registry.invalidate()
    monkeypatch.setattr(mcp_gateway, "_active_gateway", None)  # email not ready

    reached: list[str] = []

    class _S:
        slack_bot_token = "xoxb-test"

    class _FakeClient:
        def __init__(self, token: str) -> None:
            self.token = token

        async def chat_postMessage(self, channel: str, text: str) -> dict:  # type: ignore[type-arg]
            reached.append(channel)
            return {"ok": True, "ts": "1.0"}

    monkeypatch.setattr("openexecutive.config.get_settings", lambda: _S())
    monkeypatch.setattr("slack_sdk.web.async_client.AsyncWebClient", _FakeClient)
    # The anti-spam guard is a separate concern and needs settings this stub
    # does not carry; the refusal under test happens before it either way.
    monkeypatch.setattr(schedule_tools, "_guard_outbound", lambda **_kw: None)
    return reached


def test_a_demoted_team_member_is_refused_for_a_private_brief(
    slack_reached: list[str],
) -> None:
    """The case `restrict_to_principal` closes and the roster gate cannot.

    A founder DEMOTED rather than offboarded keeps a non-archived `kind =
    'team'` row, so "is this id on the People roster?" says yes — which is all
    the unconditional roster gate asks. Only the private-turn question
    ("is this id the principal's?") refuses them, and that is the question a
    brief carrying one person's mail, calendar, notes and drafts has to ask.
    The control below is the point: the same call outside the block sends.
    """
    import asyncio
    import json

    from openexecutive.orchestrator import schedule_tools
    from openexecutive.orchestrator.people_tools import (
        PRIVATE_TURN_REFUSAL,
        restrict_to_principal,
    )
    from openexecutive.people import store as people_store

    reached = slack_reached
    nick = people_store.upsert_person(
        full_name="Nick", is_principal=True, slack_user_id="UNICK"
    )
    people_store.upsert_person(
        full_name="Nick", is_principal=False, slack_user_id="UNICK", person_id=nick
    )
    row = people_store.get_person(nick)
    assert row is not None and row.kind == "team" and not row.archived

    args = {"user_id": "UNICK", "text": "PRIVATE BRIEF"}
    with restrict_to_principal():
        refused = json.loads(asyncio.run(schedule_tools.handle_send_slack_dm(args)))
    assert refused == {"error": PRIVATE_TURN_REFUSAL}
    assert reached == []

    # Control: the roster gate alone waves this very recipient through.
    allowed = json.loads(asyncio.run(schedule_tools.handle_send_slack_dm(args)))
    assert allowed["status"] == "sent"
    assert reached == ["UNICK"]


def test_an_archived_recipient_is_refused_at_the_egress_not_by_the_pre_filter(
    slack_reached: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the fan-out's revalidation stubbed to a passthrough — standing in
    for any future caller that forgets it, or for a row that goes stale after
    it ran — the Slack handler still refuses an archived recipient, and the
    network call never happens."""
    import asyncio

    from openexecutive.orchestrator.people_tools import restrict_to_principal
    from openexecutive.people import store as people_store

    reached = slack_reached
    monkeypatch.setattr(runner, "_live_principal", lambda person: person)

    gone = people_store.upsert_person(
        full_name="Gone", is_principal=True, slack_user_id="UGONE"
    )
    pinned = people_store.active_principals()
    people_store.archive_person(gone)

    with restrict_to_principal():
        results = asyncio.run(
            runner.deliver_to_each_principal("PRIVATE BRIEF", recipients=pinned)
        )

    assert reached == []  # nothing reached Slack
    assert [(p.full_name, d.ok, d.reason) for p, d in results] == [
        ("Gone", False, "send_failed")
    ]


def test_every_brief_fan_out_is_gated_at_the_egress_shared_or_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audience and content are different concerns. `PRINCIPAL_DELIVERY`
    decides what the artifact may CONTAIN; `restrict_to_principal` decides who
    it may REACH, and a brief fan-out addresses principals either way.

    An earlier revision raised the gate only for a single-recipient run, on the
    reasoning that gating a shared brief would refuse each co-founder's own leg
    because the handlers would ask "is this id *the* principal's" of a roster
    with two. That reasoning was wrong: `_dm_recipient_on_roster` asks
    `is_principal`, which every recipient out of `active_principals()` has. The
    conditional bought nothing and left a co-principal demoted mid-send still
    receiving a shared brief off a fallback leg."""
    from openexecutive.orchestrator.people_tools import turn_is_private_to_principal

    private_at_send: list[bool] = []

    async def _observe(person, text: str, **_kw: object) -> runner.PrincipalDelivery:  # type: ignore[no-untyped-def]
        private_at_send.append(turn_is_private_to_principal())
        return runner.PrincipalDelivery(True, "slack_dm → U", "delivered", "slack_dm")

    _run_brief(tmp_path, monkeypatch, deliver_person=_observe, principals=2)
    assert private_at_send == [True, True]  # shared artifact, still principal-only egress

    private_at_send.clear()
    solo = tmp_path / "solo"
    solo.mkdir()
    _run_brief(solo, monkeypatch, deliver_person=_observe, principals=1)
    assert private_at_send == [True]
    # And the flag does not outlive the send.
    assert turn_is_private_to_principal() is False


def test_a_demoted_co_principal_is_refused_by_the_shared_briefs_egress(
    sent: _Sent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gap that closing the `private_run` conditional fixes, asserted at
    the handler because that is the only place it is observable.

    Going through `deliver_to_each_principal` would prove nothing: both
    `active_principals()` and the pinned-list intersect already drop a demoted
    row, so such a test passes with the egress gate entirely removed (verified
    by mutation). The gate earns its keep only in the window those filters
    cannot see — demotion between the intersect and a later channel leg — so
    the behaviour under test is the handler's own fresh read: a demoted but
    still-active team member must be refused when the turn is restricted,
    and accepted when it is not."""
    from openexecutive.orchestrator.people_tools import restrict_to_principal
    from openexecutive.orchestrator.schedule_tools import _dm_recipient_on_roster
    from openexecutive.people import store as people_store
    from openexecutive.people.store import find_person_by_slack_id

    people_store.upsert_person(full_name="Maarten", is_principal=True, slack_user_id="UM")
    nick = people_store.upsert_person(
        full_name="Nick", is_principal=True, slack_user_id="UN"
    )
    # Demoted, NOT archived: still an active team member, so an archived-only
    # check lets him through and only `is_principal` refuses him.
    people_store.upsert_person(
        full_name="Nick", is_principal=False, slack_user_id="UN", person_id=nick
    )

    with restrict_to_principal():
        assert _dm_recipient_on_roster(find_person_by_slack_id, "UN") is False
        # The sitting principal is unaffected — the gate narrows the audience
        # to principals, it does not refuse the fan-out's own recipients.
        assert _dm_recipient_on_roster(find_person_by_slack_id, "UM") is True

    # Negative control: outside the restriction the demoted member is an
    # ordinary allowed DM recipient, so the refusal above is the gate rather
    # than a roster lookup or an archived row.
    assert _dm_recipient_on_roster(find_person_by_slack_id, "UN") is True


def test_restrict_to_principal_restores_the_prior_value() -> None:
    """Save/restore, not `Token.reset`, so a nested block and an exception both
    leave the flag as they found it — a leak would mark every later send on
    this task private and refuse teammates nowhere near a brief."""
    import pytest as _pytest

    from openexecutive.orchestrator.people_tools import (
        restrict_to_principal,
        turn_is_private_to_principal,
    )

    assert turn_is_private_to_principal() is False
    with restrict_to_principal():
        assert turn_is_private_to_principal() is True
        with restrict_to_principal():
            assert turn_is_private_to_principal() is True
        assert turn_is_private_to_principal() is True  # inner exit restores True
    assert turn_is_private_to_principal() is False

    with _pytest.raises(RuntimeError, match="brief blew up"), restrict_to_principal():
        raise RuntimeError("brief blew up")
    assert turn_is_private_to_principal() is False


def test_the_private_brief_raises_the_flag_for_every_channel_leg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag is read at the SEND, not once per run: `deliver_to_person` is
    reached inside the block, so each leg it tries — Slack, Discord, Telegram,
    the email gateway's roster allow-set — sees it on a fresh read however
    many awaits deep."""
    from openexecutive.orchestrator.people_tools import turn_is_private_to_principal

    async def _assert_private(person, text: str, **_kw: object) -> runner.PrincipalDelivery:  # type: ignore[no-untyped-def]
        import asyncio

        await asyncio.sleep(0)  # an await between the gate and the read
        assert turn_is_private_to_principal() is True
        return runner.PrincipalDelivery(True, "email → x", "delivered", "email")

    sends = _run_brief(tmp_path, monkeypatch, deliver_person=_assert_private, principals=1)
    assert len(sends) == 1


def test_one_broken_channel_is_reported_while_the_other_founder_still_gets_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial delivery reports the FAILURE, not the success: the Briefing
    notice and the Setup checks render one reason each, so collapsing to "ok
    if any succeeded" would hide a founder whose channel is broken every day
    behind the other's success. The window still advances — the recipient who
    did get it must not be replayed yesterday's brief."""
    from openexecutive.briefing import brief_state

    rows: list[dict] = []  # type: ignore[type-arg]
    monkeypatch.setattr(
        "openexecutive.audit.log_event",
        lambda *a, **k: rows.append(k.get("details") or {}),
    )

    async def _only_the_first_works(person, text: str, **_kw: object) -> runner.PrincipalDelivery:  # type: ignore[no-untyped-def]
        if person.slack_user_id == "U0":
            return runner.PrincipalDelivery(True, "slack_dm → U0", "delivered", "slack_dm")
        return runner.PrincipalDelivery(False, "nothing connected", "no_channel")

    sends = _run_brief(
        tmp_path, monkeypatch, deliver_person=_only_the_first_works, principals=2
    )

    assert len(sends) == 2
    last = brief_state.last_delivered("principal_brief_morning")
    assert last is not None and last.input_hash == "fp-123"
    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None and (outcome.reason, outcome.channel) == ("no_channel", None)
    # And the RECORD keeps it per recipient, so the surfaces can name the
    # founder rather than inferring one reason for the run. Not a channel for
    # the run either: one of the two sends never happened.
    assert outcome.recipients is not None
    assert [(r.name, r.reason, r.channel) for r in outcome.recipients] == [
        ("Founder 0", "delivered", "slack_dm"),
        ("Founder 1", "no_channel", None),
    ]
    # One audit row per recipient, so the broken one is visible on its own.
    phases = [r["phase"] for r in rows if r.get("kind") == "principal_brief_morning"]
    assert phases == ["delivered", "delivery_failed"]


def test_an_offboarded_founder_does_not_poison_the_runs_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Why the fan-out DROPS a founder who stopped being a principal instead
    of returning a reason for them. `delivery_summary` takes the worst reason
    any recipient got, and `no_owner` there already means "there was no
    principal on the roster at all" — so a per-recipient `no_owner` made a run
    that did reach the other founder store as `no_owner`, indistinguishable
    from a run with nobody to send to."""
    from openexecutive.briefing import brief_state
    from openexecutive.people import store as people_store

    archived: list[int] = []

    async def _archive_the_other_founder(person, text: str, **_kw: object):  # type: ignore[no-untyped-def]
        # Founder 1 is offboarded while Founder 0's send is in flight.
        if not archived:
            for other in people_store.active_principals():
                if other.id is not None and other.id != person.id:
                    people_store.archive_person(other.id)
                    archived.append(other.id)
        return runner.PrincipalDelivery(True, "slack_dm → U0", "delivered", "slack_dm")

    sends = _run_brief(
        tmp_path, monkeypatch, deliver_person=_archive_the_other_founder, principals=2
    )

    assert len(archived) == 1
    assert len(sends) == 1  # Founder 1 is never sent to
    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None and outcome.recipients is not None
    # The aggregate is the remaining recipient's, not `no_owner`.
    assert (outcome.reason, outcome.channel) == ("delivered", "slack_dm")
    assert [(r.name, r.reason) for r in outcome.recipients] == [("Founder 0", "delivered")]
    # And the window advanced, so the founder who got it is not replayed it.
    last = brief_state.last_delivered("principal_brief_morning")
    assert last is not None and last.input_hash == "fp-123"


def test_a_failed_roster_read_for_a_later_founder_keeps_the_earlier_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`people.store` is not exception-swallowing. A roster read that raises
    for a LATER recipient — an SQLite error under contention, say — must not
    discard the recipients already delivered to: that recorded a run-level
    `send_failed` with no recipients at all, never advanced the window, and so
    replayed the same interval to the founder who already had the brief while
    every surface said the run had failed for everyone."""
    import sqlite3

    from openexecutive.briefing import brief_state
    from openexecutive.people import store as people_store

    real_active_principals = people_store.active_principals
    armed: list[bool] = []

    def _raises_once_armed(*args: object, **kw: object) -> list[object]:
        if armed:
            raise sqlite3.OperationalError("database is locked")
        return real_active_principals(*args, **kw)  # type: ignore[arg-type]

    # Armed by the first send rather than by a call count, so the test does
    # not depend on how many times the roster is read before the loop.
    monkeypatch.setattr(people_store, "active_principals", _raises_once_armed)

    async def _deliver_then_break_the_roster(person, text: str, **_kw: object):  # type: ignore[no-untyped-def]
        armed.append(True)
        return runner.PrincipalDelivery(True, "slack_dm → U0", "delivered", "slack_dm")

    sends = _run_brief(
        tmp_path, monkeypatch, deliver_person=_deliver_then_break_the_roster, principals=2
    )

    assert len(sends) == 1  # Founder 1's revalidation raised before its send
    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None and outcome.recipients is not None
    # Founder 0's success survives, and Founder 1's failure is still visible
    # as its own recipient row rather than as a run-level verdict.
    assert [(r.name, r.reason) for r in outcome.recipients] == [
        ("Founder 0", "delivered"), ("Founder 1", "send_failed"),
    ]
    assert (outcome.reason, outcome.channel) == ("send_failed", None)
    # The window advanced for the founder who did receive it.
    last = brief_state.last_delivered("principal_brief_morning")
    assert last is not None and last.input_hash == "fp-123"


def test_the_record_says_which_channel_was_tried_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Setup light's backup-channel warning needs the channel the run
    ACTUALLY tried, not the one the plan would offer when the page is read: a
    channel connected after the run would otherwise be announced as broken
    without ever having been attempted. So the plan's first channel is
    recorded per recipient at send time."""
    from openexecutive.briefing import brief_state

    monkeypatch.setattr("openexecutive.audit.log_event", lambda *a, **k: None)

    async def _slack_fails_email_carries(person, text: str, **_kw: object):  # type: ignore[no-untyped-def]
        # The plan is Slack then email; Slack didn't send, email did.
        return runner.PrincipalDelivery(
            True, "email → f0@x.io", "delivered", "email", first_tried="slack_dm"
        )

    _run_brief(
        tmp_path, monkeypatch, deliver_person=_slack_fails_email_carries, principals=1
    )

    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None and outcome.recipients is not None
    [only] = outcome.recipients
    assert (only.channel, only.first_tried) == ("email", "slack_dm")


def test_a_founder_added_while_the_brief_runs_does_not_get_the_private_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audience is read ONCE, before the run, because it decides what the
    brief may contain. Generation takes minutes; re-reading the roster at
    delivery would hand a founder added inside that window a brief built as
    private to the sitting principal."""
    from openexecutive.people import store as people_store
    from openexecutive.workflows.base import WorkflowEvent
    from openexecutive.workflows.morning_brief import (
        PRINCIPAL_DELIVERY,
        MorningBriefInput,
        MorningBriefWorkflow,
    )

    class _AddsAFounderMidRun(MorningBriefWorkflow):
        async def run(self, inputs, store):  # type: ignore[override]
            assert PRINCIPAL_DELIVERY.get() is True  # one principal: private
            people_store.upsert_person(
                full_name="Latecomer", is_principal=True, slack_user_id="ULATE"
            )
            yield WorkflowEvent(type="result", data={
                "brief_fingerprint": "fp-123", "private_to_principal": True,
            })
            yield WorkflowEvent(type="artifact", content="PRIVATE BRIEF")

        def input_model(self):  # type: ignore[override]
            return MorningBriefInput

    sends = _run_brief(tmp_path, monkeypatch, workflow=_AddsAFounderMidRun(), principals=1)

    principals = people_store.active_principals()
    assert [p.full_name for p in principals] == ["Founder 0", "Latecomer"]  # write landed
    assert sends == [(principals[0].id, "PRIVATE BRIEF")]  # only the original


def test_no_principal_at_all_is_recorded_as_no_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh install with an empty roster: the brief is written and stored,
    nothing is sent, and the reason stays the one the Setup page reads."""
    from openexecutive.briefing import brief_state

    sends = _run_brief(tmp_path, monkeypatch, principals=0)

    assert sends == []
    assert brief_state.last_delivered("principal_brief_morning") is None
    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None and (outcome.reason, outcome.channel) == ("no_owner", None)


def test_a_founder_offboarded_while_the_brief_runs_does_not_get_the_private_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mirror of the test above, and the direction that leaks. Pinning the
    audience in BOTH directions meant a sole principal archived mid-run still
    received a brief built as private to them — their mail, calendar, notes
    and drafts. The pinned list is an upper bound, so the audience may shrink
    but never grow. The send is also wrapped in `restrict_to_principal()`, so
    an archived recipient is refused at the egress even with this pre-filter
    out of the way — see
    `test_an_archived_recipient_is_refused_at_the_egress_not_by_the_pre_filter`."""
    from openexecutive.briefing import brief_state
    from openexecutive.people import store as people_store
    from openexecutive.workflows.base import WorkflowEvent
    from openexecutive.workflows.morning_brief import (
        PRINCIPAL_DELIVERY,
        MorningBriefInput,
        MorningBriefWorkflow,
    )

    class _OffboardsTheFounderMidRun(MorningBriefWorkflow):
        async def run(self, inputs, store):  # type: ignore[override]
            assert PRINCIPAL_DELIVERY.get() is True  # one principal: private
            only = people_store.active_principals()[0]
            assert only.id is not None
            people_store.archive_person(only.id)
            yield WorkflowEvent(type="result", data={
                "brief_fingerprint": "fp-123", "private_to_principal": True,
            })
            yield WorkflowEvent(type="artifact", content="PRIVATE BRIEF")

        def input_model(self):  # type: ignore[override]
            return MorningBriefInput

    sends = _run_brief(
        tmp_path, monkeypatch, workflow=_OffboardsTheFounderMidRun(), principals=1
    )

    assert sends == []  # nothing goes to the offboarded founder
    assert people_store.active_principals() == []
    # An empty audience is the no-owner case, which is what the Setup page reads.
    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None and (outcome.reason, outcome.channel) == ("no_owner", None)


def test_a_channel_changed_while_the_brief_runs_uses_the_new_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pinned row's channel fields are stale too. Sending to the Slack id
    captured before the run would DM whoever holds the old id now."""
    from openexecutive.people import store as people_store
    from openexecutive.workflows.base import WorkflowEvent
    from openexecutive.workflows.morning_brief import MorningBriefInput, MorningBriefWorkflow

    class _MovesTheFoundersSlackMidRun(MorningBriefWorkflow):
        async def run(self, inputs, store):  # type: ignore[override]
            only = people_store.active_principals()[0]
            assert only.id is not None and only.slack_user_id == "U0"
            people_store.update_person(only.id, slack_user_id="U-MOVED")
            yield WorkflowEvent(type="result", data={"brief_fingerprint": "fp-123"})
            yield WorkflowEvent(type="artifact", content="BRIEF")

        def input_model(self):  # type: ignore[override]
            return MorningBriefInput

    captured: list[str | None] = []

    async def _see_the_person(person, text: str, **_kw: object) -> runner.PrincipalDelivery:  # type: ignore[no-untyped-def]
        captured.append(person.slack_user_id)
        return runner.PrincipalDelivery(True, "slack_dm → sent", "delivered", "slack_dm")

    _run_brief(
        tmp_path,
        monkeypatch,
        workflow=_MovesTheFoundersSlackMidRun(),
        deliver_person=_see_the_person,
        principals=1,
    )

    assert captured == ["U-MOVED"]
