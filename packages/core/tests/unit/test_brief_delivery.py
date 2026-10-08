"""Where the daily brief went, and telling the owner when it went nowhere.

  * Each brief's latest run is recorded ONE ENTRY PER RECIPIENT: who it was
    for, how it went for them, and the channel that reached them (a name,
    never an address). A row written before that shape existed holds a single
    reason and channel and reads back as a legacy row.
  * The Setup status page's "Daily brief" light says when the briefs go out
    and where, and turns amber or red when they can't — naming whom a problem
    is about whenever the roster has someone it does not apply to.
  * ``GET /today/brief-delivery`` gives the owner, and only the owner, the
    latest brief that didn't reach a principal, until its cause is fixed.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openexecutive.api.setup_checks import Snapshot, check_brief
from openexecutive.briefing import brief_state, narrative_cache
from openexecutive.briefing.brief_state import DeliveryOutcome, RecipientOutcome
from openexecutive.config import Settings
from openexecutive.people.models import Person

NOW = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)
MORNING = "principal_brief_morning"
EVENING = "principal_brief_eod"


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(narrative_cache, "DB_PATH", tmp_path / "cache.db")


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_the_latest_send_of_either_brief_is_read_back(monkeypatch: pytest.MonkeyPatch) -> None:
    assert brief_state.last_delivery_outcome() is None
    brief_state.record_delivery_outcome(MORNING, reason="delivered", channel="email")
    later = datetime.now(UTC) + timedelta(minutes=5)
    monkeypatch.setattr(narrative_cache, "utc_now_iso", lambda: later.isoformat())
    brief_state.record_delivery_outcome(EVENING, reason="no_channel", channel=None)

    latest = brief_state.last_delivery_outcome()
    assert latest is not None
    assert (latest.kind, latest.reason, latest.channel) == (EVENING, "no_channel", None)


def test_an_unreadable_record_is_ignored() -> None:
    narrative_cache.put(narrative_cache.BriefingNarrative(
        scope=f"{brief_state.DELIVERY_SCOPE_PREFIX}{MORNING}",
        input_hash="made-up-reason",
        narrative_text="",
        generated_at=narrative_cache.utc_now_iso(),
    ))
    assert brief_state.last_delivery_outcome() is None


# ---------------------------------------------------------------------------
# The record, per recipient
# ---------------------------------------------------------------------------


def _store(text: str, *, reason: str = "send_failed", kind: str = MORNING) -> None:
    """Put a raw `narrative_text` under the delivery scope, as a writer of
    another version (or a hand edit) would have left it."""
    narrative_cache.put(narrative_cache.BriefingNarrative(
        scope=f"{brief_state.DELIVERY_SCOPE_PREFIX}{kind}",
        input_hash=reason,
        narrative_text=text,
        generated_at=narrative_cache.utc_now_iso(),
    ))


def test_each_recipient_is_recorded_and_read_back() -> None:
    brief_state.record_delivery_outcome(
        MORNING,
        reason="send_failed",
        channel=None,
        recipients=[
            RecipientOutcome(person_id=3, name="Ada", reason="delivered", channel="email"),
            RecipientOutcome(person_id=7, name="Grace", reason="send_failed", channel=None),
        ],
    )
    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None and outcome.recipients is not None
    assert [(r.person_id, r.name, r.reason, r.channel) for r in outcome.recipients] == [
        (3, "Ada", "delivered", "email"),
        (7, "Grace", "send_failed", None),
    ]
    # No address anywhere in the stored row: channel NAMES only.
    stored = narrative_cache.get(f"{brief_state.DELIVERY_SCOPE_PREFIX}{MORNING}")
    assert stored is not None and "@" not in stored.narrative_text


def test_a_legacy_row_reads_back_as_one_with_no_recipients() -> None:
    """`None` recipients is what tells every reader the record cannot say who
    a problem was about — the one case left where a reason is run-level by
    ignorance rather than by nature."""
    _store("email", reason="delivered")
    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None
    assert (outcome.reason, outcome.channel, outcome.recipients) == ("delivered", "email", None)


def test_a_run_that_reached_nobody_is_not_a_legacy_row() -> None:
    """An EMPTY list is a positive fact — this run had no recipients — and
    must not be confused with a row that predates per-recipient outcomes."""
    brief_state.record_delivery_outcome(
        MORNING, reason="not_written", channel=None, recipients=[]
    )
    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None and outcome.recipients == ()


@pytest.mark.parametrize(
    "text",
    [
        "{not json at all",
        '{"channel": "email"}',  # no recipients key
        '{"channel": "email", "recipients": "Ada"}',  # not a list
        '{"channel": ["email"], "recipients": []}',  # channel of the wrong type
    ],
)
def test_a_malformed_payload_reads_as_unreadable_rather_than_raising(text: str) -> None:
    """The reason in `input_hash` still stands, so it is reported run-level,
    exactly as a legacy row's is. A truncated write must not take the Briefing
    notice and the Setup light down with it."""
    _store(text)
    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None
    assert (outcome.reason, outcome.channel, outcome.recipients) == ("send_failed", None, None)


def test_one_unreadable_entry_does_not_discard_the_others() -> None:
    """Dropping the whole list would hide the founders it could still name."""
    _store(json.dumps({
        "channel": None,
        "recipients": [
            {"person_id": 3, "name": "Ada", "reason": "not-a-reason", "channel": None},
            {"person_id": 7, "name": "Grace", "reason": "send_failed", "channel": None},
            "not even an object",
        ],
    }))
    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None and outcome.recipients is not None
    assert [r.name for r in outcome.recipients] == ["Grace"]


def test_a_channel_the_app_does_not_know_is_not_rendered_back() -> None:
    """`channel_phrase` puts it in a sentence a founder reads ("went on …"),
    so only one of `CHANNEL_NAMES` comes back out."""
    _store(json.dumps({
        "channel": None,
        "recipients": [
            {"person_id": 3, "name": "Ada", "reason": "delivered", "channel": "carrier pigeon"},
            {"person_id": 7, "name": "Grace", "reason": "delivered", "channel": "slack_dm"},
        ],
    }), reason="delivered")
    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None and outcome.recipients is not None
    assert [r.channel for r in outcome.recipients] == [None, "slack_dm"]


def test_a_json_true_person_id_is_not_read_as_person_one() -> None:
    """`isinstance(True, int)`, so an unguarded read would attribute the
    problem to whoever has id 1."""
    _store(json.dumps({
        "channel": None,
        "recipients": [{"person_id": True, "name": "", "reason": "send_failed", "channel": None}],
    }))
    outcome = brief_state.last_delivery_outcome()
    assert outcome is not None and outcome.recipients is not None
    assert outcome.recipients[0].person_id is None


@pytest.mark.parametrize(
    ("reasons", "channels", "expected"),
    [
        ([], [], ("no_owner", None)),  # no principal at all
        (["delivered"], ["email"], ("delivered", "email")),
        (["delivered", "delivered"], ["email", "email"], ("delivered", "email")),
        # Two founders on two channels: no one channel is the run's.
        (["delivered", "delivered"], ["email", "slack_dm"], ("delivered", None)),
        # A run that also failed somewhere names no channel either.
        (["delivered", "no_channel"], ["slack_dm", None], ("no_channel", None)),
        # Worst-first by a FIXED severity, so roster order cannot decide it.
        (["no_channel", "send_failed"], [None, None], ("send_failed", None)),
        (["send_failed", "no_channel"], [None, None], ("send_failed", None)),
    ],
)
def test_the_aggregate_summarises_the_fan_out_without_indexing_it(
    reasons: list[str], channels: list[str | None], expected: tuple[str, str | None]
) -> None:
    recipients = [
        RecipientOutcome(person_id=i, name=f"P{i}", reason=r, channel=c)  # type: ignore[arg-type]
        for i, (r, c) in enumerate(zip(reasons, channels, strict=True))
    ]
    assert brief_state.delivery_summary(recipients) == expected


def _reachable(name: str = "Ada", person_id: int = 3) -> Person:
    """A principal email can carry a brief to (with `email_ready=True`)."""
    return Person(
        id=person_id, full_name=name, is_principal=True, email=f"{name.lower()}@acme.io"
    )


def _unreachable(name: str = "Grace", person_id: int = 7) -> Person:
    """A principal with nothing connected at all."""
    return Person(id=person_id, full_name=name, is_principal=True)


@pytest.mark.parametrize(
    ("recorded", "roster", "email_ready", "expected"),
    [
        # A legacy row (no per-recipient list) reports its reason run-level.
        ("delivered", [_reachable()], True, []),
        ("send_failed", [_reachable()], True, [("send_failed", ())]),
        ("not_written", [_reachable()], True, [("not_written", ())]),
        # Nowhere to send it is judged from NOW, never from the record: still
        # nowhere, then fixed since so the next one will go.
        ("no_channel", [_unreachable()], True, [("no_channel", ())]),
        ("no_channel", [_reachable()], True, []),
        ("no_owner", [_reachable()], True, []),
        # A partial fix reports what is left: an owner now, still no channel.
        ("no_owner", [_unreachable()], True, [("no_channel", ())]),
        ("no_channel", [], True, [("no_owner", ())]),
        # Nobody left on the roster after a delivered run: that brief went
        # out, and the next one records `no_owner` itself.
        ("delivered", [], True, []),
    ],
)
def test_outstanding_problems_from_a_legacy_record(
    recorded: str,
    roster: list[Person],
    email_ready: bool,
    expected: list[tuple[str, tuple[str, ...]]],
) -> None:
    outcome = DeliveryOutcome(MORNING, recorded, None, NOW)  # type: ignore[arg-type]
    assert outcome.recipients is None  # the pre-per-recipient shape
    assert brief_state.outstanding_problems(
        outcome, roster, email_ready=email_ready
    ) == expected


def test_outstanding_problems_with_no_record_still_asks_the_roster() -> None:
    """`no_channel` needs no record — it is a fact about the present — so the
    Setup light reports an unreachable founder before any brief has run."""
    roster = [_reachable(), _unreachable()]
    assert brief_state.outstanding_problems(None, roster, email_ready=True) == [
        ("no_channel", ("Grace",))
    ]
    assert brief_state.outstanding_problems(None, [_reachable()], email_ready=True) == []


def _recorded(
    *entries: tuple[int, str, str, str | None] | tuple[int, str, str, str | None, str | None],
    reason: str = "send_failed",
) -> DeliveryOutcome:
    """A per-recipient record from ``(person_id, name, reason, channel)``, with
    an optional fifth element for the channel the run tried first."""
    return DeliveryOutcome(
        MORNING,
        reason,  # type: ignore[arg-type]
        None,
        NOW,
        recipients=tuple(
            RecipientOutcome(
                person_id=e[0], name=e[1], reason=e[2], channel=e[3],  # type: ignore[arg-type]
                first_tried=e[4] if len(e) > 4 else None,  # type: ignore[misc]
            )
            for e in entries
        ),
    )


def test_a_failed_send_names_the_founder_it_was_about() -> None:
    """The failure this whole change is for. The owner's channel works and the
    co-founder's Slack is broken, so the owner used to read "Your last morning
    brief wasn't sent: every way of sending it failed" every day about a brief
    she is holding. A daily false red on the surface two founders read is how
    the real failure eventually gets ignored."""
    roster = [_reachable("Ada", 3), _reachable("Grace", 7)]
    outcome = _recorded(
        (3, "Ada", "delivered", "email"), (7, "Grace", "send_failed", None)
    )
    assert brief_state.outstanding_problems(outcome, roster, email_ready=True) == [
        ("send_failed", ("Grace",))
    ]
    problem, fix = brief_state.render_problems(
        brief_state.outstanding_problems(outcome, roster, email_ready=True)
    )
    assert problem == "every way of sending it to Grace failed"
    assert fix == "The Setup status page shows which of their connections needs attention."


def test_both_founders_failing_names_neither_because_it_is_about_everyone() -> None:
    """An empty name tuple is what makes `DELIVERY_PROBLEMS`' second person
    honest: with nobody left out, "every way of sending it failed" is true for
    whoever is reading."""
    roster = [_reachable("Ada", 3), _reachable("Grace", 7)]
    outcome = _recorded(
        (3, "Ada", "send_failed", None), (7, "Grace", "send_failed", None)
    )
    assert brief_state.outstanding_problems(outcome, roster, email_ready=True) == [
        ("send_failed", ())
    ]


def test_two_different_failures_in_one_run_are_both_reported() -> None:
    """`failed[0]` made person id decide which of these anyone could see, and
    the other was rendered nowhere. Ada has no channel; Grace's sends all
    fail."""
    ada, grace = _unreachable("Ada", 3), _reachable("Grace", 7)
    outcome = _recorded((3, "Ada", "no_channel", None), (7, "Grace", "send_failed", None))
    assert brief_state.outstanding_problems(outcome, [ada, grace], email_ready=True) == [
        ("send_failed", ("Grace",)),
        ("no_channel", ("Ada",)),
    ]
    problem, fix = brief_state.render_problems(
        brief_state.outstanding_problems(outcome, [ada, grace], email_ready=True)
    )
    assert problem == (
        "every way of sending it to Grace failed; nothing is set up to send it to Ada"
    )
    # Both instructions, in order, neither swallowing the other.
    assert fix.startswith("The Setup status page shows which of their connections")
    assert "Connect Gmail" in fix


def test_the_same_recipient_listed_twice_is_named_once() -> None:
    """Not reachable through `record_delivery_outcome`, but a duplicate would
    read as "Ada and Ada" and — worse — push the count to the roster's, which
    `_names_unless_all` reads as "this is about everyone", putting the second
    person back on a problem that is about one founder."""
    ada, grace = _reachable("Ada", 3), _reachable("Grace", 7)
    outcome = _recorded(
        (3, "Ada", "send_failed", None),
        (3, "Ada", "send_failed", None),
        (7, "Grace", "delivered", "email"),
    )
    assert brief_state.outstanding_problems(
        outcome, [ada, grace], email_ready=True
    ) == [("send_failed", ("Ada",))]


def test_a_failure_the_record_cannot_attribute_is_reported_run_level() -> None:
    """The mirror of the test below, and the one that could have gone silent.
    A recipient with no person id is a failure the record ASSERTS and cannot
    attribute — unlike one whose id has left the roster, which is genuinely
    nothing to report. Dropping it would have taken the whole problem with it:
    the red light goes green on a run the record says failed."""
    ada = _reachable("Ada", 3)
    outcome = DeliveryOutcome(
        MORNING, "send_failed", None, NOW,
        recipients=(
            RecipientOutcome(person_id=None, name="Grace", reason="send_failed", channel=None),
        ),
    )
    # Run-level, not named: with nobody identified we cannot claim it is only
    # some of them, so the conservative reading stands.
    assert brief_state.outstanding_problems(outcome, [ada], email_ready=True) == [
        ("send_failed", ())
    ]


def test_an_unattributable_failure_does_not_shrink_a_named_list() -> None:
    """Naming Ada while an unidentified recipient also failed would tell the
    reader everyone else is fine — the same silent shortening `_and_list`
    exists to prevent, one layer up."""
    ada, grace = _reachable("Ada", 3), _reachable("Grace", 7)
    outcome = DeliveryOutcome(
        MORNING, "send_failed", None, NOW,
        recipients=(
            RecipientOutcome(person_id=3, name="Ada", reason="send_failed", channel=None),
            RecipientOutcome(person_id=None, name="?", reason="send_failed", channel=None),
        ),
    )
    assert brief_state.outstanding_problems(
        outcome, [ada, grace], email_ready=True
    ) == [("send_failed", ())]


def test_a_recipient_archived_since_the_run_is_dropped() -> None:
    """The same intersection `deliver_to_each_principal` makes when it sends,
    so the record and the fan-out cannot disagree about who a recipient is."""
    ada = _reachable("Ada", 3)
    outcome = _recorded(
        (3, "Ada", "delivered", "email"), (7, "Grace", "send_failed", None)
    )
    assert brief_state.outstanding_problems(outcome, [ada], email_ready=True) == []


def test_a_founder_added_since_the_run_is_still_named_by_the_roster() -> None:
    """Nothing is claimed about them from a run they were not in, but
    `no_channel` is judged from NOW, so their missing channel counts."""
    ada, lin = _reachable("Ada", 3), _unreachable("Lin", 9)
    outcome = _recorded((3, "Ada", "delivered", "email"), reason="delivered")
    assert brief_state.outstanding_problems(outcome, [ada, lin], email_ready=True) == [
        ("no_channel", ("Lin",))
    ]


def test_a_nameless_recipient_is_described_in_place() -> None:
    """Never dropped — telling a founder about one unreached co-principal when
    there are two is the same silent omission this surface exists to prevent —
    and never given its id, which is not something to show a reader."""
    ada = _reachable("Ada", 3)
    nameless = Person(id=7, full_name="", is_principal=True, email="g@acme.io")
    outcome = _recorded(
        (3, "Ada", "delivered", "email"), (7, "", "send_failed", None)
    )
    problems = brief_state.outstanding_problems(outcome, [ada, nameless], email_ready=True)
    assert problems == [("send_failed", ("",))]
    problem, _ = brief_state.render_problems(problems)
    assert problem == "every way of sending it to someone on the People list failed"


def test_a_run_whose_delivery_raised_is_reported_run_level() -> None:
    """The exception path records `send_failed` with an EMPTY recipient list:
    it raised before any recipient had an outcome, so there is nobody to
    attribute it to and the reason is the run's."""
    roster = [_reachable("Ada", 3), _reachable("Grace", 7)]
    outcome = DeliveryOutcome(MORNING, "send_failed", None, NOW, recipients=())
    assert brief_state.outstanding_problems(outcome, roster, email_ready=True) == [
        ("send_failed", ())
    ]


def test_not_written_stays_run_level_even_with_a_recipient_list() -> None:
    """The artifact did not exist for anyone, so it is legitimately
    single-valued."""
    roster = [_reachable("Ada", 3), _unreachable("Grace", 7)]
    outcome = DeliveryOutcome(MORNING, "not_written", None, NOW, recipients=())
    assert brief_state.outstanding_problems(outcome, roster, email_ready=True) == [
        ("not_written", ()),
        ("no_channel", ("Grace",)),
    ]


def test_brief_names_come_from_the_scheduler_labels() -> None:
    from openexecutive.scheduler.action_phrasing import KIND_LABEL

    for kind in brief_state.BRIEF_KINDS:
        assert brief_state.brief_name(kind) == KIND_LABEL[kind]


def test_email_is_ready_only_with_the_google_workspace_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.orchestrator import mcp_gateway
    from openexecutive.scheduler import runner

    servers: list[str] = ["notion"]
    monkeypatch.setattr(mcp_gateway, "configured_server_names", lambda _path: servers)
    monkeypatch.setattr(mcp_gateway, "_active_gateway", None)
    assert runner.email_ready() is False  # no gateway at all
    monkeypatch.setattr(mcp_gateway, "_active_gateway", object())
    assert runner.email_ready() is False  # a gateway, but no Gmail behind it
    servers.append("google_workspace")
    assert runner.email_ready() is True


# ---------------------------------------------------------------------------
# The "Daily brief" light
# ---------------------------------------------------------------------------


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "ANTHROPIC_API_KEY": "sk-ant-api03-realistic",
        "EXEC_EMAIL_ADDRESS": "exec@acme.io",
        "SCHEDULER_ENABLED": True,
    }
    return Settings(**{**base, **overrides})


def _snap(**fields: Any) -> Snapshot:
    values: dict[str, Any] = {
        "settings": _settings(),
        "now": NOW,
        "local_login": False,
        "people": [],
        "principal": Person(id=3, full_name="Ada", is_principal=True, email="ada@acme.io"),
        "last_inbound": {},
        "brief_email_ready": True,  # Gmail connected
        "brief_next_runs": {
            MORNING: datetime(2026, 9, 26, 15, 0, tzinfo=UTC),  # 08:00 in Los Angeles
            EVENING: datetime(2026, 9, 26, 1, 0, tzinfo=UTC),  # 18:00 in Los Angeles
        },
        "brief_zone": "America/Los_Angeles",
    }
    values.update(fields)
    return Snapshot(**values)


def test_the_light_says_when_and_where() -> None:
    check = check_brief(_snap())
    assert check.state == "ok"
    assert check.summary == (
        "Sent to you by email: the morning brief at 08:00 and the end-of-day digest at 18:00 "
        "(America/Los_Angeles)."
    )


def test_a_chat_preference_is_named() -> None:
    ada = Person(id=3, full_name="Ada", is_principal=True, preferred_channel="slack", slack_user_id="U1")
    assert check_brief(_snap(principal=ada)).summary.startswith("Sent to you on Slack")


def test_no_time_zone_means_utc_and_says_so() -> None:
    check = check_brief(_snap(brief_zone=None))
    assert check.state == "warn"
    assert check.summary == (
        "Sent to you by email: the morning brief at 15:00 and the end-of-day digest at 01:00, "
        "in UTC because no time zone is set."
    )
    assert (check.fix, check.link) == ("Set your time zone in Settings.", "/settings")


def test_nowhere_to_send_it() -> None:
    check = check_brief(_snap(brief_email_ready=False))  # Gmail off, no chat ids
    assert check.state == "warn"
    assert check.summary == "Kept in the app only: nothing is set up to send it to you."
    assert check.link == "/people/3"


def test_no_owner() -> None:
    check = check_brief(_snap(principal=None))
    assert (check.state, check.link) == ("warn", "/people")


def test_a_failed_send_turns_it_red() -> None:
    failed = DeliveryOutcome(EVENING, "send_failed", None, NOW)
    check = check_brief(_snap(brief_delivery=failed))
    assert check.state == "error"
    assert check.summary == "Your last end-of-day digest wasn't sent: every way of sending it failed."


def test_a_brief_that_couldnt_be_written_turns_it_red() -> None:
    failed = DeliveryOutcome(MORNING, "not_written", None, NOW)
    check = check_brief(_snap(brief_delivery=failed))
    assert check.state == "error"
    assert check.summary == "Your last morning brief wasn't sent: it couldn't be written."


def test_a_broken_first_channel_is_named_when_the_backup_carried_it() -> None:
    # Prefers Slack, Slack failed, email carried it: not all well.
    ada = Person(
        id=3, full_name="Ada", is_principal=True, preferred_channel="slack",
        slack_user_id="U1", email="ada@acme.io",
    )
    carried = DeliveryOutcome(MORNING, "delivered", "email", NOW)
    check = check_brief(_snap(principal=ada, brief_delivery=carried))
    assert check.state == "warn"
    assert check.summary == "Your last morning brief went by email, because Slack didn't work."
    assert check.fix == 'See the "Slack" light on this page.'


def test_a_fixed_cause_is_not_reported() -> None:
    # It had nowhere to go, and now it has (email is connected in this snapshot).
    stale = DeliveryOutcome(MORNING, "no_channel", None, NOW)
    assert check_brief(_snap(brief_delivery=stale)).state == "ok"


def test_off_with_the_scheduler() -> None:
    check = check_brief(_snap(settings=_settings(SCHEDULER_ENABLED=False)))
    assert check.state == "off"


# ---------------------------------------------------------------------------
# GET /today/brief-delivery
# ---------------------------------------------------------------------------


@pytest.fixture
def notice_client(monkeypatch: pytest.MonkeyPatch) -> Any:
    from openexecutive.api.routes import chat as chat_route
    from openexecutive.api.routes import today as today_route
    from openexecutive.people import store as people_store
    from openexecutive.scheduler import runner

    # Ada is reachable only by email, so `email_ready` is what decides whether
    # anything can carry the brief — the real `delivery_order` does the rest.
    state: dict[str, Any] = {
        "owner": True,
        "principals": [Person(id=3, full_name="Ada", is_principal=True, email="ada@acme.io")],
        "email_ready": False,
    }
    monkeypatch.setattr(chat_route, "_caller_is_principal_or_unclaimed", lambda _r: state["owner"])
    monkeypatch.setattr(people_store, "active_principals", lambda: state["principals"])
    monkeypatch.setattr(runner, "email_ready", lambda: state["email_ready"])
    app = FastAPI()
    app.include_router(today_route.router)
    return TestClient(app), state


def test_the_owner_hears_about_an_unsent_brief(notice_client: Any) -> None:
    client, _ = notice_client
    brief_state.record_delivery_outcome(MORNING, reason="no_channel", channel=None)
    body = client.get("/today/brief-delivery").json()
    assert body["brief"] == "morning brief"
    assert body["problem"] == "nothing is set up to send it to you"
    assert body["fix"].startswith("Connect Gmail")
    assert body["readable"] is True
    assert "@" not in str(body)


def test_the_notice_names_what_is_left_after_a_partial_fix(notice_client: Any) -> None:
    client, _ = notice_client  # an owner now, still nowhere to send it
    brief_state.record_delivery_outcome(MORNING, reason="no_owner", channel=None)
    assert client.get("/today/brief-delivery").json()["problem"] == "nothing is set up to send it to you"


def test_a_brief_that_couldnt_be_written_has_nothing_to_read(notice_client: Any) -> None:
    client, state = notice_client
    state["email_ready"] = True
    brief_state.record_delivery_outcome(EVENING, reason="not_written", channel=None)
    body = client.get("/today/brief-delivery").json()
    assert (body["brief"], body["readable"]) == ("end-of-day digest", False)
    assert body["problem"] == "it couldn't be written"


def test_no_notice_with_the_scheduler_off(
    notice_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = notice_client
    monkeypatch.setenv("SCHEDULER_ENABLED", "false")
    brief_state.record_delivery_outcome(MORNING, reason="send_failed", channel=None)
    assert client.get("/today/brief-delivery").json() is None


def test_nobody_else_does(notice_client: Any) -> None:
    client, state = notice_client
    state["owner"] = False
    brief_state.record_delivery_outcome(MORNING, reason="send_failed", channel=None)
    assert client.get("/today/brief-delivery").json() is None


@pytest.mark.parametrize("reason", ["delivered", "no_channel"])  # the second: fixed since
def test_nothing_to_say(notice_client: Any, reason: str) -> None:
    client, state = notice_client
    state["email_ready"] = True
    brief_state.record_delivery_outcome(MORNING, reason=reason, channel="email")  # type: ignore[arg-type]
    assert client.get("/today/brief-delivery").json() is None


def test_nothing_recorded_yet(notice_client: Any) -> None:
    client, _ = notice_client
    assert client.get("/today/brief-delivery").json() is None


def test_no_principal_at_all_reads_as_no_owner(notice_client: Any) -> None:
    client, state = notice_client
    state["principals"] = []
    brief_state.record_delivery_outcome(MORNING, reason="no_channel", channel=None)
    body = client.get("/today/brief-delivery").json()
    assert body["problem"] == "there's no owner on the People list to send it to"


# ---------------------------------------------------------------------------
# Co-principals: one working channel must not clear the other's failure
# ---------------------------------------------------------------------------


def _co_principals() -> list[Person]:
    """Ada reachable by email, Grace on nothing at all."""
    return [
        Person(id=3, full_name="Ada", is_principal=True, email="ada@acme.io"),
        Person(id=7, full_name="Grace", is_principal=True),
    ]


def test_the_notice_names_the_co_principal_the_brief_isnt_reaching(
    notice_client: Any,
) -> None:
    client, state = notice_client
    state["principals"] = _co_principals()
    state["email_ready"] = True  # Ada's channel works; Grace has none
    brief_state.record_delivery_outcome(MORNING, reason="no_channel", channel=None)
    body = client.get("/today/brief-delivery").json()
    # Not "send it to you": the reader may well be the founder who got it.
    assert body["problem"] == "nothing is set up to send it to Grace"
    assert body["fix"] == (
        "Connect Gmail, or add their Slack, Telegram or Discord to their People profile."
    )
    assert "@" not in str(body)


def test_the_light_names_the_co_principal_the_brief_isnt_reaching() -> None:
    people = _co_principals()
    check = check_brief(_snap(people=people, principal=people[0]))
    assert check.state == "warn"
    assert check.summary == "Not reaching everyone: nothing is set up to send it to Grace."
    assert check.link == "/people/7"


def test_the_light_names_every_unreached_co_principal() -> None:
    people = [
        *_co_principals(),
        Person(id=9, full_name="Lin", is_principal=True),
        Person(id=11, full_name="Sam", is_principal=False),  # not a principal: not a recipient
    ]
    check = check_brief(_snap(people=people, principal=people[0]))
    assert check.summary == "Not reaching everyone: nothing is set up to send it to Grace and Lin."


def test_a_failed_send_still_outranks_an_unreachable_co_principal() -> None:
    """Red outranks amber for the light's COLOUR. It no longer outranks it for
    the summary: both clauses are rendered, because dropping one for want of
    room is how a founder's recurring problem stayed invisible behind the
    other's."""
    people = _co_principals()
    failed = DeliveryOutcome(MORNING, "send_failed", None, NOW)
    check = check_brief(_snap(people=people, principal=people[0], brief_delivery=failed))
    assert check.state == "error"
    # The record is a legacy one, so the failure is run-wide ("Your last") and
    # only Grace's missing channel can be attributed.
    assert check.summary == (
        "Your last morning brief wasn't sent: every way of sending it failed; "
        "nothing is set up to send it to Grace."
    )


def test_an_unreachable_first_founder_is_named_while_the_brief_still_goes_out() -> None:
    """The mirror of the bug above: `snap.principal` is the lowest-id row, so
    when the unreachable founder is that one, judging "nowhere to send it"
    from their plan alone claims the brief reaches nobody — while it is in
    fact going to the co-principal every day."""
    people = [
        Person(id=3, full_name="Ada", is_principal=True),  # nothing connected
        Person(id=7, full_name="Grace", is_principal=True, slack_user_id="U7"),
    ]
    check = check_brief(_snap(people=people, principal=people[0], brief_email_ready=False))
    assert check.state == "warn"
    assert check.summary == "Not reaching everyone: nothing is set up to send it to Ada."
    assert check.link == "/people/3"


def test_an_unreachable_roster_is_still_app_only() -> None:
    people = [
        Person(id=3, full_name="Ada", is_principal=True),
        Person(id=7, full_name="Grace", is_principal=True),
    ]
    check = check_brief(_snap(people=people, principal=people[0], brief_email_ready=False))
    assert check.state == "warn"
    # Nobody can be reached, so "to you" is true for whoever is reading.
    assert check.summary == "Kept in the app only: nothing is set up to send it to you."
    assert check.link == "/people/3"


def test_an_unreached_founder_outranks_the_channel_and_zone_warnings() -> None:
    """All three are `warn`, and only one summary fits. A founder getting no
    brief at all is worse news than a backup channel carrying one, or than
    the briefs running on UTC, so it is the one that shows."""
    ada = Person(
        id=3,
        full_name="Ada",
        is_principal=True,
        email="ada@acme.io",
        preferred_channel="slack",
        slack_user_id="U3",
    )
    people = [ada, Person(id=7, full_name="Grace", is_principal=True)]
    # Slack was tried first and failed; email carried it. And no time zone.
    backup = DeliveryOutcome(MORNING, "delivered", "email", NOW)
    snap = _snap(people=people, principal=ada, brief_delivery=backup, brief_zone=None)
    check = check_brief(snap)
    assert check.state == "warn"
    assert check.summary == "Not reaching everyone: nothing is set up to send it to Grace."
    # Control: the backup-channel warning it masked does show on a roster
    # where it can be claimed at all — one principal, so the recorded channel
    # is unambiguously theirs. (With co-principals it stays suppressed even
    # when everyone is reachable: one record cannot be attributed to a
    # person. See `test_a_co_principal_roster_claims_no_single_channel`.)
    solo = check_brief(
        _snap(people=[ada], principal=ada, brief_delivery=backup, brief_zone=None)
    )
    assert solo.summary.startswith("Your last morning brief went by email")


def test_a_co_principal_roster_claims_no_single_channel() -> None:
    """Ada by email, Grace on Slack: both reachable, so the light is green —
    but "Sent to you by email" would be false for Grace, and the recorded
    channel (the FIRST successful recipient's) cannot be attributed to either,
    so the backup-channel warning is not claimable either."""
    ada = Person(id=3, full_name="Ada", is_principal=True, email="ada@acme.io")
    grace = Person(id=7, full_name="Grace", is_principal=True, slack_user_id="U7")
    # Recorded channel is Grace's Slack; Ada's plan starts at email. On one
    # roster that reads as "email didn't work" — here it means nothing.
    carried = DeliveryOutcome(MORNING, "delivered", "slack_dm", NOW)
    check = check_brief(_snap(people=[ada, grace], principal=ada, brief_delivery=carried))
    assert check.state == "ok"
    assert check.summary == (
        "Sent to each of you on your own channel: the morning brief at 08:00 "
        "and the end-of-day digest at 18:00 (America/Los_Angeles)."
    )
    assert "by email" not in check.summary
    assert "didn't work" not in check.summary


def test_a_reachable_roster_leaves_the_light_green() -> None:
    people = [
        Person(id=3, full_name="Ada", is_principal=True, email="ada@acme.io"),
        Person(id=7, full_name="Grace", is_principal=True, slack_user_id="U7"),
    ]
    assert check_brief(_snap(people=people, principal=people[0])).state == "ok"


def test_the_light_names_the_founder_whose_send_failed_and_not_the_reader() -> None:
    """Red, because a send that failed is red — but about Grace, not "your
    last brief". Ada is holding the brief it says wasn't sent."""
    ada, grace = _reachable("Ada", 3), _reachable("Grace", 7)
    outcome = _recorded(
        (3, "Ada", "delivered", "email"), (7, "Grace", "send_failed", None)
    )
    check = check_brief(_snap(people=[ada, grace], principal=ada, brief_delivery=outcome))
    assert check.state == "error"
    assert check.summary == (
        "The last morning brief didn't reach everyone: "
        "every way of sending it to Grace failed."
    )
    # Not addressed to the reader at all. ("you" alone would also match
    # "your", so it is no control; these are the phrasings that were wrong.)
    assert "Your last" not in check.summary
    assert "to you" not in check.summary


def test_the_notice_names_the_founder_whose_send_failed(notice_client: Any) -> None:
    client, state = notice_client
    state["principals"] = [_reachable("Ada", 3), _reachable("Grace", 7)]
    state["email_ready"] = True
    brief_state.record_delivery_outcome(
        MORNING,
        reason="send_failed",
        channel=None,
        recipients=[
            RecipientOutcome(person_id=3, name="Ada", reason="delivered", channel="email"),
            RecipientOutcome(person_id=7, name="Grace", reason="send_failed", channel=None),
        ],
    )
    body = client.get("/today/brief-delivery").json()
    assert body["problem"] == "every way of sending it to Grace failed"
    assert body["readable"] is True
    assert "@" not in str(body)


def test_the_light_reports_two_different_failures_at_once() -> None:
    """Neither clause is dropped for want of room: that is how Grace's
    recurring send failure stayed invisible behind Ada's missing channel."""
    ada, grace = _unreachable("Ada", 3), _reachable("Grace", 7)
    outcome = _recorded((3, "Ada", "no_channel", None), (7, "Grace", "send_failed", None))
    check = check_brief(_snap(people=[ada, grace], principal=grace, brief_delivery=outcome))
    assert check.state == "error"
    assert check.summary == (
        "The last morning brief didn't reach everyone: "
        "every way of sending it to Grace failed; nothing is set up to send it to Ada."
    )


def test_a_backup_channel_is_named_per_recipient_on_a_co_principal_roster() -> None:
    """Withheld until the record carried a recipient AND the channel that
    recipient's run tried first: one stored channel could only be compared
    against the lowest-id row's plan, which called a working channel broken
    whenever the record was the other founder's."""
    ada = Person(
        id=3, full_name="Ada", is_principal=True, email="ada@acme.io",
        preferred_channel="slack", slack_user_id="U3",
    )
    grace = Person(id=7, full_name="Grace", is_principal=True, slack_user_id="U7")
    # Ada's run tried Slack and email carried it; Grace's own Slack worked.
    outcome = _recorded(
        (3, "Ada", "delivered", "email", "slack_dm"),
        (7, "Grace", "delivered", "slack_dm", "slack_dm"),
        reason="delivered",
    )
    check = check_brief(_snap(people=[ada, grace], principal=ada, brief_delivery=outcome))
    assert check.state == "warn"
    assert check.summary == (
        "Slack didn't work for Ada, so the last morning brief reached them "
        "on a backup channel."
    )
    assert check.fix == 'See the "Slack" light on this page.'


def test_a_working_channel_is_never_called_broken_on_a_co_principal_roster() -> None:
    """The unsound inference the warning was withheld for. Grace was reached
    on her own first channel; the fact that Ada's plan starts elsewhere says
    nothing about it."""
    ada = Person(id=3, full_name="Ada", is_principal=True, email="ada@acme.io")
    grace = Person(id=7, full_name="Grace", is_principal=True, slack_user_id="U7")
    outcome = _recorded(
        (3, "Ada", "delivered", "email", "email"),
        (7, "Grace", "delivered", "slack_dm", "slack_dm"),
        reason="delivered",
    )
    check = check_brief(_snap(people=[ada, grace], principal=ada, brief_delivery=outcome))
    assert check.state == "ok"
    assert "didn't work" not in check.summary


def test_a_channel_connected_after_the_run_is_not_called_broken() -> None:
    """Why the attempted channel is STORED rather than recomputed. Ada's plan
    now starts at Slack, but the run that was recorded tried email and email
    sent it — Slack was never offered. Comparing against the plan as it stands
    would announce "Slack didn't work for Ada" about a channel nothing has
    ever tried, and point the founder at a light that is working."""
    ada = Person(
        id=3, full_name="Ada", is_principal=True, email="ada@acme.io",
        preferred_channel="slack", slack_user_id="U3",  # connected since the run
    )
    grace = Person(id=7, full_name="Grace", is_principal=True, slack_user_id="U7")
    outcome = _recorded(
        (3, "Ada", "delivered", "email", "email"),
        (7, "Grace", "delivered", "slack_dm", "slack_dm"),
        reason="delivered",
    )
    check = check_brief(_snap(people=[ada, grace], principal=ada, brief_delivery=outcome))
    assert check.state == "ok"
    assert "didn't work" not in check.summary


def test_a_record_with_no_attempted_channel_claims_nothing() -> None:
    """Nothing to compare, so no claim — never a guess from the plan now."""
    ada = Person(
        id=3, full_name="Ada", is_principal=True, email="ada@acme.io",
        preferred_channel="slack", slack_user_id="U3",
    )
    grace = Person(id=7, full_name="Grace", is_principal=True, slack_user_id="U7")
    outcome = _recorded(
        (3, "Ada", "delivered", "email"),  # no first_tried
        (7, "Grace", "delivered", "slack_dm"),
        reason="delivered",
    )
    check = check_brief(_snap(people=[ada, grace], principal=ada, brief_delivery=outcome))
    assert check.state == "ok"


def test_two_different_broken_first_channels_make_no_single_claim() -> None:
    """A known gap, pinned so it is a decision rather than a surprise: the
    summary is one line and its fix points at ONE light, and there is no
    honest way to name two broken channels in it. Both founders here were
    reached on a backup, and the light is still green."""
    ada = Person(
        id=3, full_name="Ada", is_principal=True, email="ada@acme.io",
        preferred_channel="slack", slack_user_id="U3",
    )
    grace = Person(
        id=7, full_name="Grace", is_principal=True, email="grace@acme.io",
        preferred_channel="telegram", telegram_chat_id="77",
    )
    outcome = _recorded(
        (3, "Ada", "delivered", "email", "slack_dm"),
        (7, "Grace", "delivered", "email", "telegram"),
        reason="delivered",
    )
    check = check_brief(_snap(people=[ada, grace], principal=ada, brief_delivery=outcome))
    assert check.state == "ok"
    assert "didn't work" not in check.summary


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["Ada"], "Ada"),
        (["Ada", "Grace"], "Ada and Grace"),
        (["Ada", "Grace", "Lin"], "Ada, Grace and Lin"),
        ([""], "someone on the People list"),  # a nameless row is never given its id
        # A blank name among named ones is described, not dropped: shortening
        # the list would hide one of the founders missing out.
        (["Grace", "   "], "Grace and someone on the People list"),
        (["", ""], "someone on the People list and someone on the People list"),
    ],
)
def test_the_unreached_are_named_in_a_readable_list(names: list[str], expected: str) -> None:
    problem, _ = brief_state.partial_delivery_problem(names)
    assert problem == f"nothing is set up to send it to {expected}"
