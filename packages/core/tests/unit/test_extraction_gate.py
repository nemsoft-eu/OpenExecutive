"""The turn gate in front of the episodic memory extractor.

Regression set for an install where the extractor ran 13 times, cost real
money, and stored nothing. The gate was a combined user+assistant floor of
1500 chars, and on real traffic it selected almost exactly the wrong turns:
of 16 exchanges it blocked 9, and those 9 held every instruction the principal
gave, while the 7 it admitted were long analytical exchanges with no
commitment in them at all.

The turns below are the real ones, with their real lengths.
"""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any, NamedTuple
from unittest import mock

import pytest

from openexecutive.memory import episodic
from openexecutive.memory.episodic import initialize_db


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    """An isolated episodic DB.

    Without it these passes write real rows into `./episodic_memory.db`, which
    other modules then read back as production data — the audit-log pollution
    trap in CLAUDE.md.
    """
    db_path = tmp_path / "episodic.db"
    initialize_db(db_path)
    return db_path

# Real turns from the tenant, with the assistant length that accompanied them.
# Each is the principal giving an instruction, and each was discarded by the
# combined-length floor.
_COMMITMENTS = [
    ("It's a mistake on the lp purchased properties worksheet.  That's it. "
     "Correct it there. And record this as fixed.", 1200),
    ("Nope.  The briefing is incorrect.  We are not under water 2.9m that is "
     "just how the finicals look as received Q2", 920),
    ("This is not relevant for us.  Don't track this", 864),
    ("Well remember you fixed it or something. So we don't keep getting. "
     "Notified.", 612),
]

# The shape a length floor cannot handle: a long analysis answered in a few
# words. A floor high enough to skip "Done" also skips "Do B.".
_SHORT_APPROVALS = ["Approve option B.", "Do B.", "Kill it. Approved.", "Yes, ship it."]


@pytest.mark.parametrize("message,assistant_len", _COMMITMENTS)
def test_real_commitments_reach_the_extractor(message: str, assistant_len: int) -> None:
    """Drives the real gate, not arithmetic on literals."""
    assert len(message) + assistant_len < 1500, "blocked by the old combined floor"
    assert episodic.should_extract(message)


@pytest.mark.parametrize("message", _SHORT_APPROVALS)
def test_short_approvals_reach_the_extractor(message: str) -> None:
    """The canonical executive decision: long analysis, three-word answer.

    A user-side length floor regressed these — "Approve option B." is 17 chars
    and the old combined floor DID admit it after a long reply. Any floor that
    skips "Done" (4) also skips "Do B." (5), so there is no floor.
    """
    assert episodic.should_extract(message)


def test_empty_turns_are_skipped() -> None:
    for blank in ("", "   ", "\n\t "):
        assert not episodic.should_extract(blank)


class _Stores(NamedTuple):
    """The three persistence calls, mocked so a test can assert on arguments."""

    decision: mock.MagicMock
    initiative: mock.MagicMock
    advice: mock.MagicMock


# --------------------------------------------------------------------- #
# Who is speaking
#
# `_is_valid_user_commitment` tests a quote against `user_message`, which is
# only meaningful when that text is the principal's own words. On a chat
# channel it is not: `Executive.chat()` is reached from Slack, Discord,
# Telegram, Google Chat and the email poller, and the message there belongs to
# whoever sent it. Removing the length floor is what makes this bite — short
# channel traffic used to fall under it incidentally.
# --------------------------------------------------------------------- #


def _person(*, is_principal: bool) -> mock.MagicMock:
    person = mock.MagicMock()
    person.is_principal = is_principal
    return person


def test_a_web_turn_needs_no_person_row() -> None:
    """No origin channel means the web app, the CLI or the API — the
    principal's own authenticated surfaces. `person_id` is legitimately None
    there in a single-user install, which is the tenant this fix is for."""
    assert episodic.should_extract("Do B.", origin_channel="", person_id=None)


@pytest.mark.parametrize("channel", ["slack", "discord", "telegram", "email"])
def test_a_channel_turn_from_the_principal_is_extracted(channel: str) -> None:
    with mock.patch(
        "openexecutive.people.store.get_person", return_value=_person(is_principal=True)
    ):
        assert episodic.should_extract(
            "Do B.", origin_channel=channel, person_id=7
        )


@pytest.mark.parametrize("channel", ["slack", "discord", "telegram", "email"])
def test_a_channel_turn_from_anyone_else_is_skipped(channel: str) -> None:
    """A teammate's line would land in `decisions` with no speaker attached,
    indistinguishable from the principal's own commitment."""
    with mock.patch(
        "openexecutive.people.store.get_person",
        return_value=_person(is_principal=False),
    ):
        assert not episodic.should_extract(
            "Let's move the deadline to Friday.", origin_channel=channel, person_id=9
        )


def test_an_unidentified_channel_speaker_is_skipped() -> None:
    """An inbound email is text the sender chose, so a verbatim self-quote is
    free. With no resolved person there is nothing to check it against."""
    assert not episodic.should_extract(
        "I approve the wire transfer.", origin_channel="email", person_id=None
    )


def test_a_failed_person_lookup_fails_closed() -> None:
    with mock.patch(
        "openexecutive.people.store.get_person", side_effect=RuntimeError("no table")
    ):
        assert not episodic.should_extract(
            "Do B.", origin_channel="slack", person_id=7
        )


def test_a_missing_person_row_fails_closed() -> None:
    with mock.patch("openexecutive.people.store.get_person", return_value=None):
        assert not episodic.should_extract(
            "Do B.", origin_channel="slack", person_id=7
        )


async def _run_pass(
    payload: Any,
    db_path: Path,
    *,
    user_message: str = "Cut burn to 400k. Tell the team.",
) -> tuple[_Stores, list[dict[str, Any]]]:
    """Drive one real extraction pass over a model payload.

    The payload is whatever the model put in its `store_memories` tool block —
    it is NOT schema-checked anywhere, so tests hand it the shapes a model
    actually emits, including the malformed ones.

    All three store functions are mocked, so every kind is asserted the same
    way — on the arguments it was called with, not just on the audit counters.
    `db_path` still points at an isolated DB so nothing here can reach the
    default `./episodic_memory.db` if a code path stops going through them.
    """
    class _Block:
        type = "tool_use"
        name = "store_memories"
        input = payload

    class _Response:
        content = [_Block()]

    rows: list[dict[str, Any]] = []

    def _capture(event_type: str, summary: str, **kw: Any) -> None:
        rows.append({"event_type": event_type, "summary": summary, **kw})

    with (
        mock.patch.object(episodic, "store_decision") as store_decision,
        mock.patch.object(episodic, "store_initiative") as store_initiative,
        mock.patch.object(episodic, "store_advice") as store_advice,
        mock.patch("openexecutive.audit.log_event", _capture),
        mock.patch("openexecutive.audit.usage.log_model_usage"),
        mock.patch("openexecutive.config.get_settings"),
        mock.patch.object(episodic, "get_active_initiatives", return_value=[]),
        mock.patch("openexecutive.providers.get_provider") as provider,
    ):
        provider.return_value.messages_create = mock.AsyncMock(
            return_value=_Response()
        )
        await episodic.extract_and_store(
            user_message, "Understood.", db_path=db_path, session_id="s-1"
        )

    return (
        _Stores(store_decision, store_initiative, store_advice),
        [r for r in rows if r["event_type"] == "memory_extraction"],
    )


@pytest.mark.asyncio
async def test_extraction_audits_proposed_stored_and_dropped(db: Path) -> None:
    """A pass that proposes items and stores none must be visible.

    That is exactly what happened in production, and it was indistinguishable
    from "the model found nothing" because drops were logger.debug and stores
    wrote no row.
    """
    stores, audit = await _run_pass(db_path=db, payload={
        "decisions": [
            # Valid: the quote is the user's own words, terminated.
            {"domain": "finance", "summary": "Cut burn to 400k",
             "user_commitment_quote": "Cut burn to 400k"},
            # Invalid: quote is not in the user message at all.
            {"domain": "finance", "summary": "Sell the building",
             "user_commitment_quote": "sell the building"},
        ],
    })

    assert stores.decision.call_count == 1
    assert stores.decision.call_args.kwargs["summary"] == "Cut burn to 400k", (
        "the item that survived must be the one with a valid quote"
    )
    assert len(audit) == 1
    assert audit[0]["details"]["proposed"]["decisions"] == 2
    assert audit[0]["details"]["stored"]["decisions"] == 1
    assert audit[0]["details"]["dropped_count"] == 1
    assert audit[0]["details"]["dropped"] == [
        {"kind": "decision", "reason": "bad_quote", "label": "Sell the building"}
    ]
    assert audit[0]["details"]["failure"] == ""


# --------------------------------------------------------------------- #
# Malformed model payloads
#
# `block.input` is whatever the model emitted; nothing validates its shape
# before the loops index into it. A `null` or a list of bare strings used to
# raise out of the whole pass — which took the other two kinds with it AND
# skipped the audit row, leaving a log indistinguishable from "extraction
# never ran". That is the exact blind spot this work exists to remove, so the
# malformed shapes must still produce a row.
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,reason",
    [
        ({"decisions": "Cut burn"}, "not_a_list"),
        ({"decisions": 7}, "not_a_list"),
        ({"decisions": ["Cut burn to 400k"]}, "item_not_a_dict"),
        ({"decisions": [None]}, "item_not_a_dict"),
    ],
    ids=["str", "int", "list-of-str", "list-of-null"],
)
async def test_malformed_items_are_dropped_not_raised(
    payload: dict[str, Any], reason: str, db: Path
) -> None:
    stores, audit = await _run_pass(payload, db)

    assert [s.call_count for s in stores] == [0, 0, 0]
    assert len(audit) == 1, "the pass must still audit itself"
    assert audit[0]["details"]["failure"] == "", "must not have raised"
    assert audit[0]["details"]["dropped"] == [
        {"kind": "decision", "reason": reason}
    ], "the kind must match the singular spelling used by per-item drops"


@pytest.mark.asyncio
async def test_a_null_kind_is_not_a_drop(db: Path) -> None:
    """The model omitting a kind is normal, not malformed."""
    _, audit = await _run_pass(
        db_path=db, payload={"decisions": None, "initiatives": []}
    )

    assert audit[0]["details"]["dropped"] == []
    assert audit[0]["details"]["proposed"] == {
        "decisions": 0, "initiatives": 0, "advice": 0
    }


@pytest.mark.asyncio
async def test_one_malformed_kind_does_not_lose_the_others(db: Path) -> None:
    """A bad `decisions` value used to abort the pass before `initiatives`
    was ever read, silently discarding good items alongside the bad one."""
    stores, audit = await _run_pass(db_path=db, payload={
        "decisions": "oops",
        "initiatives": [
            {"title": "Cut burn", "summary": "Down to 400k",
             "user_commitment_quote": "Cut burn to 400k"},
        ],
    })

    assert stores.initiative.call_args.kwargs == {
        "title": "Cut burn",
        "status": "active",
        "summary": "Down to 400k",
        "db_path": db,
    }
    assert audit[0]["details"]["stored"]["initiatives"] == 1
    assert audit[0]["details"]["dropped"] == [
        {"kind": "decision", "reason": "not_a_list"}
    ]


@pytest.mark.asyncio
async def test_a_non_dict_payload_is_dropped(db: Path) -> None:
    stores, audit = await _run_pass(["decisions"], db)

    assert [s.call_count for s in stores] == [0, 0, 0]
    assert audit[0]["details"]["dropped"] == [
        {"kind": "pass", "reason": "payload_not_a_dict"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"decisions": [{"summary": "", "user_commitment_quote": "Cut burn to 400k"}]},
         {"kind": "decision", "reason": "missing_field", "field": "summary"}),
        ({"initiatives": [{"summary": "Down to 400k",
                           "user_commitment_quote": "Cut burn to 400k"}]},
         {"kind": "initiative", "reason": "missing_field", "field": "title"}),
        ({"advice": [{"query_summary": "How much runway?",
                      "user_commitment_quote": "Cut burn to 400k"}]},
         {"kind": "advice", "reason": "missing_field", "field": "advice_summary"}),
    ],
    ids=["decision-summary", "initiative-title", "advice-summary"],
)
async def test_an_incomplete_item_is_recorded_as_a_drop(
    payload: dict[str, Any], expected: dict[str, str], db: Path
) -> None:
    """These paths counted the item as proposed and then `continue`d without a
    drop record, so `proposed == stored + dropped` did not hold and the audit
    row understated how much the model had thrown away."""
    _, audit = await _run_pass(payload, db)

    assert audit[0]["details"]["dropped"] == [expected]


@pytest.mark.asyncio
async def test_proposed_equals_stored_plus_dropped(db: Path) -> None:
    """The arithmetic an operator reads the row for. One item of every
    outcome: stored, bad quote, incomplete, and a malformed non-dict."""
    _, audit = await _run_pass(db_path=db, payload={
        "decisions": [
            {"summary": "Cut burn to 400k", "user_commitment_quote": "Cut burn to 400k"},
            {"summary": "Sell the building", "user_commitment_quote": "sell the building"},
            {"summary": "", "user_commitment_quote": "Cut burn to 400k"},
            "not a dict",
        ],
    })

    details = audit[0]["details"]
    assert details["proposed"]["decisions"] == 3, "the non-dict never becomes an item"
    assert details["stored"]["decisions"] == 1
    assert details["dropped_count"] == 3, "two rejected items plus the non-dict"

    # A malformed shape is a drop that was never a proposed item, so it is
    # excluded here. Every item that WAS proposed must be stored or dropped —
    # a path that counts one and then falls through silently is the defect.
    shape_reasons = {"not_a_list", "item_not_a_dict", "payload_not_a_dict"}
    item_drops = [d for d in details["dropped"] if d["reason"] not in shape_reasons]
    assert sum(details["proposed"].values()) == (
        sum(details["stored"].values()) + len(item_drops)
    )
    assert sorted(d["reason"] for d in details["dropped"]) == [
        "bad_quote", "item_not_a_dict", "missing_field",
    ]


@pytest.mark.asyncio
async def test_a_crashed_pass_still_audits_with_a_failure_reason(db: Path) -> None:
    """A provider outage that writes no row at all reads exactly like
    "extraction was never scheduled" — the state this row exists to rule out.
    """
    rows: list[dict[str, Any]] = []

    def _capture(event_type: str, summary: str, **kw: Any) -> None:
        rows.append({"event_type": event_type, "summary": summary, **kw})

    with (
        mock.patch("openexecutive.audit.log_event", _capture),
        mock.patch("openexecutive.config.get_settings"),
        mock.patch.object(episodic, "get_active_initiatives", return_value=[]),
        mock.patch("openexecutive.providers.get_provider") as provider,
    ):
        provider.return_value.messages_create = mock.AsyncMock(
            side_effect=RuntimeError("upstream 529")
        )
        await episodic.extract_and_store(
            "Cut burn to 400k.", "ok", db_path=db, session_id="s-1"
        )

    audit = [r for r in rows if r["event_type"] == "memory_extraction"]
    assert len(audit) == 1
    assert audit[0]["details"]["failure"] == "RuntimeError"
    assert audit[0]["summary"].startswith("FAILED(RuntimeError)")


# --------------------------------------------------------------------- #
# Field types
#
# `_iter_items` guards the container shapes; it says nothing about what is
# inside an item. A non-string `user_commitment_quote` or label used to raise
# out of the whole pass from inside the validator, which lost the OTHER kinds
# in the same payload — the regression the malformed-container tests above
# claim to pin, arriving through a door they do not cover.
# --------------------------------------------------------------------- #


_VALID_INITIATIVE = {
    "title": "Cut burn",
    "summary": "Down to 400k",
    "user_commitment_quote": "Cut burn to 400k",
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision",
    [
        {"summary": "S", "user_commitment_quote": 5},
        {"summary": 123, "user_commitment_quote": "nope"},
        {"summary": ["a"], "user_commitment_quote": None},
    ],
    ids=["int-quote", "int-summary", "list-summary"],
)
async def test_non_string_fields_do_not_abort_the_pass(
    decision: dict[str, Any], db: Path
) -> None:
    stores, audit = await _run_pass(
        db_path=db,
        payload={"decisions": [decision], "initiatives": [_VALID_INITIATIVE]},
    )

    assert audit[0]["details"]["failure"] == "", "the pass must not have raised"
    assert stores.initiative.call_count == 1, (
        "a bad decision must not cost the valid initiative beside it"
    )
    assert stores.decision.call_count == 0


@pytest.mark.asyncio
async def test_a_failed_write_is_not_counted_as_stored(db: Path) -> None:
    """`stored` counted the item before the INSERT, so a locked database
    reported `stored>0` while writing nothing.

    That inverts the one alarm this row exists for: a total store-layer
    failure would surface as a healthy-looking pass instead of the
    `proposed>0, stored=0` streak the operator is told to watch for.
    """
    class _Block:
        type = "tool_use"
        name = "store_memories"
        input = {"decisions": [
            {"summary": "Cut burn to 400k", "user_commitment_quote": "Cut burn to 400k"},
        ]}

    class _Response:
        content = [_Block()]

    rows: list[dict[str, Any]] = []

    with (
        mock.patch.object(
            episodic, "store_decision",
            side_effect=sqlite3.OperationalError("database is locked"),
        ),
        mock.patch(
            "openexecutive.audit.log_event",
            lambda event_type, summary, **kw: rows.append(
                {"event_type": event_type, "summary": summary, **kw}
            ),
        ),
        mock.patch("openexecutive.audit.usage.log_model_usage"),
        mock.patch("openexecutive.config.get_settings"),
        mock.patch.object(episodic, "get_active_initiatives", return_value=[]),
        mock.patch("openexecutive.providers.get_provider") as provider,
    ):
        provider.return_value.messages_create = mock.AsyncMock(
            return_value=_Response()
        )
        await episodic.extract_and_store(
            "Cut burn to 400k.", "ok", db_path=db, session_id="s-1"
        )

    audit = [r for r in rows if r["event_type"] == "memory_extraction"]
    assert audit[0]["details"]["stored"]["decisions"] == 0, (
        "nothing was written, so nothing may be reported as stored"
    )
    assert audit[0]["details"]["failure"] == "OperationalError"


@pytest.mark.asyncio
async def test_a_cancelled_pass_is_not_logged_as_an_empty_one(db: Path) -> None:
    """`CancelledError` is a `BaseException`, so `except Exception` misses it
    while the `finally` still writes a row.

    The row then read `proposed=0 stored=0 failure=''` — byte for byte what
    "the model proposed nothing" looks like, which is the exact ambiguity the
    row was added to remove. Cancellation must still propagate.
    """
    rows: list[dict[str, Any]] = []

    async def _hang(*args: Any, **kwargs: Any) -> Any:
        # Long enough to still be awaiting when the cancel lands, short enough
        # that a cancel which does NOT land fails this test in seconds rather
        # than hanging the CI job until its own timeout.
        await asyncio.sleep(5)
        raise AssertionError("the pass was never cancelled")

    with (
        mock.patch(
            "openexecutive.audit.log_event",
            lambda event_type, summary, **kw: rows.append(
                {"event_type": event_type, "summary": summary, **kw}
            ),
        ),
        mock.patch("openexecutive.config.get_settings"),
        mock.patch.object(episodic, "get_active_initiatives", return_value=[]),
        mock.patch("openexecutive.providers.get_provider") as provider,
    ):
        provider.return_value.messages_create = _hang
        task = asyncio.create_task(
            episodic.extract_and_store(
                "Cut burn to 400k.", "ok", db_path=db, session_id="s-1"
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    audit = [r for r in rows if r["event_type"] == "memory_extraction"]
    assert len(audit) == 1
    assert audit[0]["details"]["failure"] == "CancelledError"
    assert audit[0]["summary"].startswith("FAILED(CancelledError)")


@pytest.mark.asyncio
async def test_a_garbage_pass_does_not_read_as_an_empty_one(db: Path) -> None:
    """`proposed` counts only items the model emitted as objects, so a payload
    that was nothing but garbage would otherwise read `proposed=0 stored=0` —
    identical to a turn with nothing to extract. `malformed` separates them."""
    _, audit = await _run_pass(db_path=db, payload={"decisions": ["a", "b"]})

    details = audit[0]["details"]
    assert details["proposed"]["decisions"] == 0
    assert details["malformed_count"] == 2
    assert "malformed=2" in audit[0]["summary"]

    _, quiet = await _run_pass(db_path=db, payload={"decisions": []})
    assert quiet[0]["details"]["malformed_count"] == 0
