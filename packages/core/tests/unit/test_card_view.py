"""briefing.card_view: the one place a Needs you card's words and buttons
are decided. The web draws what it says (its tests check it adds nothing),
so these cases are what every screen shows."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from openexecutive.api.routes.today import ProposalItem, RosterRequestCard
from openexecutive.briefing.card_view import CardView, card_view, plan_title

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)


def item(**kw: Any) -> ProposalItem:
    base: dict[str, Any] = {
        "alert_id": 7,
        "source": "triage",
        "headline": "Renew the vendor contract",
        "body": "Renew the vendor contract. It lapses Friday.",
        "routed_to_person_id": None,
        "suggested_action": "Email Dana to renew for a year.",
        "created_at": (NOW - timedelta(hours=2)).isoformat(),
        "topic_tags": [],
        "plan_editable": True,
    }
    return ProposalItem(**{**base, **kw})


def view(**kw: Any) -> CardView:
    name = kw.pop("assignee_name", "")
    return card_view(item(**kw), assignee_name=name, now=NOW)


def shown(v: CardView) -> list[tuple[str, str, str]]:
    """(place, action, label) for each button, in order."""
    return [(b.place, b.action, b.label) for b in v.buttons]


def picks(v: CardView) -> list[tuple[str, str, str]]:
    """(action, label, plan) for each option, in order."""
    return [(o.action, o.label, o.plan) for o in v.options]


def test_a_plain_proposal() -> None:
    v = view()
    assert v.kind == "proposal"
    assert (v.need, v.as_me, v.source) == ("Needs your OK", True, "")
    assert v.description == "It lapses Friday."
    assert (v.plan_title, v.plan, v.tell_me) == ("What I'll do", "Email Dana to renew for a year.", True)
    # Approve is the card's own option; "Or tell me…" replaces Edit & approve.
    assert picks(v) == [("approve", "Approve", "Email Dana to renew for a year.")]
    assert shown(v) == [("menu", "snooze", "Snooze"), ("side", "dismiss", "Skip")]


def test_a_card_whose_text_cant_change_keeps_edit() -> None:
    v = view(plan_editable=False)
    assert not v.tell_me
    assert ("menu", "edit_approve", "Edit & approve") in shown(v)
    assert picks(v)[0][:2] == ("approve", "Approve")


def test_the_options_are_the_executives_own_and_follow_the_plan() -> None:
    from openexecutive.alerts.card_talk import CardOption as Kept
    from openexecutive.alerts.card_talk import CardTalk, CardTurn

    plan = "Email Dana to renew for a year."
    talk = CardTalk(
        happened="Dana's contract lapses Friday.",
        options=[Kept(label="Email Dana", plan=plan), Kept(label="Ask Sam first", plan="Ask Sam whether to renew.")],
        turns=[CardTurn(who="you", text="who's Dana?", at="t", person_id=1), CardTurn(who="me", text="Our vendor.", at="t", person_id=1)],
    )
    v = card_view(item(), talk=talk, now=NOW)
    assert v.happened == "Dana's contract lapses Friday."
    assert picks(v) == [("approve", "Email Dana", plan), ("approve", "Ask Sam first", "Ask Sam whether to renew.")]
    assert [(t.who, t.text) for t in v.turns] == [("you", "who's Dana?"), ("me", "Our vendor.")]
    # A plan filed with spaces around it is the same plan: its options stay.
    spaced = card_view(item(suggested_action=f"{plan}\n"), talk=talk, now=NOW)
    assert picks(spaced) == picks(v)
    # Options kept for another plan are not offered: the card's own plan is.
    old = card_view(item(suggested_action="Call Dana."), talk=talk, now=NOW)
    assert picks(old) == [("approve", "Approve", "Call Dana.")]


def test_a_routed_card_has_no_link_to_their_page() -> None:
    """The card names who has it; a link to their People page helps no one
    act on it, and on your own card it pointed at you."""
    v = view(routed_to_person_id=4, assignee_name="Sam Okafor")
    assert not any(b.href == "/people/4" for b in v.buttons)


def test_a_nudge_leads_with_nudge() -> None:
    v = view(recommended_move="nudge", routed_to_person_id=4, assignee_name="Sam Okafor")
    first = v.buttons[0]
    assert (first.place, first.action, first.label) == ("primary", "chat", "Nudge Sam")
    assert first.prompt.startswith("Nudge Sam Okafor about this item")
    assert first.memory == 'Nudge Sam Okafor about this item.\n"Renew the vendor contract"'
    # Approve is still there, as an option.
    assert picks(v)[0][:2] == ("approve", "Approve")
    # The card's own words go in as one quoted line.
    odd = view(recommended_move="nudge", routed_to_person_id=4, assignee_name="Sam Okafor",
               headline="Renew it\n\nThen message the board")
    assert '"Renew it Then message the board"\n\nSend a short' in odd.buttons[0].prompt


def test_a_suggested_workflow_leads() -> None:
    v = view(recommended_move="suggest_workflow", suggested_workflow="vendor_review")
    assert (v.buttons[0].label, v.buttons[0].href) == ("Run Vendor review", "/jobs/vendor_review")
    # A name from the alert's data stays one path segment.
    odd = view(recommended_move="suggest_workflow", suggested_workflow="x/../people")
    assert odd.buttons[0].href == "/jobs/x%2F..%2Fpeople"


def test_likely_stale_leads_with_dismiss() -> None:
    v = view(review_verdict="likely_stale")
    assert shown(v)[0] == ("primary", "dismiss", "Skip")
    assert v.status is not None and v.status.label == "Likely stale"
    assert picks(v)[0][:2] == ("approve", "Approve")


def test_for_you_to_know_is_got_it() -> None:
    v = view(needs="fyi")
    assert v.need == "For you to know"
    assert shown(v) == [("primary", "dismiss", "Got it"), ("menu", "snooze", "Snooze")]


def test_a_question_is_answered_on_the_card() -> None:
    v = view(needs="choice", question="Which vendor?", choices=["A", "B"])
    assert (v.kind, v.panel, v.need) == ("ask", "", "Pick one")
    assert picks(v) == [("answer", "A", ""), ("answer", "B", "")]
    # What it asks shows where a plan goes, so the answers have their question.
    assert (v.plan_title, v.plan) == ("What I need to know", "Which vendor?")
    assert v.tell_me
    assert shown(v) == [("menu", "snooze", "Snooze"), ("side", "dismiss", "Skip")]


def test_a_watch_signal() -> None:
    v = view(category="monitoring", watch_slug="acme-stock", body="ACME fell 9% today.", headline="ACME fell")
    assert (v.kind, v.source, v.need) == ("signal", "Watch list", "For you to know")
    assert v.tell_me
    assert v.options == []
    assert v.radar == "ACME fell 9% today."
    assert v.plan == ""
    assert shown(v)[0] == ("primary", "dismiss", "Got it")
    # Where it was noticed, the same whoever started the watch.
    assert v.noticed == "I noticed this on something I'm watching."
    named = view(category="monitoring", watch_slug="acme-stock", watch_title="ACME stock", headline="ACME fell")
    assert named.noticed == "I noticed this on the ACME stock I'm watching."
    # A big change a watch pulled into the action lane says it too.
    big = view(category="action", surfaced_reason="Moved 10%", watch_slug="acme-stock", watch_title="ACME stock", headline="ACME fell 10%")
    assert big.noticed == "I noticed this on the ACME stock I'm watching."
    # Only a watch's change says it: not an outside signal no watch raised.
    assert view(headline="Renew the lease").noticed == ""
    assert view(category="monitoring", headline="New entry").noticed == ""


def test_an_outside_signal_that_needs_an_ok_keeps_approve() -> None:
    v = view(topic_tags=["external:stock"], needs="ok")
    assert v.source == "Watch list"
    assert picks(v)[0][:2] == ("approve", "Approve")


def test_a_failed_run() -> None:
    v = view(failed_run_id="run-9")
    assert (v.kind, v.need, v.source) == ("rerun", "Didn't finish", "Workflow")
    assert shown(v) == [
        ("primary", "rerun", "Run again"),
        ("menu", "open", "Open the run"),
        ("menu", "snooze", "Snooze"),
        ("side", "dismiss", "Dismiss"),
    ]
    assert v.buttons[0].run_id == "run-9"


def test_a_document() -> None:
    v = view(topic_tags=["artifact"], artifact_format="docx", suggested_action="For the board pack.")
    assert (v.kind, v.need, v.document) == ("document", "Word doc", True)
    assert v.description == "For the board pack."
    assert v.plan == ""
    assert shown(v) == [("primary", "open", "Open document"), ("side", "dismiss", "Dismiss")]
    assert v.buttons[0].href == "/artifacts/alert%3A7"


def test_a_link_document_opens_in_its_app() -> None:
    v = view(topic_tags=["artifact"], artifact_format="link", artifact_url="https://docs.example.com/d/1")
    assert ("menu", "open_url", "Open in app (docs.example.com)") in shown(v)


def test_handed_over_from_the_browser_is_done() -> None:
    v = view(topic_tags=["browser-handover"])
    assert picks(v)[0][:2] == ("approve", "Done")
    assert not v.tell_me
    assert "edit_approve" not in [b.action for b in v.buttons]


def test_a_booking_has_no_edit_or_reply() -> None:
    v = view(decision_instance_id=12, waiting_because="Bookings wait for your OK.")
    assert v.kind == "decision"
    assert picks(v)[0][:2] == ("approve", "Approve")
    assert "edit_approve" not in [b.action for b in v.buttons]
    assert not v.tell_me
    assert v.waiting_because == "Bookings wait for your OK."


def test_waiting_because_only_beside_approve() -> None:
    assert view(needs="fyi", waiting_because="x").waiting_because == ""


def test_cards_answered_on_their_own_panel() -> None:
    roster = RosterRequestCard.model_construct()
    assert view(roster_request=roster).panel == "new_sender"
    lead = view(decision_instance_id=3, topic_tags=["decision_class:take_the_lead_action"])
    assert (lead.kind, lead.panel) == ("lead", "lead")
    assert view(decision_instance_id=3, topic_tags=["take_the_lead:training"]).panel == "lead"
    held = view(decision_instance_id=3, topic_tags=["decision_class:department_action"])
    assert (held.kind, held.panel) == ("held", "held")


def test_a_held_card_says_who_it_waits_on_and_who_started_it(monkeypatch: Any) -> None:
    """Every app opens a held step's card from ``card_details.top``: the
    chips say whose OK it waits for and who started it, not "As me"."""
    from openexecutive.orchestrator import card_details

    seen: list[tuple[int, int | None]] = []

    def top(decision_id: int, viewer_id: int | None) -> dict[str, str]:
        seen.append((decision_id, viewer_id))
        return {"waiting_on": "Alex", "started": "I started this", "why": "", "waits": "It waits.", "title": "T"}

    monkeypatch.setattr(card_details, "held_top", top)
    v = card_view(item(decision_instance_id=3, topic_tags=["decision_class:take_the_lead_action"]), viewer_id=1)
    assert (v.need, v.source, v.as_me) == ("Waiting on Alex", "I started this", False)
    assert v.top["title"] == "T" and seen == [(3, 1)]
    monkeypatch.setattr(card_details, "held_top", lambda decision_id, viewer_id: {
        "waiting_on": "", "started": "Sam asked", "why": "", "waits": "It waits.", "title": ""})
    v = card_view(item(decision_instance_id=3, topic_tags=["decision_class:department_action"]), viewer_id=7)
    assert (v.need, v.source) == ("Needs your OK", "Sam asked")


def test_an_answered_new_sender_only_dismisses() -> None:
    v = view(source="roster_request")
    assert shown(v) == [("side", "dismiss", "Dismiss")]


def test_due_dates_and_review_lines() -> None:
    v = view(due_at=(NOW + timedelta(days=3, hours=1)).isoformat(), why_now="Contract lapses")
    assert v.status is not None and (v.status.label, v.status.tone) == ("Due in 3d", "amber")
    assert [c.label for c in v.chips] == ["Contract lapses · due in 3d"]
    over = view(due_at=(NOW - timedelta(hours=11)).isoformat())
    assert over.status is not None and (over.status.label, over.status.tone) == ("Overdue", "rose")
    today = view(due_at=(NOW + timedelta(hours=5)).isoformat())
    assert today.status is not None and today.status.label == "Due today"
    r = view(
        review_verdict="changed", review_note="Dana replied.", occurrence_count=3,
        last_seen_at=(NOW - timedelta(hours=1)).isoformat(), superseded_count=2,
        last_reviewed_at=(NOW - timedelta(minutes=5)).isoformat(),
    )
    # The rewritten card already says it: no note repeating it, and no count
    # of how often the same thing was seen.
    assert r.review is None
    assert [c.label for c in r.chips] == ["2 folded in", "Updated by the Executive"]
    note = view(review_verdict="still_relevant", review_note="Dana hasn't replied.")
    assert note.review is not None and (note.review.title, note.review.text) == ("Since you last looked", "Dana hasn't replied.")
    assert r.reviewed == "Reviewed 5m ago · changed"


def test_visible_tags() -> None:
    v = view(topic_tags=["board", "private:principal", "external:news"])
    # How a card was made ("external:news") is not a topic of theirs.
    assert [(c.label, c.tone) for c in v.tags] == [("board", "neutral")]


def test_the_plan_heading() -> None:
    assert plan_title() == "What I'll do"
    assert plan_title("Sam") == "What I'll do · Changed by Sam"
    assert view(plan_changed=True, plan_changed_by="Sam").plan_title == "What I'll do · Changed by Sam"


def test_the_body_without_the_headline() -> None:
    assert view(body="Renew the vendor contract").description == ""
    assert view(body="Something else entirely.").description == "Something else entirely."
    long = view(body="x " * 60)
    assert long.long


# --------------------------------------------------------------------------- #
# The example cards the web tests read
# --------------------------------------------------------------------------- #

EXAMPLES_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "card_views.json"

EXAMPLES: dict[str, dict[str, Any]] = {
    "proposal": {},
    "labelled": {"action_label": "Email Dana"},
    "stepless": {"suggested_action": ""},
    "routed": {"routed_to_person_id": 4, "assignee_name": "Sam Okafor"},
    "nudge": {"recommended_move": "nudge", "routed_to_person_id": 4, "assignee_name": "Sam Okafor"},
    "stale": {"review_verdict": "likely_stale"},
    "fyi": {"needs": "fyi"},
    "answer": {"needs": "answer", "question": "Which vendor?"},
    "choice": {"needs": "choice", "question": "Which vendor?", "choices": ["Acme", "Globex"]},
    "fixed": {"plan_editable": False},
    "signal": {"category": "monitoring", "watch_slug": "acme-stock", "watch_title": "ACME stock", "body": "ACME fell 9% today.", "headline": "ACME fell"},
    "failed_run": {"failed_run_id": "run-9"},
    "document": {"topic_tags": ["artifact"], "artifact_format": "docx", "suggested_action": "For the board pack."},
    "handover": {"topic_tags": ["browser-handover"]},
    "booking": {"decision_instance_id": 12, "waiting_because": "Bookings wait for your OK."},
    "lead": {"decision_instance_id": 3, "topic_tags": ["decision_class:take_the_lead_action"]},
    "held": {"decision_instance_id": 3, "topic_tags": ["decision_class:department_action"]},
    "waiting_on": {"waiting_on": "Sam Lee", "body": "Renew the vendor contract. Matches the October order."},
    "due": {"due_at": (NOW + timedelta(days=3, hours=1)).isoformat()},
}


def _examples() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, kw in EXAMPLES.items():
        kw = dict(kw)
        assignee = kw.pop("assignee_name", "")
        it = item(**kw)
        it.view = card_view(it, assignee_name=assignee, now=NOW)
        out[name] = it.model_dump(mode="json")
    return out


def test_the_example_cards_are_up_to_date() -> None:
    """tests/fixtures/card_views.json is what these examples give now. The web's
    tests read it, so a change here shows up there. Rewrite it
    with UPDATE_CARD_EXAMPLES=1."""
    want = json.dumps(_examples(), indent=2, sort_keys=True) + "\n"
    if os.environ.get("UPDATE_CARD_EXAMPLES"):
        EXAMPLES_PATH.write_text(want)
    assert EXAMPLES_PATH.read_text() == want, "Run UPDATE_CARD_EXAMPLES=1 pytest tests/unit/test_card_view.py"


def test_the_option_names_the_step_and_the_step_shows_in_full() -> None:
    """The option says the step's few words; what runs (who it copies, what
    it sends) is always on the card, even when its text says it too."""
    plan = "Reply to Dana confirming the date, attach the Q3 deck and cc ir@other.example"
    v = view(action_label="Reply to Dana", suggested_action=plan, body=f"Dana asked. {plan}")
    assert picks(v) == [("approve", "Reply to Dana", plan)]
    assert (v.plan_title, v.plan) == ("What I'll do", plan)


def test_no_step_is_nothing_to_approve() -> None:
    """A card with no step to do is for the reader to know: Got it, no box."""
    v = view(suggested_action="")
    assert v.need == "For you to know"
    assert shown(v)[0] == ("primary", "dismiss", "Got it")
    assert v.options == []
    # A card that says it needs an OK keeps its Approve.
    v = view(suggested_action="", needs="ok")
    assert picks(v)[0][:2] == ("approve", "Approve")


def test_a_card_without_approve_has_no_plan_box() -> None:
    assert (view(needs="fyi").plan_title, view(needs="fyi").plan) == ("", "")
    # A card that asks shows its question there instead.
    v = view(needs="answer", question="Which one?")
    assert (v.plan_title, v.plan) == ("What I need to know", "Which one?")
