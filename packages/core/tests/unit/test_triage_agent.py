"""Tests for the TriageAgent — verifies prompt context assembly + decision parsing.

The Anthropic client is patched so no real API call is made.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from openexecutive.agents.triage import TRIAGE_TOOL, TriageAgent, _parse_decision
from openexecutive.alerts.models import (
    AlertChannel,
    AlertEvent,
    AlertSeverity,
)


def _fake_response(tool_input: dict):
    """Build a fake anthropic Message with one tool_use content block."""
    block = SimpleNamespace(type="tool_use", name="emit_alert_decision", input=tool_input)
    return SimpleNamespace(content=[block])


def test_triage_model_follows_routing_model(monkeypatch) -> None:
    """Triage must use the configured routing model, not a hardcoded Claude
    slug — otherwise an Anthropic-free (local / OpenRouter) deployment can't
    run the alert pipeline. The property imports get_settings from
    openexecutive.config at call time, so patch it there."""
    monkeypatch.setattr(
        "openexecutive.config.get_settings",
        lambda: SimpleNamespace(routing_model="llama3.3"),
    )
    assert TriageAgent().model == "llama3.3"


def test_triage_returns_parsed_decision_and_passes_context() -> None:
    fake_input = {
        "alert": True,
        "severity": "urgent",
        "channels": ["web", "slack_dm", "persisted"],
        "headline": "Top customer cancelling",
        "body": "ARR loss of $200k",
        "suggested_action": "Call them in the next hour",
        "topic_tags": ["customer", "churn"],
        "dedup_key": "churn-acme",
        "reason_if_suppressed": "",
    }
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=_fake_response(fake_input)))
    )

    agent = TriageAgent()
    decision = asyncio.run(
        agent.triage(
            AlertEvent(
                source="email",
                external_id="msg-1",
                subject="Cancelling our subscription",
                body="We will be cancelling next month.",
            ),
            recent_alerts=[
                {
                    "headline": "Other thing",
                    "severity": "low",
                    "dedup_key": "x",
                    "topic_tags": ["finance"],
                }
            ],
            mute_patterns=["weekly_newsletter"],
            active_initiatives=[
                SimpleNamespace(title="Reduce churn", status="active", summary="Lower 6m churn to <2%")
            ],
            client=client,
        )
    )

    assert decision.alert is True
    assert decision.severity == AlertSeverity.URGENT
    assert AlertChannel.SLACK_DM in decision.channels
    assert AlertChannel.PERSISTED in decision.channels
    assert decision.headline == "Top customer cancelling"
    assert decision.dedup_key == "churn-acme"

    # Verify the user turn included the right context tags.
    call = client.messages.create.await_args
    user_msg = call.kwargs["messages"][0]["content"]
    assert "<event>" in user_msg
    assert "Cancelling our subscription" in user_msg
    assert "<recent_alerts>" in user_msg
    assert "Other thing" in user_msg
    assert "<muted_topics>" in user_msg
    assert "weekly_newsletter" in user_msg
    assert "<active_initiatives>" in user_msg
    assert "Reduce churn" in user_msg

    # Verify forced tool-use was requested.
    assert call.kwargs["tool_choice"] == {"type": "tool", "name": "emit_alert_decision"}
    assert call.kwargs["tools"] == [TRIAGE_TOOL]


def test_triage_falls_back_when_no_tool_use() -> None:
    """If the model replies with text only (or anything non-tool_use), an
    ordinary event gets a quiet persisted-only fallback rather than a crash."""
    text_block = SimpleNamespace(type="text", text="Just chatting")
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(content=[text_block])))
    )
    agent = TriageAgent()
    decision = asyncio.run(
        agent.triage(
            AlertEvent(source="email", external_id="m-2", subject="hi", body="b"),
            client=client,
        )
    )
    assert decision.alert is False
    assert decision.channels == [AlertChannel.PERSISTED]
    assert decision.reason_if_suppressed == "no_decision"


def _raised(**kw: object) -> AlertEvent:
    """A card the Executive raised itself (create_alert)."""
    return AlertEvent(source="email", external_id="m-4", subject="Leo asks about Friday",
                      body="Open Friday afternoon?", from_executive=True, **kw)


def _client(**reply: object) -> SimpleNamespace:
    return SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(**reply)))


def test_triage_keeps_a_raised_card_when_its_answer_is_cut_off() -> None:
    """An answer stopped at max_tokens is a partial decision whose missing
    ``alert`` would read as "no card": a card the Executive raised shows as
    raised instead."""
    cut = SimpleNamespace(type="tool_use", name="emit_alert_decision", input={"severity": "medium"})
    client = _client(return_value=SimpleNamespace(content=[cut], stop_reason="max_tokens"))
    decision = asyncio.run(TriageAgent().triage(_raised(), client=client))
    assert decision.alert is True
    assert AlertChannel.WEB in decision.channels
    assert (decision.headline, decision.body) == ("Leo asks about Friday", "Open Friday afternoon?")
    assert client.messages.create.call_args.kwargs["max_tokens"] >= 1024


def test_triage_keeps_a_raised_card_when_review_fails() -> None:
    decision = asyncio.run(TriageAgent().triage(_raised(), client=_client(side_effect=RuntimeError("boom"))))
    assert decision.alert is True


def test_kept_cards_without_an_id_never_fold_into_each_other() -> None:
    """create_alert needs no external id: two different cards kept while
    review is down get different keys, and the same card again the same."""
    def kept(subject: str, body: str) -> str:
        event = AlertEvent(source="unknown", external_id="", subject=subject, body=body, from_executive=True)
        return asyncio.run(TriageAgent().triage(event, client=_client(side_effect=RuntimeError("down")))).dedup_key
    leo = kept("Leo asks about Friday", "Open Friday afternoon?")
    assert leo != kept("Invoice 4411 overdue", "Due last week.")
    assert leo == kept("Leo asks about Friday", "Open Friday afternoon?")


def test_triage_keeps_a_raised_card_when_no_tool_call() -> None:
    text = SimpleNamespace(type="text", text="Just chatting")
    decision = asyncio.run(TriageAgent().triage(
        _raised(), client=_client(return_value=SimpleNamespace(content=[text], stop_reason="end_turn")),
    ))
    assert decision.alert is True


def test_a_failed_review_never_turns_chat_lines_into_cards() -> None:
    """An outage fails review for every event: a chat line or a polled mail
    is recorded quietly, not raised (only create_alert's cards are kept)."""
    for event in (
        AlertEvent(source="slack", external_id="s-1", body="lunch?"),
        AlertEvent(source="email", external_id="m-5", subject="Weekly newsletter", body="..."),
    ):
        decision = asyncio.run(TriageAgent().triage(event, client=_client(side_effect=RuntimeError("down"))))
        assert decision.alert is False
        assert decision.channels == [AlertChannel.PERSISTED]


def test_a_muted_topic_stays_muted_when_review_fails() -> None:
    decision = asyncio.run(TriageAgent().triage(
        _raised(), mute_patterns=["friday"], client=_client(side_effect=RuntimeError("boom")),
    ))
    assert decision.alert is False


def test_a_short_mute_only_holds_back_whole_words() -> None:
    """A mute of "ai" doesn't hold back a card that says "emailed"."""
    event = AlertEvent(source="email", external_id="m-6", subject="Leo emailed about the air freight",
                       body="Open Friday afternoon?", from_executive=True)
    decision = asyncio.run(TriageAgent().triage(
        event, mute_patterns=["ai"], client=_client(side_effect=RuntimeError("boom")),
    ))
    assert decision.alert is True


def test_a_cut_off_review_that_said_raise_keeps_any_card() -> None:
    """A watch event whose review was cut off after saying alert=True still
    shows, in its own words; one cut off before deciding stays quiet."""
    event = AlertEvent(source="monitoring", external_id="w-1", subject="Competitor price drop",
                       body="Acme cut prices 20%.")
    said = SimpleNamespace(type="tool_use", name="emit_alert_decision", input={"alert": True})
    client = _client(return_value=SimpleNamespace(content=[said], stop_reason="max_tokens"))
    decision = asyncio.run(TriageAgent().triage(event, client=client))
    assert decision.alert is True
    assert decision.headline == "Competitor price drop"
    early = SimpleNamespace(type="tool_use", name="emit_alert_decision", input={})
    client = _client(return_value=SimpleNamespace(content=[early], stop_reason="max_tokens"))
    assert asyncio.run(TriageAgent().triage(event, client=client)).alert is False


def test_a_cut_off_review_keeps_the_severity_it_gave() -> None:
    event = AlertEvent(source="monitoring", external_id="w-2", subject="Site down", body="Checkout fails.")
    said = SimpleNamespace(type="tool_use", name="emit_alert_decision",
                           input={"alert": True, "severity": "urgent", "channels": ["web", "slack_dm"]})
    client = _client(return_value=SimpleNamespace(content=[said], stop_reason="max_tokens"))
    decision = asyncio.run(TriageAgent().triage(event, client=client))
    assert decision.severity == AlertSeverity.URGENT
    assert AlertChannel.SLACK_DM in decision.channels and AlertChannel.PERSISTED in decision.channels


def test_a_cut_off_review_that_said_no_card_stands() -> None:
    said = SimpleNamespace(type="tool_use", name="emit_alert_decision",
                           input={"alert": False, "severity": "low"})
    client = _client(return_value=SimpleNamespace(content=[said], stop_reason="max_tokens"))
    assert asyncio.run(TriageAgent().triage(_raised(), client=client)).alert is False


def test_a_mute_holds_back_its_plural_when_review_fails() -> None:
    event = AlertEvent(source="email", external_id="m-7", subject="Q3 newsletters ready",
                       body="Two drafts.", from_executive=True)
    decision = asyncio.run(TriageAgent().triage(
        event, mute_patterns=["newsletter"], client=_client(side_effect=RuntimeError("boom")),
    ))
    assert decision.alert is False


def test_a_muted_department_holds_back_a_kept_card() -> None:
    """ "Mute this topic" on a Sales card stores department:sales: a card
    the Executive raises for Sales stays muted while review is down."""
    event = AlertEvent(source="unknown", external_id="", subject="Pipeline update", body="Two deals moved.",
                       department="sales", from_executive=True)
    decision = asyncio.run(TriageAgent().triage(
        event, mute_patterns=["department:sales"], client=_client(side_effect=RuntimeError("boom")),
    ))
    assert decision.alert is False


def test_a_cut_off_review_keeps_its_tags_for_mutes() -> None:
    said = SimpleNamespace(type="tool_use", name="emit_alert_decision",
                           input={"alert": True, "topic_tags": ["competitor-pricing"]})
    event = AlertEvent(source="monitoring", external_id="w-3", subject="Acme price cut", body="20% off.")
    client = _client(return_value=SimpleNamespace(content=[said], stop_reason="max_tokens"))
    assert asyncio.run(TriageAgent().triage(event, client=client)).topic_tags == ["competitor-pricing"]
    client = _client(return_value=SimpleNamespace(content=[said], stop_reason="max_tokens"))
    muted = asyncio.run(TriageAgent().triage(event, mute_patterns=["competitor-pricing"], client=client))
    assert muted.alert is False


def test_a_mute_starting_with_a_symbol_holds_back_its_word() -> None:
    event = AlertEvent(source="email", external_id="m-8", subject="re: #launch plan", body="Draft.",
                       from_executive=True)
    decision = asyncio.run(TriageAgent().triage(
        event, mute_patterns=["#launch"], client=_client(side_effect=RuntimeError("boom")),
    ))
    assert decision.alert is False


def test_a_cut_off_review_that_said_raise_isnt_muted_by_words() -> None:
    """The review saw the mutes and said raise: a mute word in the body
    doesn't drop it; only the tags it gave count."""
    event = AlertEvent(source="monitoring", external_id="w-4", subject="Site down",
                       body="Checkout fails on the pricing page.")
    said = SimpleNamespace(type="tool_use", name="emit_alert_decision", input={"alert": True, "severity": "urgent"})
    client = _client(return_value=SimpleNamespace(content=[said], stop_reason="max_tokens"))
    decision = asyncio.run(TriageAgent().triage(event, mute_patterns=["pricing"], client=client))
    assert decision.alert is True


def test_a_cut_off_review_keeps_the_room_it_named() -> None:
    event = AlertEvent(source="monitoring", external_id="w-5", subject="Deal slipped", body="Acme moved to Q1.",
                       from_executive=True)
    said = SimpleNamespace(type="tool_use", name="emit_alert_decision", input={
        "alert": True, "channels": ["department_channel"], "department_slug": "sales", "broadcast_integration": "slack",
    })
    client = _client(return_value=SimpleNamespace(content=[said], stop_reason="max_tokens"))
    decision = asyncio.run(TriageAgent().triage(event, client=client))
    assert (decision.department_slug, decision.broadcast_integration) == ("sales", "slack")
    said.input = {"alert": True, "channels": ["department_channel"]}
    client = _client(return_value=SimpleNamespace(content=[said], stop_reason="max_tokens"))
    assert AlertChannel.WEB in asyncio.run(TriageAgent().triage(event, client=client)).channels


def test_outside_mail_a_cut_off_review_named_a_room_for_stays_off_the_room() -> None:
    event = AlertEvent(source="email", external_id="m-9", subject="Big order", body="Ship 40 crates.")
    said = SimpleNamespace(type="tool_use", name="emit_alert_decision", input={
        "alert": True, "severity": "high", "channels": ["department_channel", "company_broadcast"],
        "department_slug": "sales", "broadcast_integration": "slack",
    })
    client = _client(return_value=SimpleNamespace(content=[said], stop_reason="max_tokens"))
    decision = asyncio.run(TriageAgent().triage(event, client=client))
    assert decision.alert is True and decision.severity == AlertSeverity.HIGH
    assert decision.channels == [AlertChannel.WEB, AlertChannel.PERSISTED]
    assert (decision.department_slug, decision.broadcast_integration) == ("", "")


def test_a_short_mute_never_hits_a_kept_cards_department() -> None:
    event = AlertEvent(source="unknown", external_id="", subject="Store count", body="Twelve stores.",
                       department="retail", from_executive=True)
    decision = asyncio.run(TriageAgent().triage(
        event, mute_patterns=["ai"], client=_client(side_effect=RuntimeError("boom")),
    ))
    assert decision.alert is True


def test_triage_swallows_client_exceptions() -> None:
    """LLM error → quiet fallback for an ordinary event, never propagates."""
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("boom")))
    )
    agent = TriageAgent()
    decision = asyncio.run(
        agent.triage(
            AlertEvent(source="email", external_id="m-3", subject="x", body=""),
            client=client,
        )
    )
    assert decision.alert is False
    assert decision.reason_if_suppressed == "triage_error"


def test_parse_decision_coerces_persisted_in_and_caps_lengths() -> None:
    decision = _parse_decision(
        {
            "alert": True,
            "severity": "high",
            "channels": ["web"],
            "headline": "x" * 300,
            "body": "y" * 3000,
            "topic_tags": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            "dedup_key": "k" * 200,
        }
    )
    assert decision.severity == AlertSeverity.HIGH
    assert AlertChannel.PERSISTED in decision.channels
    assert len(decision.headline) <= 200
    assert len(decision.body) <= 2000
    assert len(decision.topic_tags) <= 8
    assert len(decision.dedup_key) <= 120


def test_parse_decision_unknown_severity_falls_to_low() -> None:
    decision = _parse_decision({"alert": True, "severity": "OMG_CRITICAL", "channels": []})
    assert decision.severity == AlertSeverity.LOW
    assert decision.channels == [AlertChannel.PERSISTED]


def test_triage_prompt_external_section_covers_every_registered_adapter() -> None:
    """Sync guard against drift: every monitoring adapter in the runtime
    registry must be named in the triage prompt's '## External Signals' SECTION
    (not merely somewhere in the prompt). If a new adapter's source is missing
    there, triage won't trust it or emit its `external:*` tags — which is
    exactly what kept query/edgar/page_watch out of the briefing's Monitoring
    lane. Asserting against `_REGISTRY` rather than a hardcoded list means
    adding an adapter without documenting it here fails this test."""
    from openexecutive.monitoring.sources import _REGISTRY
    from openexecutive.prompts.triage_prompt import TRIAGE_PROMPT

    start = TRIAGE_PROMPT.index("## External Signals")
    end = TRIAGE_PROMPT.index("## Output Field Requirements", start)
    section = TRIAGE_PROMPT[start:end]

    assert _REGISTRY, "expected the source adapter registry to be populated"
    for kind in _REGISTRY:
        assert kind in section, (
            f"triage '## External Signals' section omits registered adapter {kind!r}"
        )


def test_parse_decision_reads_the_buttons_words() -> None:
    decision = _parse_decision(
        {"alert": True, "severity": "medium", "suggested_action": "Email Dana to renew", "action_label": "Email Dana"}
    )
    assert decision.action_label == "Email Dana"
    assert _parse_decision({"alert": True, "severity": "medium"}).action_label == ""


def test_no_writer_hands_the_reader_a_look_up() -> None:
    """A watch card said "check the listing's attribution": a look-up the
    Executive could do. The watch's own search does it, and triage and the
    review never write one for the reader, nor make one the card's step."""
    from openexecutive.monitoring.sources.query import _QUERY_SYSTEM_PROMPT
    from openexecutive.prompts.alert_review_prompt import ALERT_REVIEW_PROMPT
    from openexecutive.prompts.triage_prompt import TRIAGE_PROMPT

    start = TRIAGE_PROMPT.index("## External Signals")
    signals = TRIAGE_PROMPT[start:TRIAGE_PROMPT.index("## Output Field Requirements", start)]
    assert "A look-up is not a step" in signals
    assert "leave `suggested_action` empty" in signals
    assert "never hands the reader a look-up" in signals
    assert "Check Stripe's status page" not in signals
    assert "never hands the reader a look-up" in ALERT_REVIEW_PROMPT
    assert "Never leave the principal a check to do" in _QUERY_SYSTEM_PROMPT
