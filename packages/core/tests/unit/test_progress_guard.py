"""The content guard for progress updates (integrations/progress_guard.py)."""
from __future__ import annotations

from typing import Any

import pytest

from openexecutive.integrations import progress_guard as pg

HOSTS = ["docs.google.com", "github.com"]

GOOD: dict[str, Any] = {
    "topic": "Q1 go-to-market plan",
    "status": "at_risk",
    "summary": (
        "Positioning draft done for the Benelux launch.\n"
        "Pricing page copy waits on the 2026-11-20 10:00 review.\n"
        "Budget so far: €12,500 of the €40k plan; v2.3 of the deck is out."
    ),
    "ask": "Pick one of the two taglines by Friday",
    "link": "https://docs.google.com/document/d/1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789/edit",
    "idempotency_key": "gtm-plan-2026-11-15",
}


def check(**overrides: Any) -> pg.CleanUpdate:
    payload = {**GOOD, **overrides}
    payload = {k: v for k, v in payload.items() if v is not ...}
    return pg.check(payload, link_hosts=HOSTS)


def rejected(**overrides: Any) -> pg.Rejected:
    with pytest.raises(pg.Rejected) as info:
        check(**overrides)
    return info.value


def test_a_realistic_update_is_accepted_and_cleaned() -> None:
    clean = check(topic="  Q1   go-to-market\tplan ", summary="  Positioning draft done.  \n\n  Next:  pricing.  ")
    assert clean.topic == "Q1 go-to-market plan"
    assert clean.topic_key == "q1 go-to-market plan"
    assert clean.status == "at_risk"
    # spaces collapse and lines are trimmed; line breaks stay
    assert clean.summary == "Positioning draft done.\n\nNext: pricing."
    assert clean.ask == "Pick one of the two taglines by Friday"
    assert clean.link == GOOD["link"]
    assert clean.idempotency_key == "gtm-plan-2026-11-15"
    assert check().summary == GOOD["summary"]


def test_optional_fields_may_be_absent_or_blank() -> None:
    clean = check(ask=..., link=..., idempotency_key=...)
    assert (clean.ask, clean.link, clean.idempotency_key) == (None, None, None)
    clean = check(ask="   ", link=None, idempotency_key=None)
    assert (clean.ask, clean.link) == (None, None)


def test_crlf_line_breaks_are_accepted() -> None:
    assert check(summary="First line of it.\r\nSecond line.").summary == "First line of it.\nSecond line."


@pytest.mark.parametrize(
    ("field", "text"),
    [
        ("summary", "Done with the draft.\rSecond"),  # a lone carriage return
        ("summary", "Done with the\x07 draft."),  # a bell
        ("summary", "Done with the​ draft."),  # a zero-width space
        ("topic", "GTM⁠plan"),  # a word joiner
        ("ask", "Approve­ it"),  # a soft hyphen
        ("link", "https://docs.google.com/​doc"),
    ],
)
def test_hidden_characters_are_rejected_not_stripped(field: str, text: str) -> None:
    err = rejected(**{field: text})
    assert err.field == field
    assert err.reason == "contains a control or invisible character"


def test_a_newline_in_a_one_line_field_is_rejected() -> None:
    assert rejected(topic="GTM\nplan").reason == "must be one line"
    assert rejected(ask="Pick one\nby Friday").reason == "must be one line"


def test_a_nine_line_summary_is_rejected_not_collapsed() -> None:
    eight = "\n".join(f"Line {i} of the update." for i in range(8))
    assert check(summary=eight).summary == eight
    # blank lines count too: the check runs on the raw text
    err = rejected(summary=eight + "\n\nLine nine.")
    assert (err.field, err.reason) == ("summary", "has more than 8 lines")


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Sent the brief to jane.doe@example.org for review.", "contains an email address"),
        ("Call the agency at +32 470 12 34 56 tomorrow.", "contains a phone number"),
        ("Call the agency at 0470 12 34 56 tomorrow.", "contains a phone number"),
        ("Call the agency at (555) 123-4567 tomorrow.", "contains a phone number"),
        ("Call the agency at 555-123-4567 tomorrow.", "contains a phone number"),
        ("Pay the invoice to BE68 5390 0754 7034 this week.", "contains a bank account number (IBAN)"),
        ("Pay the invoice to GB82WEST12345698765432 THIS WEEK.", "contains a bank account number (IBAN)"),
        ("Use key sk-ant-api03-abcdefghijklmnop to run it.", "contains something shaped like a credential"),
        ("The bot token xoxb-1234567890-abcdefghij works.", "contains something shaped like a credential"),
        ("Pushed with ghp_abcdefghijklmnopqrstuvwxyz0123 today.", "contains something shaped like a credential"),
        ("The AWS key AKIAABCDEFGHIJKLMNOP is set.", "contains something shaped like a credential"),
        ("Session eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.sig leaked.", "contains something shaped like a credential"),
        ("-----BEGIN RSA PRIVATE KEY----- is in the repo.", "contains something shaped like a credential"),
        ("Staging login: password = hunter22 for now.", "contains something shaped like a credential"),
        ("Use Zx9Kq2LmN8pR4tV7wY1bC3dF6gH0jK5s to sign in.", "contains something shaped like a credential"),
        ("Header is <b>bold</b> now.", "contains < or >"),
        ("Fake header > quoted line.", "contains < or >"),
        ("Full-width ＜b＞ still counts.", "contains < or >"),
        ("Draft is at https://evil.example/x for now.", "contains a URL; put one link in the link field"),
        ("Draft is at www.example.org for now.", "contains a URL; put one link in the link field"),
        ("Draft is at acme-launch.com for now.", "contains a URL; put one link in the link field"),
    ],
)
def test_each_deny_pattern_rejects(text: str, reason: str) -> None:
    err = rejected(summary=text)
    assert (err.field, err.reason) == ("summary", reason)
    # the reason never quotes the text back
    assert text not in str(err)


@pytest.mark.parametrize(
    "text",
    [
        "Jane from the agency reviewed the draft.",  # a name
        "The launch budget is €12,500 in total.",
        "Ads ran at $99.00 per day last week.",
        "The pipeline is now worth £1.2m this quarter.",
        "The starter plan stays at $49/month for now.",
        "We spent twelve thousand euros on the event.",
        "Review on 05/11/2026 10:30 with the team.",  # a date and a time
        "Merged 3f9a2c4e5b6d7f8091a2b3c4d5e6f708192a3b4c into main.",  # a commit hash
        "Target FY26 PLAN WITH MORE REACH in Q3 2026.",  # capitals in prose
        "Shipped the Q3-Go-To-Market-Plan-For-Benelux-2026 page copy.",  # a slug
        "Our secret: start a week earlier.",
    ],
)
def test_documented_misses_are_accepted(text: str) -> None:
    assert check(summary=text).summary == text


@pytest.mark.parametrize(
    ("overrides", "field", "reason"),
    [
        ({"topic": "ab"}, "topic", "must be at least 3 characters"),
        ({"topic": "x" * 81}, "topic", "must be at most 80 characters"),
        ({"topic": "   "}, "topic", "must be at least 3 characters"),
        ({"topic": ...}, "topic", "is required"),
        ({"summary": "Too short"}, "summary", "must be at least 10 characters"),
        ({"summary": "x" * 801}, "summary", "must be at most 800 characters"),
        ({"ask": "x" * 301}, "ask", "must be at most 300 characters"),
        ({"status": "great"}, "status", "must be one of on_track, at_risk, blocked, done, info"),
        ({"status": ...}, "status", "must be one of on_track, at_risk, blocked, done, info"),
        ({"topic": 42}, "topic", "must be a string"),
        ({"idempotency_key": "short"}, "idempotency_key", "must be 8-64 of A-Z, a-z, 0-9, _ and -"),
        ({"idempotency_key": "has space in it"}, "idempotency_key", "must be 8-64 of A-Z, a-z, 0-9, _ and -"),
        ({"source": "someone-else"}, "source", "is not a field of a progress update"),
    ],
)
def test_limits_and_types(overrides: dict[str, Any], field: str, reason: str) -> None:
    err = rejected(**overrides)
    assert (err.field, err.reason) == (field, reason)


def test_limits_apply_to_the_cleaned_form() -> None:
    # 80 visible characters padded with runs of spaces still fit
    assert len(check(topic="  " + "x" * 40 + "      " + "y" * 39 + "  ").topic) == 80


@pytest.mark.parametrize(
    ("link", "reason"),
    [
        ("http://docs.google.com/doc", "must be an https URL"),
        ("javascript:alert(1)", "must be an https URL"),
        ("https://evil.example/doc", "points at a host that is not allowed"),
        ("https://docs.google.com.evil.example/doc", "points at a host that is not allowed"),
        ("https://user:pw@docs.google.com/doc", "must not carry credentials or a port"),
        ("https://docs.google.com:8443/doc", "must not carry credentials or a port"),
        ("https://docs.google.com/a b", "must be a single URL"),
        ("https://docs.google.com/<x>", "must be a single URL"),
        ("https://docs.google.com:99999/doc", "is not a valid URL"),
    ],
)
def test_link_rules(link: str, reason: str) -> None:
    err = rejected(link=link)
    assert (err.field, err.reason) == ("link", reason)


def test_link_host_matches_without_case_and_needs_a_list() -> None:
    assert check(link="https://GitHub.com/org/repo/pull/1").link == "https://GitHub.com/org/repo/pull/1"
    with pytest.raises(pg.Rejected, match="not allowed"):
        pg.check({**GOOD}, link_hosts=[])


def test_link_hosts_default_to_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROGRESS_LINK_HOSTS", "docs.google.com, github.com")
    assert pg.check(dict(GOOD)).link == GOOD["link"]
    monkeypatch.setenv("PROGRESS_LINK_HOSTS", "")
    with pytest.raises(pg.Rejected, match="not allowed"):
        pg.check(dict(GOOD))
