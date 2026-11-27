"""A forwarded email is read, not dropped as quoted history.

Apple Mail and iPhone forward with "Begin forwarded message:" and quote the
whole forwarded email with ">" below it, so ``sender_new_text`` (the
sender's own words) left nothing of it: the Executive opened a forward and
saw only its logo placeholders. ``readable_text`` keeps the forward, unquoted,
under "Forwarded message:", while reply history stays out.
"""
from __future__ import annotations

from types import SimpleNamespace

from openexecutive.integrations.email_poller import (
    FORWARDED_HEADING,
    _email_memory_text,
    readable_text,
    sender_new_text,
)

APPLE_FORWARD = (
    "\r\n\r\n> Begin forwarded message:\r\n> \r\n"
    "> From: \"Power Co\" <letters@power.example>\r\n"
    "> Subject: Tree trimming easement - Parcel 12\r\n"
    "> Date: November 3, 2026\r\n> \r\n"
    "> Dear owner,\r\n> The easement covers 640 feet along Mill Road.\r\n"
    "> ￼\r\n> Call Dana Ruiz at 555-0100.\r\n"
)


def test_an_apple_mail_forward_keeps_the_forwarded_email() -> None:
    text = readable_text(APPLE_FORWARD, "Fwd: Tree trimming easement")
    assert text.startswith(FORWARDED_HEADING)
    assert "The easement covers 640 feet along Mill Road." in text
    assert "Call Dana Ruiz at 555-0100." in text
    assert "From: \"Power Co\" <letters@power.example>" in text
    # Its own ">" quoting is taken off; the marker line isn't repeated.
    assert "\n>" not in text and "Begin forwarded message" not in text


def test_the_senders_note_comes_before_the_forward() -> None:
    body = "Can you answer this one?\n\n" + APPLE_FORWARD
    text = readable_text(body)
    assert text.startswith("Can you answer this one?\n\nForwarded message:\n")
    # Their own words alone are unchanged.
    assert sender_new_text(body) == "Can you answer this one?"


def test_a_gmail_forward_keeps_the_forwarded_email() -> None:
    body = "FYI.\n\n---------- Forwarded message ---------\nFrom: Bank\nDate: Mon\n\nRate is 5%"
    assert readable_text(body) == "FYI.\n\nForwarded message:\nFrom: Bank\nDate: Mon\n\nRate is 5%"


def test_an_outlook_forward_keeps_the_forwarded_email_by_its_subject() -> None:
    body = "See below.\n\n________________________________\nFrom: Bank\nSent: Monday\n\nRate is 5%"
    assert readable_text(body, "FW: Rates").endswith("Forwarded message:\nFrom: Bank\nSent: Monday\n\nRate is 5%")
    # The same block under a reply is history, still left out.
    assert readable_text(body, "RE: Rates") == "See below."


def test_reply_history_is_still_left_out() -> None:
    body = "Sounds good.\n\nOn Mon, Dana <d@example.com> wrote:\n> Shall we meet Friday?\n> Begin forwarded message:\n> old"
    assert readable_text(body, "Re: Friday") == "Sounds good."


def test_each_part_has_its_own_cap() -> None:
    body = "x" * 50 + "\n\n---------- Forwarded message ---------\n" + "y" * 500
    text = readable_text(body, max_chars=10, forwarded_chars=100)
    assert text == "x" * 10 + "\n\nForwarded message:\n" + "y" * 100


def test_peer_memory_still_notes_an_apple_forward_without_its_text() -> None:
    raw = "Message ID: 1\nSubject: Fwd: Tree trimming\nFrom: a@example.com\n\n--- BODY ---\n" + APPLE_FORWARD
    assert "640 feet" not in _email_memory_text(raw)
    assert "[Forwarded an earlier message]" in _email_memory_text(raw)


def test_read_my_email_shows_a_forwarded_email() -> None:
    from openexecutive.orchestrator.mail_read_tools import _render_thread

    message = SimpleNamespace(
        text=APPLE_FORWARD, subject="Fwd: Tree trimming easement", date="Tue", received_at="",
        labels=["INBOX"], from_addr="ann@example.com", from_name="Ann", to=["me@example.com"], cc=[],
    )
    rendered = _render_thread([message], 1, "me@example.com")
    assert "The easement covers 640 feet along Mill Road." in rendered


def test_the_inbox_drafter_sees_a_forwarded_email() -> None:
    from openexecutive.delegation.inbox_classifier import render_email

    message = SimpleNamespace(
        text=APPLE_FORWARD, subject="Fwd: Tree trimming easement", from_name="Ann",
        from_addr="ann@example.com", sender_authenticated=True, to=["me@example.com"], cc=[],
        attachments=[], has_attachments=False,
    )
    assert "640 feet along Mill Road" in render_email(message, relation="teammate")


def test_a_forward_they_sent_stays_out_of_their_own_words() -> None:
    from openexecutive.orchestrator.mail_read_tools import _render_thread

    message = SimpleNamespace(
        text="Please look.\n\n" + APPLE_FORWARD, subject="Fwd: Tree trimming easement", date="Tue",
        received_at="", labels=["SENT"], from_addr="me@example.com", from_name="Me", to=["x@example.com"], cc=[],
    )
    rendered = _render_thread([message], 1, "me@example.com")
    yours, _, rest = rendered.partition("<untrusted_content")
    assert "Yours" in yours and "Please look." in yours
    assert "640 feet" not in yours
    assert "640 feet" in rest


def test_a_note_below_an_apple_forward_stays_the_senders() -> None:
    from openexecutive.integrations.email_poller import split_forward

    body = "> Begin forwarded message:\n> From: A\n> body\n\nmy note below"
    own, forwarded = split_forward(body, "Fwd: x")
    assert own == "my note below"
    assert forwarded == "From: A\nbody"
    # Own words alone (voice learning, emailed answers) keep it too.
    assert sender_new_text(body) == "my note below"


def test_the_principals_unquoted_iphone_forward_is_not_their_own_words() -> None:
    from openexecutive.integrations.email_poller import _forwarded, _proven_email_text

    raw = (
        "Message ID: 1\nSubject: Fwd: Invoice\nFrom: owner@example.com\n\n--- BODY ---\n"
        "Look\n\nBegin forwarded message:\n\nFrom: A\n\nPay invoice"
    )
    assert _forwarded(raw)
    text = _proven_email_text(raw, "owner@example.com")
    own, _, rest = text.partition("<untrusted_content")
    assert "Look" in own and "Pay invoice" not in own
    assert "Pay invoice" in rest


def test_a_forward_with_its_marker_in_another_language_is_read() -> None:
    from openexecutive.integrations.email_poller import split_forward

    body = "Bitte ansehen.\n\n> Anfang der weitergeleiteten Nachricht:\n>\n> Von: A\n> Rechnung 42"
    own, forwarded = split_forward(body, "WG: Rechnung")
    assert own == "Bitte ansehen."
    assert "Rechnung 42" in forwarded and ">" not in forwarded
    # A reply's quote is still history, whatever the subject.
    assert split_forward("Ja.\n\nOn Mon, A <a@example.com> wrote:\n> old", "WG: Rechnung") == ("Ja.", "")


def test_max_chars_alone_caps_the_whole_text() -> None:
    body = "x" * 50 + "\n\n---------- Forwarded message ---------\n" + "y" * 500
    assert len(readable_text(body, max_chars=100)) == 100


def test_a_note_below_a_forward_in_another_language_shows_once() -> None:
    from openexecutive.integrations.email_poller import split_forward

    body = "FYI see below\n\n> Anfang der weitergeleiteten Nachricht:\n> Von: X\n> hello there\n\nThanks, Bob"
    own, forwarded = split_forward(body, "WG: report")
    assert own == "FYI see below\n\nThanks, Bob"
    assert forwarded == "Anfang der weitergeleiteten Nachricht:\nVon: X\nhello there"


def test_a_lead_in_with_a_date_or_address_keeps_the_forward() -> None:
    quote = "> Begin forwarded message:\n>\n> From: Bank\n> Rate is 5%"
    for lead in (
        "Can you handle this before the Nov 3 meeting:",
        "See below from dana@x.com:",
        "Can you handle this before our 10:30 call:",
        "Forwarding this from Dana <d@x.com>:",
        "Please review by 11/20/2026:",
    ):
        text = readable_text(f"{lead}\n\n{quote}", "Fwd: Rates")
        assert text == f"{lead}\n\nForwarded message:\nFrom: Bank\nRate is 5%"


def test_a_lead_in_above_a_forward_in_another_language_keeps_it() -> None:
    from openexecutive.integrations.email_poller import split_forward

    quote = "> Anfang der weitergeleiteten Nachricht:\n>\n> Von: Bank\n> Rate is 5%"
    for lead in ("Please review by 11/20/2026:", "Bitte bis 10:30 ansehen:", "Von Dana <d@x.com>:"):
        own, forwarded = split_forward(f"{lead}\n\n{quote}", "WG: Rates")
        assert own == lead
        assert forwarded.endswith("Von: Bank\nRate is 5%")


def test_the_principals_quoted_apple_forward_counts_as_forwarded() -> None:
    from openexecutive.integrations.email_poller import _forwarded, _proven_email_text

    raw = (
        "Message ID: 1\nSubject: Fwd: Tree trimming easement\nFrom: owner@example.com\n\n--- BODY ---\n"
        "Can you handle this?\n" + APPLE_FORWARD
    )
    assert _forwarded(raw)
    own, _, rest = _proven_email_text(raw, "owner@example.com").partition("<untrusted_content")
    assert "Can you handle this?" in own and "640 feet" not in own
    assert "640 feet" in rest
    # A plain reply with quoted history is not a forward.
    reply = "Message ID: 2\nSubject: Re: x\nFrom: owner@example.com\n\n--- BODY ---\nOK\n\nOn Mon, A <a@example.com> wrote:\n> hi"
    assert not _forwarded(reply)


def test_a_lead_in_that_says_wrote_keeps_the_forward() -> None:
    body = (
        "Look at what the county wrote:\n\n> Begin forwarded message:\n>\n"
        "> From: County <c@x>\n> Subject: Permit\n>\n> Your permit is denied.\n"
    )
    text = readable_text(body, "Fwd: Permit")
    assert text.startswith("Look at what the county wrote:\n\nForwarded message:\n")
    assert "Your permit is denied." in text
    assert sender_new_text(body) == "Look at what the county wrote:"
    # Even when it names a date, a Begin forwarded marker right below wins.
    dated = body.replace("county wrote:", "county wrote on 11/3:")
    assert "Your permit is denied." in readable_text(dated, "Fwd: Permit")


def test_a_quote_under_a_reply_in_another_language_is_history() -> None:
    from openexecutive.integrations.email_poller import _new_text_lines, split_forward

    body = "Merci.\n\nLe lun. 3 nov. 2026 \u00e0 10:00, Dana <d@x> a \u00e9crit :\n> old"
    own, forwarded = split_forward(body, "RE: x")
    assert forwarded == "" and "old" not in own
    # A forward marker further down a quote is that history's, in any language.
    body = body.replace("> old", "> Voici\n> Begin forwarded message:\n> From: Bank\n> secret old")
    own, forwarded = split_forward(body, "Re: x")
    assert forwarded == "" and "secret old" not in own
    assert _new_text_lines(body.splitlines())[1] is False


def test_lead_ins_keep_the_forward_and_stay_the_senders() -> None:
    from openexecutive.integrations.email_poller import split_forward

    for body, subject in (
        ("Schau, was Dana am 3.11. schrieb:\n\n> Anfang der weitergeleiteten Nachricht:\n>\n> Von: Bank\n> Rate 5%", "WG: Rate"),
        ("See what Dana wrote on 11/3:\n> D\u00e9but du message r\u00e9exp\u00e9di\u00e9 :\n> De: Bank\n> Rate 5%", "TR: Rate"),
    ):
        own, forwarded = split_forward(body, subject)
        assert own == body.split("\n")[0]
        assert forwarded.endswith("Bank\nRate 5%")
    assert sender_new_text("Look at what Dana wrote on 11/3:\n> the quote\nThoughts?") == (
        "Look at what Dana wrote on 11/3:\nThoughts?"
    )
    assert sender_new_text("As Dana <d@x> wrote yesterday:\n> we ship Friday\nCan we?") == (
        "As Dana <d@x> wrote yesterday:\nCan we?"
    )


def test_a_reply_whose_quote_opens_with_a_forward_marker_is_history() -> None:
    from openexecutive.integrations.email_poller import _forwarded, split_forward

    body = (
        "Gracias.\n\nEl lun, 3 nov 2026 a las 10:00, Dana <d@x> escribi\u00f3:\n"
        "> Begin forwarded message:\n> From: Bank\n> old text"
    )
    own, forwarded = split_forward(body, "Re: Rates")
    assert forwarded == "" and "old text" not in own
    raw = "Message ID: 1\nSubject: AW: Rates\nFrom: owner@example.com\n\n--- BODY ---\n" + body
    assert not _forwarded(raw)
    # The same quote in a forward is the forwarded email.
    assert split_forward(body, "Fwd: Rates")[1] == "From: Bank\nold text"


def test_an_inline_quote_above_a_forward_is_not_the_forward() -> None:
    from openexecutive.integrations.email_poller import split_forward

    body = (
        "Look at what he said:\n> we will ship Friday\nThoughts?\n\n"
        "> Anfang der weitergeleiteten Nachricht:\n> Von: Bank\n> Rate 5%"
    )
    own, forwarded = split_forward(body, "WG: status")
    assert own == "Look at what he said:\nThoughts?"
    assert forwarded.endswith("Von: Bank\nRate 5%") and "Friday" not in forwarded


def test_the_drafting_thread_shares_one_budget_for_forwards() -> None:
    from openexecutive.delegation import threads

    def msg(text: str) -> SimpleNamespace:
        return SimpleNamespace(
            text=text, subject="Fwd: x", labels=["INBOX"], from_name="Ann", from_addr="a@example.com", date="Tue",
        )

    old = msg("> Begin forwarded message:\n> " + "o" * 5000)
    new = msg("> Begin forwarded message:\n> " + "n" * 5000)
    text = threads.thread_text(SimpleNamespace(messages=[old, new]), "me@example.com")
    assert text.count("n") >= 5000
    assert text.count("o") <= threads.THREAD_FORWARDED_CHARS - 5000 + 50


def test_an_unquoted_marker_over_a_quoted_forward_is_unquoted() -> None:
    from openexecutive.integrations.email_poller import split_forward

    body = "FYI\n\nBegin forwarded message:\n\n> From: Bob\n> Subject: hi\n> \n> secret plan\n"
    assert split_forward(body, "Fwd: hi") == ("FYI", "From: Bob\nSubject: hi\n\nsecret plan")


def test_a_note_below_an_iphone_forward_stays_the_senders() -> None:
    from openexecutive.integrations.email_poller import split_forward

    body = (
        "See below.\n\nBegin forwarded message:\n\n> From: Dana\n> Subject: Q3\n>\n"
        "> Please pay invoice 42.\n\nAlso: call me about this tomorrow."
    )
    own, forwarded = split_forward(body, "Fwd: Q3")
    assert own == "See below.\n\nAlso: call me about this tomorrow."
    assert forwarded == "From: Dana\nSubject: Q3\n\nPlease pay invoice 42."
    words = sender_new_text(body)
    assert "call me about this tomorrow" in words and "invoice" not in words


def test_a_second_forward_below_a_quoted_one_is_kept() -> None:
    from openexecutive.integrations.email_poller import split_forward

    body = (
        "Note\n\n> Begin forwarded message:\n> From: A\n> hi\n\n"
        "---------- Forwarded message ---------\nFrom: B\nbody B"
    )
    own, forwarded = split_forward(body, "Fwd: x")
    assert own == "Note"
    assert forwarded.startswith("From: A\nhi") and forwarded.endswith("From: B\nbody B")

