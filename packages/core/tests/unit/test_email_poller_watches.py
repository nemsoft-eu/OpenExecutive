"""Every email that reaches the Executive is checked against the email
watches (monitoring.mailbox.note_executive_mail), even one that gets no
turn: "tell me every time a sign-up email comes in" is mostly automated
mail."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import openexecutive.integrations.email_poller as poller
from openexecutive.integrations.mail_sources import InboundMail
from openexecutive.integrations.workspace.google import GoogleMail
from openexecutive.integrations.workspace.mail import InboundMessage


def _noting(noted: list[dict[str, Any]], result: int) -> Any:
    async def note(**kw: Any) -> int:
        header_id = kw.pop("header_id", None)
        kw_header = await header_id() if header_id is not None else None
        noted.append({**kw, "header": kw_header})
        return result

    return note


async def _no_send(_to: str, _body: str, _in_thread: bool) -> str | None:
    return None


def test_automated_mail_gets_no_turn_but_is_checked_against_the_watches(monkeypatch: Any) -> None:
    noted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "openexecutive.monitoring.mailbox.note_executive_mail", _noting(noted, 1),
    )
    run = AsyncMock(side_effect=AssertionError("a turn ran"))
    monkeypatch.setattr(poller, "_run_executive", run)
    mail = InboundMail(
        message=InboundMessage(
            message_id="18c0ffee00000002", thread_id="th9", from_addr="noreply@app.example",
            from_name="App", subject="New sign up: Acme Dental",
            raw_text="From: App <noreply@app.example>\nMessage-ID: <su-9@app.example>\n\nMessage-ID: <forged@x>",
        ),
        source="mailbox",
        own_address="exec@example.com",
        send_answer=_no_send,
    )
    outcome = asyncio.run(poller._handle_mail(mail, MagicMock(), mailbox=GoogleMail(), done=AsyncMock()))
    assert outcome == poller._SKIPPED
    run.assert_not_awaited()
    assert noted == [{
        "message_id": "18c0ffee00000002", "thread_id": "th9", "from_addr": "noreply@app.example",
        "from_name": "App", "subject": "New sign up: Acme Dental", "private": True, "mail_of": None,
        "proven": None, "header": "<su-9@app.example>",
    }]


def test_a_name_that_is_only_the_address_is_no_name(monkeypatch: Any) -> None:
    noted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "openexecutive.monitoring.mailbox.note_executive_mail", _noting(noted, 0),
    )
    msg = InboundMessage(message_id="m1", thread_id="", from_addr="robin@hale.example",
                         from_name="Robin@Hale.example", subject="Rent")
    asyncio.run(poller._note_watches(msg, private=True, mail_of=None, proven=None))
    assert noted[0]["from_name"] == "" and noted[0]["private"] is True and noted[0]["header"] is None


def test_the_message_id_is_read_from_the_header_lines_never_the_body() -> None:
    assert poller._message_id_in("From: a@b.example\nMessage-Id:  <m-1@b.example>\n\nhi") == "<m-1@b.example>"
    assert poller._message_id_in("From: a@b.example\n\nMessage-ID: <forged@x>") == ""


def _msg(raw: str | None = None) -> InboundMessage:
    return InboundMessage(message_id="m7", thread_id="t7", from_addr="robin@hale.example",
                          from_name="Robin Hale", subject="Rent", raw_text=raw)


def _provider(monkeypatch: Any, name: str) -> None:
    monkeypatch.setattr(
        "openexecutive.integrations.workspace.registry.get_mail_provider", lambda: SimpleNamespace(name=name),
    )


def test_without_it_in_the_text_gmail_reads_the_message_id_from_the_raw_message(monkeypatch: Any) -> None:
    from openexecutive.integrations import fact_confirmation

    _provider(monkeypatch, "google")
    asked: list[str] = []

    async def read_raw(gateway: Any, message_id: str) -> str:
        asked.append(message_id)
        return "Received: by 10.0.0.1\r\nMessage-ID: <rent-1@hale.example>\r\nFrom: Robin\r\n\r\nMessage-ID: <body@x>"

    monkeypatch.setattr(fact_confirmation, "read_raw", read_raw)
    read = poller._header_id_reader(object(), _msg("From: Robin Hale <robin@hale.example>\nSubject: Rent"), in_mailbox=True)
    assert asyncio.run(read()) == "<rent-1@hale.example>" and asked == ["m7"]


def test_on_outlook_the_message_id_comes_from_its_headers(monkeypatch: Any) -> None:
    from email.message import Message

    from openexecutive.integrations import fact_confirmation

    _provider(monkeypatch, "microsoft")
    headers = Message()
    headers["Message-ID"] = " <rent-1@hale.example> "

    async def read_headers(gateway: Any, message_id: str) -> Message:
        return headers

    monkeypatch.setattr(fact_confirmation, "read_outlook_headers", read_headers)
    assert asyncio.run(poller._header_id_reader(object(), _msg(), in_mailbox=True)()) == "<rent-1@hale.example>"


def test_mail_not_in_the_executives_mailbox_is_never_fetched_for_it(monkeypatch: Any) -> None:
    from openexecutive.integrations import fact_confirmation

    async def never(*_: Any) -> str:
        raise AssertionError("fetched")

    monkeypatch.setattr(fact_confirmation, "read_raw", never)
    monkeypatch.setattr(fact_confirmation, "read_outlook_headers", never)
    assert asyncio.run(poller._header_id_reader(object(), _msg(), in_mailbox=False)()) == ""
    assert asyncio.run(poller._header_id_reader(None, _msg(), in_mailbox=True)()) == ""
