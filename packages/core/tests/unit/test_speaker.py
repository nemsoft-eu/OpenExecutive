"""Who is asking on a chat turn, and whether only they read it
(orchestrator/speaker.py): one answer for every surface."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from openexecutive.memory import episodic
from openexecutive.orchestrator.session import Session
from openexecutive.orchestrator.speaker import own_private_speaker, speaker
from openexecutive.people import registry as people_registry
from openexecutive.people import store as people_store


@pytest.fixture(autouse=True)
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    path = tmp_path / "episodic_memory.db"
    monkeypatch.setattr(episodic, "DB_PATH", path)
    monkeypatch.setattr(people_store, "DB_PATH", path)
    episodic.initialize_db(path)
    people_store.initialize_db(path)
    people_registry.invalidate()
    yield
    people_registry.invalidate()


@pytest.fixture
def owner() -> int:
    pid = people_store.upsert_person(full_name="Olivia Owner", is_principal=True, slack_user_id="U_OWN")
    people_registry.invalidate()
    return pid


@pytest.fixture
def teammate() -> int:
    pid = people_store.upsert_person(full_name="Ben Teammate", email="ben@co.example")
    people_registry.invalidate()
    return pid


def _call(person_id: int) -> Session:
    return Session(session_id="call-1", origin_channel="meeting", origin_channel_ref="app:c1",
                   caller_person_id=person_id, private_chat=True, speaker_verified=True)


def test_the_same_person_on_every_private_surface(owner: int) -> None:
    surfaces = [
        Session(session_id="web-1", from_web_chat=True, caller_person_id=owner, web_caller_signed_in=True),
        Session(session_id="slack:dm:U_OWN", origin_channel="slack", origin_channel_ref="U_OWN",
                caller_person_id=owner),
        _call(owner),
    ]
    for session in surfaces:
        assert getattr(speaker(session), "id", None) == owner, session.session_id
        assert getattr(own_private_speaker(session), "id", None) == owner, session.session_id


def test_a_teammate_in_their_own_call_is_themselves(teammate: int) -> None:
    assert getattr(own_private_speaker(_call(teammate)), "id", None) == teammate


def test_verified_but_where_others_read_is_not_private(owner: int) -> None:
    channel = Session(session_id="slack:channel:C1:U_OWN", origin_channel="slack",
                      origin_channel_ref="U_OWN", caller_person_id=owner)
    shared = Session(session_id="web-1", from_web_chat=True, caller_person_id=owner, shared_chat=True,
                     web_caller_signed_in=True)
    for session in (channel, shared):
        assert speaker(session) is not None
        assert own_private_speaker(session) is None


def test_nobody_speaks_on_a_run_mail_or_an_unverified_call(owner: int) -> None:
    unverified = Session(session_id="call-1", origin_channel="meeting", origin_channel_ref="app:c1",
                         caller_person_id=owner, private_chat=True)
    for session in (
        Session(session_id="run-1", unattended=True, caller_person_id=owner),
        Session(session_id="brief", private_to_principal=True, caller_person_id=owner),
        unverified,
        None,
    ):
        assert own_private_speaker(session) is None


def test_a_web_or_ai_app_turn_without_a_sign_in_is_nobody(owner: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from openexecutive.utils import deployment

    monkeypatch.setattr(deployment, "is_local_login", lambda: False)
    for session in (
        Session(session_id="web-1", from_web_chat=True, caller_person_id=owner),
        Session(session_id="mcp-1", from_mcp_client=True, caller_person_id=owner),
    ):
        assert speaker(session) is None
    # A server people sign in to locally has no signed callers.
    monkeypatch.setattr(deployment, "is_local_login", lambda: True)
    assert getattr(speaker(Session(session_id="web-1", from_web_chat=True, caller_person_id=owner)), "id", None) == owner


def _mail(person_id: int, *, proven: bool, owner: bool = False, alone: bool = False) -> Session:
    proof = "email_authenticated" if owner else "email_sender_authenticated"
    return Session(
        session_id="email:t1", origin_channel="email", caller_person_id=person_id,
        email_from="olivia@co.example" if owner else "ben@co.example",
        email_one_on_one=alone, **{proof: proven},
    )


def test_proven_mail_from_a_teammate_is_them_asking(teammate: int) -> None:
    from openexecutive.delegation.settings import private_conversation, speaker_surface_ok

    alone = _mail(teammate, proven=True, alone=True)
    assert speaker(alone).id == teammate
    assert own_private_speaker(alone).id == teammate
    assert private_conversation(alone)
    assert speaker_surface_ok(alone, people_store.get_person(teammate))
    # Others on the mail: still them asking, but not their private conversation.
    with_others = _mail(teammate, proven=True)
    assert speaker(with_others).id == teammate
    assert own_private_speaker(with_others) is None
    # A From line nothing proved is nobody.
    unproven = _mail(teammate, proven=False, alone=True)
    assert speaker(unproven) is None
    assert not private_conversation(unproven)


def test_the_owners_proven_mail_to_the_executive_alone_is_their_own(owner: int) -> None:
    session = _mail(owner, proven=True, owner=True, alone=True)
    assert speaker(session).id == owner
    assert own_private_speaker(session).id == owner
