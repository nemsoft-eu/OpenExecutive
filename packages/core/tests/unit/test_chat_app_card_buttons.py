"""Approve and Skip buttons on cards in a Slack or Discord DM
(integrations/channel_cards.py, slack_bot.handle_card_tap,
discord_bot.handle_card_tap)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from openexecutive.integrations import channel_cards, discord_bot, slack_bot
from openexecutive.orchestrator.session import Session

PERSON = SimpleNamespace(id=1, email="olivia@co.example")


@pytest.fixture(autouse=True)
def chat_apps(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, Any, bool]]:

    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_CARD_BUTTONS", "true")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "discord-test")
    monkeypatch.setattr("openexecutive.audit.log_event", lambda *_a, **_k: None)
    resolved: list[tuple[int, Any, bool]] = []

    async def resolve(decision_id: int, person: Any, *, approve: bool, allow: bool = False, shown_at: str | None = None, shown: str | None = None) -> str:
        resolved.append((decision_id, person, approve))
        return "Approved." if approve else "Skipped."

    monkeypatch.setattr(channel_cards, "resolve", resolve)
    yield resolved


def _dm(channel: str, session_id: str, person_id: int | None = 1) -> Session:
    return Session(session_id=session_id, origin_channel=channel, origin_channel_ref="U1",
                   caller_person_id=person_id)


def test_slack_and_discord_dms_show_cards(monkeypatch: pytest.MonkeyPatch) -> None:

    assert channel_cards.shows_cards(_dm("slack", "slack:dm:U1"))
    assert channel_cards.shows_cards(_dm("discord", "discord:dm:42"))
    # A channel or thread is everyone in it; nobody on the roster proves nothing.
    assert not channel_cards.shows_cards(_dm("slack", "slack:C1:123.4"))
    assert not channel_cards.shows_cards(_dm("discord", "discord:thread:9"))
    assert not channel_cards.shows_cards(_dm("slack", "slack:dm:U1", person_id=None))
    # Slack needs Interactivity, so the buttons wait for SLACK_CARD_BUTTONS.
    monkeypatch.setenv("SLACK_CARD_BUTTONS", "false")
    assert not channel_cards.shows_cards(_dm("slack", "slack:dm:U1"))


def test_a_google_chat_dm_with_the_app_alone_shows_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the DM the adapter marks private (the app and its person alone,
    verified): a space or a two-person DM is read by others too."""
    monkeypatch.setenv("GOOGLE_CHAT_PROJECT_NUMBER", "123456789012")
    monkeypatch.setenv("GOOGLE_CHAT_SERVICE_ACCOUNT_FILE", "/fake/key.json")

    def gchat(*, private: bool, verified: bool = True, person_id: int | None = 1) -> Session:
        session = _dm("google_chat", "google_chat:spaces/DM1", person_id)
        session.speaker_verified = verified
        session.private_chat = private
        return session

    assert channel_cards.shows_cards(gchat(private=True))
    assert not channel_cards.shows_cards(gchat(private=False))
    assert not channel_cards.shows_cards(gchat(private=True, verified=False))
    assert not channel_cards.shows_cards(gchat(private=True, person_id=None))
    # Not set up: no webhook to take the tap.
    monkeypatch.delenv("GOOGLE_CHAT_PROJECT_NUMBER")
    assert not channel_cards.shows_cards(gchat(private=True))


class FakeSlack:
    def __init__(self, dm: str = "D1") -> None:
        self.dm = dm
        self.updates: list[dict[str, Any]] = []
        self.posts: list[dict[str, Any]] = []
        self.ephemeral: list[dict[str, Any]] = []

    async def conversations_open(self, users: str) -> dict[str, Any]:
        return {"channel": {"id": self.dm}}

    async def chat_update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    async def chat_postMessage(self, **kwargs: Any) -> None:  # noqa: N802 - Slack's name
        self.posts.append(kwargs)

    async def chat_postEphemeral(self, **kwargs: Any) -> None:  # noqa: N802 - Slack's name
        self.ephemeral.append(kwargs)


def _slack_tap(channel: str = "D1", value: str = "card:7:a:0000beef") -> dict[str, Any]:
    return {
        "user": {"id": "U1"}, "channel": {"id": channel},
        "container": {"message_ts": "111.2"},
        "message": {"ts": "111.2", "blocks": slack_bot._card_blocks("Change the workflow", 7)},
        "actions": [{"action_id": "oe_card_approve", "value": value}],
    }


def test_a_slack_tap_in_its_own_dm_decides_the_card(
    chat_apps: list[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("openexecutive.people.store.find_person_by_slack_id", lambda uid: PERSON)
    # The card as its message shows it (no ledger in this test).
    monkeypatch.setattr(channel_cards, "card_text", lambda decision_id: "The card")
    monkeypatch.setattr(channel_cards, "version", lambda decision_id, text: "0000beef")
    client = FakeSlack()
    asyncio.run(slack_bot.handle_card_tap(_slack_tap(), client))
    assert chat_apps == [(7, PERSON, True)]
    update = client.updates[0]
    assert update["ts"] == "111.2" and "Approved." in update["text"]
    assert [b["type"] for b in update["blocks"]] == ["section"]  # no buttons left


def test_a_slack_tap_anywhere_else_is_ignored(chat_apps: list[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("openexecutive.people.store.find_person_by_slack_id", lambda uid: PERSON)
    # A channel, or a DM that isn't this user's with the bot.
    asyncio.run(slack_bot.handle_card_tap(_slack_tap(channel="C1"), FakeSlack()))
    elsewhere = FakeSlack(dm="D1")
    asyncio.run(slack_bot.handle_card_tap(_slack_tap(channel="D9"), elsewhere))
    assert elsewhere.ephemeral[0]["text"] == "This card is no longer there." and elsewhere.updates == []
    asyncio.run(slack_bot.handle_card_tap(_slack_tap(value="junk"), FakeSlack()))
    monkeypatch.setattr("openexecutive.people.store.find_person_by_slack_id", lambda uid: None)
    asyncio.run(slack_bot.handle_card_tap(_slack_tap(), FakeSlack()))
    assert chat_apps == []


def test_slack_cards_go_out_as_plain_text_with_buttons(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(channel_cards, "card_text", lambda decision_id: f"*card* {decision_id}")
    # The ledger isn't set up in this module; the labels have their own tests.
    monkeypatch.setattr(channel_cards, "card_options", lambda decision_id, shown: (("Approve", f"card:{decision_id}:a:0000beef"), ("Skip", f"card:{decision_id}:s:0000beef")))
    client = FakeSlack()
    asyncio.run(slack_bot._send_cards(client, "D1", [3]))
    blocks = client.posts[0]["blocks"]
    assert blocks[0]["text"]["type"] == "plain_text"
    assert [e["value"] for e in blocks[1]["elements"]] == ["card:3:a:0000beef", "card:3:s:0000beef"]


def test_discord_cards_go_out_literally() -> None:
    # The fence goes on at display (send_card, send_dm_card, a re-shown
    # card), never before card_options: the buttons version the card's words.
    text = "x" * (channel_cards.CARD_MAX - 40) + "||``` [bank.com](https://evil.example)"
    shown = discord_bot._card_content(text)
    assert shown.startswith("```\n") and shown.endswith("\n```") and shown.count("```") == 2


class FakeResponse:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.deferred = False

    async def send_message(self, text: str, **_: Any) -> None:
        self.sent.append(text)

    async def defer(self) -> None:
        self.deferred = True


def _discord_tap(dm: bool = True, recipient: str = "42", custom_id: str = "card:7:s:0000beef",
                 partial: Any = None, guild_id: int | None = None) -> Any:
    import discord

    edits: list[dict[str, Any]] = []
    channel: Any
    if partial is not None:
        channel = SimpleNamespace(type=partial)  # not cached: no recipient
    elif dm:
        channel = discord.DMChannel.__new__(discord.DMChannel)
        channel.recipients = [SimpleNamespace(id=int(recipient))]
    else:
        channel = SimpleNamespace()

    async def edit_original_response(**kwargs: Any) -> None:
        edits.append(kwargs)

    return SimpleNamespace(
        data={"custom_id": custom_id}, user=SimpleNamespace(id=42), channel=channel, guild_id=guild_id,
        message=SimpleNamespace(content="Change the workflow"), response=FakeResponse(),
        edit_original_response=edit_original_response, edits=edits,
    )


def test_a_discord_tap_in_its_own_dm_decides_the_card(
    chat_apps: list[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("openexecutive.people.store.find_person_by_discord_id", lambda uid: PERSON)
    tap = _discord_tap()
    asyncio.run(discord_bot.handle_card_tap(tap))
    assert chat_apps == [(7, PERSON, False)]
    assert tap.response.deferred and tap.edits[0]["view"] is None
    assert "Skipped." in tap.edits[0]["content"]


def test_a_discord_tap_still_counts_after_a_restart(
    chat_apps: list[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    import discord

    monkeypatch.setattr("openexecutive.people.store.find_person_by_discord_id", lambda uid: PERSON)
    asyncio.run(discord_bot.handle_card_tap(_discord_tap(partial=discord.ChannelType.private)))
    assert chat_apps == [(7, PERSON, False)]
    for tap in (_discord_tap(partial=discord.ChannelType.group),
                _discord_tap(partial=discord.ChannelType.private, guild_id=5)):
        asyncio.run(discord_bot.handle_card_tap(tap))
        assert tap.response.sent == ["This card is no longer there."]
    assert len(chat_apps) == 1


def test_a_discord_tap_anywhere_else_is_turned_away(
    chat_apps: list[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("openexecutive.people.store.find_person_by_discord_id", lambda uid: PERSON)
    for tap in (_discord_tap(dm=False), _discord_tap(recipient="99")):
        asyncio.run(discord_bot.handle_card_tap(tap))
        assert tap.response.sent == ["This card is no longer there."]
    other = _discord_tap(custom_id="something_else")
    asyncio.run(discord_bot.handle_card_tap(other))
    assert other.response.sent == ["This card is no longer there."] and chat_apps == []


_AGAIN = channel_cards.Tapped(
    "Changed who it goes from. Check it, then send it.", "Reply from Ada",
    (("Send from Ada, cc me", "card:7:a:1234abcd"), ("Send from me", "card:7:x:1234abcd"),
     ("Skip", "card:7:s:1234abcd")),
)


async def _again(tapped: Any, who: Any) -> channel_cards.Tapped:
    return _AGAIN


def test_a_slack_card_that_still_waits_is_shown_again_with_its_new_buttons(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("openexecutive.people.store.find_person_by_slack_id", lambda uid: PERSON)
    monkeypatch.setattr(channel_cards, "tap", _again)
    client = FakeSlack()
    asyncio.run(slack_bot.handle_card_tap(_slack_tap(value="card:7:x:0000beef"), client))
    update = client.updates[0]
    assert update["text"] == f"Reply from Ada\n\n{_AGAIN.line}"
    elements = update["blocks"][1]["elements"]
    assert [e["value"] for e in elements] == [data for _, data in _AGAIN.options]
    assert [e["action_id"] for e in elements] == ["oe_card_approve", "oe_card_switch", "oe_card_skip"]


def test_a_discord_card_that_still_waits_is_shown_again_with_its_new_buttons(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("openexecutive.people.store.find_person_by_discord_id", lambda uid: PERSON)
    monkeypatch.setattr(channel_cards, "tap", _again)
    tap = _discord_tap(custom_id="card:7:x:0000beef")
    asyncio.run(discord_bot.handle_card_tap(tap))
    edit = tap.edits[0]
    assert edit["content"] == f"```\nReply from Ada\n```\n\n{_AGAIN.line}"
    assert [b.custom_id for b in edit["view"].children] == [data for _, data in _AGAIN.options]
    assert [b.label for b in edit["view"].children] == [label for label, _ in _AGAIN.options]


def test_a_long_discord_card_gets_buttons_for_all_of_its_words(monkeypatch: pytest.MonkeyPatch) -> None:
    long = "x" * (discord_bot._MAX_CARD_TEXT + 50)
    monkeypatch.setattr(channel_cards, "card_text", lambda decision_id: long)
    sent: list[str] = []

    async def send_card(decision_id: int, text: str) -> None:
        sent.append(text)

    asyncio.run(discord_bot._send_cards(send_card, [3]))
    # The buttons carry the version of the whole card, as a tap checks it;
    # only the message cuts it short.
    assert sent == [long]


def _card_message(user: str = "UBOT", value: str = "card:7:a:0000beef") -> dict[str, Any]:
    return {"ts": "111.2", "user": user, "bot_id": "B1",
            "blocks": slack_bot._card_blocks("Reply to Ada", 7, (("Send from me", value), ("Skip", "card:7:s:0000beef")))}


_SAID = channel_cards.Tapped(
    "I made it shorter.", "Reply to Ada\nShort", (("Send from me", "card:7:a:1111beef"), ("Skip", "card:7:s:1111beef")),
)


@pytest.fixture
def slack_say(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    monkeypatch.setattr(slack_bot, "_bot_user_id", "UBOT")
    monkeypatch.setattr("openexecutive.people.store.find_person_by_slack_id", lambda uid: PERSON)
    monkeypatch.setattr(channel_cards, "takes_words", lambda decision_id: True)
    said: list[Any] = []

    async def say(card: Any, who: Any, text: str) -> channel_cards.Tapped | None:
        said.append((card, who, text))
        return _SAID

    monkeypatch.setattr(channel_cards, "say", say)
    return said


def _slack_reply(event_said: list[dict[str, Any]]) -> Any:
    async def reply(**kwargs: Any) -> None:
        event_said.append(kwargs)

    return reply


_REPLY = {"user": "U1", "channel": "D1", "ts": "112.0", "thread_ts": "111.2"}


def test_a_reply_to_a_slack_card_in_its_own_dm_is_said_on_it(slack_say: list[Any]) -> None:
    client, answers = FakeSlack(), []
    assert asyncio.run(slack_bot._say_on_card(_REPLY, "make it shorter", _card_message(), _slack_reply(answers), client))
    assert slack_say == [(channel_cards.Choice(7, "a", "0000beef"), PERSON, "make it shorter")]
    # The card as it reads now, with its new buttons and how to say more.
    update = client.updates[0]
    assert update["ts"] == "111.2" and update["text"] == "Reply to Ada\nShort"
    assert [e["value"] for e in update["blocks"][1]["elements"]] == ["card:7:a:1111beef", "card:7:s:1111beef"]
    assert update["blocks"][2]["elements"][0]["text"] == channel_cards.TELL_ME
    # What the Executive said back goes in the thread.
    assert answers == [{"text": "I made it shorter.", "thread_ts": "111.2"}]


def test_a_go_said_on_a_slack_card_leaves_it_with_its_outcome(
    slack_say: list[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def say(card: Any, who: Any, text: str) -> channel_cards.Tapped:
        return channel_cards.Tapped("Done. I sent it.", settled=True)

    monkeypatch.setattr(channel_cards, "say", say)
    client, answers = FakeSlack(), []
    assert asyncio.run(slack_bot._say_on_card(_REPLY, "send it", _card_message(), _slack_reply(answers), client))
    update = client.updates[0]
    assert update["text"] == "Reply to Ada\n\nDone. I sent it."
    assert [b["type"] for b in update["blocks"]] == ["section"]
    assert answers[0]["text"] == "Done. I sent it."


def test_a_reply_that_isnt_on_a_slack_card_of_its_own_dm_is_an_ordinary_message(
    slack_say: list[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers: list[dict[str, Any]] = []
    # Someone else's message, one with no card's buttons, or no parent read.
    for parent in (_card_message(user="U2"), {**_card_message(), "blocks": slack_bot._card_blocks("Hi")}, None):
        assert not asyncio.run(slack_bot._say_on_card(_REPLY, "hi", parent, _slack_reply(answers), FakeSlack()))
    # A DM that isn't this person's with the bot.
    assert not asyncio.run(slack_bot._say_on_card(_REPLY, "hi", _card_message(), _slack_reply(answers), FakeSlack(dm="D9")))
    assert slack_say == [] and answers == []
    # A card that takes no words, or isn't theirs, leaves them to the turn.
    async def no(card: Any, who: Any, text: str) -> None:
        return None

    monkeypatch.setattr(channel_cards, "say", no)
    client = FakeSlack()
    assert not asyncio.run(slack_bot._say_on_card(_REPLY, "hi", _card_message(), _slack_reply(answers), client))
    assert client.updates == [] and answers == []
