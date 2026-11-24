from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from openexecutive.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


_GOOGLE_CHAT_ISSUER = "chat@system.gserviceaccount.com"
_GOOGLE_CHAT_CERTS_URL = f"https://www.googleapis.com/service_accounts/v1/metadata/x509/{_GOOGLE_CHAT_ISSUER}"


def verify_google_jwt(token: str, project_number: str) -> Mapping[str, Any]:
    """Verify a Bearer JWT from a Google Chat webhook.

    Google Chat sets the JWT audience to the GCP project number (numeric string)
    and iss to chat@system.gserviceaccount.com.
    Raises google.auth.exceptions.TransportError or ValueError on failure.
    """
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    # Chat signs with its own service account's keys, not Google's OAuth
    # ones, which ``verify_token`` checks by default.
    claims = id_token.verify_token(
        token, google_requests.Request(), audience=project_number, certs_url=_GOOGLE_CHAT_CERTS_URL
    )
    if claims.get("iss") != _GOOGLE_CHAT_ISSUER:
        raise ValueError(f"Unexpected issuer: {claims.get('iss')!r}")
    return claims


_CHAT_SCOPES = ["https://www.googleapis.com/auth/chat.bot"]


def _build_credentials(
    service_account_file: str | None,
    service_account_email: str | None,
) -> Any:
    """Return Google credentials for the Chat REST API.

    Auth method is selected by which env vars are set:

    1. Key file  — GOOGLE_CHAT_SERVICE_ACCOUNT_FILE is set.
       Standard service-account JSON key.  Blocked by the org policy
       ``iam.disableServiceAccountKeyCreation``.

    2. Impersonation — GOOGLE_CHAT_SERVICE_ACCOUNT_EMAIL is set (no key file).
       Uses Application Default Credentials (ADC) to impersonate the service
       account.  Works when the org policy blocks key creation.
       Local setup: ``gcloud auth application-default login``, then grant your
       user account ``roles/iam.serviceAccountTokenCreator`` on the SA.

    3. ADC direct — neither var is set.
       Uses the ambient credential (metadata server on Cloud Run / GCE / GKE,
       or whatever ``gcloud auth application-default login`` produced locally).
       The executing identity must already be the Chat bot SA or have chat.bot
       scope.
    """
    if service_account_file:
        from google.oauth2 import service_account as sa_module

        return sa_module.Credentials.from_service_account_file(
            service_account_file, scopes=_CHAT_SCOPES
        )

    import google.auth

    if service_account_email:
        # Impersonation requires the source credential to call the IAM Credentials
        # API, which needs cloud-platform scope — NOT chat.bot scope.
        from google.auth import impersonated_credentials

        base_creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return impersonated_credentials.Credentials(
            source_credentials=base_creds,
            target_principal=service_account_email,
            target_scopes=_CHAT_SCOPES,
            lifetime=3600,
        )

    # ADC direct: works on Cloud Run / GCE / GKE where the compute identity
    # is already the Chat bot service account.
    base_creds, _ = google.auth.default(scopes=_CHAT_SCOPES)
    return base_creds


def send_reply(
    space_name: str,
    thread_name: str,
    text: str,
    service_account_file: str | None,
    service_account_email: str | None = None,
) -> None:
    """Post a reply to a Google Chat thread via the REST API.

    This is a synchronous function; call it via asyncio.to_thread() from async contexts.
    thread_name may be empty for DMs — in that case no thread key is sent.
    """
    from googleapiclient.discovery import build as google_build

    creds = _build_credentials(service_account_file, service_account_email)
    service = google_build("chat", "v1", credentials=creds, cache_discovery=False)
    body: dict[str, Any] = {"text": text}
    if thread_name:
        body["thread"] = {"name": thread_name}
    service.spaces().messages().create(
        parent=space_name,
        body=body,
        messageReplyOption="REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD",
    ).execute()


_SPACE_NAME = re.compile(r"^spaces/[A-Za-z0-9_-]{1,128}$")
_MAX_SPACES = 100
# The last listing of the app's spaces, so a post or an approval card
# doesn't ask Google (or the control plane) every time: (monotonic time, rows).
_spaces_cache: tuple[float, list[dict[str, Any]]] | None = None


class NotAppSpace(ValueError):
    """A post to a chat that isn't one of the app's group spaces."""

_KNOWN_FOR_S = 600.0


def _list_spaces() -> list[dict[str, Any]]:
    settings = get_settings()
    if not settings.google_chat_project_number:
        return []
    from googleapiclient.discovery import build as google_build

    creds = _build_credentials(
        settings.google_chat_service_account_file, settings.google_chat_service_account_email,
    )
    service = google_build("chat", "v1", credentials=creds, cache_discovery=False)
    default = (settings.google_chat_default_space or "").strip()
    rows: list[dict[str, Any]] = []
    token = ""
    while len(rows) < _MAX_SPACES:
        kwargs: dict[str, Any] = {"filter": 'spaceType = "SPACE"', "pageSize": _MAX_SPACES}
        if token:
            kwargs["pageToken"] = token
        listed = service.spaces().list(**kwargs).execute()
        rows += [
            {"space": s.get("name"), "title": s.get("displayName") or "", "announce": s.get("name") == default}
            for s in listed.get("spaces") or []
        ]
        token = listed.get("nextPageToken") or ""
        if not token:
            break
    return rows


def group_spaces(max_age_s: float = 0.0) -> list[dict[str, Any]]:
    """The group spaces (not DMs) the Chat app is in, for posts and the
    department picker: ``[{"space", "title", "announce"}]``. ``announce``
    marks the space company announcements go to. ``max_age_s`` accepts a
    listing that recent instead of asking again. Synchronous; raises when
    Google Chat can't be reached."""
    global _spaces_cache
    import time

    cached = _spaces_cache
    if cached is not None and time.monotonic() - cached[0] <= max_age_s:
        return [dict(r) for r in cached[1]]
    out: list[dict[str, Any]] = []
    for row in _list_spaces()[:_MAX_SPACES]:
        space = str(row.get("space") or "")
        if _SPACE_NAME.fullmatch(space):
            out.append({
                "space": space, "title": str(row.get("title") or "")[:200], "announce": bool(row.get("announce")),
            })
    _spaces_cache = (time.monotonic(), out)
    return [dict(r) for r in out]


def known_spaces() -> list[dict[str, Any]] | None:
    """The last listing of the app's spaces, if one from the last few
    minutes is at hand; never asks Google. For a card's wording."""
    import time

    cached = _spaces_cache
    if cached is None or time.monotonic() - cached[0] > _KNOWN_FOR_S:
        return None
    return [dict(r) for r in cached[1]]


def announce_space(spaces: list[dict[str, Any]] | None = None) -> str | None:
    """Where company announcements go in Google Chat: ``GOOGLE_CHAT_DEFAULT_SPACE``,
    else the app's only group space. None when there's no such space.
    ``spaces`` is a listing already at hand. Synchronous."""
    default = (get_settings().google_chat_default_space or "").strip()
    if default:
        if _SPACE_NAME.fullmatch(default):
            return default
        # Set but not a space name: never fall back to a space nobody chose.
        logger.warning("GOOGLE_CHAT_DEFAULT_SPACE is not a space name (spaces/...)")
        return None
    spaces = group_spaces() if spaces is None else spaces
    picked = next((s["space"] for s in spaces if s["announce"]), None)
    if picked is not None:
        return str(picked)
    return str(spaces[0]["space"]) if len(spaces) == 1 else None


def space_title(space: str, spaces: list[dict[str, Any]] | None = None) -> str:
    """The name of one of the app's group spaces ("" when unnamed or not
    one of them). Synchronous; raises when Google Chat can't be reached."""
    spaces = group_spaces() if spaces is None else spaces
    return next((str(s["title"]) for s in spaces if s["space"] == space), "")


def post_to_space(space: str, text: str) -> None:
    """Post a new message as the Chat app in one of its group spaces, never
    a DM or a chat it isn't in. Chat's link markup is made inert, so the
    post can't show words over a link that goes elsewhere. Synchronous."""
    if not _SPACE_NAME.fullmatch(space or ""):
        raise NotAppSpace("not a Google Chat space the app is in")
    # A recent listing will do; a space it doesn't have is asked about again.
    if space not in {s["space"] for s in group_spaces(max_age_s=300)} and space not in {
        s["space"] for s in group_spaces()
    }:
        raise NotAppSpace("not a Google Chat space the app is in")
    settings = get_settings()
    send_reply(
        space, "", card_safe(text),
        settings.google_chat_service_account_file, settings.google_chat_service_account_email,
    )


# A card's button runs this function (``onClick.action.function``); its tap
# comes back as a CARD_CLICKED event naming it, with the button's data
# (``channel_cards.callback_data``) under ``_CARD_PARAM``.
_CARD_FUNCTION = "oe_card"
_CARD_PARAM = "card"
# What a card's message says while its tap is being decided.
_WORKING = "Working on it…"
# What it says when a tap couldn't be confirmed (``verify_tap``).
_UNCONFIRMED = "That tap couldn't be checked, so nothing was done. The card is still waiting on Today."
# How long the webhook waits on Chat to confirm a tap (Chat waits 30s).
_TAP_CHECK_S = 20.0


def _card_buttons(decision_id: int, options: tuple[tuple[str, str], ...]) -> list[dict[str, Any]]:
    """A card's buttons as one Chat ``cardsV2`` card, the Executive's pick
    first and Skip last (``channel_cards.card_options``)."""
    buttons = [
        {
            "text": card_safe(label),
            "onClick": {"action": {
                "function": _CARD_FUNCTION,
                "parameters": [{"key": _CARD_PARAM, "value": data}],
            }},
        }
        for label, data in options
    ]
    return [{"cardId": f"oe-card-{decision_id}", "card": {"sections": [{"widgets": [{"buttonList": {"buttons": buttons}}]}]}}]


def card_safe(text: str) -> str:
    """A card's words with Chat's link markup made inert. A card can quote
    what someone else wrote (a meeting title from an email), and Chat turns
    ``<https://…|words>`` into a link that shows only the words, so an
    approval card could hide where it points."""
    return text.replace("<", "\u2039")


def send_card(
    space_name: str,
    thread_name: str,
    decision_id: int,
    text: str,
    options: tuple[tuple[str, str], ...],
    service_account_file: str | None,
    service_account_email: str | None = None,
) -> None:
    """Post a card the reply left (``channel_cards``) as a message: its words,
    then its buttons. Synchronous; call it via asyncio.to_thread()."""
    from googleapiclient.discovery import build as google_build

    creds = _build_credentials(service_account_file, service_account_email)
    service = google_build("chat", "v1", credentials=creds, cache_discovery=False)
    body: dict[str, Any] = {"text": card_safe(text), "cardsV2": _card_buttons(decision_id, options)}
    if thread_name:
        body["thread"] = {"name": thread_name}
    service.spaces().messages().create(
        parent=space_name,
        body=body,
        messageReplyOption="REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD",
    ).execute()


def update_card(
    message_name: str,
    text: str,
    service_account_file: str | None,
    service_account_email: str | None = None,
    decision_id: int | None = None,
    options: tuple[tuple[str, str], ...] = (),
) -> None:
    """Rewrite a card's message once its tap is decided: what it said with
    the outcome under it, and no buttons; or, a card that still waits with
    new words, those words and its ``options``. The app may edit only its
    own messages. Synchronous; call it via asyncio.to_thread()."""
    from googleapiclient.discovery import build as google_build

    creds = _build_credentials(service_account_file, service_account_email)
    service = google_build("chat", "v1", credentials=creds, cache_discovery=False)
    cards = _card_buttons(decision_id, options) if decision_id is not None and options else []
    service.spaces().messages().patch(
        name=message_name, updateMask="text,cardsV2", body={"text": card_safe(text), "cardsV2": cards},
    ).execute()


def _obj(body: dict[str, Any], key: str) -> dict[str, Any]:
    value = body.get(key)
    return value if isinstance(value, dict) else {}


def _tapped(body: dict[str, Any]) -> str | None:
    """The button data of a CARD_CLICKED event on one of our cards, or None."""
    common = _obj(body, "common")
    action = _obj(body, "action")
    function = common.get("invokedFunction") or action.get("actionMethodName")
    if function != _CARD_FUNCTION:
        return None
    params = common.get("parameters")
    if isinstance(params, dict) and isinstance(params.get(_CARD_PARAM), str):
        return str(params[_CARD_PARAM])
    for item in action.get("parameters") or []:
        if isinstance(item, dict) and item.get("key") == _CARD_PARAM and isinstance(item.get("value"), str):
            return str(item["value"])
    return None


def _card_shown(body: dict[str, Any]) -> str:
    """What the tapped card's message says, without a line a tap added."""
    message = _obj(body, "message")
    return _strip_working(str(message.get("text") or ""))


def _strip_working(text: str) -> str:
    return text.removesuffix(f"\n\n{_WORKING}")


def verify_tap(
    space_name: str,
    message_name: str,
    data: str,
    user_name: str,
    service_account_file: str | None,
    service_account_email: str | None = None,
) -> str | None:
    """Whether a tap's event matches what Chat itself says, read as the app:
    the tapped message is the app's own, in that space, carrying this very
    button; the space is a DM between the app and one person alone; and that
    person is the event's user. The webhook's token proves Google issued it,
    not what the body says, so the tap counts only when Chat agrees.
    Returns the card message's words as Chat has them, or None for a no.
    Synchronous; call it via asyncio.to_thread(). Never raises: any doubt is
    a no."""
    import json

    from googleapiclient.discovery import build as google_build

    if not (space_name and user_name and message_name.startswith(f"{space_name}/messages/")):
        return None
    try:
        creds = _build_credentials(service_account_file, service_account_email)
        service = google_build("chat", "v1", credentials=creds, cache_discovery=False)
        message = _own_dm_message(
            service, space_name, message_name, user_name,
            carries=lambda m: json.dumps(data) in json.dumps(m.get("cardsV2") or []),
        )
        return None if message is None else _strip_working(str(message.get("text") or ""))
    except Exception:
        logger.warning("Google Chat: couldn't check a tap with Chat; it is refused", exc_info=True)
        return None


def _own_dm_message(
    service: Any, space_name: str, message_name: str, user_name: str, *, carries: Any,
) -> dict[str, Any] | None:
    """The app's own message ``message_name``, read from Chat as the app,
    when ``carries`` says it holds what is asked of it, and ``space_name``
    is a DM between the app and ``user_name`` alone; else None."""
    message = service.spaces().messages().get(name=message_name).execute()
    sender = message.get("sender") if isinstance(message.get("sender"), dict) else {}
    if sender.get("type") != "BOT" or not carries(message):
        return None
    space = service.spaces().get(name=space_name).execute()
    if space.get("singleUserBotDm") is not True:
        return None
    members = service.spaces().members().list(parent=space_name).execute()
    humans = [
        str((m.get("member") or {}).get("name") or "")
        for m in members.get("memberships") or []
        if isinstance(m, dict) and (m.get("member") or {}).get("type") == "HUMAN"
    ]
    return message if humans == [user_name] else None


def _card_data(cards: Any) -> list[str]:
    """The data on a message's buttons, as ``_card_buttons`` makes them."""
    out: list[str] = []
    for card in cards if isinstance(cards, list) else []:
        sections = ((card or {}).get("card") or {}).get("sections") if isinstance(card, dict) else None
        for section in sections if isinstance(sections, list) else []:
            for widget in (section or {}).get("widgets") or [] if isinstance(section, dict) else []:
                listed = ((widget or {}).get("buttonList") or {}).get("buttons") if isinstance(widget, dict) else None
                for button in listed if isinstance(listed, list) else []:
                    action = ((button or {}).get("onClick") or {}).get("action") if isinstance(button, dict) else None
                    for param in (action or {}).get("parameters") or [] if isinstance(action, dict) else []:
                        if isinstance(param, dict) and param.get("key") == _CARD_PARAM:
                            out.append(str(param.get("value") or ""))
    return out


def said_on(message: dict[str, Any], user_name: str, quoted: str, words: str) -> bool:
    """Whether ``message``, as Chat has it, is ``user_name``'s, quotes
    ``quoted`` and says ``words`` (its @mentions left out)."""
    sender, meta = _obj(message, "sender"), _obj(message, "quotedMessageMetadata")
    text = re.sub(r"<users/\d+>", "", str(message.get("text") or "")).strip()
    return bool(user_name) and sender.get("name") == user_name and meta.get("name") == quoted and text == words


def quoted_card(
    space_name: str,
    message_name: str,
    user_name: str,
    said_name: str,
    words: str,
    service_account_file: str | None,
    service_account_email: str | None = None,
) -> tuple[str, str] | None:
    """One of the app's cards that a message (``said_name``) quotes
    (``message_name``), with a tap's checks (``verify_tap``): read from Chat
    as the app, the quoting message is ``user_name``'s, quotes that card and
    says ``words`` (``said_on``: the event's body isn't bound to Google's
    token), and the card is the app's own message in that space, still with
    a card's buttons, in a DM between the app and ``user_name`` alone. Its
    buttons' data and its words, or None for a no. Synchronous; call it via
    asyncio.to_thread(). Never raises: any doubt is a no."""
    from googleapiclient.discovery import build as google_build

    if not (space_name and user_name and message_name.startswith(f"{space_name}/messages/")
            and said_name.startswith(f"{space_name}/messages/")):
        return None
    try:
        creds = _build_credentials(service_account_file, service_account_email)
        service = google_build("chat", "v1", credentials=creds, cache_discovery=False)
        said = service.spaces().messages().get(name=said_name).execute()
        if not said_on(said, user_name, message_name, words):
            return None
        message = _own_dm_message(
            service, space_name, message_name, user_name, carries=lambda m: bool(_card_data(m.get("cardsV2"))),
        )
        if message is None:
            return None
        return _card_data(message.get("cardsV2"))[0], str(message.get("text") or "")
    except Exception:
        logger.warning("Google Chat: couldn't check a quoted card with Chat", exc_info=True)
        return None


def _quoted_name(message: dict[str, Any]) -> str:
    """The message ``message`` replies to by quoting it, or ""."""
    meta = message.get("quotedMessageMetadata")
    if not isinstance(meta, dict) or str(meta.get("quoteType") or "REPLY") != "REPLY":
        return ""
    return str(meta.get("name") or "")


async def _say_on_card(
    words: str,
    person: Any,
    space_name: str,
    thread_name: str,
    quoted: str,
    user_name: str,
    said_name: str,
    service_account_file: str | None,
    service_account_email: str | None,
) -> bool:
    """A reply that quotes one of the app's cards (``quoted``), in a DM
    between the app and its person alone: words said on that card ("Or
    tell me…", ``channel_cards.say``), once Chat confirms the card is the
    app's own there (``quoted_card``). The card is shown again as it reads
    now, and what the Executive said back is posted. False when it isn't
    such a reply: the words are then an ordinary message."""
    from openexecutive.audit import log_event as audit_log
    from openexecutive.integrations import channel_cards

    found = await asyncio.to_thread(
        quoted_card, space_name, quoted, user_name, said_name, words, service_account_file, service_account_email,
    )
    card = channel_cards.parse_choice(found[0]) if found else None
    if card is None or found is None:
        return False
    said = await channel_cards.say(card, person, words)
    if said is None:
        return False
    decision_id = card.decision_id
    await asyncio.to_thread(
        audit_log,
        "integration_inbound",
        f"Google Chat card {decision_id}: said on it: {said.line[:120]}",
        actor="google_chat",
        details={"channel": "google_chat", "decision_id": decision_id, "choice": "say",
                 "outcome": channel_cards.outcome_of(said.line), "person_id": person.id,
                 "space_name": space_name, "message_name": quoted, "google_user": user_name},
    )
    try:
        if said.options:
            # It waits: shown again as it reads now, with its buttons.
            text = await asyncio.to_thread(channel_cards.with_tell_me, decision_id, said.text)
            await asyncio.to_thread(
                update_card, quoted, text, service_account_file, service_account_email, decision_id, said.options,
            )
        elif said.settled:
            # Carried out: the card keeps what it said, with the outcome under it.
            await asyncio.to_thread(
                update_card, quoted, channel_cards.with_line(_strip_working(found[1]), said.line),
                service_account_file, service_account_email,
            )
    except Exception:
        logger.exception("Google Chat: couldn't update card %d after words on it", decision_id)
    try:
        await asyncio.to_thread(
            send_reply, space_name, thread_name, said.line, service_account_file, service_account_email,
        )
    except Exception:
        logger.exception("Google Chat: couldn't answer words on card %d", decision_id)
    return True


async def _confirm_tap(
    data: str,
    space_name: str,
    message_name: str,
    sender_email: str,
    user_name: str,
    service_account_file: str | None,
    service_account_email: str | None,
) -> tuple[Any, str] | None:
    """Who tapped, and the card's words as Chat has them, once Chat confirms
    the tap (``verify_tap``); None, audited, for a no. Asked before the
    buttons come off, since the check looks for the tapped button on the
    message."""
    from openexecutive.audit import log_event as audit_log
    from openexecutive.integrations import channel_cards
    from openexecutive.people.store import find_person_by_email

    parsed = channel_cards.parse_choice(data)
    if parsed is None:
        return None
    decision_id = parsed.decision_id
    try:
        person = await asyncio.to_thread(find_person_by_email, sender_email) if sender_email else None
    except Exception:
        logger.exception("Google Chat: couldn't look up who tapped card %d", decision_id)
        person = None
    confirmed = None
    if person is not None and person.id is not None:
        confirmed = await asyncio.to_thread(
            verify_tap, space_name, message_name, data, user_name, service_account_file, service_account_email,
        )
    if person is not None and person.id is not None and confirmed is not None:
        return person, confirmed
    logger.warning("Google Chat: a tap on card %d was refused (%s)", decision_id,
                   "not on People" if person is None or person.id is None else "Chat didn't confirm it")
    await asyncio.to_thread(
        audit_log,
        "integration_inbound",
        f"Rejected: Google Chat tap on card {decision_id}",
        actor="google_chat",
        details={"channel": "google_chat", "decision_id": decision_id, "rejected": True,
                 "person_id": getattr(person, "id", None), "space_name": space_name,
                 "message_name": message_name, "google_user": user_name},
    )
    return None


async def _decide_tap(
    data: str,
    person: Any,
    shown: str,
    space_name: str,
    message_name: str,
    user_name: str,
    service_account_file: str | None,
    service_account_email: str | None,
) -> None:
    """Decide a tap Chat confirmed (``_confirm_tap``) for ``person``, whose
    DM with the app it is (``channel_cards.tap``), then rewrite the card:
    from its confirmed words (``shown``) with the outcome under them, or,
    a card that still waits with new words, those and its buttons."""
    from openexecutive.audit import log_event as audit_log
    from openexecutive.integrations import channel_cards

    parsed = channel_cards.parse_choice(data)
    if parsed is None:
        return
    decision_id = parsed.decision_id
    where = {"space_name": space_name, "message_name": message_name, "google_user": user_name}
    tapped = await channel_cards.tap(parsed, person)
    line = tapped.line
    name = channel_cards.CHOICE_NAMES[parsed.choice]
    await asyncio.to_thread(
        audit_log,
        "integration_inbound",
        f"Google Chat card {decision_id}: tapped {name.capitalize()}: {line[:120]}",
        actor="google_chat",
        details={"channel": "google_chat", "decision_id": decision_id, "choice": name,
                 "outcome": channel_cards.outcome_of(line), "person_id": person.id, **where},
    )
    if line == channel_cards.BUSY:
        return
    try:
        if tapped.options:
            text = await asyncio.to_thread(
                channel_cards.with_tell_me, decision_id, channel_cards.with_line(tapped.text, line),
            )
            await asyncio.to_thread(
                update_card, message_name, text, service_account_file, service_account_email, decision_id,
                tapped.options,
            )
        else:
            await asyncio.to_thread(
                update_card, message_name, channel_cards.with_line(shown, line),
                service_account_file, service_account_email,
            )
    except Exception:
        logger.exception("Google Chat: couldn't update card %d after the tap", decision_id)


async def _take_tap(body: dict[str, Any], background_tasks: BackgroundTasks, settings: Any) -> dict[str, Any]:
    """A CARD_CLICKED event: a tap on a card's button. It counts only in a DM
    between the app and the tapper alone (``singleUserBotDm``), from a human
    on People, as a Slack or Discord tap counts only in the tapper's own DM;
    ``verify_tap`` checks that with Chat itself before the answer, since the
    token isn't bound to the body and the check needs the button still on
    the message, and ``decisions.resolve_in_chat`` decides whether the card
    is theirs.
    The card shows it is being worked on at once, without its buttons, since
    deciding can take longer than Chat waits for an answer, and is rewritten
    when it is done. A tap Chat doesn't confirm decides nothing, and the card
    says so (only a fixed line: an unconfirmed tap's event may not be this
    card's)."""
    data = _tapped(body)
    if data is None:
        return {}
    user = _obj(body, "user")
    space = _obj(body, "space")
    message = _obj(body, "message")
    shown = _card_shown(body)
    if not _is_one_on_one(space) or user.get("type") != "HUMAN":
        return {"actionResponse": {"type": "UPDATE_MESSAGE"}, "text": shown, "cardsV2": []}
    space_name = str(space.get("name") or "")
    message_name = str(message.get("name") or "")
    user_name = str(user.get("name") or "")
    sa_file = settings.google_chat_service_account_file
    sa_email = settings.google_chat_service_account_email
    try:
        # Inside Chat's wait for the answer: a check that outlasts it is a no,
        # so a late yes can't decide a card whose tapper saw an error.
        confirmed = await asyncio.wait_for(_confirm_tap(
            data, space_name, message_name, str(user.get("email") or ""), user_name, sa_file, sa_email,
        ), timeout=_TAP_CHECK_S)
    except TimeoutError:
        from openexecutive.audit import log_event as audit_log

        logger.warning("Google Chat: checking a tap took too long; it is refused")
        await asyncio.to_thread(
            audit_log,
            "integration_inbound",
            "Rejected: Google Chat tap (Chat didn't confirm it in time)",
            actor="google_chat",
            details={"channel": "google_chat", "rejected": True, "timed_out": True, "space_name": space_name,
                     "message_name": message_name, "google_user": user_name},
        )
        confirmed = None
    if confirmed is None:
        return {"actionResponse": {"type": "UPDATE_MESSAGE"}, "text": _UNCONFIRMED, "cardsV2": []}
    person, words = confirmed
    background_tasks.add_task(
        _decide_tap,
        data=data,
        person=person,
        shown=words,
        space_name=space_name,
        message_name=message_name,
        user_name=user_name,
        service_account_file=sa_file,
        service_account_email=sa_email,
    )
    return {
        "actionResponse": {"type": "UPDATE_MESSAGE"},
        "text": f"{words}\n\n{_WORKING}" if words else _WORKING,
        "cardsV2": [],
    }


# Per message: bounds the downloads (and OCR / model work) one message can
# cause. Files are read only for a sender on the People roster (matched by
# the Google account email the event carries): the webhook authenticates
# Google, not the sender, so anyone in a space the bot is in can post.
_MAX_ATTACHMENTS = 5
_MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024


def _message_attachments(message: dict) -> list[dict]:
    """The attachments on a Chat MESSAGE event, capped per message."""
    raw = message.get("attachment")
    if not isinstance(raw, list):
        return []
    return [a for a in raw if isinstance(a, dict)][:_MAX_ATTACHMENTS]


def _attachment_name(attachment: dict) -> str:
    return str(attachment.get("contentName") or "file")


def _sender_on_roster(email: str) -> bool:
    """Whether a Chat sender is a team member (not just a contact), by the
    Google account email on the event. No email — a bot, or an event without
    one — is not."""
    if not email:
        return False
    try:
        from openexecutive.people.store import find_person_by_email

        return find_person_by_email(email) is not None
    except Exception:
        logger.exception("Google Chat: roster lookup failed")
        return False


def download_attachment(
    resource_name: str,
    service_account_file: str | None,
    service_account_email: str | None = None,
    max_bytes: int = _MAX_ATTACHMENT_BYTES,
) -> bytes:
    """Download an uploaded Chat attachment's bytes (``media.download``, the
    ``chat.bot`` scope the replies already use). Raises ``ValueError`` past
    ``max_bytes``, checked per chunk so an oversized file is never buffered
    whole. Synchronous; call it via asyncio.to_thread()."""
    import io

    from googleapiclient.discovery import build as google_build
    from googleapiclient.http import MediaIoBaseDownload

    creds = _build_credentials(service_account_file, service_account_email)
    service = google_build("chat", "v1", credentials=creds, cache_discovery=False)
    request = service.media().download_media(resourceName=resource_name)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request, chunksize=_DOWNLOAD_CHUNK_BYTES)
    done = False
    while not done:
        _status, done = downloader.next_chunk()
        if buf.tell() > max_bytes:
            raise ValueError("attachment too large")
    return buf.getvalue()


async def _read_attachments(
    attachments: list[dict],
    service_account_file: str | None,
    service_account_email: str | None,
) -> tuple[str, list[dict]]:
    """Read a message's attachments the way the other chat channels do:
    documents to text (a scanned PDF converted), images to vision blocks.

    A Drive file shared into the message has no bytes here — it is left to
    the Drive tools, with its id so the Executive can fetch it."""
    from openexecutive.integrations.attachments import build_attachment_output

    parts: list[str] = []
    blocks: list[dict] = []
    limit_mb = _MAX_ATTACHMENT_BYTES // (1024 * 1024)
    for att in attachments:
        name = _attachment_name(att)
        drive = att.get("driveDataRef")
        data_ref = att.get("attachmentDataRef")
        if isinstance(drive, dict) and drive.get("driveFileId"):
            parts.append(
                f"(Attached {name}: a Google Drive file, id {drive['driveFileId']}"
                " — read it with the Drive tools)"
            )
            continue
        resource = data_ref.get("resourceName") if isinstance(data_ref, dict) else None
        if not resource:
            parts.append(f"(Could not download {name})")
            continue
        try:
            data = await asyncio.to_thread(
                download_attachment, str(resource), service_account_file, service_account_email
            )
        except ValueError:
            parts.append(f"(Skipped {name}: file too large — limit {limit_mb} MB)")
            continue
        except Exception:
            logger.exception("Google Chat: attachment download failed for %s", name)
            parts.append(f"(Could not download {name})")
            continue
        try:
            text, image_blocks = await build_attachment_output(
                name, data, str(att.get("contentType") or "")
            )
        except Exception:
            logger.exception("Google Chat: attachment processing failed for %s", name)
            parts.append(f"(Could not process {name})")
            continue
        if text:
            parts.append(text)
        blocks.extend(image_blocks)
    return "\n\n".join(parts), blocks


def _is_direct(space: object) -> bool:
    """The event's space is a DM with the app (old ``type`` or new ``spaceType``)."""
    if not isinstance(space, dict):
        return False
    return (
        space.get("type") == "DM"
        or space.get("spaceType") == "DIRECT_MESSAGE"
        or space.get("singleUserBotDm") is True
    )


def _is_one_on_one(space: object) -> bool:
    """The event's space is a DM between the app and one person alone
    (``singleUserBotDm``). A ``DIRECT_MESSAGE`` can also be two people's DM
    the app was added to, which the other person reads too."""
    return isinstance(space, dict) and space.get("singleUserBotDm") is True


def _load_history(session_id: str) -> list[dict[str, Any]]:
    from openexecutive.memory.session_store import load_messages

    try:
        return load_messages(session_id)
    except Exception:
        logger.exception("Google Chat: couldn't load the conversation %s", session_id)
        return []


async def _send_cards(
    space_name: str,
    thread_name: str,
    cards: list[int],
    service_account_file: str | None,
    service_account_email: str | None,
) -> None:
    """Post the cards a reply left, each as its own message under the reply."""
    from openexecutive.integrations import channel_cards

    for decision_id in cards[: channel_cards.MAX_CARDS]:
        try:
            text = await asyncio.to_thread(channel_cards.card_text, decision_id)
            if not text:
                continue
            options = await asyncio.to_thread(channel_cards.card_options, decision_id, text)
            shown = await asyncio.to_thread(channel_cards.with_tell_me, decision_id, text)
            await asyncio.to_thread(
                send_card, space_name, thread_name, decision_id, shown, options,
                service_account_file, service_account_email,
            )
        except Exception:
            logger.exception("Google Chat: couldn't send card %d", decision_id)


async def _process_and_reply(
    message_text: str,
    sender_name: str,
    space_name: str,
    thread_name: str,
    message_name: str,
    service_account_file: str | None,
    service_account_email: str | None,
    attachments: list[dict] | None = None,
    sender_email: str = "",
    reply_context: str = "",
    direct: bool = False,
    one_on_one: bool = False,
    quoted: str = "",
    user_name: str = "",
) -> None:
    """One Chat message's turn. ``reply_context`` is the message it quotes
    (``integrations.reply_context``), put in front of the words. ``direct``:
    a DM with the app, which is one conversation, whatever thread each
    message lands in. ``one_on_one``: a DM between the app and the sender
    alone (``singleUserBotDm``), the only conversation private to them.
    ``quoted``: the message it quotes, and ``user_name`` the sender's
    ``users/{id}``: a quote of one of the app's cards there is words said on
    that card (``_say_on_card``)."""
    if not space_name:
        logger.error("Google Chat: missing space_name for message %s, cannot reply", message_name)
        return

    # Deterministic per-thread session id so every audit row from this inbound
    # (chat_turn, specialist_consult, tool_invocation) shares a grouping key
    # with the integration_inbound row.
    # A DM is one conversation (each message there can start its own thread);
    # in a space, each thread is one.
    session_id = f"google_chat:{space_name}" if direct else f"google_chat:{space_name}:{thread_name}"
    # The sender's People row, by the Google account email on the event (the
    # JWT proves the event is Google's, so the email is Google's word). None
    # for someone who isn't on People: their turn is unverified, and the
    # sends and edits wait on the owner's card (levels.unrostered_speaker).
    sender_person = None
    if sender_email:
        try:
            from openexecutive.people.store import find_person_by_email

            sender_person = find_person_by_email(sender_email)
        except Exception:
            logger.exception("Google Chat: failed to look up the sender on People")
    if one_on_one and quoted and sender_person is not None and not attachments and await _say_on_card(
        message_text, sender_person, space_name, thread_name, quoted, user_name, message_name,
        service_account_file, service_account_email,
    ):
        return

    # The rows written before the turn binds its session (the inbound row,
    # alert triage, retrieval) follow the sender's turn, as on the other chat
    # channels (``people_tools.audit_rows_on_senders_turn``): a row naming one
    # of the principal's contacts is theirs alone on their own turn, and a
    # teammate's rows are that teammate's.
    from openexecutive.audit.context import principal_turn_rows, rows_for_person
    from openexecutive.orchestrator.people_tools import sent_by_principal, sent_by_teammate

    principal, teammate = False, None
    if sender_person is not None:
        try:
            principal = sent_by_principal("google_chat", space_name, sender_person, speaker_verified=True)
            teammate = sent_by_teammate("google_chat", space_name, sender_person, speaker_verified=True)
        except Exception:
            logger.exception("Google Chat: couldn't tell whose turn this is; rows follow the ordinary rule")
    with (
        principal_turn_rows(principal),
        rows_for_person(teammate) if teammate is not None else contextlib.nullcontext(),
    ):
        await _answer(
            message_text=message_text, sender_name=sender_name, space_name=space_name,
            thread_name=thread_name, message_name=message_name,
            service_account_file=service_account_file, service_account_email=service_account_email,
            attachments=attachments, sender_email=sender_email, reply_context=reply_context,
            one_on_one=one_on_one, session_id=session_id, sender_person=sender_person,
        )


async def _answer(
    *,
    message_text: str,
    sender_name: str,
    space_name: str,
    thread_name: str,
    message_name: str,
    service_account_file: str | None,
    service_account_email: str | None,
    attachments: list[dict] | None,
    sender_email: str,
    reply_context: str,
    one_on_one: bool,
    session_id: str,
    sender_person: Any,
) -> None:
    """The rest of ``_process_and_reply``, once the sender is known."""
    from openexecutive.audit import log_event as audit_log
    audit_log(
        "integration_inbound",
        f"Inbound google_chat from {sender_name} (space={space_name}): {message_text[:160]}",
        actor="google_chat",
        session_id=session_id,
        details={
            "channel": "google_chat",
            "space_name": space_name,
            "thread_name": thread_name,
            "message_name": message_name,
            "sender": sender_name,
            # Who it is by People, not by display name: the brief tells the
            # owner's own chat from a namesake's by this (briefing.live_signals).
            "sender_person_id": sender_person.id if sender_person is not None else None,
            "text_len": len(message_text),
        },
    )

    # Fork into alert triage pipeline (fire-and-forget, same pattern as Slack/email).
    try:
        from openexecutive.alerts.pipeline import schedule_chat_message_evaluation

        schedule_chat_message_evaluation(
            source="google_chat",
            external_id=message_name,
            body=message_text,
            sender=sender_person,
            fallback_name=sender_name,
        )
    except Exception:
        logger.exception("Google Chat: failed to schedule alert evaluation")

    try:
        from openexecutive.integrations import channel_turn

        session_facts: dict[str, bool] = {
            # Anywhere but the DM between the app and the sender alone, others
            # write into the conversation this turn reads, and unlike Slack's
            # channels not only people on People (Google Chat turns no one
            # away). So it is a shared chat, as a web chat shared with
            # teammates is.
            "shared_chat": not one_on_one,
            # And anyone may have written last, so the turn is one that read
            # outside words: the Approvals level decides what its steps do.
            "others_spoke_last": not one_on_one,
        }
        if sender_person is not None:
            # A sender on People is signed in here, as on Slack: Google
            # signed the event (the JWT checked above), and its `user.email`
            # is the Google account that sent it, never anything the message
            # says. So the turn is theirs, verified, in a space or a DM.
            session_facts["speaker_verified"] = True
            # A DM between the app and the sender alone is read by them
            # alone, like a Slack DM, so what is private to them may be
            # offered there. Anything else never: a space, or a DM between
            # two people the app was added to.
            session_facts["private_chat"] = one_on_one

        own_text = message_text
        # Quoted only for people on the team: the quote is text the sender
        # chose, as an attachment's is.
        reply_context = reply_context if _sender_on_roster(sender_email) else ""
        if reply_context:
            from openexecutive.integrations.reply_context import with_context

            message_text = with_context(reply_context, message_text)
        chat_message = message_text
        # What the sender wrote and the files' names, as Slack records it: any
        # attachment text below and the quote above go into the prompt only.
        from openexecutive.integrations.attachments import attached_files_note
        from openexecutive.integrations.attachments import own_words as _own_words

        names = [_attachment_name(a) for a in attachments or ()]
        # A file-only message already arrives as that note (the webhook).
        own_words = (
            own_text if own_text == attached_files_note(names)
            else _own_words(own_text, names)
        )
        image_blocks: list[dict] = []
        # The webhook authenticates Google, not the sender: anyone in a space
        # with the bot reaches here, so only a team member's files are read
        # and only their links can be opened.
        on_roster = _sender_on_roster(sender_email)
        if attachments and on_roster:
            att_text, image_blocks = await _read_attachments(
                attachments, service_account_file, service_account_email
            )
            if att_text:
                # Before the words, as the other chat channels inline it.
                chat_message = f"{att_text}\n\n{message_text}"
        elif attachments:
            chat_message = (
                f"(Attached files, not read — files are read only from people on "
                f"the team: {', '.join(names)})\n\n{message_text}"
            )

        result = await channel_turn.run(
            channel_turn.ChannelTurn(
                channel="google_chat",
                channel_ref=space_name,
                session_id=session_id,
                own_text=own_text,
                turn_text=own_text,
                model_text=chat_message,
                person=sender_person,
                # The open cards, in the DM of a sender on People only.
                is_dm=one_on_one and sender_person is not None,
                # The sender's own words: never attachment text or the quote.
                memory_text=own_words,
                fetch_own_urls=on_roster,
                image_blocks=image_blocks or None,
                # A DM between the app and its person alone shows the reply's
                # cards with Approve and Skip buttons (integrations.channel_cards).
                offers_cards=True,
                session_facts=session_facts,
                # The conversation so far, so a follow-up ("add a few more
                # bullets") has what it follows.
                history=await asyncio.to_thread(_load_history, session_id),
            )
        )
        response = result.response
        cards = result.cards
        await asyncio.to_thread(
            channel_turn.save_turn,
            session_id=session_id,
            title=f"Google Chat {sender_name}",
            created_at=result.session.created_at.isoformat(),
            owner_person_id=result.person_id,
            user_text=own_text,
            assistant_text=response,
            sender_person_id=result.person_id,
        )
        from openexecutive.orchestrator.disclosure import with_chat_line

        await asyncio.to_thread(
            send_reply, space_name, thread_name, with_chat_line(response, session_id),
            service_account_file, service_account_email,
        )
        if cards:
            await _send_cards(space_name, thread_name, cards, service_account_file, service_account_email)
    except Exception:
        logger.exception("Google Chat: handler error for message %s", message_name)
        try:
            await asyncio.to_thread(
                send_reply,
                space_name,
                thread_name,
                "I encountered an error processing your request. Please try again.",
                service_account_file,
                service_account_email,
            )
        except Exception:
            logger.exception("Google Chat: also failed to send error reply")


@router.post("/webhook/google-chat", status_code=200)
async def google_chat_webhook(request: Request, background_tasks: BackgroundTasks) -> dict:
    settings = get_settings()

    if not settings.google_chat_project_number or not (
        settings.google_chat_service_account_file
        or settings.google_chat_service_account_email
    ):
        raise HTTPException(status_code=503, detail="Google Chat integration not configured")

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    try:
        verify_google_jwt(auth.removeprefix("Bearer "), settings.google_chat_project_number)
    except Exception:
        logger.warning("Google Chat JWT verification failed")
        raise HTTPException(status_code=401, detail="Invalid JWT") from None

    body = await request.json()
    event_type = body.get("type", "")

    if event_type == "ADDED_TO_SPACE":
        logger.info("Google Chat bot added to space: %s", body.get("space", {}).get("name"))
        return {"text": "Hello! I'm your AI executive. Mention me anytime with a question."}

    if event_type == "REMOVED_FROM_SPACE":
        logger.info("Google Chat bot removed from space: %s", body.get("space", {}).get("name"))
        return {}

    if event_type == "CARD_CLICKED":
        return await _take_tap(body, background_tasks, settings)

    if event_type != "MESSAGE":
        return {}

    message = body.get("message", {})
    raw_text = message.get("text", "")
    # Strip structured @mention tokens (format: <users/USER_ID>).
    cleaned = re.sub(r"<users/\d+>", "", raw_text).strip()
    attachments = _message_attachments(message)
    if not cleaned and not attachments:
        return {}
    if not cleaned:
        # A file-only message: its names stand in for the words, as the web
        # upload route records one.
        from openexecutive.integrations.attachments import attached_files_note

        cleaned = attached_files_note(map(_attachment_name, attachments))

    from openexecutive.integrations import reply_context

    background_tasks.add_task(
        _process_and_reply,
        message_text=cleaned,
        sender_name=body.get("user", {}).get("displayName", "Unknown"),
        space_name=body.get("space", {}).get("name", ""),
        thread_name=message.get("thread", {}).get("name", ""),
        message_name=message.get("name", ""),
        reply_context=reply_context.google_chat(
            message,
            reply_context.google_chat_app_id(message),
            # A DM between the app and one person has two in it: the sender and the app.
            two_party=_is_one_on_one(body.get("space", {})),
        ),
        direct=_is_direct(body.get("space", {})),
        one_on_one=_is_one_on_one(body.get("space", {})),
        quoted=_quoted_name(message),
        user_name=str(body.get("user", {}).get("name") or ""),
        service_account_file=settings.google_chat_service_account_file,
        service_account_email=settings.google_chat_service_account_email,
        attachments=attachments,
        sender_email=str(body.get("user", {}).get("email") or ""),
    )
    return {}
