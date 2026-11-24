"""Home's Needs you rows for connections that stopped working.

A connection whose sign-in died stops whatever runs through it, quietly:
Home said "All clear" while the person's own mailbox no longer opened. So
each broken connection the reader can fix is one row in ``GET /today``'s
``waiting_elsewhere``, with **Reconnect** to the page that connects it, and
the row goes away by itself once the connection works again.

- **Your mailbox** (and the calendar on it): the reader's own mailbox for
  As you, from the record a status check keeps while its saved sign-in is
  refused (``delegation.sign_in_notice.broken_since``), while a saved
  sign-in is still there to reconnect.
- **My Google or Microsoft account**: the Executive's own accounts, which
  the tools gateway marks when they turn its sign-in away
  (``MCPGateway.needs_reconnect``). Only someone who may reconnect them sees
  these.

One wording for a broken connection everywhere it shows:
:data:`NEEDS_RECONNECTING` and :data:`RECONNECT`. Never raises.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

NEEDS_RECONNECTING = "Needs reconnecting"
RECONNECT = "Reconnect"

MAILBOX_KEY = "reconnect:mailbox"
MAILBOX_TITLE = "Your mailbox needs reconnecting"
MAILBOX_DETAIL = "I can't read or answer your email, or see your calendar, until you do."

# The Executive's accounts, by the gateway server that uses them.
_ACCOUNTS: dict[str, str] = {
    "google_workspace": "Google",
    "microsoft_365": "Microsoft",
}
ACCOUNT_DETAIL = "I can't use its email, calendar or files until you do."


def mailbox_row(email: str | None, href: str) -> dict[str, Any] | None:
    """The row for ``email``'s own mailbox, or None while it works (or was
    never connected, or was disconnected on purpose)."""
    if not email:
        return None
    try:
        from openexecutive.delegation import sign_in_notice
        from openexecutive.delegation.gmail import credential_path, normalize_email

        address = normalize_email(email)
        since = sign_in_notice.broken_since(address)
        if since is None or not credential_path(address).exists():
            return None
    except Exception:
        logger.warning("needs_reconnecting: couldn't read the mailbox's state", exc_info=True)
        return None
    return {
        "key": MAILBOX_KEY,
        "title": MAILBOX_TITLE,
        "detail": MAILBOX_DETAIL,
        "count": 1,
        "href": href,
        "action": RECONNECT,
        "at": since,
    }


def account_rows(href: str, gateway: object | None = None) -> list[dict[str, Any]]:
    """One row per Executive account whose sign-in the gateway found dead."""
    try:
        from openexecutive.orchestrator.mcp_gateway import (
            gateway_reconnect_needed,
            get_active_gateway,
        )

        dead = gateway_reconnect_needed(gateway if gateway is not None else get_active_gateway())
    except Exception:
        logger.warning("needs_reconnecting: couldn't read the gateway", exc_info=True)
        return []
    return [
        {
            "key": f"reconnect:{server}",
            "title": f"My {_ACCOUNTS[server]} account needs reconnecting",
            "detail": ACCOUNT_DETAIL,
            "count": 1,
            "href": href,
            "action": RECONNECT,
            "at": "",
        }
        for server in dead
        if server in _ACCOUNTS
    ]
