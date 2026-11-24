"""Links inside the app stay out of what the model reads.

A tool result may name where something lives in the app ("url":
"/artifacts/alert:57") for the screens that show it: the chat's chips and
cards link there themselves. Given to the model, such a path only ends up
written out as text in a reply, where nobody can open it (in an email or a
chat app it means nothing). So every tool result loses those fields at the
one point results reach the model (``Executive`` tool loop), whatever tool
wrote them. A full web address (``https://…``) is kept: it opens anywhere.
"""

from __future__ import annotations

import json
from typing import Any

# The fields tools use for a place in the app.
_LINK_KEYS = frozenset({"url", "link", "href"})


def _in_app(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("/") and not value.startswith("//")


def _without(value: Any) -> tuple[Any, bool]:
    if isinstance(value, dict):
        changed = False
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in _LINK_KEYS and _in_app(item):
                changed = True
                continue
            out[key], inner = _without(item)
            changed = changed or inner
        return out, changed
    if isinstance(value, list):
        items = [_without(item) for item in value]
        return [item for item, _ in items], any(changed for _, changed in items)
    return value, False


def for_model(content: Any) -> Any:
    """``content`` without links inside the app, when it is a JSON tool
    result; anything else as it is. Never raises."""
    if not isinstance(content, str) or not content.lstrip().startswith(("{", "[")):
        return content
    try:
        cleaned, changed = _without(json.loads(content))
        return json.dumps(cleaned, ensure_ascii=False) if changed else content
    except Exception:  # noqa: BLE001 - not JSON, or nested too deep: as it is
        return content
