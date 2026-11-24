"""Links inside the app never reach the model (orchestrator.app_links)."""

from __future__ import annotations

import json

from openexecutive.orchestrator import app_links


def test_a_path_inside_the_app_is_left_out_and_the_rest_kept() -> None:
    result = json.dumps({
        "ok": True, "artifact_id": "alert:57", "title": "Plan", "url": "/artifacts/alert:57",
        "external_url": "https://docs.example.com/d/1",
        "items": [{"id": 3, "link": "/meetings/3"}, {"id": 4, "href": "https://example.com/x"}],
    })
    cleaned = json.loads(app_links.for_model(result))
    assert cleaned == {
        "ok": True, "artifact_id": "alert:57", "title": "Plan",
        "external_url": "https://docs.example.com/d/1",
        "items": [{"id": 3}, {"id": 4, "href": "https://example.com/x"}],
    }


def test_anything_else_is_untouched() -> None:
    for content in ("It's at /artifacts/alert:57", '{"url": "//cdn.example.com/a"}', "{not json", [{"type": "text"}]):
        assert app_links.for_model(content) == content
    unchanged = json.dumps({"ok": True, "path_hint": "/x"})
    assert app_links.for_model(unchanged) is unchanged
