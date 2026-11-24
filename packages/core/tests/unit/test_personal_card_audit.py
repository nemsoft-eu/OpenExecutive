"""Approving a personal card (``alerts.models.PERSONAL_SOURCES``): its
audit rows are private to the person it was for."""
from __future__ import annotations

from typing import Any

import pytest

from openexecutive.alerts import approvals
from openexecutive.alerts.models import Alert
from openexecutive.api.routes import alerts as alerts_route


def test_approving_a_personal_card_writes_a_row_only_its_person_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: list[dict[str, Any]] = []
    monkeypatch.setattr("openexecutive.audit.log_event", lambda *a, **k: rows.append(k))
    card = Alert(id=7, external_id="h1", source="browser_handover", severity="medium",
                 headline="Waiting on you: sign in to the pharmacy", body="b",
                 created_at="2026-11-16T09:30:00+00:00", routed_to_person_id=12)
    team_card = card.model_copy(update={"source": "system"})
    approval = approvals.AlertApproval(id=1, alert_id=7, approved_by_person_id=12,
                                       approved_at="2026-11-16T09:31:00+00:00")

    alerts_route._audit_approval("alert_approved", "Approved", card, approval)
    alerts_route._audit_approval("alert_approved", "Approved", team_card, approval)

    assert [r["private_to_person"] for r in rows] == [12, None]
