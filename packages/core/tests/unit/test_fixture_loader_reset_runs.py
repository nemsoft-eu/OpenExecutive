"""``reset_all_state`` must also wipe the run/audit history tables.

Before this change, the reset endpoint cleared chat/alerts/scheduled-actions
but left ``workflow_runs``, ``audit_log`` and ``eval_runs`` intact. The /jobs
"Runs" tab and /audit page then kept showing the previous fixture's executions
and the demo-page reset looked broken.

This test reaches into ``memory.company_tables.wipe_tables``, which
``reset_all_state`` delegates to, seeds one row in each of the three tables
via their real persistence helpers, and confirms they're empty afterwards.
``test_company_tables.py`` runs the whole ``reset_all_state`` against a row
in every registered table.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from openexecutive.memory.company_tables import PER_COMPANY_TABLES, wipe_tables


@pytest.fixture
def _episodic_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the episodic DB at tmp and initialize the base schema."""
    from openexecutive.memory import episodic

    db_path = tmp_path / "episodic.db"
    monkeypatch.setattr(episodic, "DB_PATH", db_path)
    episodic.initialize_db(db_path)
    return db_path


def test_reset_clears_workflow_runs_audit_log_and_eval_runs(
    _episodic_db: Path,
) -> None:
    """Seeding one row into each run/audit table and then running the same
    deletion ``reset_all_state`` uses must leave all three tables empty.
    """
    from openexecutive.audit.logger import AuditLogger
    from openexecutive.evals.persistence import create_eval_run
    from openexecutive.workflows.persistence import create_run

    # --- seed ---
    create_run(
        run_id="run-1",
        workflow_name="board_strategy",
        title="Board strategy run",
        inputs={"foo": "bar"},
        db_path=_episodic_db,
    )

    logger = AuditLogger(db_path=_episodic_db)
    logger.log(
        event_type="chat",
        actor="user",
        summary="seeded audit row",
    )

    create_eval_run(
        run_id="eval-1",
        kind="smoke",
        scenario_ids=["scenario-a"],
        db_path=_episodic_db,
    )

    # sanity: rows landed
    with sqlite3.connect(str(_episodic_db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM workflow_runs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0] == 1

    # --- act: invoke the same helper reset_all_state uses ---
    cleared = wipe_tables(_episodic_db, PER_COMPANY_TABLES)

    # --- assert: counts returned + tables actually empty ---
    assert {t: cleared[t] for t in ("workflow_runs", "audit_log", "eval_runs")} == {
        "workflow_runs": 1,
        "audit_log": 1,
        "eval_runs": 1,
    }
    with sqlite3.connect(str(_episodic_db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM workflow_runs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0] == 0


def test_reset_all_state_wipe_list_includes_run_history_tables() -> None:
    """Guard against silently dropping a table from the wipe list.

    ``reset_all_state`` wipes every table in ``memory.company_tables``
    (proven with a row in every table in ``test_company_tables.py``), so the
    run/audit history must stay registered there.
    """
    for table in ("workflow_runs", "audit_log", "eval_runs"):
        assert table in PER_COMPANY_TABLES, (
            f"{table!r} dropped out of memory.company_tables — the demo-page "
            "reset will stop clearing it and previously-run jobs/audit/eval "
            "rows will reappear after reset."
        )


def test_reset_all_state_wipes_the_decision_ledger_and_act_as_me() -> None:
    """A reset box has proposed nothing, and keeps nobody's Act as me state.
    Act as me's tables come from its own list, so a new one is wiped too."""
    from openexecutive.delegation.schema import TABLES as DELEGATION_TABLES

    for table in ("decision_instances", "decision_class_state", *DELEGATION_TABLES):
        assert table in PER_COMPANY_TABLES
