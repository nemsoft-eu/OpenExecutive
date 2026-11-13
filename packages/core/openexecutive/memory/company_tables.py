"""Which SQLite tables belong to one company — the list every company swap reads.

Three paths replace the live company: switching into a blank client slot
(``clients.slots``), the factory reset and fixture load/unload (both in
``cli.fixture_loader``). Each used to keep its own hand-written table list, and
a table added to one list and not the others leaked one company's rows into the
next (department notes served under a shared department slug, a previous
company's mail threads, Act as me state). They now all read this module.

Every table the app creates is in exactly one of :data:`PER_COMPANY_TABLES` or
:data:`COMPANY_INDEPENDENT_TABLES`; ``tests/unit/test_company_tables.py`` scans
the source for ``CREATE TABLE`` and fails on a table that is in neither. A new
table therefore needs a decision here, not a guess in each swap path.

The order of :data:`PER_COMPANY_TABLES` does not matter: :func:`wipe_tables`
reads the foreign keys from the database and deletes children before parents.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from openexecutive.delegation import sign_in_notice
from openexecutive.delegation.schema import TABLES as DELEGATION_TABLES
from openexecutive.memory.history_schema import TABLES as HISTORY_TABLES
from openexecutive.orchestrator.take_the_lead import TABLES as TAKE_THE_LEAD_TABLES
from openexecutive.people.roster_requests import TABLES as ROSTER_REQUEST_TABLES

PER_COMPANY_TABLES: tuple[str, ...] = (
    # Derived caches. Regenerable, but /today serves them from cache, so a
    # surviving row renders the outgoing company's text under the incoming one.
    #
    # Do not assume person_insights' input_hash makes it self-guarding. The hash
    # (people.insights.build_insight_input_hash) covers role, is_principal,
    # status, awaiting_count, awaiting_reply_count, overdue, on_leave_until,
    # reachable_now, authority_scope, department_slugs, the hour-bucketed
    # soonest_sla_at / oldest_awaiting_reply_at / next_window_at /
    # last_contact_at, and the UTC day — every one of them a per-person signal,
    # and NOT the person's name, the person's id, or the company. The cache key
    # is person_id, a reused autoincrement PK. So an outgoing principal and a
    # freshly seeded incoming principal hash identically whenever those signals
    # coincide: both principals, same role string, no awaiting work, no leave,
    # same default departments, same UTC day. That is an ordinary steady state
    # for a seeded slot, not a collision.
    "briefing_narrative",
    "person_insights",
    # Chat history and what the Executive learned from it.
    "chat_messages",
    "chat_topics",
    "session_shares",
    "session_drive_files",
    "session_drive_searches",
    "sessions",
    "chat_review_turns",
    "chat_review_days",
    "chat_review_weeks",
    "chat_review_lessons",
    "chat_review_lesson_uses",
    "chat_review_lesson_surfaces",
    "chat_review_replays",
    "chat_review_replay_runs",
    "knowledge_gaps",
    "knowledge_gap_asks",
    "worked_out",
    "worked_out_passes",
    # Episodic memory and standing facts.
    "decisions",
    "initiatives",
    "advice_given",
    "facts",
    "facts_meta",
    "fact_confirmations",
    "fact_approval_rules",
    "scheduled_actions",
    "outbound_context",
    "voice_personas",
    # Alerts and the Today page.
    "alerts",
    "alert_approvals",
    "alert_held_actions",
    "mute_topics",
    "user_preferences",
    "brief_requests",
    "briefing_point_dismissed",
    "briefing_point_seen",
    "briefing_point_source",
    "briefing_reader_seen",
    "right_now_dismissed",
    # Solo/team mode, the user's time zone and the principal's role.
    "workspace_settings",
    "outside_content_mode",
    # Runs, audit and authored workflows.
    "workflow_runs",
    "workflow_approved_targets",
    "dynamic_workflows",
    "saved_tools",
    "saved_tool_versions",
    "saved_tool_runs",
    "repeat_asks",
    "repeat_ask_offers",
    "audit_log",
    "eval_runs",
    "eval_scenarios",
    "artifact_shares",
    # External monitoring.
    "external_signals",
    "watchlist",
    "watchlist_declines",
    "watchlist_policy_outcomes",
    "page_watch_state",
    # An in-progress setup interview: the unedited description and answers.
    "onboarding_interviews",
    # Attunement and gated decisions.
    "attunement_usage",
    "proactive_outcomes",
    "attunement_profiles",
    "attunement_profile_history",
    "decision_instances",
    "decision_class_state",
    # Act as me, Always in the loop, Take the lead.
    *DELEGATION_TABLES,
    sign_in_notice.TABLE,
    *HISTORY_TABLES,
    *TAKE_THE_LEAD_TABLES,
    # Chat-app and mail state: who wrote, what was asked, which cards went out.
    "email_thread_replies",
    "chat_asks",
    "chat_card_pushes",
    "telegram_chats",
    # The Agent Council's per-company configuration.
    "agent_override_history",
    "agent_overrides",
    # Knowledge review decisions (a rejection suppresses retrieval).
    "review_annotations",
    "review_items",
    # People and what is known about them.
    *ROSTER_REQUEST_TABLES,
    "held_messages",
    "contact_changes",
    "contact_keeper_blocks",
    "contact_keeper_reads",
    "contact_keeper_state",
    "contact_notes",
    "contact_note_tries",
    "reply_habit_kept",
    "person_notification_settings",
    "person_details",
    "person_emails",
    "person_authority_scope",
    "person_availability",
    "people",
    # Departments.
    "department_goals",
    "department_notes",
    "departments",
    "departments_meta",
    # Legacy talent / staff-onboarding tables. Nothing creates or writes these
    # any more, but upgraded installs may still hold them with candidate PII
    # (names, employers, screening summaries, offer comp). Wiping skips a table
    # that does not exist.
    "onboarding_tasks",
    "onboarding_plans",
    "onboarding_templates",
    "offers",
    "candidates",
    "engagements",
)

# Tables that hold nothing about any one company, with the reason. Wipes leave
# them alone; a restore of saved rows brings back the saved migration markers
# (MIGRATION_MARKER_TABLES below), and a whole-DB slot restore carries the
# slot's own copy of all of them except the operator switches in
# ``clients.slots._GLOBAL_TABLES``.
COMPANY_INDEPENDENT_TABLES: dict[str, str] = {
    "generated_fixtures": "the operator's demo-fixture library, kept across every swap",
    "executive_control": "the operator's pause switch; a swap must never un-pause the Executive",
    "architecture_sections": "the /architecture page cache, derived from the repo, not company data",
    "app_migrations": "markers of the one-shot data sweeps already run on this database file",
    "monitoring_migrations": "markers of the monitoring store's one-shot migrations on this file",
    "review_meta": "markers of the review store's one-shot backfills on this database file",
    "departments_new": "scratch table of a departments migration, renamed in the same transaction",
    "department_goals_new": "scratch table of a goals migration, renamed in the same transaction",
}

# Company-independent tables that record what was done to the rows of this
# database file (one-shot sweeps and backfills), so they belong with those rows:
# when a saved copy's per-company rows are restored, its markers come back too,
# or a sweep marked done on the outgoing rows would never run on the restored
# ones.
MIGRATION_MARKER_TABLES: tuple[str, ...] = ("app_migrations", "monitoring_migrations", "review_meta")


def _existing_tables(conn: sqlite3.Connection, schema: str = "main") -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            f"SELECT name FROM {schema}.sqlite_master WHERE type='table'"  # noqa: S608 — fixed schema names
        )
    }


def _children_first(conn: sqlite3.Connection, tables: list[str]) -> list[str]:
    """``tables`` ordered so no table is deleted before one that references it."""
    wanted = set(tables)
    parents = {
        t: {row[2] for row in conn.execute(f'PRAGMA foreign_key_list("{t}")')} & wanted - {t}
        for t in tables
    }
    ordered: list[str] = []
    remaining = list(tables)
    while remaining:
        ready = [t for t in remaining if not any(t in parents[o] for o in remaining)]
        # A reference cycle: delete in the given order and let the foreign key
        # error surface rather than guess.
        ready = ready or remaining[:1]
        ordered += ready
        remaining = [t for t in remaining if t not in ready]
    return ordered


def wipe_tables(db_path: Path, tables: tuple[str, ...] = PER_COMPANY_TABLES) -> dict[str, int]:
    """Delete every row of each of ``tables`` that exists; return rows deleted per table.

    Never creates the database, and skips a table this database does not have
    (a store not initialized yet, a legacy table never created). Runs with
    ``PRAGMA foreign_keys = ON`` like the rest of the app, children first.
    """
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        existing = _existing_tables(conn)
        present = [t for t in dict.fromkeys(tables) if t in existing]
        counts: dict[str, int] = {}
        for table in _children_first(conn, present):
            counts[table] = int(conn.execute(f'DELETE FROM "{table}"').rowcount)  # noqa: S608 — registry names
        conn.commit()
    finally:
        conn.close()
    return counts


def swap_tables(
    db_path: Path,
    tables: tuple[str, ...],
    saved_path: Path | None = None,
    *,
    restore_also: tuple[str, ...] = (),
) -> tuple[dict[str, int], dict[str, int]]:
    """Wipe ``tables``, then restore them (and ``restore_also``) from a saved copy.

    One transaction, so no other writer sees the tables empty and a failure
    leaves them as they were. Returns ``(rows deleted, rows restored)`` per
    table. With ``saved_path`` None this only wipes.

    For a saved copy of this same database: a table the copy lacks is left
    empty; one the live database lacks is created from the copy's own
    definition. Only the columns both have are copied, so a copy from before a
    column was added still restores. Foreign keys are off for the transaction:
    it writes back a consistent snapshot as it was, orphans included, and
    every table that references a swapped one is swapped too (a test checks).
    Raises ``FileNotFoundError`` if ``saved_path`` does not exist, and
    ``ValueError`` (changing nothing) if it holds none of ``tables``: a caller
    that meant to restore must never be left with the tables wiped instead.
    """
    if saved_path is not None and not saved_path.exists():
        raise FileNotFoundError(saved_path)
    if not db_path.exists():
        return {}, {}
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        if saved_path is not None:
            conn.execute("ATTACH DATABASE ? AS saved", (str(saved_path),))
        conn.execute("BEGIN IMMEDIATE")
        try:
            live = _existing_tables(conn)
            cleared: dict[str, int] = {}
            restored: dict[str, int] = {}
            targets = list(dict.fromkeys(tables))
            if saved_path is not None:
                targets += [t for t in restore_also if t not in targets]
            for table in targets:
                if table in live:
                    cleared[table] = int(conn.execute(f'DELETE FROM main."{table}"').rowcount)  # noqa: S608
            if saved_path is not None:
                saved_sql = dict(
                    conn.execute("SELECT name, sql FROM saved.sqlite_master WHERE type='table'").fetchall()
                )
                if not any(t in saved_sql for t in tables):
                    # An empty or truncated copy: restoring it would be a wipe.
                    raise ValueError(f"{saved_path} holds none of the tables to restore")
                for table in targets:
                    if table not in saved_sql:
                        continue
                    if table not in live:
                        conn.execute(saved_sql[table])
                    live_cols = {r[1] for r in conn.execute(f'PRAGMA main.table_info("{table}")')}
                    cols = [
                        r[1] for r in conn.execute(f'PRAGMA saved.table_info("{table}")') if r[1] in live_cols
                    ]
                    col_list = ", ".join(f'"{c}"' for c in cols)
                    restored[table] = int(
                        conn.execute(
                            f'INSERT INTO main."{table}" ({col_list}) SELECT {col_list} FROM saved."{table}"'  # noqa: S608
                        ).rowcount
                    )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        if saved_path is not None:
            conn.execute("DETACH DATABASE saved")
    finally:
        conn.close()
    return cleared, restored
