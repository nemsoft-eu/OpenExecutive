"""Which SQLite tables belong to one company — the list every company swap wipes.

Three paths replace the live company: switching into a blank client slot
(``clients.slots``), the factory reset and fixture load/unload (both in
``cli.fixture_loader``). Each used to keep its own hand-written table list, and
a table added to one list and not the others leaked one company's rows into the
next (fixture load kept ``department_notes``, so a shared department slug served
the previous company's notes). They now all wipe from this module.

Every table the app creates is in exactly one of :data:`PER_COMPANY_TABLES` or
:data:`COMPANY_INDEPENDENT_TABLES`; ``tests/unit/test_company_tables.py`` scans
the source for ``CREATE TABLE`` and fails on a table that is in neither. A new
table therefore needs a decision here, not an edit to each swap path.

A path may leave a per-company table in place only through
:data:`KEPT_ON_EVERY_SWAP` or :data:`KEPT_ON_FIXTURE_SWAP`, each entry with its
reason. The order of :data:`PER_COMPANY_TABLES` does not matter:
:func:`wipe_per_company_tables` reads the foreign keys from the database and
deletes children before parents.

The fixture path wipes what its YAML backup (``_user_backup/``) does not carry
as well, so those tables come back empty after a demo load and unload; unload
restores only what the backup's files hold.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Collection
from pathlib import Path

from openexecutive.delegation import sign_in_notice
from openexecutive.delegation.schema import TABLES as DELEGATION_TABLES
from openexecutive.memory import department_notes
from openexecutive.memory.history_schema import TABLES as HISTORY_TABLES
from openexecutive.orchestrator.take_the_lead import TABLES as TAKE_THE_LEAD_TABLES
from openexecutive.people.roster_requests import TABLES as ROSTER_REQUEST_TABLES

PER_COMPANY_TABLES: tuple[str, ...] = (
    # Derived caches. Regenerable, but /today serves them from cache and only
    # regenerates in a background task, so a surviving row renders the
    # outgoing company's text under the incoming one.
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
    # Episodic memory and standing facts. Decisions live in ``facts`` now
    # (``decisions`` is the older table), and ``facts_meta`` holds the
    # copied-once marker, so emptying both also does what
    # ``memory.facts.drop_decisions`` does.
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
    # Runs, audit, and what was authored for this company.
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
    # Chat-app and mail state: who wrote, what was asked, which cards went out,
    # and what the Executive mailed back (found by the recipient's address).
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
    # Departments and each one's own notes.
    "department_goals",
    department_notes.TABLE,
    "departments",
    "departments_meta",
    # Legacy talent / staff-onboarding tables. Nothing creates or writes these
    # any more, but upgraded installs may still hold them with candidate PII
    # (names, employers, screening summaries, offer comp). The wipe skips a
    # table that does not exist.
    "onboarding_tasks",
    "onboarding_plans",
    "onboarding_templates",
    "offers",
    "candidates",
    "engagements",
)

# Tables that hold nothing about any one company, with the reason. No swap
# wipes them.
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

# Per-company tables that no swap path wipes, with the reason. Each is a known
# leak between companies, kept until the backups carry them.
KEPT_ON_EVERY_SWAP: dict[str, str] = {
    "dynamic_workflows": (
        "authored, not derived, and no backup carries it: the fixture backup and "
        "a blank slot's first-activation snapshot hold only profile, docs, "
        "memory.json, people, departments and workspace, so a wipe would destroy "
        "the operator's own workflows"
    ),
    "eval_scenarios": "authored eval scenarios, not carried by any backup (as dynamic_workflows)",
    "saved_tools": (
        "step scripts the Executive kept to run again by name: authored, and not "
        "carried by any backup (as dynamic_workflows)"
    ),
    "saved_tool_versions": "earlier versions of those saved tools (as saved_tools)",
}

# Per-company tables fixture load and unload leave in place, on top of
# KEPT_ON_EVERY_SWAP. The blank switch and the factory reset wipe them.
KEPT_ON_FIXTURE_SWAP: dict[str, str] = {
    "audit_log": (
        "load and unload keep the audit history by design; only the research "
        "run-history rows that gate the seeded scan are cleared "
        "(monitoring.research.scheduler.clear_research_run_history)"
    ),
    "workspace_settings": (
        "set by the path itself from workspace.yaml (cli.fixture_loader._apply_workspace): "
        "a fixture resets it to the defaults first, and a user backup from before "
        "workspace.yaml keeps the live row"
    ),
    "review_items": (
        "review state of the knowledge files, which a fixture load does not swap; "
        "with no row a rejected upload of the user's would be retrievable again"
    ),
    "review_annotations": "SME notes on those same knowledge files (as review_items)",
    "outside_content_mode": (
        "the owner's Safety level; no backup carries it and a missing row reads as "
        "Balanced, so a wipe would silently lower a Careful owner's checking after a demo"
    ),
}


def _children_first(conn: sqlite3.Connection, tables: list[str]) -> list[str]:
    """``tables`` ordered so no table is deleted before one that references it."""
    # SQLite table names are case-insensitive, and a REFERENCES clause keeps
    # whatever case it was written in, so match parents by their lowered name.
    by_lower = {t.lower(): t for t in tables}
    parents = {
        t: {
            by_lower[ref]
            for ref in (row[2].lower() for row in conn.execute(f'PRAGMA foreign_key_list("{t}")'))
            if ref in by_lower and by_lower[ref] != t
        }
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


def wipe_per_company_tables(db_path: Path, *, keep: Collection[str] = ()) -> dict[str, int]:
    """Delete every row of each per-company table but ``keep``; return rows deleted per table.

    Never creates the database, and skips a table this database does not have
    (a store not initialized yet, a legacy table never created). Runs with
    ``PRAGMA foreign_keys = ON`` like the rest of the app, children first, in
    one transaction.
    """
    if not db_path.exists():
        return {}
    # The slot restore's 30 seconds: a scheduler or bot write holding the lock
    # must not abort a swap.
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        present = [t for t in PER_COMPANY_TABLES if t in existing and t not in keep]
        counts: dict[str, int] = {}
        for table in _children_first(conn, present):
            counts[table] = int(conn.execute(f'DELETE FROM "{table}"').rowcount)  # noqa: S608 — registry names
        conn.commit()
    finally:
        conn.close()
    # The Agent Council's overrides are cached in-process with no expiry, so
    # the wiped rows would otherwise keep applying until the next restart.
    from openexecutive.agents.overrides import invalidate_cache

    invalidate_cache()
    return counts
