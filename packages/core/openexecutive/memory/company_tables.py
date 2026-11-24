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

The wipe itself (:func:`wipe_per_company_tables`) empties every table in the
database that is not company-independent, read from ``sqlite_master`` rather
than from :data:`PER_COMPANY_TABLES`. So a table that ships unclassified anyway
is wiped, the safe default, instead of leaking; the test is what keeps the
classification honest. A company-independent table is one a swap must never
empty: an instance-level row, or a one-time migration's marker (emptied, the
migration would run again over the next company's data).

A path may leave a per-company table in place only through
:data:`KEPT_ON_EVERY_SWAP` or :data:`KEPT_ON_FIXTURE_SWAP`, each entry with its
reason. The order of :data:`PER_COMPANY_TABLES` does not matter: the wipe reads
the foreign keys from the database and deletes children before parents.

The fixture path wipes what its YAML backup (``_user_backup/``) does not carry
as well; unload then puts the user's whole saved database
(``_user_backup/state.db``) back over it.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Collection
from pathlib import Path

from openexecutive.delegation import sign_in_notice
from openexecutive.delegation.schema import TABLES as DELEGATION_TABLES
from openexecutive.delegation.schema import UPGRADES_TABLE as DELEGATION_UPGRADES_TABLE
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
    # Names of the Drive files this company's chats opened (memory.drive_reads).
    "drive_file_names",
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
    "scheduled_actions",
    "outbound_context",
    "voice_personas",
    # Alerts and the Today page.
    "alerts",
    "alert_approvals",
    "alert_held_actions",
    # A card's options and the talk on it (alerts.card_talk), its plan's
    # changes (alerts.replan), one card per topic and the mail it came from
    # (alerts.one_card), and whose own chat a decision's card went to
    # (alerts.own_chat): all keyed on this company's alerts and decisions.
    "alert_card_options",
    "alert_card_turns",
    "alert_plan_changes",
    "card_topics",
    "mail_conversations",
    "decision_chat_owners",
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
    "artifact_shares",
    # External monitoring. mailbox_executive_mail references watchlist; the
    # wipe deletes it first.
    "external_signals",
    "mailbox_executive_mail",
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
    # Act as me, Always in the loop, Take the lead. Act as me's upgrade
    # markers are company-independent (below).
    *(t for t in DELEGATION_TABLES if t != DELEGATION_UPGRADES_TABLE),
    sign_in_notice.TABLE,
    *HISTORY_TABLES,
    *TAKE_THE_LEAD_TABLES,
    # Chat-app and mail state: who wrote, what was asked, which cards went out,
    # and what the Executive mailed back (found by the recipient's address).
    "email_thread_replies",
    "progress_updates",
    "chat_asks",
    "chat_card_pushes",
    "telegram_chats",
    # Fingerprints of the mail answered for a day (integrations.email_repeats)
    # and the mail the Executive sent for someone (orchestrator.teammate_mail).
    "email_repeats",
    "mail_sent_for",
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
    "eval_scenarios": (
        "the eval scenarios people wrote: the instance's, carried across every "
        "client switch and unload (clients.slots._GLOBAL_TABLES)"
    ),
    "install_owner": (
        "who runs the install, whichever company is live (people.install_owner); "
        "a reset writes the principal back against it"
    ),
    "app_migrations": "markers of the one-shot data sweeps already run on this database file",
    "monitoring_migrations": "markers of the monitoring store's one-shot migrations on this file",
    "review_meta": "markers of the review store's one-shot backfills on this database file",
    "dynamic_workflows_meta": (
        "marker of the one-shot move of workflow cadences to company time "
        "(workflows.dynamic_store); emptied, it would shift the next company's again"
    ),
    "departments_meta": (
        "marker of the one-shot move of check-in cadences to company time "
        "(departments.store._cadences_to_company_time); emptied, the move would run "
        "again over the next company's departments and shift their check-ins. It "
        "also holds the default-departments-seeded marker, which only "
        "seed_default_departments reads, and nothing calls that at runtime"
    ),
    DELEGATION_UPGRADES_TABLE: (
        "markers of Act as me's one-shot upgrades (delegation.schema); emptied, "
        "_keep_the_old_opt_in would run again and put the next company's people "
        "In training"
    ),
    "departments_new": "scratch table of a departments migration, renamed in the same transaction",
    "department_goals_new": "scratch table of a goals migration, renamed in the same transaction",
}

# Per-company tables that no swap path wipes, with the reason. None at present:
# the user's snapshot and each client slot hold the whole database
# (``state.db``), so a reset or a blank switch may wipe authored tables too.
KEPT_ON_EVERY_SWAP: dict[str, str] = {}

# Per-company tables fixture load and unload leave in place, on top of
# KEPT_ON_EVERY_SWAP. The blank switch and the factory reset wipe them.
KEPT_ON_FIXTURE_SWAP: dict[str, str] = {
    "dynamic_workflows": (
        "the operator's own workflows: authored, not derived. They stay live "
        "through a demo, and unload's state.db restore brings back the saved copy; "
        "a backup taken before state.db existed holds none, so a wipe here would "
        "destroy them"
    ),
    "saved_tools": (
        "step scripts the Executive kept to run again by name: authored, kept as "
        "dynamic_workflows is"
    ),
    "saved_tool_versions": "earlier versions of those saved tools (as saved_tools)",
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


# The tables SQLite's full-text search (FTS3/4/5) and R-tree modules keep
# behind each virtual table, as "<virtual table>_<part>".
_SHADOW_TABLE_PARTS = (
    "config", "content", "data", "docsize", "idx", "segdir", "segments",
    "stat", "node", "parent", "rowid",
)


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
    """Delete every row of every table but the company-independent ones and
    ``keep``; return rows deleted per table.

    The tables are read from the database, so one missing from
    :data:`PER_COMPANY_TABLES` is wiped too, and one the database does not
    have (a store not initialized yet, a legacy table) is skipped. SQLite's own
    tables (``sqlite_*``) are never touched, so ids are never handed out twice.
    Never creates the database. Runs with ``PRAGMA foreign_keys = ON`` like the
    rest of the app, children first, in one transaction.
    """
    if not db_path.exists():
        return {}
    # The slot restore's 30 seconds: a scheduler or bot write holding the lock
    # must not abort a swap.
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master"
            " WHERE type = 'table' AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'"
        ).fetchall()
        # A virtual table (full-text search, R-tree) is emptied through itself;
        # its shadow tables hold its structure and break it if emptied directly.
        shadows = {
            f"{name}_{part}"
            for name, sql in rows
            if (sql or "").upper().startswith("CREATE VIRTUAL TABLE")
            for part in _SHADOW_TABLE_PARTS
        }
        present = sorted(
            name
            for name, _sql in rows
            if name not in shadows and name not in COMPANY_INDEPENDENT_TABLES and name not in keep
        )
        counts: dict[str, int] = {}
        for table in _children_first(conn, present):
            counts[table] = int(conn.execute(f'DELETE FROM "{table}"').rowcount)  # noqa: S608 — names from sqlite_master
        conn.commit()
    finally:
        conn.close()
    # The Agent Council's overrides are cached in-process with no expiry, so
    # the wiped rows would otherwise keep applying until the next restart.
    from openexecutive.agents.overrides import invalidate_cache

    invalidate_cache()
    return counts
