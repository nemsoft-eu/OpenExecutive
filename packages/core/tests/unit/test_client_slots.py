"""Client-slot round-trip tests: save → switch → switch back must be lossless.

The slot mechanism's whole contract is "a slot is a faithful save file" —
these tests prove the SQLite state (decisions, scheduled actions, the people
roster), the company artifacts (profile, docs, mcp_servers.json), and the
operator-level tables behave correctly across switches. The vector and
Honcho layers are stubbed: they're side effects of a switch, not part of the
round-trip contract under test.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openexecutive.clients import slots
from openexecutive.clients.slots import (
    ClientSlotConflictError,
    ClientSlotNotFoundError,
    activate_client_slot,
    create_client_slot,
    delete_client_slot,
    get_active_client,
    list_client_slots,
    save_active_client,
)


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Isolated company dir + episodic DB, with vector/Honcho layers stubbed."""
    company = tmp_path / "company"
    company.mkdir()
    settings = SimpleNamespace(
        company_profile_path=company / "profile.yaml",
        vector_store_path=tmp_path / "chroma",
        mcp_servers_config_path=company / "mcp_servers.json",
        honcho_workspace_id="default-ws",
    )

    db_path = tmp_path / "episodic.db"
    from openexecutive.departments import store as dept_store
    from openexecutive.memory import episodic
    from openexecutive.people import store as people_store

    monkeypatch.setattr(episodic, "DB_PATH", db_path)
    monkeypatch.setattr(people_store, "DB_PATH", db_path)
    monkeypatch.setattr(dept_store, "DB_PATH", db_path)
    episodic.initialize_db(db_path)
    people_store.initialize_db(db_path)
    dept_store.initialize_db(db_path)

    async def _no_vector(_settings: Any, _app_state: Any) -> int:
        return 0

    reseed_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(slots, "_rebuild_vector_state", _no_vector)
    monkeypatch.setattr(slots, "_set_honcho_client_workspace", lambda _slug: None)
    monkeypatch.setattr(
        slots, "_reseed_blank_defaults", lambda **kw: reseed_calls.append(kw)
    )
    return SimpleNamespace(
        settings=settings, db_path=db_path, company=company, reseed_calls=reseed_calls
    )


def _insert_decision(db_path: Path, summary: str) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO decisions (timestamp, domain, summary, rationale, outcome, tags, department) "
            "VALUES ('2026-06-01T00:00:00', 'strategy', ?, '', '', '', '')",
            (summary,),
        )
        conn.commit()
    finally:
        conn.close()


def _decision_summaries(db_path: Path) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return [r[0] for r in conn.execute("SELECT summary FROM decisions").fetchall()]
    finally:
        conn.close()


def _seed_live_company(env: SimpleNamespace, name: str = "Acme Corp") -> None:
    env.settings.company_profile_path.write_text(f"name: {name}\n")
    docs = env.company / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "strategy.md").write_text(f"# {name} strategy")
    env.settings.mcp_servers_config_path.write_text('{"servers": {"crayon": {}}}')
    _insert_decision(env.db_path, f"{name} decision")


async def test_create_from_current_captures_state_and_activates(env: SimpleNamespace) -> None:
    _seed_live_company(env, "Acme Corp")

    result = await create_client_slot(
        env.settings, display_name="Acme Corp", source="current"
    )

    assert result["slug"] == "acme_corp"
    assert result["active"] is True
    assert get_active_client(env.settings) == "acme_corp"

    slot = env.company / "_client_slots" / "acme_corp"
    assert (slot / "profile.yaml").exists()
    assert (slot / "docs" / "strategy.md").exists()
    assert (slot / "mcp_servers.json").exists()
    assert (slot / "state.db").exists()

    listed = list_client_slots(env.settings)
    assert [s["slug"] for s in listed] == ["acme_corp"]
    assert listed[0]["has_state"] is True
    assert listed[0]["has_mcp_config"] is True


async def test_switch_round_trip_is_lossless(env: SimpleNamespace) -> None:
    # Client A: full company with a roster entry and a scheduled action.
    _seed_live_company(env, "Acme Corp")
    from openexecutive.people import store as people_store

    people_store.upsert_person(
        full_name="Dana Acme",
        role="CFO",
        email="dana@acme.example",
        db_path=env.db_path,
    )
    conn = sqlite3.connect(str(env.db_path))
    conn.execute(
        "INSERT INTO scheduled_actions (created_at, run_at, channel, channel_ref, "
        "intent_text, status, attempts, last_error, department, kind) "
        "VALUES ('2026-06-01T00:00:00', '2099-01-01T00:00:00', 'any', '', "
        "'follow up with Acme board', 'pending', 0, '', '', 'ad_hoc')"
    )
    conn.commit()
    conn.close()

    await create_client_slot(env.settings, display_name="Acme Corp", source="current")
    await create_client_slot(env.settings, display_name="Beta Inc", source="blank")

    # Switch to Beta: live state must be Beta's (empty), not Acme's.
    await activate_client_slot(env.settings, "beta_inc")
    assert get_active_client(env.settings) == "beta_inc"
    assert _decision_summaries(env.db_path) == []
    # Acme's roster must not leak into Beta — the other half of the witness.
    assert people_store.find_person_by_email("dana@acme.example", env.db_path) is None
    assert not env.settings.mcp_servers_config_path.exists()
    assert "Beta Inc" in env.settings.company_profile_path.read_text()

    # Do Beta-specific work, then switch back to Acme.
    _insert_decision(env.db_path, "Beta decision")
    (env.company / "docs").mkdir(exist_ok=True)
    (env.company / "docs" / "beta.md").write_text("# Beta")

    await activate_client_slot(env.settings, "acme_corp")
    assert get_active_client(env.settings) == "acme_corp"
    assert _decision_summaries(env.db_path) == ["Acme Corp decision"]
    assert (env.company / "docs" / "strategy.md").exists()
    assert not (env.company / "docs" / "beta.md").exists()
    assert env.settings.mcp_servers_config_path.exists()
    assert people_store.find_person_by_email("dana@acme.example", env.db_path) is not None
    conn = sqlite3.connect(str(env.db_path))
    actions = conn.execute("SELECT intent_text FROM scheduled_actions").fetchall()
    conn.close()
    assert actions == [("follow up with Acme board",)]

    # And Beta's work survived its park.
    await activate_client_slot(env.settings, "beta_inc")
    assert _decision_summaries(env.db_path) == ["Beta decision"]
    assert (env.company / "docs" / "beta.md").exists()
    assert not (env.company / "docs" / "strategy.md").exists()


async def test_generated_fixtures_survive_switches(env: SimpleNamespace) -> None:
    """The fixture library is operator-level — never swapped with the client."""
    from openexecutive.fixtures import store as fixtures_store

    fixtures_store.initialize_db(env.db_path)
    _seed_live_company(env)
    await create_client_slot(env.settings, display_name="Acme", source="current")
    await create_client_slot(env.settings, display_name="Beta", source="blank")

    conn = sqlite3.connect(str(env.db_path))
    conn.execute(
        "INSERT INTO generated_fixtures (name, display_name, created_at, updated_at) "
        "VALUES ('halcyon_test', 'Halcyon Test', '2026-06-01', '2026-06-01')"
    )
    conn.commit()
    conn.close()

    await activate_client_slot(env.settings, "beta")

    conn = sqlite3.connect(str(env.db_path))
    rows = conn.execute("SELECT name FROM generated_fixtures").fetchall()
    conn.close()
    assert rows == [("halcyon_test",)]


async def test_first_activation_snapshots_original_company(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Activating with no active client preserves the user's company first."""
    _seed_live_company(env, "My Real Co")
    snapshots: list[str] = []
    monkeypatch.setattr(
        slots, "snapshot_user_state", lambda s: snapshots.append("taken")
    )

    await create_client_slot(env.settings, display_name="Beta", source="blank")
    await activate_client_slot(env.settings, "beta")

    assert snapshots == ["taken"]
    assert get_active_client(env.settings) == "beta"


async def test_create_from_current_snapshots_original_company(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entering client mode via create-from-current must leave a _user_backup
    restore point so POST /fixtures/unload remains a working exit path."""
    _seed_live_company(env, "My Real Co")
    snapshots: list[str] = []
    monkeypatch.setattr(
        slots, "snapshot_user_state", lambda s: snapshots.append("taken")
    )

    await create_client_slot(env.settings, display_name="My Real Co", source="current")
    assert snapshots == ["taken"]

    # An older backup is never reused: a later create saves your company
    # again, so unload brings back this copy and not the first one.
    backup = env.company / "_user_backup"
    backup.mkdir(exist_ok=True)
    (backup / "profile.yaml").write_text("name: Original\n")
    # Exit client mode, then re-enter via a fresh create-from-current.
    (env.company / "_client_slots" / ".active_client").unlink()
    await create_client_slot(env.settings, display_name="Second", source="current")
    assert snapshots == ["taken", "taken"]


async def test_create_from_current_stops_when_the_snapshot_fails(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entering client mode without a fresh copy of your company would let
    unload bring back an older one, so a failed snapshot stops the create
    before any client exists."""
    _seed_live_company(env, "My Real Co")

    def _fail(_settings: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(slots, "snapshot_user_state", _fail)
    with pytest.raises(slots.ClientSnapshotError, match="Nothing was changed"):
        await create_client_slot(env.settings, display_name="Acme", source="current")
    assert not (env.company / "_client_slots" / "acme").exists()
    assert get_active_client(env.settings) is None


async def test_save_active_client_checkpoints_without_switching(env: SimpleNamespace) -> None:
    _seed_live_company(env)
    await create_client_slot(env.settings, display_name="Acme", source="current")

    _insert_decision(env.db_path, "late decision")
    result = await save_active_client(env.settings)
    assert result["saved"] is True

    # The slot's state.db now contains the late decision.
    slot_db = env.company / "_client_slots" / "acme" / "state.db"
    conn = sqlite3.connect(str(slot_db))
    rows = conn.execute("SELECT summary FROM decisions ORDER BY id").fetchall()
    conn.close()
    assert ("late decision",) in rows


async def test_save_with_no_active_client_raises(env: SimpleNamespace) -> None:
    with pytest.raises(ClientSlotConflictError):
        await save_active_client(env.settings)


async def test_operations_refuse_while_fixture_active(env: SimpleNamespace) -> None:
    backup = env.company / "_user_backup"
    backup.mkdir()
    (backup / ".fixture_active").write_text("halcyon_motors")

    with pytest.raises(ClientSlotConflictError):
        await create_client_slot(env.settings, display_name="Acme", source="current")
    with pytest.raises(ClientSlotConflictError):
        await save_active_client(env.settings)


async def test_create_from_current_refused_when_client_active(env: SimpleNamespace) -> None:
    _seed_live_company(env)
    await create_client_slot(env.settings, display_name="Acme", source="current")
    with pytest.raises(ClientSlotConflictError):
        await create_client_slot(env.settings, display_name="Other", source="current")


async def test_delete_refuses_active_then_deletes_parked(env: SimpleNamespace) -> None:
    _seed_live_company(env)
    await create_client_slot(env.settings, display_name="Acme", source="current")
    await create_client_slot(env.settings, display_name="Beta", source="blank")

    with pytest.raises(ClientSlotConflictError):
        await delete_client_slot(env.settings, "acme")

    result = await delete_client_slot(env.settings, "beta")
    assert result == {"deleted": True, "slug": "beta"}
    assert [s["slug"] for s in list_client_slots(env.settings)] == ["acme"]


async def test_activate_unknown_slug_raises_not_found(env: SimpleNamespace) -> None:
    with pytest.raises(ClientSlotNotFoundError):
        await activate_client_slot(env.settings, "nope")


async def test_activate_already_active_is_a_noop(env: SimpleNamespace) -> None:
    _seed_live_company(env)
    await create_client_slot(env.settings, display_name="Acme", source="current")
    result = await activate_client_slot(env.settings, "acme")
    assert result == {"slug": "acme", "already_active": True}


async def test_sentinel_garbage_is_ignored(env: SimpleNamespace) -> None:
    root = env.company / "_client_slots"
    root.mkdir()
    (root / ".active_client").write_text("../../etc/passwd")
    assert get_active_client(env.settings) is None


# ── Generated (engagement-intake) seed slots ────────────────────────────────


def _intake_bundle() -> dict[str, Any]:
    """A minimal valid engagement bundle (FixtureBundle shape)."""
    return {
        "profile": {
            "name": "Meridian Solar",
            "industry": "Commercial solar",
            "stage": "Private",
            "mission": "Margin-positive installs.",
        },
        "people": [
            {"full_name": "Dana Reyes", "role": "CEO", "is_principal": True},
            {"full_name": "Lee Park", "role": "VP Ops"},
        ],
        "departments": [
            {
                "slug": "operations",
                "title": "Operations",
                "head_person_name": "Lee Park",
            }
        ],
        "memory": {
            "decisions": [
                {
                    "timestamp": "2026-05-01T00:00:00",
                    "domain": "operations",
                    "summary": "Standardized on single-vendor inverters",
                }
            ],
            "initiatives": [],
            "advice_given": [],
            "alerts": [],
        },
        "docs": [
            {"filename": "intake_brief.md", "content": "# Intake brief"},
            {"filename": "open_questions.md", "content": "# Open questions"},
        ],
    }


async def test_generated_seed_slot_create_does_not_touch_live(env: SimpleNamespace) -> None:
    _seed_live_company(env, "My Real Co")

    result = await create_client_slot(
        env.settings,
        display_name="Meridian Solar",
        source="generated",
        bundle=_intake_bundle(),
        intake_description="kickoff call notes",
    )

    assert result["origin"] == "generated"
    assert result["active"] is False
    assert get_active_client(env.settings) is None
    # Live state untouched.
    assert "My Real Co" in env.settings.company_profile_path.read_text()
    assert _decision_summaries(env.db_path) == ["My Real Co decision"]

    slot = env.company / "_client_slots" / "meridian_solar"
    assert (slot / "profile.yaml").exists()
    assert (slot / "people.yaml").exists()
    assert (slot / "departments.yaml").exists()
    assert (slot / "memory.json").exists()
    assert sorted(p.name for p in (slot / "docs").glob("*.md")) == [
        "intake_brief.md",
        "open_questions.md",
    ]
    assert not (slot / "state.db").exists()

    listed = list_client_slots(env.settings)
    assert listed[0]["origin"] == "generated"


# "≈", "→" and "✅" sit outside cp1252, so an unencoded write_text() raises
# UnicodeEncodeError on a default Windows install. Accented Latin-1 characters
# alone would NOT reproduce it — they encode fine in cp1252.
_NON_ASCII_DOC = "Orçamento ≈ R$ 300 → prioridade: presença digital ✅"


async def test_generated_slot_pins_encoding_on_every_text_write(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slot creation must never write or read text at the platform default.

    ``Path.write_text()`` with no ``encoding`` uses the C-level locale
    (cp1252 on a default Windows install), which raised mid-write and
    surfaced as ``ClientSlotError: Failed to write client slot: 'charmap'
    codec can't encode ...`` for any client whose docs were not
    cp1252-representable. Doc bodies are the only artifact written verbatim —
    profile/people/departments YAML and memory.json go through
    ``yaml.safe_dump`` / ``json.dumps``, which escape non-ASCII and are pure
    ASCII on disk — but every site in this path is pinned so the next verbatim
    write is safe by default.

    This asserts the *contract* (each call passes an explicit encoding) rather
    than simulating a locale, because the locale cannot be faked from Python:
    patching ``locale.getencoding`` does not affect ``Path.write_text``, which
    reads the C-level locale. A symptom-based test would therefore be inert on
    the UTF-8 platform CI runs on, and would police nothing.
    """
    real_write, real_read = Path.write_text, Path.read_text
    offenders: list[str] = []

    def guarded_write(self, data, encoding=None, errors=None, newline=None):  # type: ignore[no-untyped-def]
        if encoding is None:
            offenders.append(f"write_text: {self.name}")
        return real_write(self, data, encoding=encoding or "utf-8", errors=errors, newline=newline)

    # No `newline` here: Path.read_text only accepts it on Python 3.13+, and CI
    # runs 3.11 — forwarding it unconditionally would TypeError the moment a
    # read_text call enters the guarded path (exactly the regression this guard
    # exists to catch). **kwargs keeps the forwarding correct on every version.
    def guarded_read(self, encoding=None, **kwargs):  # type: ignore[no-untyped-def]
        if encoding is None:
            offenders.append(f"read_text: {self.name}")
        return real_read(self, encoding=encoding or "utf-8", **kwargs)

    monkeypatch.setattr(Path, "write_text", guarded_write)
    monkeypatch.setattr(Path, "read_text", guarded_read)

    bundle = _intake_bundle()
    bundle["docs"] = [{"filename": "plano.md", "content": _NON_ASCII_DOC}]
    await create_client_slot(
        env.settings,
        display_name="Consultório Ação — Psicologia",
        slug="consultorio_acao",
        source="generated",
        bundle=bundle,
    )
    monkeypatch.undo()

    assert offenders == []

    # And the bytes that landed really are UTF-8, losslessly.
    slot = env.company / "_client_slots" / "consultorio_acao"
    assert (slot / "docs" / "plano.md").read_bytes().decode("utf-8") == _NON_ASCII_DOC


async def test_seed_slot_activation_clears_derived_caches(env: SimpleNamespace) -> None:
    """The outgoing client's cached narrative/insights must not survive a switch.

    /today serves the cached briefing narrative unconditionally and only
    regenerates in a background task (api/routes/today.py::_attach_narrative),
    so a surviving row renders the PREVIOUS client's narrative under the
    incoming one — observed live, one client's briefing appearing verbatim
    under another. Both tables are regenerable caches, so wiping is free.
    """
    from openexecutive.briefing import narrative_cache
    from openexecutive.people import insights_cache

    _seed_live_company(env, "Outgoing Co")
    narrative_cache.put(
        narrative_cache.BriefingNarrative(
            scope="principal",
            input_hash="stale-hash",
            narrative_text="**Outgoing Co is straining** — do not show this elsewhere.",
            generated_at="2026-09-01T00:00:00+00:00",
        )
    )
    insights_cache.put(
        insights_cache.PersonInsight(
            person_id=1,
            input_hash="stale-hash",
            insight_text="Outgoing Co founder is unreachable.",
            generated_at="2026-09-01T00:00:00+00:00",
        )
    )
    # Both caches are warm for the outgoing client.
    assert narrative_cache.get("principal") is not None
    assert insights_cache.get(1) is not None

    await create_client_slot(
        env.settings,
        display_name="Incoming Co",
        source="generated",
        bundle=_intake_bundle(),
    )
    await activate_client_slot(env.settings, "incoming_co")

    assert narrative_cache.get("principal") is None
    assert insights_cache.get(1) is None


async def test_switch_preserves_outgoing_caches_in_its_slot(env: SimpleNamespace) -> None:
    """Wiping the caches on switch must not DESTROY them — only unpublish them.

    The wipe has to happen after the outgoing client's VACUUM INTO save-back, so
    its cached narrative lands in state.db and returns on reactivation. Ordering
    the wipe before the save-back would lose it permanently, and the sibling
    test above cannot catch that: there, no client is active, so the save-back
    branch of activate_client_slot never runs.
    """
    from openexecutive.briefing import narrative_cache
    from openexecutive.people import insights_cache

    _seed_live_company(env, "Client A")
    await create_client_slot(env.settings, display_name="Client A", source="current")
    narrative_cache.put(
        narrative_cache.BriefingNarrative(
            scope="principal",
            input_hash="a-hash",
            narrative_text="**Client A narrative**",
            generated_at="2026-09-01T00:00:00+00:00",
        )
    )
    insights_cache.put(
        insights_cache.PersonInsight(
            person_id=1,
            input_hash="a-hash",
            insight_text="Client A insight",
            generated_at="2026-09-01T00:00:00+00:00",
        )
    )
    await save_active_client(env.settings)

    await create_client_slot(env.settings, display_name="Client B", source="blank")
    await activate_client_slot(env.settings, "client_b")
    assert narrative_cache.get("principal") is None

    await activate_client_slot(env.settings, "client_a")
    restored = narrative_cache.get("principal")
    assert restored is not None
    assert restored.narrative_text == "**Client A narrative**"
    restored_insight = insights_cache.get(1)
    assert restored_insight is not None
    assert restored_insight.insight_text == "Client A insight"


async def test_generated_seed_slot_activation_seeds_org_and_memory(
    env: SimpleNamespace,
) -> None:
    _seed_live_company(env, "My Real Co")
    await create_client_slot(env.settings, display_name="My Real Co", source="current")
    await create_client_slot(
        env.settings,
        display_name="Meridian Solar",
        source="generated",
        bundle=_intake_bundle(),
    )

    await activate_client_slot(env.settings, "meridian_solar")

    # Seeded people, departments, and memory landed in the live DB.
    assert _decision_summaries(env.db_path) == [
        "Standardized on single-vendor inverters"
    ]
    conn = sqlite3.connect(str(env.db_path))
    people = sorted(
        r[0] for r in conn.execute("SELECT full_name FROM people").fetchall()
    )
    depts = [
        r[0] for r in conn.execute("SELECT slug FROM departments").fetchall()
    ]
    conn.close()
    assert people == ["Dana Reyes", "Lee Park"]
    assert depts == ["operations"]
    # Only the draft's departments: the boot-time defaults add none.
    assert env.reseed_calls[-1] == {}
    assert "Meridian Solar" in env.settings.company_profile_path.read_text()
    assert (env.company / "docs" / "intake_brief.md").exists()


async def test_seed_slot_save_back_prefers_state_db(env: SimpleNamespace) -> None:
    """Once a seed slot has been saved back, state.db wins over seed files."""
    _seed_live_company(env, "My Real Co")
    await create_client_slot(env.settings, display_name="My Real Co", source="current")
    await create_client_slot(
        env.settings,
        display_name="Meridian Solar",
        source="generated",
        bundle=_intake_bundle(),
    )
    await activate_client_slot(env.settings, "meridian_solar")
    _insert_decision(env.db_path, "engagement week-1 decision")

    # Park Meridian (save-back writes state.db), then return.
    await activate_client_slot(env.settings, "my_real_co")
    slot = env.company / "_client_slots" / "meridian_solar"
    assert (slot / "state.db").exists()
    assert (slot / "people.yaml").exists()  # birth record kept

    await activate_client_slot(env.settings, "meridian_solar")
    # state.db restore: both the seeded and the new decision survive, and the
    # seeders did NOT run again (people not duplicated).
    assert sorted(_decision_summaries(env.db_path)) == [
        "Standardized on single-vendor inverters",
        "engagement week-1 decision",
    ]
    conn = sqlite3.connect(str(env.db_path))
    n_people = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
    conn.close()
    assert n_people == 2


async def test_generated_requires_valid_bundle(env: SimpleNamespace) -> None:
    with pytest.raises(slots.ClientSlotError):
        await create_client_slot(
            env.settings, display_name="X", source="generated", bundle=None
        )
    # Referential failure (department head not in roster) → rejected, no husk.
    bad = _intake_bundle()
    bad["departments"][0]["head_person_name"] = "Ghost"
    with pytest.raises(slots.ClientSlotError):
        await create_client_slot(
            env.settings, display_name="Ghost Co", source="generated", bundle=bad
        )
    assert not (env.company / "_client_slots" / "ghost_co").exists()


async def test_executive_pause_survives_switch_to_a_pre_pause_slot(
    env: SimpleNamespace,
) -> None:
    """The pause switch is operator-level, and a slot saved before the switch
    existed (its state.db has no `executive_control` table) must not
    silently un-pause the Executive when it is activated."""
    from openexecutive.scheduler import pause as pause_store

    _seed_live_company(env)
    await create_client_slot(env.settings, display_name="Acme", source="current")
    await create_client_slot(env.settings, display_name="Beta", source="blank")
    await activate_client_slot(env.settings, "beta")

    pause_store.pause("ceo@example.com", "holiday")
    await activate_client_slot(env.settings, "acme")

    state = pause_store.get_pause_state()
    assert state.paused is True
    assert state.reason == "holiday"


async def test_switch_forgets_the_old_clients_agent_overrides(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overrides are cached with no expiry; a switch must drop them."""
    from openexecutive.agents import overrides

    monkeypatch.setattr(overrides, "DB_PATH", env.db_path)
    overrides.initialize_overrides_db(env.db_path)
    _seed_live_company(env)
    await create_client_slot(env.settings, display_name="Beta", source="blank")
    overrides.set_override("finance", instructions="Acme's voice.", instructions_set=True)
    await create_client_slot(env.settings, display_name="Acme", source="current")
    assert overrides.get_override("finance") is not None  # now in memory
    await activate_client_slot(env.settings, "beta")
    try:
        assert overrides.get_override("finance") is None
    finally:
        overrides.invalidate_cache()


async def test_every_slot_restore_forgets_the_cached_overrides(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rotation's recovery restores a slot without activate_client_slot,
    so the restore itself must drop the cache."""
    from openexecutive.agents import overrides

    monkeypatch.setattr(overrides, "DB_PATH", env.db_path)
    overrides.initialize_overrides_db(env.db_path)
    _seed_live_company(env)
    await create_client_slot(env.settings, display_name="Beta", source="blank")
    overrides.set_override("finance", instructions="Acme's voice.", instructions_set=True)
    assert overrides.get_override("finance") is not None  # now in memory
    await slots._restore_slot_state(env.settings, slots._slot_dir(env.settings, "beta"))
    try:
        assert overrides.get_override("finance") is None
    finally:
        overrides.invalidate_cache()


async def test_blank_switch_empties_a_table_nobody_listed(env: SimpleNamespace) -> None:
    """A blank slot gets the factory reset's sweep, so a table added later
    doesn't carry the previous client's rows into it."""
    _seed_live_company(env)
    with sqlite3.connect(str(env.db_path)) as conn:
        conn.execute("CREATE TABLE brand_new_feature (note TEXT)")
        conn.execute("INSERT INTO brand_new_feature VALUES ('acme only')")
    await create_client_slot(env.settings, display_name="Acme", source="current")
    await create_client_slot(env.settings, display_name="Beta", source="blank")
    await activate_client_slot(env.settings, "beta")
    with sqlite3.connect(str(env.db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM brand_new_feature").fetchone()[0] == 0


async def test_blank_switch_then_unload_brings_every_table_back(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The first switch from your own company into a blank client sweeps
    every table; unload must bring all of them back, not only the ones the
    snapshot files cover."""
    from openexecutive.cli import fixture_loader
    from openexecutive.knowledge.store import ChromaDBStore

    monkeypatch.setattr("openexecutive.audit.log_event", lambda *a, **k: None)

    class _NoStore:
        def __init__(self, *a: Any, **k: Any) -> None: ...

        def __getattr__(self, _name: str) -> Any:
            return lambda *a, **k: None

    for name in dir(ChromaDBStore):
        if name.endswith("_COLLECTION"):
            setattr(_NoStore, name, getattr(ChromaDBStore, name))
    monkeypatch.setattr("openexecutive.knowledge.store.ChromaDBStore", _NoStore)

    _seed_live_company(env, "My Real Co")
    with sqlite3.connect(str(env.db_path)) as conn:
        conn.execute("CREATE TABLE brand_new_feature (note TEXT)")
        conn.execute("INSERT INTO brand_new_feature VALUES ('mine')")
        conn.execute("CREATE TABLE IF NOT EXISTS eval_scenarios (note TEXT)")
        conn.execute("INSERT INTO departments_meta (key, value) VALUES ('mine_only', 'x')")
    # A job mid-run when the copy is taken.
    from openexecutive.memory import episodic

    episodic.insert_scheduled_action(
        run_at="2099-01-01T00:00:00+00:00", channel="email", channel_ref="pat@example.com",
        intent_text="Call the bank", kind="reminder",
    )
    rotation_id = episodic.insert_scheduled_action(
        run_at="2099-01-01T00:00:00+00:00", channel="__internal__", channel_ref="practice",
        intent_text="Rotate clients", kind="client_rotation",
    )
    # Pending when saved, its time passed before unload: it would fire late.
    overdue_id = episodic.insert_scheduled_action(
        run_at="2020-01-01T00:00:00+00:00", channel="email", channel_ref="pat@example.com",
        intent_text="Follow up with Pat", kind="ad_hoc",
    )
    future_id = episodic.insert_scheduled_action(
        run_at="2099-01-01T00:00:00+00:00", channel="email", channel_ref="pat@example.com",
        intent_text="Renewal reminder", kind="ad_hoc",
    )
    # Only a little late (a short time away, like the overnight rotation):
    # it still runs.
    just_due_id = episodic.insert_scheduled_action(
        run_at=(datetime.now(UTC) - timedelta(minutes=10)).isoformat(), channel="email",
        channel_ref="pat@example.com", intent_text="Call Pat back", kind="ad_hoc",
    )
    # Still running in this process when the copy comes back: its handler
    # finishes it and chains the next run, so the restore leaves it alone.
    in_flight_id = episodic.insert_scheduled_action(
        run_at="2020-01-01T00:00:00+00:00", channel="__internal__", channel_ref="weekly",
        intent_text="Scheduled run of dynamic workflow: Weekly", kind="dynamic_workflow",
    )
    from openexecutive.scheduler import runner

    in_flight = episodic.get_scheduled_action(in_flight_id)
    assert in_flight is not None
    # Same id, another job: a row of another company's copy is not in flight.
    monkeypatch.setattr(runner, "_inflight_actions", {
        (in_flight_id, "dynamic_workflow", in_flight.run_at),
        (rotation_id, "reminder", "2099-01-01T00:00:00+00:00"),
    })
    with sqlite3.connect(str(env.db_path)) as conn:
        conn.execute(
            "UPDATE scheduled_actions SET status = 'running' "
            "WHERE kind IN ('reminder', 'client_rotation', 'dynamic_workflow')"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS audit_log (note TEXT)")
        conn.execute("INSERT INTO audit_log VALUES ('mine')")
        conn.execute("CREATE TABLE IF NOT EXISTS dynamic_workflows (name TEXT)")
        conn.execute("INSERT INTO dynamic_workflows VALUES ('mine')")
    await create_client_slot(env.settings, display_name="Beta", source="blank")
    await activate_client_slot(env.settings, "beta")
    with sqlite3.connect(str(env.db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM brand_new_feature").fetchone()[0] == 0
        # Written while away: not the company's, so unload keeps it.
        conn.execute("INSERT INTO eval_scenarios VALUES ('written meanwhile')")
        # A blank client starts with no workflows; yours are in the snapshot.
        assert conn.execute("SELECT COUNT(*) FROM dynamic_workflows").fetchone()[0] == 0
        # The client's own: they stay in its slot, not in your company.
        conn.execute("INSERT INTO audit_log VALUES ('beta')")
        # A one-time marker describes the data saved with it, so the saved
        # copy's own comes back, not whatever is live at unload.
        conn.execute("INSERT OR REPLACE INTO departments_meta (key, value) VALUES ('mine_only', 'x')")
        conn.execute("DELETE FROM departments_meta WHERE key = 'mine_only'")
        conn.execute("INSERT INTO dynamic_workflows VALUES ('beta')")

    summary = await fixture_loader.unload_fixture(env.settings)

    assert summary["database_restored"] is True
    # Used once: a second unload can't roll your work back to it.
    assert not (env.company / "_user_backup" / "state.db").exists()
    with sqlite3.connect(str(env.db_path)) as conn:
        assert conn.execute("SELECT note FROM brand_new_feature").fetchall() == [("mine",)]
    assert _decision_summaries(env.db_path) == ["My Real Co decision"]
    assert get_active_client(env.settings) is None
    with sqlite3.connect(str(env.db_path)) as conn:
        assert conn.execute("SELECT note FROM eval_scenarios").fetchall() == [("written meanwhile",)]
        assert conn.execute("SELECT note FROM audit_log").fetchall() == [("mine",)]
        assert conn.execute("SELECT name FROM dynamic_workflows").fetchall() == [("mine",)]
        assert conn.execute(
            "SELECT value FROM departments_meta WHERE key = 'mine_only'"
        ).fetchall() == [("x",)]
        # A job running when the copy was saved finished here: it is closed,
        # never run a second time (a rotation, a send).
        assert conn.execute(
            "SELECT status FROM scheduled_actions WHERE kind = 'reminder'"
        ).fetchall() == [("cancelled",)]
        assert conn.execute(
            "SELECT status FROM scheduled_actions WHERE id = ?", (rotation_id,)
        ).fetchall() == [("cancelled",)]
        # Long-overdue rows are closed; one still to come stays the user's.
        assert conn.execute(
            "SELECT status FROM scheduled_actions WHERE id = ?", (overdue_id,)
        ).fetchall() == [("cancelled",)]
        assert conn.execute(
            "SELECT status FROM scheduled_actions WHERE id = ?", (future_id,)
        ).fetchall() == [("pending",)]
        assert conn.execute(
            "SELECT status FROM scheduled_actions WHERE id = ?", (just_due_id,)
        ).fetchall() == [("pending",)]
        assert conn.execute(
            "SELECT status FROM scheduled_actions WHERE id = ?", (in_flight_id,)
        ).fetchall() == [("running",)]


async def test_eval_scenarios_stay_across_every_switch(env: SimpleNamespace) -> None:
    """The scenarios people wrote belong to the instance: switching into a
    saved client keeps the live ones, as a blank switch and unload do."""
    from openexecutive.evals.persistence import create_user_scenario

    _seed_live_company(env)
    await create_client_slot(env.settings, display_name="Acme", source="current")
    await create_client_slot(env.settings, display_name="Beta", source="blank")
    await activate_client_slot(env.settings, "beta")
    create_user_scenario("written-in-beta", "chat", "name: x\n", db_path=env.db_path)
    # A copy saved before the table existed must still take the live rows.
    with sqlite3.connect(str(env.company / "_client_slots" / "acme" / "state.db")) as conn:
        conn.execute("DROP TABLE IF EXISTS eval_scenarios")
    await activate_client_slot(env.settings, "acme")  # a saved slot
    with sqlite3.connect(str(env.db_path)) as conn:
        assert conn.execute("SELECT id FROM eval_scenarios").fetchall() == [("written-in-beta",)]


async def test_every_first_activation_saves_your_company_again(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An older backup is never reused: unload must bring back the company
    as it was when you left it, not as it was the first time."""
    _seed_live_company(env, "My Real Co")
    backup = env.company / "_user_backup"
    backup.mkdir()
    (backup / "profile.yaml").write_text("name: Older copy\n")
    snapshots: list[str] = []
    monkeypatch.setattr(slots, "snapshot_user_state", lambda s: snapshots.append("taken"))

    await create_client_slot(env.settings, display_name="Beta", source="blank")
    await activate_client_slot(env.settings, "beta")

    assert snapshots == ["taken"]


async def test_a_failed_snapshot_keeps_the_last_one_and_stops_the_switch(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The earlier backup stays until a new copy is complete, and a switch
    that couldn't save your company doesn't go on to replace it."""
    from openexecutive.cli import fixture_loader

    _seed_live_company(env, "My Real Co")
    fixture_loader.snapshot_user_state(env.settings)
    saved = env.company / "_user_backup" / "state.db"
    before = saved.read_bytes()

    real_connect = sqlite3.connect

    class _FailingVacuum:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

        def execute(self, sql: str, *args: Any) -> Any:
            if sql.startswith("VACUUM INTO"):
                raise sqlite3.OperationalError("database or disk is full")
            return self._conn.execute(sql, *args)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._conn, name)

    monkeypatch.setattr(
        fixture_loader.sqlite3, "connect", lambda *a, **k: _FailingVacuum(real_connect(*a, **k))
    )
    await create_client_slot(env.settings, display_name="Beta", source="blank")
    with pytest.raises(slots.ClientSlotError):
        await activate_client_slot(env.settings, "beta")
    monkeypatch.setattr(fixture_loader.sqlite3, "connect", real_connect)

    assert saved.read_bytes() == before
    assert get_active_client(env.settings) is None
    assert _decision_summaries(env.db_path) == ["My Real Co decision"]


async def test_the_scheduler_is_held_while_a_saved_company_is_put_back(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No tick may claim from the copy between the swap and the settle."""
    import asyncio

    from openexecutive.cli.fixture_loader import snapshot_user_state
    from openexecutive.memory import episodic
    from openexecutive.scheduler import runner

    _seed_live_company(env, "My Real Co")
    snapshot_user_state(env.settings)
    held: list[bool] = []
    real_restore = slots._restore_db_from_file
    real_settle = episodic.settle_restored_queue

    def _restore(src: Path) -> None:
        held.append(runner._claim_lock.locked())
        real_restore(src)

    def _settle(*a: Any, **k: Any) -> int:
        held.append(runner._claim_lock.locked())
        return real_settle(*a, **k)

    monkeypatch.setattr(slots, "_restore_db_from_file", _restore)
    monkeypatch.setattr(episodic, "settle_restored_queue", _settle)
    await asyncio.to_thread(
        slots.restore_company_db,
        env.company / "_user_backup" / "state.db",
    )
    assert held == [True, True]
    assert not runner._claim_lock.locked()
