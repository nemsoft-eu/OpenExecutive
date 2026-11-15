"""Every company swap empties ``progress_updates``: one company's agents'
reports must not reach the next company's principal.

The table is in ``memory.company_tables.PER_COMPANY_TABLES``, which the three
swap paths wipe from; ``test_company_tables.py`` proves that for every table
with a stand-in row. This pins it for a real row written by the store, read
back through ``latest_per_topic``, on every path.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from openexecutive.cli import fixture_loader
from openexecutive.integrations import progress_store
from openexecutive.integrations.progress_guard import CleanUpdate
from openexecutive.memory.company_tables import PER_COMPANY_TABLES

from .test_company_tables import env  # noqa: F401


def _admit(db: Path) -> None:
    update = CleanUpdate(
        topic="GTM plan", topic_key="gtm plan", status="on_track",
        summary="Positioning draft is done.", ask=None, link=None, idempotency_key=None,
    )
    assert progress_store.admit("marketing-planner", update, db_path=db).id is not None
    assert progress_store.latest_per_topic(db_path=db)


def test_progress_updates_is_a_per_company_table() -> None:
    assert progress_store.TABLE in PER_COMPANY_TABLES


def _blank_slot_switch(env: Any) -> None:  # noqa: F811
    from openexecutive.clients import slots

    slot = env.settings.company_profile_path.parent / "_client_slots" / "fresh"
    slot.mkdir(parents=True)
    asyncio.run(slots._restore_slot_state(env.settings, slot))


def _factory_reset(env: Any) -> None:  # noqa: F811
    asyncio.run(fixture_loader.reset_all_state(env.settings))


def _fixture_load(env: Any) -> None:  # noqa: F811
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))


@pytest.mark.parametrize("swap", [_blank_slot_switch, _factory_reset, _fixture_load])
def test_a_swap_empties_the_updates(swap: Any, env: Any) -> None:  # noqa: F811
    _admit(env.db)

    swap(env)

    assert progress_store.latest_per_topic(db_path=env.db) == []


@pytest.mark.parametrize("with_memory_json", [False, True])
def test_a_fixture_unload_empties_what_the_demo_received(
    with_memory_json: bool, env: Any,  # noqa: F811
) -> None:
    """With or without a ``memory.json`` in the fixture: the wipe must not
    hang off the fixture's own files."""
    if with_memory_json:
        (fixture_loader.FIXTURES_ROOT / "demo" / "memory.json").write_text("{}\n")
    _admit(env.db)
    asyncio.run(fixture_loader.load_fixture("demo", env.settings))
    assert progress_store.latest_per_topic(db_path=env.db) == []

    _admit(env.db)
    asyncio.run(fixture_loader.unload_fixture(env.settings))

    assert progress_store.latest_per_topic(db_path=env.db) == []
