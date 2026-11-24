"""Whose card is being carried out (``people_tools.approved_speaker``) and
what it grants: the asker's own powers, and nothing for a card nobody on the
team asked for."""
from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from openexecutive.memory import episodic
from openexecutive.orchestrator import levels, people_tools
from openexecutive.people import registry as people_registry
from openexecutive.people import store as people_store

_UNATTENDED = SimpleNamespace(unattended=True)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    db = tmp_path / "test.db"
    monkeypatch.setattr(people_store, "DB_PATH", db)
    monkeypatch.setattr(episodic, "DB_PATH", db)
    episodic.initialize_db(db)
    people_store.initialize_db(db)
    people_registry.invalidate()
    yield
    people_registry.invalidate()


@pytest.fixture
def team() -> SimpleNamespace:
    ids = SimpleNamespace(
        principal=people_store.upsert_person(full_name="Pat Principal", is_principal=True,
                                             email="pat@northwind.test"),
        sara=people_store.upsert_person(full_name="Sara Kim", email="sara@northwind.test"),
        carl=people_store.upsert_person(full_name="Carl Contact", email="carl@else.test",
                                        kind="contact"),
        gina=people_store.upsert_person(full_name="Gina Gone", email="gina@northwind.test"),
    )
    people_store.archive_person(ids.gina)
    people_registry.invalidate()
    return ids


def test_no_speaker_outside_an_approved_step(team: SimpleNamespace) -> None:
    assert people_tools.approved_speaker() is None


def test_the_principals_own_card_acts_as_the_principal(team: SimpleNamespace) -> None:
    with levels.approved(team.principal):
        assert people_tools.approved_speaker().id == team.principal
        assert people_tools.is_principal_on_verified_surface(_UNATTENDED) is True
        assert people_tools.teammate_on_verified_surface(_UNATTENDED) is None


def test_a_teammates_own_card_acts_as_that_teammate(team: SimpleNamespace) -> None:
    with levels.approved(team.sara):
        assert people_tools.teammate_on_verified_surface(_UNATTENDED).id == team.sara
        assert people_tools.is_principal_on_verified_surface(_UNATTENDED) is False


@pytest.mark.parametrize("who", ["carl", "gina", "missing", "nobody"])
def test_a_card_nobody_on_the_team_asked_for_grants_nothing(
    team: SimpleNamespace, who: str
) -> None:
    """A contact, an archived person, someone no longer on People, or no
    asker at all (mail from outside) gets no speaker, and the unattended
    session underneath answers no."""
    requester = {"carl": team.carl, "gina": team.gina, "missing": 9999, "nobody": None}[who]
    with levels.approved(requester):
        assert people_tools.approved_speaker() is None
        assert people_tools.is_principal_on_verified_surface(_UNATTENDED) is False
        assert people_tools.teammate_on_verified_surface(_UNATTENDED) is None


def test_an_unreadable_roster_grants_nothing(
    team: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_pid: int) -> None:
        raise RuntimeError("db gone")

    monkeypatch.setattr(people_store, "get_person", boom)
    with levels.approved(team.principal):
        assert people_tools.approved_speaker() is None
        assert people_tools.is_principal_on_verified_surface(_UNATTENDED) is False
        assert people_tools.teammate_on_verified_surface(_UNATTENDED) is None


def test_the_speaker_ends_with_the_approved_step_and_stays_in_it(team: SimpleNamespace) -> None:
    """The card's asker is bound only around the card's own run: not after
    it, and not in a turn running beside it."""
    with levels.approved(team.sara):
        pass
    assert people_tools.approved_speaker() is None

    async def card() -> int | None:
        with levels.approved(team.sara):
            await asyncio.sleep(0.01)
            speaker = people_tools.approved_speaker()
            return speaker.id if speaker else None

    async def other_turn() -> object:
        await asyncio.sleep(0.005)
        return people_tools.approved_speaker()

    async def both() -> list[object]:
        return list(await asyncio.gather(card(), other_turn()))

    assert list(asyncio.run(both())) == [team.sara, None]
