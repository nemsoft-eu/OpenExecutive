"""GET /schedule: everything the Executive does without being asked, in one
list, soonest first, read from the stores that already hold it."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openexecutive.api.routes import schedule as schedule_route
from openexecutive.api.routes import today as today_route
from openexecutive.departments import registry as dept_registry
from openexecutive.departments import store as dept_store
from openexecutive.memory import episodic
from openexecutive.monitoring import store as monitoring_store
from openexecutive.monitoring.models import MODE_DRY_RUN, ORIGIN_RESEARCH_PROPOSED
from openexecutive.people import registry as people_registry
from openexecutive.people import store as people_store
from openexecutive.workflows import dynamic_store
from openexecutive.workflows.dynamic_models import DynamicWorkflowDef


@pytest.fixture()
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "schedule.db"
    monkeypatch.setattr(episodic, "DB_PATH", path)
    monkeypatch.setattr(dept_store, "DB_PATH", path)
    monkeypatch.setattr(people_store, "DB_PATH", path)
    monkeypatch.setattr(dynamic_store, "DB_PATH", path)
    from openexecutive.audit import logger as audit_logger

    monkeypatch.setattr(audit_logger, "_default_logger", audit_logger.AuditLogger(db_path=path))
    monkeypatch.setenv("OE_LOCAL_LOGIN", "1")
    episodic.initialize_db(path)
    dept_store.initialize_db(path)
    people_store.initialize_db(path)
    monitoring_store.initialize_db(path)
    dynamic_store.initialize_dynamic_workflows_db(path)
    dept_registry.invalidate()
    people_registry.invalidate()
    yield path
    dept_registry.invalidate()
    people_registry.invalidate()


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(schedule_route.router)
    app.include_router(today_route.router)
    return TestClient(app)


def _in(hours: float) -> str:
    return (datetime.now(UTC) + timedelta(hours=hours)).isoformat()


def _workflow(name: str, title: str, cadence: str | None, active: bool = True) -> DynamicWorkflowDef:
    return DynamicWorkflowDef.model_validate({
        "name": name,
        "title": title,
        "cadence": cadence,
        "cadence_person_id": 1 if cadence else None,
        "is_active": active,
        "steps": [
            {"kind": "specialist", "id": "research", "title": "Research", "specialist": "cso", "goal": "Go."},
            {"kind": "synthesis", "id": "assemble", "title": "Assemble"},
        ],
    })


def test_schedule_lists_all_five_kinds_soonest_first(db: Path) -> None:
    from openexecutive.orchestrator import take_the_lead

    jordan = people_store.upsert_person(full_name="Jordan", is_principal=True, email="jordan@example.com", db_path=db)
    priya = people_store.upsert_person(full_name="Priya Nair", email="priya@example.com", db_path=db)
    sam = people_store.upsert_person(full_name="Sam Lee", email="sam@example.com", kind="team", db_path=db)
    people_registry.invalidate()

    follow_up = episodic.insert_scheduled_action(
        run_at=_in(0.5), channel="slack_dm", channel_ref="U1", intent_text="Ask Priya about the pre-read",
        assigned_to_person_id=priya, created_by_person_id=jordan,
    )
    # One a teammate asked for is theirs alone, never on the owner's list.
    episodic.insert_scheduled_action(
        run_at=_in(0.2), channel="slack_dm", channel_ref="U2", intent_text="Sam's own follow-up",
        created_by_person_id=sam,
    )
    episodic.insert_scheduled_action(
        run_at=_in(5), channel="__internal__", channel_ref="principal", intent_text="brief",
        kind="principal_brief_morning",
    )
    # Plumbing never shows.
    episodic.insert_scheduled_action(
        run_at=_in(1), channel="__internal__", channel_ref="x", intent_text="scan", kind="nudge_scan",
    )
    dynamic_store.upsert_definition(_workflow("price_check", "Weekly price check", "weekly@mon@09:00"))
    dynamic_store.upsert_definition(_workflow("one_off", "Run when asked", None))
    episodic.insert_scheduled_action(
        run_at=_in(30), channel="__internal__", channel_ref="price_check",
        intent_text="Scheduled run", kind="dynamic_workflow", assigned_to_person_id=1,
    )
    watch_id = monitoring_store.insert_watchlist_item(
        slug="acme-news", signal_type="rss", target="https://acme.example/feed",
        config={"feed_label": "Acme"}, cadence="daily",
    )
    checked = (datetime.now(UTC) - timedelta(hours=23)).isoformat()
    monitoring_store.mark_polled(watch_id, at=datetime.fromisoformat(checked))
    # A suggestion waiting for a yes stays on the Watch list.
    monitoring_store.insert_watchlist_item(
        slug="maybe", signal_type="rss", target="https://maybe.example/feed",
        mode=MODE_DRY_RUN, origin=ORIGIN_RESEARCH_PROPOSED,
    )
    monitoring_store.insert_watchlist_item(
        slug="off", signal_type="rss", target="https://off.example/feed", enabled=False,
    )
    take_the_lead.add_job("Send Alex the open items every Friday", created_by="person:1")

    body = _client().get("/schedule", headers={"x-caller-email": "jordan@example.com"}).json()
    kinds = [(i["kind"], i["title"]) for i in body["items"]]
    assert kinds[0] == ("follow_up", "Ask Priya about the pre-read")
    assert kinds[1] == ("watch", "Acme news")  # checks again in about an hour
    assert kinds[2] == ("always", "Morning brief")
    assert kinds[3] == ("workflow", "Weekly price check")
    assert body["counts"] == {"follow_up": 1, "standing": 1, "workflow": 1, "watch": 1, "always": 1}

    by_kind = {i["kind"]: i for i in body["items"]}
    assert by_kind["follow_up"]["stop_key"] == f"action:{follow_up}"
    assert by_kind["follow_up"]["detail"] == "I'll follow up with Priya Nair"
    assert by_kind["standing"]["stop_key"].startswith("job:")
    assert by_kind["standing"]["href"] == "/delegate#as-me"
    assert by_kind["workflow"] == {
        **by_kind["workflow"], "href": "/jobs/price_check", "cadence": "every week", "stop_key": None,
    }
    assert by_kind["watch"]["last_checked"] == checked
    assert by_kind["watch"]["cadence"] == "once a day"
    assert by_kind["watch"]["href"] == "/watchlist/acme-news"
    assert by_kind["always"]["cadence"] == "every day"


def test_schedule_stop_uses_the_watching_next_stop(db: Path) -> None:
    people_store.upsert_person(full_name="Jordan", is_principal=True, email="jordan@example.com", db_path=db)
    people_registry.invalidate()
    action = episodic.insert_scheduled_action(
        run_at=_in(3), channel="slack_dm", channel_ref="U1", intent_text="Chase the contract",
    )
    owner = {"x-caller-email": "jordan@example.com"}
    client = _client()
    item = client.get("/schedule", headers=owner).json()["items"][0]
    assert client.post("/today/watching/stop", json={"key": item["stop_key"]}, headers=owner).json() == {
        "stopped": True,
    }
    assert episodic.get_scheduled_action(action).status == "cancelled"  # type: ignore[union-attr]
    assert client.get("/schedule", headers=owner).json()["items"] == []


def test_teammate_sees_their_own_and_the_teams_not_the_owners_briefs(db: Path) -> None:
    people_store.upsert_person(full_name="Jordan", is_principal=True, email="jordan@example.com", db_path=db)
    sam = people_store.upsert_person(full_name="Sam Lee", email="sam@example.com", kind="team", db_path=db)
    people_registry.invalidate()
    episodic.insert_scheduled_action(
        run_at=_in(2), channel="slack_dm", channel_ref="U1", intent_text="The owner's follow-up",
    )
    episodic.insert_scheduled_action(
        run_at=_in(3), channel="slack_dm", channel_ref="U2", intent_text="Sam's follow-up",
        created_by_person_id=sam,
    )
    episodic.insert_scheduled_action(
        run_at=_in(5), channel="__internal__", channel_ref="principal", intent_text="brief",
        kind="principal_brief_morning",
    )
    monitoring_store.insert_watchlist_item(slug="acme", signal_type="rss", target="https://acme.example/feed")

    client = _client()
    mine = client.get("/schedule", headers={"x-caller-email": "sam@example.com"}).json()
    assert [(i["kind"], i["title"]) for i in mine["items"]][:1] == [("follow_up", "Sam's follow-up")]
    assert mine["counts"]["always"] == 0 and mine["counts"]["watch"] == 1
    stranger = client.get("/schedule", headers={"x-caller-email": "nobody@example.com"}).json()
    assert stranger["items"] == []
    assert stranger["counts"] == dict.fromkeys(schedule_route.KINDS, 0)


def test_department_check_in_names_the_department(db: Path) -> None:
    people_store.upsert_person(full_name="Jordan", is_principal=True, email="jordan@example.com", db_path=db)
    people_registry.invalidate()
    dept_store.seed_default_departments(db_path=db)
    dept = dept_store.list_departments()[0]
    episodic.insert_scheduled_action(
        run_at=_in(4), channel="__internal__", channel_ref=dept.config.slug, intent_text="check-in",
        kind="dept_cadence", department=dept.config.slug,
    )
    items = _client().get("/schedule", headers={"x-caller-email": "jordan@example.com"}).json()["items"]
    assert [(i["kind"], i["title"], i["href"]) for i in items] == [
        ("always", f"{dept.config.title} check-in", f"/departments/{dept.config.slug}"),
    ]


def test_mixed_time_forms_sort_by_the_moment() -> None:
    a = schedule_route.ScheduleItem(key="a", kind="always", title="A", at="2026-11-15T09:30:00Z")
    b = schedule_route.ScheduleItem(key="b", kind="watch", title="B", at="2026-11-15T09:00:00+00:00")
    c = schedule_route.ScheduleItem(key="c", kind="workflow", title="C", at="2026-11-15T09:15:00")
    none = schedule_route.ScheduleItem(key="n", kind="watch", title="N")
    assert [i.key for i in sorted([none, a, c, b], key=schedule_route._sort_key)] == ["b", "c", "a", "n"]


def test_no_kinds_lists_nothing(db: Path) -> None:
    episodic.insert_scheduled_action(run_at=_in(1), channel="slack_dm", channel_ref="U1", intent_text="x")
    assert episodic.list_scheduled_actions(status="pending", kinds=[]) == []
    assert len(episodic.list_scheduled_actions(status="pending", kinds=["ad_hoc"])) == 1


def test_owner_keeps_their_follow_ups_behind_many_private_ones(db: Path) -> None:
    """Teammates' own follow-ups due sooner are read past, not counted
    against the owner's page."""
    jordan = people_store.upsert_person(full_name="Jordan", is_principal=True, email="jordan@example.com", db_path=db)
    sam = people_store.upsert_person(full_name="Sam Lee", email="sam@example.com", kind="team", db_path=db)
    people_registry.invalidate()
    for n in range(900):
        episodic.insert_scheduled_action(
            run_at=_in(1), channel="slack_dm", channel_ref="U2", intent_text=f"Sam's {n}",
            created_by_person_id=sam,
        )
    mine = episodic.insert_scheduled_action(
        run_at=_in(2), channel="slack_dm", channel_ref="U1", intent_text="Chase the contract",
        created_by_person_id=jordan,
    )
    items = _client().get("/schedule", headers={"x-caller-email": "jordan@example.com"}).json()["items"]
    assert [i["stop_key"] for i in items if i["kind"] == "follow_up"] == [f"action:{mine}"]
