"""An Approve card shows the step in plain words, never a file's bytes."""
from __future__ import annotations

import base64
import json

from openexecutive.integrations import channel_cards
from openexecutive.memory.decision_ledger import DecisionInstance
from openexecutive.orchestrator import card_details

_PDF = base64.b64encode(b"%PDF-1.7 " + b"x" * 300_000).decode()
_DRIVE_CALL = {
    "user_google_email": "ai@example.com",
    "file_name": "Harbor Lane Q3 Report.pdf",
    "base64_content": _PDF,
    "content_mime_type": "application/pdf",
    "base64_sha256": "ab" * 32,
    "folder_id": "1UX0G1Z8qUk8cmaeXp0CCG3eeyeF-HlnL",
}


def test_a_drive_upload_shows_name_kind_size_and_folder() -> None:
    shown = card_details.plain("google_workspace__create_drive_file", _DRIVE_CALL)
    assert shown == [
        {"field": "File name", "value": "Harbor Lane Q3 Report.pdf"},
        {"field": "File", "value": "PDF file, 293 KB"},
        {"field": "Folder", "value": "Open the folder",
         "href": "https://drive.google.com/drive/folders/1UX0G1Z8qUk8cmaeXp0CCG3eeyeF-HlnL"},
    ]
    text = json.dumps(shown)
    assert _PDF[:50] not in text and "user_google_email" not in text and "ab" * 32 not in text


def test_bytes_are_caught_by_their_look_too() -> None:
    shown = card_details.plain("some_app__upload", {"name": "scan.png", "payload": _PDF})
    assert shown[1] == {"field": "File", "value": "Image file, 293 KB"}


def test_words_under_a_bytes_like_key_are_still_shown() -> None:
    words = "Wire $50,000 to account 1234"
    assert card_details.plain("x__send", {"data": words, "base64_note": words}) == [
        {"field": "Data", "value": words}, {"field": "Base64 note", "value": words},
    ]


def test_plumbing_is_hidden_but_a_non_default_calendar_is_shown() -> None:
    call = {"user_google_email": "ai@example.com", "thread_id": "t1", "calendar_id": "primary", "title": "Sync"}
    assert card_details.plain("google_workspace__create_event", call) == [{"field": "Title", "value": "Sync"}]
    call["calendar_id"] = "team@group.calendar.google.com"
    assert {"field": "Calendar id", "value": "team@group.calendar.google.com"} in card_details.plain(
        "google_workspace__create_event", call,
    )


def test_the_account_is_shown_for_an_app_that_isnt_google() -> None:
    assert card_details.plain("microsoft_365__send-mail", {"user_email": "ceo@other.example"}) == [
        {"field": "User email", "value": "ceo@other.example"},
    ]


def test_a_call_that_only_names_its_target_still_shows_it() -> None:
    call = {"user_google_email": "ai@example.com", "draft_id": "r123"}
    assert card_details.plain("google_workspace__send_gmail_draft", call) == [{"field": "Draft id", "value": "r123"}]


def test_words_are_shown_whole() -> None:
    long = "word " * 1000
    [shown] = card_details.plain("send_email", {"body": long})
    assert shown == {"field": "Message", "value": long}
    [short] = card_details.plain("send_email", {"subject": "Final offers"})
    assert short == {"field": "Subject", "value": "Final offers"}


def test_lists_read_as_a_list() -> None:
    assert card_details.plain("send_email", {"to": ["a@example.com", "b@example.com"]}) == [
        {"field": "To", "value": "a@example.com, b@example.com"},
    ]


def test_served_payload_drops_the_bytes_and_rewrites_old_details() -> None:
    payload = {
        "tool": "google_workspace__create_drive_file", "input": dict(_DRIVE_CALL), "summary": "Create it",
        "details": [{"field": "base64_content", "value": _PDF}],
    }
    out = card_details.served(payload)
    assert out["input"]["base64_content"] == "" and out["input"]["file_name"] == _DRIVE_CALL["file_name"]
    assert out["details"][1] == {"field": "File", "value": "PDF file, 293 KB"}
    # The stored copy, which Approve reads, is untouched.
    assert payload["input"]["base64_content"] == _PDF


def test_chat_app_card_shows_plain_details(monkeypatch) -> None:
    payload = {
        "tool": "google_workspace__create_drive_file", "input": dict(_DRIVE_CALL),
        "summary": "Create Harbor Lane Q3 Report.pdf (Google Drive)",
        "reason": "the folder it goes in is shared outside the company", "held": True,
    }
    instance = DecisionInstance(
        id=5, decision_class="department_action", created_at="2026-11-04T00:00:00+00:00", department="executive",
        originating_session_id=None, proposed_payload_json=json.dumps(payload), idempotency_key=None,
        gate_mode="propose", approver_person_id=1, confidence=None, status="proposed", resolved_at=None,
        resolver_person_id=None, final_payload_json=None, external_event_id=None, reversal_reason=None, severity="",
    )
    from openexecutive.api.routes import decisions
    from openexecutive.memory import decision_ledger

    monkeypatch.setattr(decision_ledger, "get_decision_instance", lambda _id: instance)
    monkeypatch.setattr(decisions, "shown_in_chat", lambda _i: True)
    text = channel_cards.card_text(5)
    assert text is not None
    assert "File name: Harbor Lane Q3 Report.pdf" in text and "File: PDF file, 293 KB" in text
    assert _PDF[:50] not in text and "user_google_email" not in text


def test_the_decisions_route_serves_the_plain_copy() -> None:
    from openexecutive.api.routes import decisions

    payload = {"tool": "google_workspace__create_drive_file", "input": dict(_DRIVE_CALL), "summary": "Create it"}
    instance = DecisionInstance(
        id=6, decision_class="department_action", created_at="2026-11-04T00:00:00+00:00", department="executive",
        originating_session_id=None, proposed_payload_json=json.dumps(payload), idempotency_key=None,
        gate_mode="propose", approver_person_id=1, confidence=None, status="proposed", resolved_at=None,
        resolver_person_id=None, final_payload_json=None, external_event_id=None, reversal_reason=None, severity="",
    )
    shown = decisions._served(instance)
    assert _PDF[:50] not in shown.proposed_payload_json
    assert json.loads(shown.proposed_payload_json)["details"][0] == {"field": "File name", "value": "Harbor Lane Q3 Report.pdf"}
    other = instance.model_copy(update={"decision_class": "meeting_scheduling"})
    assert decisions._served(other) is other


def test_a_carried_out_step_serves_no_bytes_either() -> None:
    # final_payload_json copies the input with the result once Approve ran.
    from openexecutive.api.routes import decisions

    payload = {"tool": "google_workspace__create_drive_file", "input": dict(_DRIVE_CALL), "summary": "Create it"}
    instance = DecisionInstance(
        id=7, decision_class="department_action", created_at="2026-11-04T00:00:00+00:00", department="executive",
        originating_session_id=None, proposed_payload_json=json.dumps(payload), idempotency_key=None,
        gate_mode="propose", approver_person_id=1, confidence=None, status="approved",
        resolved_at="2026-11-04T01:00:00+00:00", resolver_person_id=1,
        final_payload_json=json.dumps({**payload, "result": "created"}), external_event_id=None,
        reversal_reason=None, severity="",
    )
    shown = decisions._served(instance)
    assert _PDF[:50] not in shown.proposed_payload_json
    assert shown.final_payload_json is not None and _PDF[:50] not in shown.final_payload_json
    assert json.loads(shown.final_payload_json)["result"] == "created"


# --------------------------------------------------------------------------- #
# Every app's card reads plainly: a file by its name, rows as a table, a Doc
# edit as a line, anything with parts as "Name: value" lines.
# --------------------------------------------------------------------------- #
_SHEET_ID = "1mOCHqPWC2OZT9I0QBskZf0KKVgFp0tdro8dNAb1zi3s"
_SHEET_CALL = {
    "user_google_email": "ai@example.com",
    "spreadsheet_id": _SHEET_ID,
    "range_name": "A1:C3",
    "values": [["Workstream", "Status", "Owner"], ["Launch date", "Nov 16 passed", "Sam"], ["Waitlist", "Wave 1", "Ari"]],
}


def _names(monkeypatch, names: dict[str, str]) -> None:
    monkeypatch.setattr(card_details, "file_name", lambda file_id: names.get(file_id, ""))


def test_a_sheet_update_shows_the_sheet_by_name_its_cells_and_a_table(monkeypatch) -> None:
    _names(monkeypatch, {_SHEET_ID: "Launch tracker"})
    shown = card_details.plain("google_workspace__modify_sheet_values", _SHEET_CALL)
    assert shown[0] == {"field": "Sheet", "value": "Launch tracker", "href": f"https://drive.google.com/open?id={_SHEET_ID}"}
    assert shown[1] == {"field": "Cells", "value": "Rows 1 to 3, columns A to C (A1:C3)"}
    assert shown[2]["field"] == "Values"
    assert shown[2]["table"] == _SHEET_CALL["values"] and shown[2]["header"] is True
    # Where a table can't be drawn (a chat app), every cell as a line.
    assert shown[2]["value"].splitlines() == [
        "3 rows", "Workstream | Status | Owner", "Launch date | Nov 16 passed | Sam", "Waitlist | Wave 1 | Ari",
    ]
    assert "[[" not in json_text(shown)


def json_text(shown: list) -> str:
    return "\n".join(str(d["value"]) for d in shown)


def test_a_sheet_whose_name_isnt_known_says_open_the_sheet(monkeypatch) -> None:
    _names(monkeypatch, {})
    shown = card_details.plain("google_workspace__modify_sheet_values", _SHEET_CALL)
    assert shown[0]["value"] == "Open the sheet" and shown[0]["href"].endswith(_SHEET_ID)


def test_rows_sent_as_a_json_string_are_a_table_too(monkeypatch) -> None:
    _names(monkeypatch, {})
    call = {**_SHEET_CALL, "values": json.dumps(_SHEET_CALL["values"])}
    assert card_details.plain("google_workspace__modify_sheet_values", call)[2]["table"] == _SHEET_CALL["values"]


def test_ranges_read_as_words_and_never_understate() -> None:
    assert card_details.cells("Sheet1!A1:F13") == "Sheet1, rows 1 to 13, columns A to F (A1:F13)"
    assert card_details.cells("'Q4 Plan'!A:G") == "Q4 Plan, all rows, columns A to G (A:G)"
    assert card_details.cells("B2") == "Cell B2 (B2)"
    assert card_details.cells("2:5") == "Rows 2 to 5, all columns (2:5)"
    # An open end runs to the end of the sheet.
    assert card_details.cells("A1:F") == "Rows 1 to the end, columns A to F (A1:F)"
    assert card_details.cells("B2:B") == "Rows 2 to the end, column B (B2:B)"
    # What it can't read whole is shown as written: a named range, a mixed one.
    assert card_details.cells("Tax") == "Tax"
    assert card_details.cells("A:F10") == "A:F10"
    assert card_details.cells("Pipeline") == "Pipeline"


def test_a_doc_edit_reads_as_a_line(monkeypatch) -> None:
    _names(monkeypatch, {"1docdocdocdoc": "Board memo"})
    shown = card_details.plain("google_workspace__modify_doc_text", {
        "document_id": "1docdocdocdoc", "start_index": 10, "end_index": 24, "bold": True, "italic": True,
    })
    assert shown[0] == {"field": "Change", "value": "Make characters 10 to 24 bold and italic"}
    assert shown[1]["value"] == "Board memo"
    assert {"field": "Bold", "value": "Yes"} in shown
    assert not any(d["field"] in ("From character", "To character") for d in shown)
    added = card_details.plain("google_workspace__modify_doc_text", {"document_id": "1docdocdocdoc", "start_index": 1, "text": "Hi"})
    assert added[0] == {"field": "Change", "value": "Add the text below at character 1"}


def test_anything_with_parts_reads_as_name_value_lines_never_brackets() -> None:
    shown = card_details.plain("crm__update_deal", {
        "properties": {"dealname": "Harbor Lane", "closed": False},
        "associations": [{"to": {"id": 7}, "types": ["primary"]}],
    })
    assert shown[0] == {"field": "Properties", "value": "Dealname: Harbor Lane\nClosed: No"}
    assert shown[1] == {"field": "Associations", "value": "- To:\n    Id: 7\n  Types: primary"}
    assert not any("{" in d["value"] or "[" in d["value"] for d in shown)


def test_the_page_and_chat_apps_name_the_file_in_the_title(monkeypatch) -> None:
    from openexecutive.orchestrator import take_the_lead

    _names(monkeypatch, {_SHEET_ID: "Launch tracker"})
    tool = "google_workspace__modify_sheet_values"
    payload = {
        "tool": tool, "mcp": True, "input": _SHEET_CALL, "kind": "money", "reason": "it's about money",
        "headline": take_the_lead.title(tool, _SHEET_CALL, mcp=True, private=True),
        "summary": take_the_lead.summarize(tool, _SHEET_CALL, mcp=True),
    }
    rows = len(_SHEET_CALL["values"])
    assert payload["headline"] == f"Write {rows} rows to a spreadsheet (Google Sheets)"
    served = card_details.served(payload)
    assert served["headline"] == f"Write {rows} rows to Launch tracker (Google Sheets)"
    assert served["summary"].startswith("Update Launch tracker")
    # No amount in it: nothing is quoted.
    assert not any(d["field"] == "The amount it saw" for d in served["details"])


def test_a_money_card_quotes_the_amount_it_saw(monkeypatch) -> None:
    _names(monkeypatch, {})
    call = {**_SHEET_CALL, "values": [["Plan", "Price"], ["Growth", "Growth plan is $200 a month for teams"]]}
    served = card_details.served({"tool": "google_workspace__modify_sheet_values", "mcp": True, "input": call, "kind": "money"})
    assert served["details"][-1] == {"field": "The amount it saw", "value": "“Growth plan is $200 a month for teams”"}


def test_a_chat_app_card_names_the_sheet_and_lists_its_rows(monkeypatch) -> None:
    from openexecutive.api.routes import decisions
    from openexecutive.memory import decision_ledger
    from openexecutive.orchestrator import take_the_lead

    _names(monkeypatch, {_SHEET_ID: "Launch tracker"})
    tool = "google_workspace__modify_sheet_values"
    payload = {
        "tool": tool, "mcp": True, "input": dict(_SHEET_CALL), "held": True, "reason": "a rule says to wait",
        "summary": take_the_lead.summarize(tool, _SHEET_CALL, mcp=True),
    }
    instance = DecisionInstance(
        id=8, decision_class="take_the_lead_action", created_at="2026-11-04T00:00:00+00:00", department="executive",
        originating_session_id=None, proposed_payload_json=json.dumps(payload), idempotency_key=None,
        gate_mode="propose", approver_person_id=1, confidence=None, status="proposed", resolved_at=None,
        resolver_person_id=None, final_payload_json=None, external_event_id=None, reversal_reason=None, severity="",
    )
    monkeypatch.setattr(decision_ledger, "get_decision_instance", lambda _id: instance)
    monkeypatch.setattr(decisions, "shown_in_chat", lambda _i: True)
    text = channel_cards.card_text(8)
    assert text is not None
    assert "Update Launch tracker (Google Sheets)" in text
    assert f"Sheet: Launch tracker (https://drive.google.com/open?id={_SHEET_ID})" in text
    assert "Cells: Rows 1 to 3, columns A to C (A1:C3)" in text
    assert "Launch date | Nov 16 passed | Sam" in text
    assert "[[" not in text


def test_rows_written_below_row_one_have_no_header(monkeypatch) -> None:
    _names(monkeypatch, {})
    call = {**_SHEET_CALL, "range_name": "Sheet1!A5:C7"}
    assert card_details.plain("google_workspace__modify_sheet_values", call)[2]["header"] is False


def test_an_emptied_field_is_shown_never_dropped() -> None:
    shown = card_details.plain("crm__update_deal", {"properties": {"owner": "", "closedate": None, "amount": 1}})
    assert shown == [{"field": "Properties", "value": "Owner: (empty)\nClosedate: (empty)\nAmount: 1"}]
    flat = card_details.plain("google_workspace__find_and_replace_doc", {"find_text": "Acme Corp", "replace_text": ""})
    assert {"field": "Replace with", "value": "(nothing: it deletes what it finds)"} in flat


def test_another_apps_range_and_offsets_keep_their_own_words() -> None:
    shown = card_details.plain("crm__report", {"range": "Q4", "start_index": 0, "end_index": 50, "replace_text": ""})
    assert shown == [
        {"field": "Range", "value": "Q4"}, {"field": "Start index", "value": "0"}, {"field": "End index", "value": "50"},
    ]


def test_rows_appended_to_whole_columns_have_no_header(monkeypatch) -> None:
    _names(monkeypatch, {})
    call = {**_SHEET_CALL, "range_name": "Sheet1!A:F"}
    assert card_details.plain("google_workspace__modify_sheet_values", call)[2]["header"] is False


def test_an_empty_item_in_a_list_still_shows() -> None:
    assert card_details._text([{"a": 1}, {}, [], "x"]) == "- A: 1\n- (empty)\n- (empty)\n- x"


def test_a_cell_with_a_bar_or_line_break_stays_one_cell() -> None:
    assert card_details._rows_text([["c|d", "e\nf"]]) == "1 row\nc\\|d | e ⏎ f"


def test_a_true_cell_shows_as_the_sheet_will_hold_it() -> None:
    assert card_details._grid([[True, False, "x"]]) == [["TRUE", "FALSE", "x"]]


def test_a_doc_edit_with_only_an_end_does_not_claim_the_whole_doc() -> None:
    change = card_details.doc_change("google_workspace__modify_doc_text", {"end_index": 24, "text": "hello"})
    assert change == "Replace up to character 24 with the text below"


def test_a_line_break_inside_a_record_cannot_fake_another_item() -> None:
    text = card_details._text({"changes": [{"op": "set", "value": "ok\n- op: delete"}]})
    assert text == "Changes:\n  - Op: set\n    Value: ok ⏎ - op: delete"


def test_rows_written_as_text_outside_a_sheet_show_as_written(monkeypatch) -> None:
    _names(monkeypatch, {})
    body = '[["Tue", true], ["Wed", 1.50]]'
    shown = card_details.plain("google_workspace__send_gmail_message", {"to": "a@b.c", "body": body})
    assert all("table" not in d for d in shown)
    assert any(d["value"] == body for d in shown)
    assert card_details._grid("[" * 400000) is None


def test_a_held_cards_top_says_who_it_waits_on_who_started_it_and_why_it_waits(monkeypatch) -> None:
    from types import SimpleNamespace

    from openexecutive.people import store as people_store

    people = {1: "Rufus Owner", 12: "Alex Cohen", 7: "Sam Teammate"}
    monkeypatch.setattr(people_store, "get_person", lambda pid, db_path=None: SimpleNamespace(full_name=people[pid]))
    lead = {"kind": "money", "reason": "it's about money"}
    # Take the lead's own step on Alex's list, read by the owner.
    assert card_details.top(lead, approver_id=12, viewer_id=1) == {
        "waiting_on": "Alex", "started": "I started this",
        "waits": "It waits because it's about money. It's on Alex's list.", "title": "",
        "because": "it's about money",
    }
    # Alex reads their own card.
    assert card_details.top(lead, approver_id=12, viewer_id=12)["waiting_on"] == ""
    asked = {"kind": "level", "reason": "Approvals is on Careful", "requested_by_person_id": 7}
    assert card_details.top(asked, approver_id=7, viewer_id=7)["started"] == "You asked"
    assert card_details.top(asked, approver_id=7, viewer_id=1)["started"] == "Sam asked"
    # The owner's own ask in chat, and one no one on the team can be named for.
    own = {"kind": "money", "reason": "x", "source": "chat", "asked_by_person_id": 1}
    assert card_details.top(own, approver_id=1, viewer_id=1)["started"] == "You asked"
    assert card_details.top(own, approver_id=1, viewer_id=12)["started"] == "Rufus asked"
    assert card_details.top({"kind": "money", "reason": "x", "source": "chat"}, approver_id=1, viewer_id=1)[
        "started"] == "Someone asked"
    training = card_details.top({"kind": "training", "reason": "x"}, approver_id=1, viewer_id=1)
    assert training["waits"] == "It's a new kind of step for me, so it waits for an OK once."
    assert training["because"] == "it's a new kind of step"


def test_our_own_tools_name_people_departments_and_goals_never_a_number(monkeypatch) -> None:
    from types import SimpleNamespace

    from openexecutive.departments import store as dept_store
    from openexecutive.people import registry

    people = {4: "Annamarie Braden", 7: "Leo Park"}
    monkeypatch.setattr(
        registry, "get_person",
        lambda pid, include_contacts=False: SimpleNamespace(full_name=people[pid], email="") if pid in people else None,
    )
    monkeypatch.setattr(
        dept_store, "get_department",
        lambda slug: SimpleNamespace(config=SimpleNamespace(title="Sales")) if slug == "sales" else None,
    )
    monkeypatch.setattr(dept_store, "get_goal", lambda gid: SimpleNamespace(key_result="Close 10 deals") if gid == 3 else None)

    shown = card_details.plain("message_person", {"person_id": 4, "text": "Hi"})
    assert shown == [{"field": "To", "value": "Annamarie Braden"}, {"field": "Message", "value": "Hi"}]

    shown = card_details.plain("assign_open_loop", {
        "assigned_to_person_id": 7, "loop_id": 12, "department_slug": "sales", "goal_id": 3, "task": "Call Acme",
    })
    assert shown == [
        {"field": "Assigned to", "value": "Leo Park"}, {"field": "Department", "value": "Sales"},
        {"field": "Goal", "value": "Close 10 deals"}, {"field": "Task", "value": "Call Acme"},
    ]
    # A number that names no one, or only the system reads, isn't shown.
    assert card_details.plain("message_person", {"person_id": 99, "loop_id": 12, "text": "Hi"}) == [
        {"field": "To", "value": "Someone not on People"}, {"field": "Message", "value": "Hi"},
    ]
    # Two people with the same kind of label are told apart.
    assert [d["field"] for d in card_details.plain("x", {"target_person_id": 4, "cadence_person_id": 7})] == [
        "Who", "Who (cadence person id)",
    ]
    assert card_details.plain("x", {"alert_id": 5, "status": "done"}) == [{"field": "Status", "value": "done"}]
    # A call that names only a number still says what it acts on.
    assert card_details.plain("cancel_followup", {"action_id": 5}) == [{"field": "Action id", "value": "5"}]


def test_a_connected_apps_ids_are_still_shown() -> None:
    # A connected app's step may act only on its own record: that stays.
    assert card_details.plain("asana__update_task", {"task_id": "123"}) == [{"field": "Task id", "value": "123"}]


def test_a_chat_app_message_names_who_it_goes_to(monkeypatch) -> None:
    from types import SimpleNamespace

    from openexecutive.people import store

    monkeypatch.setattr(
        store, "find_person_by_slack_id",
        lambda sid, include_contacts=False: SimpleNamespace(full_name="Ria Lopez", email="") if sid == "U_RIA" else None,
    )
    monkeypatch.setattr(store, "find_person_by_telegram_chat_id", lambda cid, include_contacts=False: None)
    assert card_details.plain("send_slack_dm", {"user_id": "U_RIA", "text": "Hi"})[0] == {"field": "To", "value": "Ria Lopez"}
    # Not on People: the chat's id is all there is to say who it reaches.
    assert card_details.plain("send_telegram_message", {"chat_id": "-100123", "text": "Hi"})[0] == {
        "field": "To", "value": "-100123",
    }


def test_an_account_or_a_yes_no_a_step_sets_is_still_shown(monkeypatch) -> None:
    monkeypatch.setattr(card_details, "_names", lambda value: "Dana Lee")
    shown = card_details.plain("upsert_person", {"person_id": 4, "full_name": "Dana", "slack_user_id": "U_OTHER"})
    assert {"field": "Slack user id", "value": "U_OTHER"} in shown
    shown = card_details.plain("resolve_roster_request", {"decision": "link", "person_id": 4, "replace_channel_id": True})
    assert {"field": "Replace channel id", "value": "Yes"} in shown


def test_a_person_given_by_something_other_than_a_number_shows_as_given() -> None:
    assert card_details.plain("message_person", {"person_id": "dana@x.com"})[0] == {"field": "To", "value": "dana@x.com"}


def test_a_list_of_people_with_one_not_on_people_still_shows_everyone(monkeypatch) -> None:
    monkeypatch.setattr(card_details, "_names", lambda value: "Annamarie Braden" if value == 4 else None)
    shown = card_details.plain("create_calendar_event", {"title": "Sync", "attendee_person_ids": [4, 99]})
    assert {"field": "With", "value": "Annamarie Braden, someone not on People"} in shown
