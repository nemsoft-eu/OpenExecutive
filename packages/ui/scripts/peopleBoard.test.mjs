import assert from "node:assert/strict";
import test from "node:test";

import {
  addFormKind,
  contactLine,
  contactsByRecent,
  detailsLine,
  hasSeenable,
  roleBadge,
  sameEmailPairs,
  splitChanges,
  teamInOrder,
  teamLine,
} from "../src/lib/peopleBoard.ts";

const person = (over = {}) => ({ id: 1, full_name: "Avery Example", role: "", email: null, email_aliases: [], is_principal: false, on_leave_until: null, ...over });
const change = (over = {}) => ({ id: 1, person_id: 7, action: "add", field: "contact", email: "p@x.test", full_name: "Priya Shah", old_value: "", new_value: "", reason: "emailed", evidence: 2, status: "applied", seen: false, created_at: "", decided_at: null, ...over });

test("the owner is marked; everyone else is not", () => {
  assert.equal(roleBadge(person({ is_principal: true })), "Owner");
  assert.equal(roleBadge(person()), null);
});

test("a team member's line: leave, then the role, else nothing", () => {
  const today = "2026-11-15";
  assert.deepEqual(teamLine(person({ role: "CFO", on_leave_until: "2026-11-20" }), today), { text: "On leave until Nov 20", warn: true });
  assert.deepEqual(teamLine(person({ role: "CFO", on_leave_until: "2026-11-01" }), today), { text: "CFO", warn: false });
  // No "—", no channel, no reply time: an empty role says nothing.
  assert.equal(teamLine(person({ role: "  " }), today), null);
});

test("a contact's line is the role and when you were last in touch", () => {
  assert.deepEqual(contactLine(person({ role: "Northwind" }), "Emailed Nov 12"), { text: "Northwind · Emailed Nov 12", warn: false });
  assert.deepEqual(contactLine(person(), "Emailed Nov 12"), { text: "Emailed Nov 12", warn: false });
  assert.deepEqual(contactLine(person({ role: "Advisor" }), ""), { text: "Advisor", warn: false });
  assert.equal(contactLine(person(), ""), null);
});

test("the same address twice is one pair, whichever address matched", () => {
  const a = person({ id: 1, full_name: "Marco Diaz", email: "m@x.test" });
  const b = person({ id: 2, full_name: "Marco D.", email: "M@x.test", email_aliases: ["marco@home.test"] });
  const c = person({ id: 3, full_name: "Marco Home", email: "marco@home.test" });
  const d = person({ id: 4, full_name: "Other", email: "o@x.test" });
  const pairs = sameEmailPairs([a, b, c, d]).map((p) => [p.person.id, p.other.id]);
  assert.deepEqual(pairs, [[1, 2], [2, 3]]);
  assert.deepEqual(sameEmailPairs([a, { ...a, id: 5, email_aliases: ["m@x.test"] }]).length, 1);
});

test("changes split: people added or suggested are rows, details fold into one line", () => {
  const rows = [
    change({ id: 1 }),
    change({ id: 2, seen: true }),
    change({ id: 3, status: "suggested", person_id: null }),
    change({ id: 4, status: "undone" }),
    change({ id: 5, action: "update", field: "role" }),
    change({ id: 6, action: "update", field: "phone", seen: true }),
  ];
  const split = splitChanges(rows);
  assert.deepEqual(split.people.map((c) => c.id), [1, 3]);
  assert.deepEqual(split.details.map((c) => c.id), [5]);
  assert.equal(hasSeenable(rows), true);
  assert.equal(hasSeenable([change({ status: "suggested" })]), false);
  assert.equal(detailsLine(1), "I filled in 1 detail from your email");
  assert.equal(detailsLine(3), "I filled in 3 details from your email");
});

test("contacts by last in touch, then name; the team owner first", () => {
  const last = new Map([[2, { person_id: 2, at: "2026-11-12", how: "emailed" }], [3, { person_id: 3, at: "2026-10-29", how: "met" }]]);
  const people = [person({ id: 1, full_name: "Zed" }), person({ id: 2, full_name: "Bea" }), person({ id: 3, full_name: "Al" }), person({ id: 4, full_name: "Cy" })];
  assert.deepEqual(contactsByRecent(people, last).map((p) => p.id), [2, 3, 4, 1]);
  assert.deepEqual(teamInOrder([person({ id: 1, full_name: "Al" }), person({ id: 2, full_name: "Zed", is_principal: true })]).map((p) => p.id), [2, 1]);
});

test("Add starts on a contact for someone on their own, else a team member", () => {
  assert.equal(addFormKind({ mode: "solo", seesContacts: true }), "contact");
  assert.equal(addFormKind({ mode: "team", seesContacts: true }), "team");
  assert.equal(addFormKind({ mode: "solo", seesContacts: false }), "team");
});
