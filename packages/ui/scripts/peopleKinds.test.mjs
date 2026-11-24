import assert from "node:assert/strict";
import test from "node:test";
import {
  canClaimPrincipal,
  effectiveKind,
  hiddenTeamCount,
  isContact,
  peopleForTab,
  shouldSwitchToTeam,
} from "../src/lib/peopleKinds.ts";

const person = (id, kind, is_principal = false) => ({ id, kind, is_principal });

const PRINCIPAL = person(1, "team", true);
const TEAMMATE = person(2, "team");
const CLIENT = person(3, "contact");
const LEGACY = { id: 4, is_principal: false }; // an older server sends no kind
const ROSTER = [PRINCIPAL, TEAMMATE, CLIENT, LEGACY];

test("a row with no kind is a team member, not a contact", () => {
  assert.equal(isContact(LEGACY), false);
  assert.equal(isContact(CLIENT), true);
});

test("the Team tab lists the whole team in team mode", () => {
  assert.deepEqual(peopleForTab(ROSTER, "team", "team").map((p) => p.id), [1, 2, 4]);
});

test("the Team tab is just the principal in solo mode", () => {
  assert.deepEqual(peopleForTab(ROSTER, "team", "solo").map((p) => p.id), [1]);
  assert.equal(hiddenTeamCount(ROSTER, "solo"), 2);
  assert.equal(hiddenTeamCount(ROSTER, "team"), 0);
});

test("the Contacts tab lists only contacts, in either mode", () => {
  assert.deepEqual(peopleForTab(ROSTER, "contacts", "team").map((p) => p.id), [3]);
  assert.deepEqual(peopleForTab(ROSTER, "contacts", "solo").map((p) => p.id), [3]);
});

test("the principal is always on the team", () => {
  assert.equal(effectiveKind("contact", true), "team");
  assert.equal(effectiveKind("contact", false), "contact");
});

test("adding a teammate in solo switches to team mode; a contact or the principal does not", () => {
  assert.equal(shouldSwitchToTeam("solo", "team", false), true);
  assert.equal(shouldSwitchToTeam("solo", "contact", false), false);
  assert.equal(shouldSwitchToTeam("solo", "team", true), false);
  assert.equal(shouldSwitchToTeam("solo", "contact", true), false);
  assert.equal(shouldSwitchToTeam("team", "team", false), false);
});

test("This is me is offered only while nobody is the principal", () => {
  assert.equal(canClaimPrincipal([TEAMMATE, CLIENT], false), true);
  assert.equal(canClaimPrincipal([], false), true);
  assert.equal(canClaimPrincipal(ROSTER, false), false);
  // Not while the list is loading: it may hold the principal.
  assert.equal(canClaimPrincipal([], true), false);
});
