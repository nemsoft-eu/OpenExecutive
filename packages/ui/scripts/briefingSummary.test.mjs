import assert from "node:assert/strict";
import test from "node:test";
import {
  ageLabel,
  briefingSummary,
  daysUntil,
  greeting,
  handAsks,
  homeHeadline,
  teamOffTrack,
  waitingOnTeam,
  watchingRow,
} from "../src/lib/briefingSummary.ts";

// Local-time constructors, so the labels hold in any test-runner time zone.
const at = (h) => new Date(2026, 9, 3, h, 30);

test("greeting follows the time of day and drops a missing name", () => {
  assert.equal(greeting("Jordan", at(8)), "Good morning, Jordan");
  assert.equal(greeting("Jordan", at(13)), "Good afternoon, Jordan");
  assert.equal(greeting("Jordan", at(19)), "Good evening, Jordan");
  assert.equal(greeting(undefined, at(8)), "Good morning");
  assert.equal(greeting("  ", at(8)), "Good morning");
});

const zero = {
  needsYou: 0,
  handledOvernight: 0,
  peopleOverdue: 0,
  peopleNeedReply: 0,
  deptAtRisk: 0,
  inFlight: 0,
  monitoring: 0,
};

test("an all-clear day has no summary phrases", () => {
  assert.deepEqual(briefingSummary(zero), []);
});

test("the summary leads with what needs you, then what was handled", () => {
  const parts = briefingSummary({ ...zero, needsYou: 3, handledOvernight: 12, inFlight: 7, deptAtRisk: 1 });
  assert.deepEqual(
    parts.map((p) => p.text),
    ["3 things need you", "The Executive handled 12 on its own", "1 department at risk", "7 under way"],
  );
  assert.deepEqual(parts[0].target, { kind: "needsYou" });
  assert.deepEqual(parts[1].target, { kind: "panel", panel: "handled" });
});

test("singular wording, people phrases and the solo goals link", () => {
  const parts = briefingSummary({
    ...zero,
    needsYou: 1,
    peopleOverdue: 2,
    peopleNeedReply: 1,
    goalsAtRisk: 1,
    monitoring: 1,
  });
  assert.deepEqual(
    parts.map((p) => p.text),
    ["1 thing needs you", "2 overdue", "1 awaiting reply", "1 goal at risk", "1 signal monitored"],
  );
  assert.deepEqual(parts[1].target, { kind: "panel", panel: "people" });
  assert.deepEqual(parts[3].target, { kind: "href", href: "/goals" });
});

test("ageLabel and daysUntil", () => {
  const now = new Date(2026, 9, 3, 12, 0);
  assert.equal(ageLabel(new Date(2026, 9, 3, 11, 30).toISOString(), now), "30m");
  assert.equal(ageLabel(new Date(2026, 9, 2, 12, 0).toISOString(), now), "24h");
  assert.equal(ageLabel(new Date(2026, 8, 24, 12, 0).toISOString(), now), "9d");
  assert.equal(ageLabel("nonsense", now), "");
  assert.equal(daysUntil(new Date(2026, 9, 3, 1, 0).toISOString(), now), -1);
  assert.equal(daysUntil(new Date(2026, 9, 3, 20, 0).toISOString(), now), 0);
  assert.equal(daysUntil(null, now), null);
});


test("Home's headline says what needs you, then what it handled and watches", () => {
  assert.deepEqual(homeHeadline(1, 6, 2), {
    lead: "One thing needs you.",
    rest: "I handled 6 since yesterday and I'm watching 2 for you.",
    clear: false,
  });
  assert.deepEqual(homeHeadline(3, 4, 0), { lead: "3 things need you.", rest: "I handled 4 since yesterday.", clear: false });
  assert.deepEqual(homeHeadline(2, 0, 0), { lead: "2 things need you.", rest: "", clear: false });
  assert.deepEqual(homeHeadline(0, 0, 0), { lead: "All clear.", rest: "Nothing needs you.", clear: true });
  assert.equal(homeHeadline(0, 0, 2).rest, "Nothing needs you. I'm watching 2 things for you.");
  assert.equal(homeHeadline(0, 0, 1).rest, "Nothing needs you. I'm watching 1 thing for you.");
  assert.equal(homeHeadline(0, 3, 0).rest, "Nothing needs you. I handled 3 since yesterday.");
});

test("hand me something lists the asks with words, each once, four at most", () => {
  assert.deepEqual(handAsks(["Plan my week", "  ", "", 7, null, " Draft the board update ", "Plan my week"]), [
    "Plan my week",
    "Draft the board update",
  ]);
  assert.deepEqual(handAsks(["a", "b", "c", "d", "e", "f"]), ["a", "b", "c", "d"]);
  assert.deepEqual(handAsks(["a", "b", "c"], 2), ["a", "b"]);
});

test("hand me something has no rows when there are no asks to show", () => {
  for (const nothing of [undefined, null, [], "Plan my week", { prompts: ["a"] }, ["", "   "]]) {
    assert.deepEqual(handAsks(nothing), []);
  }
});

const roster = [
  { id: 1, full_name: "Jordan Lee" },
  { id: 2, full_name: "Alex Cohen" },
  { id: 3, full_name: "Dana Ruiz" },
  { id: 4, full_name: "Sam Park" },
];
const item = (headline, routed_to_person_id) => ({ headline, routed_to_person_id });

test("the teammates row says who it waits on and what", () => {
  assert.equal(waitingOnTeam([], roster), null);
  assert.deepEqual(waitingOnTeam([item("Sign off on the launch post", 2)], roster), {
    title: "Waiting on Alex",
    detail: "Sign off on the launch post",
    line: "Waiting on Alex: Sign off on the launch post",
  });
  assert.deepEqual(waitingOnTeam([item("Sign off on the launch post", 2), item("Renew the lease", 2), item("Pick a venue", 3)], roster), {
    title: "Waiting on Alex and Dana",
    detail: "Sign off on the launch post, and 2 more",
    line: "3 waiting on Alex and Dana",
  });
  assert.equal(
    waitingOnTeam([item("a", 2), item("b", 3), item("c", 4), item("d", 1)], roster).title,
    "Waiting on Alex, Dana and 2 others",
  );
  assert.equal(waitingOnTeam([item("a", 2), item("b", 3), item("c", 4)], roster).title, "Waiting on Alex, Dana and 1 other");
});

test("the teammates row still reads when it can't tell who", () => {
  assert.equal(waitingOnTeam([item("Pick a venue", 99)], roster).title, "Waiting on someone else");
  assert.equal(waitingOnTeam([item("a", null), item("b", 99)], roster).title, "Waiting on teammates");
  assert.equal(waitingOnTeam([item("a", null), item("b", 3)], roster).title, "Waiting on Dana and others");
  assert.equal(waitingOnTeam([item("a", 2), item("b", 3), item("c", 99)], roster).title, "Waiting on Alex, Dana and others");
  assert.equal(waitingOnTeam([item("a", 2), item("b", 3), item("c", 4), item("d", 99)], roster).title, "Waiting on Alex, Dana and others");
  assert.deepEqual(waitingOnTeam([item("  ", 2)], roster), { title: "Waiting on Alex", detail: "", line: "Waiting on Alex" });
});

test("the teammates row tells apart two people who share a first name", () => {
  const two = [...roster, { id: 5, full_name: "Alex Moreno" }];
  assert.equal(waitingOnTeam([item("a", 2), item("b", 5)], two).title, "Waiting on Alex Cohen and Alex Moreno");
  assert.equal(waitingOnTeam([item("a", 2), item("b", 2)], two).title, "Waiting on Alex");
});

test("the headline doesn't say all clear while something is off track", () => {
  assert.deepEqual(homeHeadline(0, 0, 0, true), { lead: "Nothing needs you.", rest: "", clear: false });
  assert.deepEqual(homeHeadline(0, 3, 1, true), {
    lead: "Nothing needs you.",
    rest: "I handled 3 since yesterday and I'm watching 1 for you.",
    clear: false,
  });
  assert.deepEqual(homeHeadline(2, 0, 0, true), { lead: "2 things need you.", rest: "", clear: false });
  assert.equal(homeHeadline(0, 0, 0, false).lead, "All clear.");
});

test("off track is a goal at risk or off track, or an answer someone waits on", () => {
  const dept = { at_risk_count: 0, off_track_count: 0, awaiting_count: 0 };
  assert.equal(teamOffTrack([], []), false);
  assert.equal(teamOffTrack([dept], [{ awaiting_count: 0 }, {}]), false);
  assert.equal(teamOffTrack([dept, { ...dept, at_risk_count: 2 }], []), true);
  assert.equal(teamOffTrack([{ ...dept, off_track_count: 1 }], []), true);
  assert.equal(teamOffTrack([{ ...dept, awaiting_count: 1 }], []), true);
  assert.equal(teamOffTrack([dept], [{ awaiting_count: 1 }]), true);
});

test("the watching row says what changed, and counts several", () => {
  assert.equal(watchingRow([]), null);
  const noticed = "I noticed this on the Acme pricing page I'm watching.";
  assert.deepEqual(watchingRow([{ headline: "Acme cut its Pro price to $40", view: { noticed } }]), {
    title: "Acme cut its Pro price to $40",
    detail: noticed,
    line: "Acme cut its Pro price to $40",
  });
  assert.deepEqual(
    watchingRow([{ headline: "Acme cut its Pro price to $40" }, { headline: "Initech filed a 10-Q" }, { headline: "x" }]),
    {
      title: "3 things changed on what I'm watching",
      detail: "Acme cut its Pro price to $40, and 2 more",
      line: "3 things changed on what I'm watching",
    },
  );
});

test("a waiting row says the job, who started it, why it waits and how long", async () => {
  const { waitingRow, waitingNote, waitedFor } = await import("../src/lib/briefingSummary.ts");
  const now = new Date("2026-11-24T14:00:00Z");
  const held = {
    headline: "Message Dana Price",
    created_at: "2026-11-19T15:07:45Z",
    routed_to_person_id: 12,
    view: { top: { title: "Message Dana Price", started: "I started this", because: "it's about money" } },
  };
  assert.deepEqual(waitingRow(held, now), {
    title: "Message Dana Price",
    started: "I started this",
    because: "Waits: it's about money",
    age: "4 days",
  });
  // A card with no held step keeps its headline and says nothing it can't.
  assert.deepEqual(waitingRow({ headline: "Reply to Dana", created_at: "2026-11-24T09:00:00Z" }, now), {
    title: "Reply to Dana",
    started: "",
    because: "",
    age: "today",
  });
  const people = [{ id: 12, full_name: "Dana Price" }, { id: 7, full_name: "Sam Lee" }];
  assert.equal(waitingNote([held], people, now), "Nothing for you to do. Dana has to OK it. Waiting 4 days.");
  assert.equal(
    waitingNote([held, { ...held, created_at: "2026-11-23T10:00:00Z" }], people, now),
    "Nothing for you to do. Dana has to OK these. Oldest 4 days.",
  );
  assert.equal(
    waitingNote([held, { ...held, routed_to_person_id: 7, created_at: "2026-11-24T10:00:00Z" }], people, now),
    "Nothing for you to do. Each waits for its person's OK. Oldest 4 days.",
  );
  assert.equal(waitingNote([], people, now), "");
  assert.equal(
    waitingNote([{ ...held, routed_to_person_id: 99 }], [], now),
    "Nothing for you to do. It waits for someone else's OK. Waiting 4 days.",
  );
  assert.equal(waitedFor("2026-11-23T13:00:00Z", now), "1 day");
});
