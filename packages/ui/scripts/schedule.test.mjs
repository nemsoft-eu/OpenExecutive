import assert from "node:assert/strict";
import { test } from "node:test";

import { checkedLabel, filterFor, filterItems, nextLabel, scheduleHref, scheduleLine } from "../src/lib/schedule.ts";

const now = new Date(2026, 10, 10, 15, 0); // Tue Nov 10, 3 pm local

const row = (over) => ({
  key: "k", kind: "workflow", title: "t", detail: "", at: null, last_checked: null,
  cadence: "", href: null, stop_key: null, ...over,
});

test("says when it runs next", () => {
  assert.equal(nextLabel(new Date(2026, 10, 10, 17, 30).toISOString(), now), "Today, 5:30 pm");
  assert.equal(nextLabel(new Date(2026, 10, 11, 8, 0).toISOString(), now), "Tomorrow, 8:00 am");
  assert.equal(nextLabel(new Date(2026, 10, 13, 16, 0).toISOString(), now), "Fri, 4:00 pm");
  assert.equal(nextLabel(new Date(2026, 10, 30, 9, 0).toISOString(), now), "Nov 30, 9:00 am");
  assert.equal(nextLabel(new Date(2026, 10, 10, 14, 0).toISOString(), now), "Soon");
  assert.equal(nextLabel(null, now), "");
});

test("a watch says when it last checked", () => {
  assert.equal(checkedLabel(new Date(2026, 10, 9, 16, 0).toISOString(), now), "Checked 23h ago");
  assert.equal(checkedLabel(new Date(2026, 10, 5, 15, 0).toISOString(), now), "Checked 5d ago");
  assert.equal(checkedLabel(new Date(2026, 10, 10, 14, 50).toISOString(), now), "Checked 10m ago");
  assert.equal(checkedLabel(null, now), "Not checked yet");
});

test("the line joins when and how often", () => {
  const at = new Date(2026, 10, 11, 9, 0).toISOString();
  assert.equal(scheduleLine(row({ at, cadence: "every week" }), now), "Tomorrow, 9:00 am · every week");
  assert.equal(scheduleLine(row({ kind: "follow_up", at }), now), "Tomorrow, 9:00 am");
  const checked = new Date(2026, 10, 9, 16, 0).toISOString();
  assert.equal(
    scheduleLine(row({ kind: "watch", at, last_checked: checked, cadence: "once a day" }), now),
    "Checked 23h ago · once a day",
  );
});

test("filters by kind, from the link's ?show=", () => {
  assert.equal(filterFor("watch"), "watch");
  assert.equal(filterFor("jobs"), "all");
  assert.equal(filterFor(null), "all");
  assert.equal(scheduleHref("standing"), "/schedule?show=standing");
  assert.equal(scheduleHref(), "/schedule");
  const items = [row({ key: "a", kind: "watch" }), row({ key: "b", kind: "always" })];
  assert.deepEqual(filterItems(items, "watch").map((i) => i.key), ["a"]);
  assert.equal(filterItems(items, "all").length, 2);
});
