import assert from "node:assert/strict";
import test from "node:test";
import {
  BRIEFING_REFRESH_INTERVAL_MS,
  BRIEFING_TICK_MS,
  FOCUS_REFRESH_MIN_GAP_MS,
  narrativeRepollDelay,
  newestAnswer,
  narrativeUpdatedLabel,
  shouldRefreshOnFocus,
  shouldRefreshOnTick,
} from "../src/lib/narrativeFreshness.ts";

test("re-polls a few times while the header is rewritten, then stops", () => {
  assert.equal(narrativeRepollDelay(0), 4000);
  assert.equal(narrativeRepollDelay(1), 10000);
  assert.equal(narrativeRepollDelay(2), 20000);
  assert.equal(narrativeRepollDelay(3), null);
  assert.equal(narrativeRepollDelay(-1), null);
  assert.equal(narrativeRepollDelay(1.5), null);
});

test("focus refresh is throttled", () => {
  assert.equal(shouldRefreshOnFocus(0, FOCUS_REFRESH_MIN_GAP_MS - 1), false);
  assert.equal(shouldRefreshOnFocus(0, FOCUS_REFRESH_MIN_GAP_MS), true);
});

test("timed refresh comes a minute after the last pull of any kind", () => {
  assert.equal(shouldRefreshOnTick(0, BRIEFING_REFRESH_INTERVAL_MS - 1), false);
  assert.equal(shouldRefreshOnTick(0, BRIEFING_REFRESH_INTERVAL_MS), true);
  // A pull at 31 s (Approve, focus): the next timed one is due at 91 s and
  // the tick catches it within one tick, so nothing waits much past a minute.
  const due = 31_000 + BRIEFING_REFRESH_INTERVAL_MS;
  const tick = Math.ceil(due / BRIEFING_TICK_MS) * BRIEFING_TICK_MS;
  assert.equal(shouldRefreshOnTick(31_000, tick), true);
  assert.ok(tick - 31_000 < BRIEFING_REFRESH_INTERVAL_MS + BRIEFING_TICK_MS);
});

test("an older /today answer never overwrites a newer one already shown", () => {
  const pulls = newestAnswer();
  const timed = pulls.begin();
  const approve = pulls.begin();
  assert.equal(pulls.accept(approve), true); // Approve's pull lands first: card gone
  assert.equal(pulls.accept(timed), false); // the older timed pull is dropped
});

test("an older answer still shows while the newer pull hasn't landed or failed", () => {
  const pulls = newestAnswer();
  const first = pulls.begin(); // slow first load
  pulls.begin(); // a focus pull starts, then fails (never accepted)
  assert.equal(pulls.accept(first), true);
  const later = pulls.begin();
  assert.equal(pulls.accept(later), true);
});

test("a change made on the page drops every pull already under way", () => {
  const pulls = newestAnswer();
  const timed = pulls.begin(); // left before the row was dismissed
  pulls.changedHere();
  assert.equal(pulls.accept(timed), false);
  const next = pulls.begin();
  assert.equal(pulls.accept(next), true);
});

test("updated label reads as recency, then as a local clock time", () => {
  // Local-time constructors, so the labels hold in any test-runner time zone.
  const written = new Date(2026, 8, 28, 14, 4);
  const at = (minutes) => new Date(written.getTime() + minutes * 60_000);
  const iso = written.toISOString();
  assert.equal(narrativeUpdatedLabel(iso, at(0.5)), "Updated just now");
  assert.equal(narrativeUpdatedLabel(iso, at(12)), "Updated 12 min ago");
  assert.equal(narrativeUpdatedLabel(iso, at(90)), "Updated 2:04 PM");
  const morning = new Date(2026, 8, 28, 0, 7);
  assert.equal(
    narrativeUpdatedLabel(morning.toISOString(), new Date(2026, 8, 28, 9, 0)),
    "Updated 12:07 AM",
  );
  assert.equal(narrativeUpdatedLabel(null, at(0)), null);
  assert.equal(narrativeUpdatedLabel("not a date", at(0)), null);
});
