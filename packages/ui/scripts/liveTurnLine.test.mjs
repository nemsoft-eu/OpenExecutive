// The running turn's one live line (lib/turnSteps.ts liveLineKind). Run by `npm test`.
import assert from "node:assert/strict";
import test from "node:test";

import { liveLineKind } from "../src/lib/turnSteps.ts";

const base = { isLoading: true, steps: [], hasProgress: false, inCommittee: false, statusShown: true };

test("a long turn shows its steps line only, never the plain status too", () => {
  assert.equal(liveLineKind({ ...base, steps: ["Opening the page…", "Looking at the page…"] }), "steps");
  assert.equal(liveLineKind({ ...base, hasProgress: true }), "steps");
});

test("a short turn shows the plain status when it has something to say", () => {
  assert.equal(liveLineKind({ ...base, steps: ["Looking at the page…"] }), "status");
  assert.equal(liveLineKind({ ...base, statusShown: false }), null);
  assert.equal(liveLineKind({ ...base, steps: ["a", "b"], inCommittee: true }), "status");
  assert.equal(liveLineKind({ ...base, isLoading: false, steps: ["a", "b"], statusShown: false }), null);
});

test("the chat's hand-over ask takes the server's words and only an id it minted", async () => {
  const { toHandOverAsk } = await import("../src/lib/handOverAsk.ts");
  const id = "a".repeat(32);
  const ask = toHandOverAsk({
    type: "hand_over", id, text: "Waiting on you:  enter the code", detail: "Do it in your browser.",
    done_label: "Done", stop_label: "Stop",
  });
  assert.deepEqual(ask, { id, text: "Waiting on you: enter the code", detail: "Do it in your browser.", doneLabel: "Done", stopLabel: "Stop" });
  assert.equal(toHandOverAsk({ type: "hand_over", id: "../x", text: "a", detail: "", done_label: "", stop_label: "" }), null);
  assert.equal(toHandOverAsk({ type: "hand_over", id, text: " ", detail: "", done_label: "", stop_label: "" }), null);
});
