import assert from "node:assert/strict";
import test from "node:test";
import { APPROVALS_LEVELS, approvalsTitle } from "../src/lib/approvalsLevel.ts";

test("the three levels, in order", () => {
  assert.deepEqual(
    APPROVALS_LEVELS.map((l) => l.title),
    ["Careful", "Balanced", "Hands-off"],
  );
});

test("a level's name, and Careful for anything unknown", () => {
  assert.equal(approvalsTitle("outsiders"), "Balanced");
  assert.equal(approvalsTitle("trust"), "Hands-off");
  assert.equal(approvalsTitle("nope"), "Careful");
});

test("one source for level names and order, used by every picker", async () => {
  const { LEVEL_ORDER, LEVEL_NAMES, atMost } = await import("../src/lib/approvalsLevel.ts");
  assert.deepEqual(LEVEL_ORDER, ["ask", "outsiders", "trust"]);
  assert.deepEqual(LEVEL_NAMES, { ask: "Careful", outsiders: "Balanced", trust: "Hands-off" });
  assert.equal(atMost("outsiders", "trust"), true);
  assert.equal(atMost("trust", "outsiders"), false);
  assert.equal(atMost("ask", "ask"), true);
});

test("onboarding's department levels use the level names", async () => {
  const { AUTHORITY_LEVELS } = await import("../src/lib/approvalsLevel.ts");
  assert.deepEqual(
    AUTHORITY_LEVELS.map((a) => [a.value, a.label]),
    [
      ["escalate", "Careful"],
      ["propose_only", "Balanced"],
      ["auto_execute", "Hands-off"],
    ],
  );
});
