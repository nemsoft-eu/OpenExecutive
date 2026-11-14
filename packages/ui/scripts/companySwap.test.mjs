import assert from "node:assert/strict";
import test from "node:test";
import { loadNeedsConfirm, returnNote } from "../src/lib/companySwap.ts";

test("loading over your own company asks first", () => {
  assert.equal(loadNeedsConfirm(null), true);
  assert.equal(loadNeedsConfirm(""), true);
});

test("loading over another fixture does not ask", () => {
  assert.equal(loadNeedsConfirm("acme"), false);
});

test("the return note names what comes back and what comes back empty", () => {
  assert.equal(
    returnNote("Unload"),
    "Unload restores your profile, documents, decisions, initiatives, advice, people, departments and workspace settings from the snapshot. Chat history, learned facts, department notes, and Act as me, Take the lead and Always in the loop settings come back empty.",
  );
  assert.ok(returnNote("Going back to it").startsWith("Going back to it restores your profile"));
});
