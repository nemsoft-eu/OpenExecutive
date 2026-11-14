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
    "Unload restores your profile, Markdown documents, decisions, initiatives, advice, team roster, departments and workspace settings from the snapshot. Everything else comes back empty, such as chat history, uploaded PDF, Word, Excel, CSV and text files, contacts, learned facts, department notes, and Act as me, Take the lead and Always in the loop settings.",
  );
});
