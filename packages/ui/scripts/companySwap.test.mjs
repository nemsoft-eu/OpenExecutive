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
    "Unload restores your profile, Markdown documents and the whole company database (chat history, decisions, team and contacts, learned facts, departments and every setting) from the snapshot. Everything else comes back empty, such as uploaded PDF, Word, Excel, CSV and text files.",
  );
});
