// One wording for a connection that stopped working (lib/reconnect.ts).
import assert from "node:assert/strict";
import test from "node:test";

import { MAILBOX_BROKEN, RECONNECT, mailboxNeedsReconnecting } from "../src/lib/reconnect.ts";

test("a refused or mismatched sign-in needs reconnecting; anything else doesn't", () => {
  assert.equal(mailboxNeedsReconnecting("needs_reconnect"), true);
  assert.equal(mailboxNeedsReconnecting("mismatch"), true);
  for (const s of ["connected", "not_configured", "error", null, undefined]) assert.equal(mailboxNeedsReconnecting(s), false);
});

test("the words match Home's row from the server", () => {
  // briefing/needs_reconnecting.py MAILBOX_TITLE and RECONNECT.
  assert.equal(MAILBOX_BROKEN, "Your mailbox needs reconnecting");
  assert.equal(RECONNECT, "Reconnect");
});

