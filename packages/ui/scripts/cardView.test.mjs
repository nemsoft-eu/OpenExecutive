// Needs you cards: what a card says and which buttons it has come from the
// server (ProposalItem.view, briefing/card_view.py), the same view every
// screen draws. The card decides nothing from a card's raw fields, so no two
// screens can drift apart; a card that starts reading them again fails here.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const CARD = new URL("../src/components/briefing/ProposalCard.tsx", import.meta.url);
const EXAMPLES = new URL("../../core/tests/fixtures/card_views.json", import.meta.url);

test("the web card works out nothing from a card's raw fields", () => {
  const src = readFileSync(CARD, "utf8");
  for (const field of ["topic_tags", "recommended_move", "review_verdict", "failed_run_id", "artifact_format", ".category", "suggested_workflow"]) {
    assert.ok(!src.includes(field), `ProposalCard reads ${field}`);
  }
});

test("the card handles every action and place the server sends", () => {
  const src = readFileSync(CARD, "utf8");
  const examples = JSON.parse(readFileSync(EXAMPLES, "utf8"));
  const actions = new Set();
  const places = new Set();
  for (const item of Object.values(examples)) {
    for (const b of item.view.buttons) {
      actions.add(b.action);
      places.add(b.place);
    }
  }
  for (const action of actions) assert.ok(src.includes(`case "${action}":`), `ProposalCard doesn't carry out ${action}`);
  for (const place of places) assert.ok(src.includes(`b.place === "${place}"`), `ProposalCard doesn't show ${place} buttons`);
});

test("an option or Or tell me shows only when this page can carry it out", () => {
  const src = readFileSync(CARD, "utf8");
  assert.ok(src.includes('(o.action === "approve" ? onApprove : onAnswer) !== undefined'), "options show without a handler");
  assert.ok(src.includes("tell_me: view.tell_me && (onApprove !== undefined || onAnswer !== undefined)"), "Or tell me shows without a handler");
});

test("an email reply card reads like every other card", () => {
  const src = readFileSync(new URL("../src/components/RepliesWaiting.tsx", import.meta.url), "utf8");
  for (const part of ['title="What happened"', 'title="What I\'ll do"', "See reply to ${firstNames(toList)}", "<SayBox", '"Skip"', ", cc me`"]) {
    assert.ok(src.includes(part), `the reply card has no ${part}`);
  }
  // A go said in Or tell me brings up the same confirm a tap on Send does.
  assert.ok(src.includes("if (updated.send_now) {"), "a go on a reply card does nothing");
  assert.ok(!src.includes("Don't send"), "the reply card still says Don't send");
});
