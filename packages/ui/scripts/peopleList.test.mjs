import assert from "node:assert/strict";
import test from "node:test";
import { lastContactLabel, matchesSearch } from "../src/lib/peopleList.ts";

const p = (id, full_name, role = "", email = null, email_aliases = []) => ({ id, full_name, role, email, email_aliases });

const PRIYA = p(1, "Priya Shah", "Partner, Northwind Legal LLP (outside counsel)", "priya@northwind.example");
const ANA = p(6, "Ana Ruiz", "Ana Ruiz, Ruiz & Co, P.C.", "ana@isp.example", ["ana@ruizco.example"]);

test("search matches name, role and any address, every word", () => {
  assert.ok(matchesSearch(PRIYA, "priya"));
  assert.ok(matchesSearch(PRIYA, "northwind partner"));
  assert.ok(matchesSearch(ANA, "ruizco"));
  assert.ok(!matchesSearch(PRIYA, "contoso"));
  assert.ok(matchesSearch(PRIYA, "   "));
});

test("last contact reads as a verb and a day", () => {
  assert.equal(lastContactLabel({ person_id: 1, at: "2026-10-29T15:00:00Z", how: "emailed" }), "Emailed Oct 29");
  assert.equal(lastContactLabel({ person_id: 1, at: "2026-10-22T15:00:00Z", how: "met" }), "Met Oct 22");
  assert.equal(lastContactLabel({ person_id: 1, at: "2026-10-30T15:00:00Z", how: "heard" }), "Heard from Oct 30");
  assert.equal(lastContactLabel(undefined), "");
  assert.equal(lastContactLabel({ person_id: 1, at: "nope", how: "met" }), "");
});
