import assert from "node:assert/strict";
import test from "node:test";
import { cardDetail, cardDetails } from "../src/lib/cardDetails.ts";

test("a detail keeps a Google link and rows of cells", () => {
  const d = cardDetail({
    field: "Sheet",
    value: "Launch tracker",
    href: "https://drive.google.com/open?id=1abcdefghijk",
  });
  assert.deepEqual(d, { field: "Sheet", value: "Launch tracker", href: "https://drive.google.com/open?id=1abcdefghijk" });
  const rows = cardDetail({ field: "Values", value: "2 rows", table: [["A", 1], ["B", null]] });
  assert.deepEqual(rows.table, [["A", "1"], ["B", ""]]);
  assert.equal(rows.header, true);
  assert.equal(cardDetail({ field: "Values", value: "x", table: [["A"]], header: false }).header, false);
});

test("any other link, or a table that isn't rows, is dropped", () => {
  assert.equal(cardDetail({ field: "x", value: "y", href: "javascript:alert(1)" }).href, undefined);
  assert.equal(cardDetail({ field: "x", value: "y", href: "https://evil.example/drive.google.com/" }).href, undefined);
  assert.equal(cardDetail({ field: "x", value: "y", table: ["not", "rows"] }).table, undefined);
  assert.deepEqual(cardDetails("nope"), []);
  assert.equal(cardDetails([{ label: "To", value: "a@b.c" }])[0].field, "To");
});
