/** An email's words in runs: plain text, and the Markdown pipe tables in it
 * (a header row, its separator row, then the rows with a pipe under them).
 * The same rule the email's own HTML part uses (delegation/email_html.py
 * `_blocks`), so a card shows the table the way the email will. Only the
 * tables are read as Markdown: every other line, and every cell, reads as
 * written. */

export type EmailBlock =
  | { kind: "text"; text: string }
  | { kind: "table"; header: string[]; rows: string[][]; align: ("left" | "center" | "right")[] };

// A table's separator row: pipes and dashes, optional colons for alignment.
const SEPARATOR = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;

function cells(row: string): string[] {
  let inner = row.trim();
  if (inner.startsWith("|")) inner = inner.slice(1);
  if (inner.endsWith("|")) inner = inner.slice(0, -1);
  return inner.split("|").map((c) => c.trim());
}

function alignOf(cell: string): "left" | "center" | "right" {
  const left = cell.startsWith(":");
  const right = cell.endsWith(":");
  return left && right ? "center" : right ? "right" : "left";
}

// Where a forwarded message starts (email_html.py `split_forward`): its
// "Forwarded message" line, or the "From:" / "Sent:" header Outlook's
// forward starts with. The email sends what is under it as written, so the
// card shows it as written too, tables and all.
const FORWARDED = /^-{2,}\s*Forwarded message\s*-{2,}\s*$/im;
const OUTLOOK_FORWARDED = /^From:[^\n]*\n(?:[ \t]*\n)*Sent:/m;

export function emailBlocks(text: string): EmailBlock[] {
  const all = text || "";
  const at = all.search(FORWARDED) >= 0 ? all.search(FORWARDED) : all.search(OUTLOOK_FORWARDED);
  if (at > 0) {
    const note = all.slice(0, at).replace(/\n+$/, "");
    return [...emailBlocks(note), { kind: "text", text: "\n" + all.slice(at) }];
  }
  if (at === 0) return [{ kind: "text", text: all }];
  const lines = all.split("\n");
  const blocks: EmailBlock[] = [];
  let plain: string[] = [];
  const flush = () => {
    if (plain.length) blocks.push({ kind: "text", text: plain.join("\n") });
    plain = [];
  };
  let i = 0;
  while (i < lines.length) {
    const line = lines[i] ?? "";
    const next = lines[i + 1] ?? "";
    if (line.includes("|") && next.includes("|") && SEPARATOR.test(next) && cells(line).length === cells(next).length) {
      const header = cells(line);
      const align = cells(next).map(alignOf);
      let end = i + 2;
      const rows: string[][] = [];
      while (end < lines.length && (lines[end] ?? "").includes("|") && (lines[end] ?? "").trim()) {
        // A row has the header's cells: short rows are filled, long ones cut.
        const row = cells(lines[end] ?? "");
        rows.push(header.map((_, c) => row[c] ?? ""));
        end += 1;
      }
      flush();
      blocks.push({ kind: "table", header, rows, align });
      i = end;
      continue;
    }
    plain.push(line);
    i += 1;
  }
  flush();
  return blocks;
}
