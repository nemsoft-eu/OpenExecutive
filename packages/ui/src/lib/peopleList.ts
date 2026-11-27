// Finding someone on the People page: the search, and when you were last in
// touch with a contact.
//
// Type-only imports, so `npm test` can exercise this under
// `node --experimental-strip-types` (see scripts/peopleList.test.mjs).

import type { Person } from "@/lib/api";

/** When you last emailed, heard from or met someone (GET /contacts/last-contact). */
export interface LastContact {
  person_id: number;
  at: string;
  how: "emailed" | "heard" | "met" | string;
}

type Row = Pick<Person, "full_name" | "role" | "email"> & { id?: number | null; email_aliases?: string[] };

function addresses(p: Row): string[] {
  return [p.email ?? "", ...(p.email_aliases ?? [])].map((a) => a.trim().toLowerCase()).filter(Boolean);
}

/** Does the person match what was typed? Name, role or any of their addresses. */
export function matchesSearch(p: Row, query: string): boolean {
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length === 0) return true;
  const hay = [p.full_name, p.role ?? "", ...addresses(p)].join(" ").toLowerCase();
  return words.every((w) => hay.includes(w));
}

/** "Emailed Oct 29", "Heard from Oct 30", "Met Oct 22", or "" when unknown. */
export function lastContactLabel(lc: LastContact | undefined): string {
  if (!lc) return "";
  const when = new Date(lc.at);
  if (Number.isNaN(when.getTime())) return "";
  const day = when.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  const verb = lc.how === "emailed" ? "Emailed" : lc.how === "met" ? "Met" : "Heard from";
  return `${verb} ${day}`;
}
