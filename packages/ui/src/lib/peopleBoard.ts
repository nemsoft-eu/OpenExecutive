// People as one page: what needs a look first (people the Executive added or
// suggests, details it filled in, the same address twice), then the team,
// then the contacts. One short row per person: a second line only when it
// says something.
// Type-only imports, so `npm test` runs it under --experimental-strip-types
// (scripts/peopleBoard.test.mjs).
import type { ContactChange } from "@/lib/api";
import type { LastContact } from "@/lib/peopleList";

export interface BoardPerson {
  id?: number | null;
  full_name: string;
  role?: string | null;
  email?: string | null;
  email_aliases?: string[];
  is_principal: boolean;
  on_leave_until?: string | null;
}

/** Contacts listed before "All N". */
export const CONTACTS_PREVIEW = 3;

/** "Nov 6", or "" for a date that doesn't read. */
export function dayLabel(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso.length === 10 ? `${iso}T12:00:00` : iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

/** The mark by a team member's name: the owner. */
export function roleBadge(person: Pick<BoardPerson, "is_principal">): "Owner" | null {
  return person.is_principal ? "Owner" : null;
}

export interface RowLine {
  text: string;
  /** Something to notice (on leave), shown in amber. */
  warn: boolean;
}

/** A team member's second line: on leave, else their role, else nothing. */
export function teamLine(person: Pick<BoardPerson, "role" | "on_leave_until">, today: string): RowLine | null {
  const leave = person.on_leave_until?.slice(0, 10);
  if (leave && leave >= today) return { text: `On leave until ${dayLabel(leave) || leave}`, warn: true };
  const role = (person.role ?? "").trim();
  return role ? { text: role, warn: false } : null;
}

/** A contact's second line: their role and when you were last in touch ("Emailed Nov 12"), whichever are known. */
export function contactLine(person: Pick<BoardPerson, "role">, lastLabel: string): RowLine | null {
  const text = [(person.role ?? "").trim(), lastLabel].filter(Boolean).join(" · ");
  return text ? { text, warn: false } : null;
}

function addresses(p: BoardPerson): string[] {
  return [...new Set([p.email ?? "", ...(p.email_aliases ?? [])].map((a) => a.trim().toLowerCase()).filter(Boolean))];
}

/** Two people with the same address: likely one person twice. One pair each, the first-listed first. */
export function sameEmailPairs<P extends BoardPerson>(people: P[]): { person: P; other: P }[] {
  const byAddress = new Map<string, P[]>();
  for (const p of people) {
    if (p.id == null) continue;
    for (const a of addresses(p)) byAddress.set(a, [...(byAddress.get(a) ?? []), p]);
  }
  const seen = new Set<string>();
  const out: { person: P; other: P }[] = [];
  for (const group of byAddress.values()) {
    for (let i = 1; i < group.length; i++) {
      const person = group[0]!;
      const other = group[i]!;
      const key = [person.id, other.id].sort().join(":");
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ person, other });
    }
  }
  return out;
}

/**
 * The Executive's contact changes, split for the page: people it added or
 * suggests are rows of their own; details it filled in fold into one line
 * ("I filled in 3 details"), since there can be many.
 */
export function splitChanges(changes: ContactChange[]): { people: ContactChange[]; details: ContactChange[] } {
  const open = changes.filter((c) => c.status === "suggested" || (c.status === "applied" && !c.seen));
  return {
    people: open.filter((c) => c.status === "suggested" || c.action === "add"),
    details: open.filter((c) => c.status === "applied" && c.action !== "add"),
  };
}

/** Whether Got it has anything to clear (suggestions stay until answered). */
export function hasSeenable(changes: ContactChange[]): boolean {
  return changes.some((c) => c.status === "applied" && !c.seen);
}

export function detailsLine(n: number): string {
  return `I filled in ${n === 1 ? "1 detail" : `${n} details`} from your email`;
}

/** Contacts in the order the page lists them: last in touch first, then by name. */
export function contactsByRecent<P extends BoardPerson>(contacts: P[], last: Map<number, LastContact>): P[] {
  const at = (p: P) => (p.id == null ? "" : (last.get(p.id)?.at ?? ""));
  return [...contacts].sort(
    (a, b) => at(b).localeCompare(at(a)) || a.full_name.localeCompare(b.full_name, undefined, { sensitivity: "base" }),
  );
}

/** The team in the order the page lists them: the owner first, then by name. */
export function teamInOrder<P extends BoardPerson>(team: P[]): P[] {
  return [...team].sort((a, b) =>
    a.is_principal !== b.is_principal
      ? a.is_principal
        ? -1
        : 1
      : a.full_name.localeCompare(b.full_name, undefined, { sensitivity: "base" }),
  );
}


/**
 * What Add starts on. Someone using Open Executive just for themselves
 * mostly adds contacts, so Add starts there (a team member would switch to
 * team mode). Otherwise it starts on a team member.
 */
export function addFormKind(v: { mode: "solo" | "team"; seesContacts: boolean }): "team" | "contact" {
  return v.seesContacts && v.mode === "solo" ? "contact" : "team";
}
