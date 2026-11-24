// The Home screen's wording: the greeting, the one-line summary that replaced
// the row of status pills, and the single status chip a collapsed proposal
// card keeps. Kept apart from the components so `npm test` can check it (see
// scripts/briefingSummary.test.mjs). No imports, so the test can load this
// under `node --experimental-strip-types`.

/** "Good morning" before noon, "Good afternoon" until 6pm, then "Good evening". */
export function greeting(firstName: string | undefined, now: Date = new Date()): string {
  const h = now.getHours();
  const part = h < 12 ? "morning" : h < 18 ? "afternoon" : "evening";
  const name = firstName?.trim();
  return name ? `Good ${part}, ${name}` : `Good ${part}`;
}

/** Where a summary phrase leads: a lane on the page, one of the tiles' side
 * panels, or another route. */
export type SummaryTarget =
  | { kind: "needsYou" }
  | { kind: "panel"; panel: "handled" | "people" | "departments" | "inFlight" | "monitoring" }
  | { kind: "href"; href: string };

export interface SummaryPart {
  text: string;
  target: SummaryTarget;
}

export interface SummaryCounts {
  needsYou: number;
  handledOvernight: number;
  peopleOverdue: number;
  peopleNeedReply: number;
  deptAtRisk: number;
  /** Solo only: at-risk + off-track goals, which have no tile on the page. */
  goalsAtRisk?: number;
  inFlight: number;
  monitoring: number;
}

const plural = (n: number, one: string, many: string) => (n === 1 ? one : many);

/** The summary under the greeting, as phrases in the order they are read:
 * what needs you first, then what the Executive did on its own (trust and
 * relief), then the rest by urgency. Zero counts are dropped; an empty list
 * means all clear. Each phrase opens the detail it counts. */
export function briefingSummary(c: SummaryCounts): SummaryPart[] {
  const parts: SummaryPart[] = [];
  if (c.needsYou > 0)
    parts.push({
      text: `${c.needsYou} ${plural(c.needsYou, "thing needs", "things need")} you`,
      target: { kind: "needsYou" },
    });
  if (c.handledOvernight > 0)
    parts.push({
      text: `The Executive handled ${c.handledOvernight} on its own`,
      target: { kind: "panel", panel: "handled" },
    });
  if (c.peopleOverdue > 0)
    parts.push({ text: `${c.peopleOverdue} overdue`, target: { kind: "panel", panel: "people" } });
  if (c.peopleNeedReply > 0)
    parts.push({
      text: `${c.peopleNeedReply} awaiting reply`,
      target: { kind: "panel", panel: "people" },
    });
  if (c.deptAtRisk > 0)
    parts.push({
      text: `${c.deptAtRisk} ${plural(c.deptAtRisk, "department", "departments")} at risk`,
      target: { kind: "panel", panel: "departments" },
    });
  if (c.goalsAtRisk && c.goalsAtRisk > 0)
    parts.push({
      text: `${c.goalsAtRisk} ${plural(c.goalsAtRisk, "goal", "goals")} at risk`,
      target: { kind: "href", href: "/goals" },
    });
  if (c.inFlight > 0)
    parts.push({ text: `${c.inFlight} under way`, target: { kind: "panel", panel: "inFlight" } });
  if (c.monitoring > 0)
    parts.push({
      text: `${c.monitoring} ${plural(c.monitoring, "signal", "signals")} monitored`,
      target: { kind: "panel", panel: "monitoring" },
    });
  return parts;
}

/** Compact age label ("3h", "9d") for a card chip; "" for an unparseable stamp. */
export function ageLabel(iso: string | null | undefined, now: Date = new Date()): string {
  if (!iso) return "";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const mins = Math.max(0, Math.round((now.getTime() - t) / 60000));
  if (mins < 60) return `${mins}m`;
  const hours = Math.round(mins / 60);
  if (hours < 48) return `${hours}h`;
  return `${Math.round(hours / 24)}d`;
}

/** Whole days between now and an ISO stamp: 0 = due within the day, negative =
 * past (floor, so an 11-hour-old deadline is -1 → "overdue", never "due
 * today"). null if unparseable. */
export function daysUntil(iso: string | null | undefined, now: Date = new Date()): number | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return null;
  return Math.floor((t - now.getTime()) / 86400000);
}

export type ChipTone = "rose" | "amber" | "neutral";

/** Home's headline, in the Executive's words: what needs you first, then
 *  what it handled in the last day and what it's watching. `lead` is the
 *  part Home colors (and, when something needs you, the link to Needs you).
 *  `clear` says nothing needs you and nothing is off track: with a goal at
 *  risk or someone waiting on an answer (`offTrack`), it says only that
 *  nothing needs you, never "All clear". */
export function homeHeadline(
  needsYou: number,
  done: number,
  watching: number,
  offTrack: boolean = false,
): { lead: string; rest: string; clear: boolean } {
  const none = needsYou <= 0;
  const clear = none && !offTrack;
  const lead = clear
    ? "All clear."
    : none
      ? "Nothing needs you."
      : needsYou === 1
        ? "One thing needs you."
        : `${needsYou} things need you.`;
  const did =
    done > 0 && watching > 0
      ? `I handled ${done} since yesterday and I'm watching ${watching} for you.`
      : done > 0
        ? `I handled ${done} since yesterday.`
        : watching > 0
          ? `I'm watching ${watching} ${plural(watching, "thing", "things")} for you.`
          : "";
  return { lead, rest: [clear ? "Nothing needs you." : "", did].filter(Boolean).join(" "), clear };
}

/** Whether something is off track that isn't an item in Needs you: a
 *  department with a goal at risk or off track or an answer it waits on, or
 *  a person waiting on one. Home doesn't say "All clear" then. */
export function teamOffTrack(
  departments: { at_risk_count?: number; off_track_count?: number; awaiting_count?: number }[],
  people: { awaiting_count?: number }[],
): boolean {
  return (
    departments.some((d) => (d.at_risk_count ?? 0) > 0 || (d.off_track_count ?? 0) > 0 || (d.awaiting_count ?? 0) > 0) ||
    people.some((p) => (p.awaiting_count ?? 0) > 0)
  );
}

/** Hand me something's rows: the suggested asks that have words, trimmed,
 *  each once, the first few. Anything else the server sends (not a list, a
 *  blank, a non-string) yields no row. */
export const HAND_ASKS_MAX = 4;
export function handAsks(prompts: unknown, max: number = HAND_ASKS_MAX): string[] {
  if (!Array.isArray(prompts)) return [];
  const asks: string[] = [];
  for (const p of prompts) {
    const text = typeof p === "string" ? p.trim() : "";
    if (text && !asks.includes(text)) asks.push(text);
  }
  return asks.slice(0, max);
}

/** The first word of a name, as Home says it ("Waiting on Alex"). */
function firstName(full: string): string {
  return full.trim().split(/\s+/)[0] ?? "";
}

/** Home's line for what waits on teammates: who (first names, in the order
 *  their items come, two named and the rest counted) and what (the first
 *  item, and how many more). `line` is the same in one line, for the web's
 *  at-a-glance row. Null when nothing waits on anyone else. */
export function waitingOnTeam(
  team: { headline: string; routed_to_person_id?: number | null }[],
  people: { id: number; full_name: string }[],
): { title: string; detail: string; line: string } | null {
  if (team.length === 0) return null;
  // Each person once, in the order their items come.
  const waited: { id: number; full_name: string }[] = [];
  // An item whose person isn't on the list still waits on someone.
  let unnamed = false;
  for (const p of team) {
    const who = people.find((x) => x.id === p.routed_to_person_id);
    if (!who || !firstName(who.full_name)) unnamed = true;
    else if (!waited.some((x) => x.id === who.id)) waited.push(who);
  }
  // First names, or the whole name for two people who share one.
  const names = waited.map((w) => {
    const first = firstName(w.full_name);
    return waited.some((x) => x.id !== w.id && firstName(x.full_name) === first) ? w.full_name.trim() : first;
  });
  const others = names.length - 2;
  // Who is left after the names it says: a count when it knows them all.
  const rest = unnamed ? "others" : others > 0 ? `${others} ${others === 1 ? "other" : "others"}` : "";
  const named = names.slice(0, 2);
  const whom =
    names.length === 0
      ? team.length === 1
        ? "someone else"
        : "teammates"
      : rest
        ? `${named.join(", ")} and ${rest}`
        : named.join(" and ");
  const title = `Waiting on ${whom}`;
  const first = (team[0]?.headline ?? "").trim();
  const more = team.length - 1;
  const detail = more > 0 ? (first ? `${first}, and ${more} more` : `${team.length} things`) : first;
  const line = team.length === 1 ? (first ? `${title}: ${first}` : title) : `${team.length} waiting on ${whom}`;
  return { title, detail, line };
}

/** Home's row for what changed on the watches: one change says what it is,
 *  with where it was noticed under it (the card's `noticed`); several are
 *  counted, with the first under them. `line` is the same in one line, for
 *  the web's at-a-glance row. Null when nothing changed. */
export function watchingRow(
  signals: { headline: string; view?: { noticed?: string } | null }[],
): { title: string; detail: string; line: string } | null {
  if (signals.length === 0) return null;
  const first = (signals[0]?.headline ?? "").trim();
  if (signals.length === 1) {
    const title = first || "Something changed on what I'm watching";
    return { title, detail: signals[0]?.view?.noticed ?? "", line: title };
  }
  const title = `${signals.length} things changed on what I'm watching`;
  const more = signals.length - 1;
  return { title, detail: first ? `${first}, and ${more} more` : "", line: title };
}

/** What a teammate's waiting card carries that Home's Waiting on list reads:
 *  its title, when it came, and how its held step opens (the server's
 *  `card_details.top`). */
export interface WaitingItem {
  headline: string;
  created_at: string;
  view?: { top?: Record<string, string> | null } | null;
}

/** "today", "1 day", "5 days": how long something has waited. */
export function waitedFor(iso: string | null | undefined, now: Date = new Date()): string {
  const t = iso ? Date.parse(iso) : NaN;
  if (Number.isNaN(t)) return "";
  const days = Math.floor(Math.max(0, now.getTime() - t) / 86_400_000);
  return days === 0 ? "today" : `${days} ${days === 1 ? "day" : "days"}`;
}

/** One row of Home's Waiting on list: the job in words, who started it,
 *  why it waits ("Waits: it's about money") and how long. */
export function waitingRow(
  item: WaitingItem,
  now: Date = new Date(),
): { title: string; started: string; because: string; age: string } {
  const top = item.view?.top ?? {};
  const because = (top.because ?? "").trim();
  return {
    title: (top.title ?? "").trim() || item.headline.trim(),
    started: (top.started ?? "").trim(),
    because: because ? `Waits: ${because}` : "",
    age: waitedFor(item.created_at, now),
  };
}

/** The line under Home's "Waiting on …" heading: that it needs nothing from
 *  the reader, whose OK it waits for, and how old the oldest is. */
export function waitingNote(
  team: { created_at: string; routed_to_person_id?: number | null }[],
  people: { id: number; full_name: string }[],
  now: Date = new Date(),
): string {
  if (team.length === 0) return "";
  const ids = new Set(team.map((p) => p.routed_to_person_id ?? null));
  const one = ids.size === 1 ? people.find((x) => x.id === [...ids][0]) : undefined;
  const who = one ? firstName(one.full_name) || one.full_name.trim() : "";
  const it = team.length === 1 ? "it" : "these";
  const whose = who
    ? `${who} has to OK ${it}.`
    : ids.size === 1
      ? `${team.length === 1 ? "It waits" : "They wait"} for someone else's OK.`
      : `Each waits for its person's OK.`;
  const oldest = team
    .map((p) => p.created_at)
    .reduce((a, b) => (Date.parse(b) < Date.parse(a) ? b : a));
  const age = waitedFor(oldest, now);
  const since =
    age === "" ? "" : age === "today" ? " Since today." : team.length === 1 ? ` Waiting ${age}.` : ` Oldest ${age}.`;
  return `Nothing for you to do. ${whose}${since}`;
}
