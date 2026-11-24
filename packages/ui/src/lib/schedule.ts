// The Schedule page (GET /schedule): what a row's tag says, the filters, and
// the line under its title ("Tomorrow, 8:00 am · every week", "Checked 23h
// ago · once a day"). Pure (type imports only), so scripts/schedule.test.mjs
// runs it under plain node. The phone's packages/mobile/src/lib/schedule.ts
// says the same words.
import type { ScheduleItem, ScheduleKind } from "./api";

export const SCHEDULE_KINDS: ScheduleKind[] = ["follow_up", "standing", "workflow", "watch", "always"];

export type ScheduleFilter = "all" | ScheduleKind;

/** The tag on a row. */
export const KIND_TAG: Record<ScheduleKind, string> = {
  follow_up: "Follow-up",
  standing: "Standing",
  workflow: "Workflow",
  watch: "Watch",
  always: "Always",
};

/** The filter chips, in order. */
export const FILTER_LABEL: Record<ScheduleFilter, string> = {
  all: "All",
  follow_up: "Follow-ups",
  standing: "Standing work",
  workflow: "Workflows",
  watch: "Watches",
  always: "Always",
};

export const SCHEDULE_FILTERS: ScheduleFilter[] = ["all", ...SCHEDULE_KINDS];

/** `/schedule?show=watch`: the filter a link opens on; anything else is All. */
export function filterFor(show: string | null | undefined): ScheduleFilter {
  return (SCHEDULE_KINDS as string[]).includes(show ?? "") ? (show as ScheduleKind) : "all";
}

/** The page's address with a filter on (the five places link here). */
export function scheduleHref(kind?: ScheduleKind): string {
  return kind ? `/schedule?show=${kind}` : "/schedule";
}

export function filterItems(items: ScheduleItem[], filter: ScheduleFilter): ScheduleItem[] {
  return filter === "all" ? items : items.filter((item) => item.kind === filter);
}

const WEEKDAY = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DAY_MS = 86_400_000;

function clock(d: Date): string {
  const h = d.getHours();
  const hour = h % 12 === 0 ? 12 : h % 12;
  return `${hour}:${String(d.getMinutes()).padStart(2, "0")} ${h < 12 ? "am" : "pm"}`;
}

function startOfDay(d: Date): number {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
}

/** "Today, 3:00 pm", "Tomorrow, 8:00 am", "Fri, 4:00 pm", "Nov 20, 9:00 am";
 *  "Soon" once it is due; "" for no time. */
export function nextLabel(iso: string | null | undefined, now: Date = new Date()): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  if (d.getTime() <= now.getTime()) return "Soon";
  const days = Math.round((startOfDay(d) - startOfDay(now)) / DAY_MS);
  if (days === 0) return `Today, ${clock(d)}`;
  if (days === 1) return `Tomorrow, ${clock(d)}`;
  if (days < 7) return `${WEEKDAY[d.getDay()]}, ${clock(d)}`;
  return `${MONTH[d.getMonth()]} ${d.getDate()}, ${clock(d)}`;
}

/** "Checked 23h ago", "Checked 5d ago", "Checked just now";
 *  "Not checked yet" for no time. */
export function checkedLabel(iso: string | null | undefined, now: Date = new Date()): string {
  const at = iso ? new Date(iso).getTime() : Number.NaN;
  if (Number.isNaN(at)) return "Not checked yet";
  const mins = Math.max(0, Math.round((now.getTime() - at) / 60_000));
  if (mins < 1) return "Checked just now";
  if (mins < 60) return `Checked ${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 48) return `Checked ${hours}h ago`;
  return `Checked ${Math.round(hours / 24)}d ago`;
}

/** The line under a row's title: when it acts next (a watch: when it last
 *  checked), then how often. */
export function scheduleLine(item: ScheduleItem, now: Date = new Date()): string {
  const when = item.kind === "watch" ? checkedLabel(item.last_checked, now) : nextLabel(item.at, now);
  return [when, item.cadence].filter(Boolean).join(" · ");
}
