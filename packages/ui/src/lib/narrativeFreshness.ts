// Keeping the Briefing's "What's going on" header current.
//
// /today serves the cached header and, when the picture has moved, rewrites
// it in the background (`narrative_stale: true` on the response). The page
// used to fetch once on mount, so it always showed the header from the
// previous visit. These helpers drive a short re-poll while a rewrite is in
// flight, a refresh when the tab comes back, and the "Updated …" stamp.
//
// No imports, so `npm test` can exercise this under
// `node --experimental-strip-types` (see scripts/narrativeFreshness.test.mjs).

/** Delays before each re-poll while the header is being rewritten. The model
 * call takes a few seconds; after the last one the page waits for the next
 * focus or periodic refresh rather than polling forever. */
export const NARRATIVE_REPOLL_DELAYS_MS = [4_000, 10_000, 20_000] as const;

/** How often an open, visible Briefing re-pulls /today on its own, so a
 * card answered on another device leaves Home within a minute. */
export const BRIEFING_REFRESH_INTERVAL_MS = 60_000;

/** A tab or window coming back to the foreground refreshes at most this often. */
export const FOCUS_REFRESH_MIN_GAP_MS = 10_000;

/** The delay before re-poll number `attempt` (0-based), or null when the
 * page should stop polling. */
export function narrativeRepollDelay(attempt: number): number | null {
  if (!Number.isInteger(attempt) || attempt < 0) return null;
  return NARRATIVE_REPOLL_DELAYS_MS[attempt] ?? null;
}

/** How often the open Briefing checks whether its timed refresh is due. */
export const BRIEFING_TICK_MS = 15_000;

/** Whether a timed check should re-pull /today: a minute since the last
 * pull of any kind (focus, Approve or timed), so no change waits longer
 * than a minute and a tick never follows another refresh closely. */
export function shouldRefreshOnTick(lastFetchMs: number, nowMs: number): boolean {
  return nowMs - lastFetchMs >= BRIEFING_REFRESH_INTERVAL_MS;
}

/** Orders overlapping pulls of /today: `begin` numbers a pull as it starts,
 * and `accept` says whether its answer may be shown when it lands. An answer
 * older than one already shown is dropped (it would put old counts back); an
 * older one still shows while the newer pull hasn't landed or has failed.
 * `changedHere` drops every pull already under way, for a change made on the
 * page itself (a dismissed row) that those answers would undo. */
export function newestAnswer(): { begin(): number; accept(seq: number): boolean; changedHere(): void } {
  let started = 0;
  let shown = 0;
  return {
    begin: () => ++started,
    accept: (seq) => {
      if (seq <= shown) return false;
      shown = seq;
      return true;
    },
    changedHere: () => {
      shown = started;
    },
  };
}

/** Whether a tab that just became visible should re-pull /today. */
export function shouldRefreshOnFocus(lastFetchMs: number, nowMs: number): boolean {
  return nowMs - lastFetchMs >= FOCUS_REFRESH_MIN_GAP_MS;
}

/** "Updated just now" / "Updated 12 min ago" / "Updated 9:42 AM" — or null
 * when there is no (readable) timestamp. */
export function narrativeUpdatedLabel(generatedAt: string | null | undefined, now: Date): string | null {
  if (!generatedAt) return null;
  const written = new Date(generatedAt);
  const ms = now.getTime() - written.getTime();
  if (Number.isNaN(ms)) return null;
  if (ms < 60_000) return "Updated just now";
  const minutes = Math.floor(ms / 60_000);
  if (minutes < 60) return `Updated ${minutes} min ago`;
  const hours = written.getHours();
  const h12 = hours % 12 === 0 ? 12 : hours % 12;
  const mm = String(written.getMinutes()).padStart(2, "0");
  return `Updated ${h12}:${mm} ${hours < 12 ? "AM" : "PM"}`;
}
