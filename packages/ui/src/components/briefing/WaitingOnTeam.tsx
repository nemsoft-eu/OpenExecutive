"use client";

import type { ProposalItem } from "@/lib/api";
import { waitingNote, waitingOnTeam, waitingRow } from "@/lib/briefingSummary";

// What waits on someone else on the team, out of Needs you (it needs nothing
// from the reader): a heading naming whose OK it waits for, a line saying so
// and how old the oldest is, then one row per card with the job in words,
// who started it, why it waits and how long, the first few only, then
// "N more". A row opens the full cards.
// The phone app shows the same.
const SHOWN = 3;

export function WaitingOnTeam({
  proposals,
  people,
  onOpen,
}: {
  proposals: ProposalItem[];
  people: { id: number; full_name: string }[];
  onOpen: () => void;
}) {
  const heading = waitingOnTeam(proposals, people);
  if (!heading) return null;
  const more = proposals.length - SHOWN;
  return (
    <section aria-labelledby="waiting-on-team" className="mb-8">
      <h2 id="waiting-on-team" className="text-lg font-semibold text-fg">
        {heading.title}
        <span className="ml-1.5 font-normal text-fg-muted">({proposals.length})</span>
      </h2>
      <p className="mb-3 text-sm text-fg-muted">{waitingNote(proposals, people)}</p>
      <div className="divide-y divide-line rounded-2xl border border-line bg-surface-elevated px-4">
        {proposals.slice(0, SHOWN).map((p) => {
          const row = waitingRow(p);
          return (
            <button
              key={p.alert_id}
              type="button"
              onClick={onOpen}
              aria-haspopup="dialog"
              className="group flex w-full items-center gap-3 py-3 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
            >
              <div className="min-w-0 flex-1">
                <div className="text-[15px] font-medium text-fg transition-colors group-hover:text-accent line-clamp-2">
                  {row.title}
                </div>
                <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-fg-muted">
                  {row.started && (
                    <span className="rounded-md bg-surface px-2 py-0.5 font-medium text-fg">{row.started}</span>
                  )}
                  {row.because && (
                    <span className="rounded-md bg-amber-500/15 px-2 py-0.5 font-medium text-amber-700 dark:text-amber-300">
                      {row.because}
                    </span>
                  )}
                  {row.age && <span>{row.age}</span>}
                </div>
              </div>
              <span aria-hidden className="text-lg text-fg-muted">
                ›
              </span>
            </button>
          );
        })}
        {more > 0 && (
          <button
            type="button"
            onClick={onOpen}
            aria-haspopup="dialog"
            className="flex w-full items-center gap-3 py-3 text-left text-sm font-medium text-accent hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
          >
            {more} more
          </button>
        )}
      </div>
    </section>
  );
}
