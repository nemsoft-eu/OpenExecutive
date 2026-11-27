"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { getActivity, type ActivityItem } from "@/lib/api";
import Icon from "@/components/Icon";
import { EmptyState, LivePulse, Skeleton, formatRunAt } from "./shared";
import FeatureName from "@/components/FeatureName";

// Memory's Activity screen (`RecentActivity`): what the Executive did on its
// own, newest first, and the words each row reads. What it will do next is
// the Schedule page (app/schedule), one list for everything on its schedule.

export function activityLine(item: ActivityItem): { verb: string; subject: string } {
  switch (item.kind) {
    case "dm_sent":
      return { verb: "DM'd", subject: item.target ?? "a colleague" };
    case "email_sent":
      return { verb: "emailed", subject: item.target ?? "a colleague" };
    case "nudge_sent":
      return { verb: "nudged", subject: item.target ?? "a stalled item" };
    case "cadence_sent":
      return { verb: "checked in with", subject: item.department ?? "a team" };
    case "workflow_resumed":
      return { verb: "resumed workflow with", subject: item.target ?? "someone" };
    case "proposal_routed":
      // propose_only path — backend marked the action done without dispatching,
      // so describe the intent ("proposed to X") rather than implying a send.
      return { verb: "proposed to", subject: item.target ?? "an approver" };
    case "decision_logged":
      return { verb: "logged decision:", subject: item.summary };
    case "advice_given":
      return { verb: "advised on", subject: item.summary };
    case "workflow_done":
      // The summary already says what it did ("Researched your market").
      return { verb: "", subject: item.summary };
    case "initiative_started":
      return { verb: "kicked off initiative:", subject: item.summary };
    case "decision_resolved":
      return { verb: "resolved decision:", subject: item.summary };
    case "alert_raised":
      return { verb: "raised alert:", subject: item.summary };
    case "took_the_lead":
      return { verb: "took the lead:", subject: item.summary };
    case "sent_as_you":
      return { verb: "sent as you:", subject: item.summary };
    case "contacts_kept":
      // The summary says it; the detail has the counts.
      return { verb: "", subject: item.summary };
    default:
      return { verb: "acted on", subject: item.summary };
  }
}

// Kinds whose `subject` IS the full summary: the body row renders the text, so
// the inline subject span is suppressed to avoid repeating it (only the verb
// shows inline). Kept beside activityLine so adding a kind is a one-place edit.
export const SUMMARY_KINDS = new Set([
  "decision_logged",
  "advice_given",
  "action",
  "workflow_done",
  "initiative_started",
  "decision_resolved",
  "alert_raised",
  "took_the_lead",
  "sent_as_you",
  "contacts_kept",
]);

/** Where a finished run's write-up opens, when it kept one. */
export function writeUpHref(item: ActivityItem): string | null {
  if (item.kind === "contacts_kept") return "/people?tab=contacts";
  return item.artifact_id ? `/artifacts/${encodeURIComponent(item.artifact_id)}` : null;
}

/** What the link after a row's detail says. */
export function writeUpLabel(item: ActivityItem): string {
  return item.kind === "contacts_kept" ? "See changes" : "Read it";
}

// Take the lead and Handle it for me rows say whose name it acted in.
export function actingAs(item: ActivityItem): string | null {
  if (item.kind === "sent_as_you") return "As you";
  if (item.kind === "took_the_lead") return "As the Executive";
  return null;
}

// Live activity refreshes on this cadence so new events stream into the feed
// without a reload (mirrors the departments page poll). Kept modest — the feed
// is read-only and the payload is small.
const ACTIVITY_POLL_INTERVAL_MS = 20_000;

export function RecentActivity() {
  const [items, setItems] = useState<ActivityItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    // The Pulse is the single home for recent activity (removed from the
    // Briefing), so pull a generous slice (the list scrolls internally) and
    // then refresh on an interval so the feed stays live.
    const load = () => {
      getActivity(100)
        .then((res) => {
          if (!cancelled) setItems(res.items);
        })
        .catch(() => {
          // Swallow transient errors — keep the last good list and let the
          // next tick retry rather than blanking the feed.
        })
        .finally(() => {
          // Only the first load drives the skeleton; interval refreshes update
          // silently so the feed doesn't flash on every tick.
          if (!cancelled) setLoading(false);
        });
    };

    load();
    const interval = window.setInterval(load, ACTIVITY_POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  if (loading) {
    return (
      <div className="rounded-2xl border border-line bg-surface-elevated p-4 sm:p-5 space-y-2">
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-5/6" />
      </div>
    );
  }
  if (items.length === 0) {
    return (
      <div className="rounded-2xl border border-line bg-surface-elevated">
        <EmptyState message="Nothing yet. Actions the Executive takes on its own show up here as they happen." />
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-line bg-surface-elevated p-4 sm:p-5">
      <div className="flex items-center justify-between gap-2 mb-4">
        <div className="flex items-center gap-2">
          <Icon name="activity" size="w-5 h-5" className="text-emerald-500" />
          <h3 className="text-base font-semibold text-fg">Recent activity</h3>
        </div>
        <LivePulse />
      </div>
      {/* The timeline line lives on the <ol> (sized to the full list). */}
      <div>
        <ol className="relative space-y-4 before:absolute before:left-[3px] before:top-1.5 before:bottom-1.5 before:w-px before:bg-line">
          {items.map((it, i) => {
            const { relative } = formatRunAt(it.at);
            const { verb, subject } = activityLine(it);
            // For summary-style kinds the subject IS the summary, so don't
            // repeat it inline — the body row below carries the text.
            const subjectIsSummary = SUMMARY_KINDS.has(it.kind);
            return (
              <li key={`${it.at}-${it.kind}-${i}`} className="relative pl-5">
                <span
                  className="absolute left-0 top-2 h-[7px] w-[7px] rounded-full bg-emerald-500/80 ring-2 ring-surface-elevated"
                  aria-hidden="true"
                />
                <div className="flex items-baseline gap-2 flex-wrap">
                  {relative && (
                    <span className="text-xs text-fg-subtle tabular-nums">{relative}</span>
                  )}
                  {actingAs(it) && (
                    <>
                      <FeatureName
                        feature={it.kind === "sent_as_you" ? "handle_it" : "take_the_lead"}
                        className="text-xs"
                      />
                      <span className="text-xs font-semibold text-fg-muted">{actingAs(it)}</span>
                    </>
                  )}
                  {verb && <span className="text-sm text-fg-muted">{verb}</span>}
                  {!subjectIsSummary && (
                    <span className="text-sm text-fg" title={subject}>{subject}</span>
                  )}
                  {it.department && (
                    <span className="ml-auto text-xs text-fg-subtle hidden sm:inline capitalize">
                      {it.department}
                    </span>
                  )}
                </div>
                {it.summary && (
                  <p className="text-[15px] text-fg leading-snug break-words line-clamp-2 mt-0.5" title={it.summary}>
                    {it.summary}
                  </p>
                )}
                {(it.detail || writeUpHref(it)) && (
                  <p className="text-sm text-fg-muted leading-snug break-words mt-0.5">
                    {it.detail}
                    {writeUpHref(it) && (
                      <>
                        {it.detail && " "}
                        <Link href={writeUpHref(it)!} className="font-semibold text-accent hover:underline whitespace-nowrap">
                          {writeUpLabel(it)}
                        </Link>
                      </>
                    )}
                  </p>
                )}
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Read-only rhythm cards (recurring briefs, dept check-ins, awaiting items).
// ---------------------------------------------------------------------------
