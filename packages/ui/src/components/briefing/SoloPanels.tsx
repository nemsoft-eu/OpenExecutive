"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import InfoTip from "@/components/InfoTip";
import {
  getTopThree,
  getWeeklyReview,
  type TopThreeToday,
  type WeeklyReviewSummary,
} from "@/lib/api";
import { loopText } from "@/lib/dueSoon";
import { reviewExcerpt, reviewRanLabel, topThreeSlot, topThreeWhy } from "@/lib/rhythmCards";

import { PanelIntro, RankBadge } from "./shared";

// Solo's parts of Home: this week's review (behind its line at the foot of
// Home) and the top three.

// ── This week's review ────────────────────────────────────────────────────
// The latest weekly review (GET /today/weekly-review) — when it ran and next
// week's top three, with a link to the whole review on its run page. Null
// until a review has run, and for anyone but the owner.
export function useWeeklyReview(enabled: boolean) {
  const [review, setReview] = useState<WeeklyReviewSummary | null>(null);
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    getWeeklyReview(controller.signal)
      .then(setReview)
      .catch(() => { /* no card is better than a broken one */ });
    return () => controller.abort();
  }, [enabled]);
  return review;
}

export function weeklyReviewMeta(review: WeeklyReviewSummary): string {
  return [review.period, reviewRanLabel(review.completed_at)].filter(Boolean).join(" · ");
}

export function WeeklyReviewPanelBody({ review }: { review: WeeklyReviewSummary }) {
  const excerpt = reviewExcerpt(review);
  const meta = weeklyReviewMeta(review);
  return (
    <>
      <PanelIntro>
        Once a week the Executive looks back on your week — your goals by area, what&apos;s due, projects that went
        quiet and the decisions you made — and picks next week&apos;s top three. It&apos;s also sent to you when a
        chat app or email is connected.
      </PanelIntro>
      {meta && <p className="mb-3 text-sm text-fg-muted">{meta}</p>}
      {excerpt.heading && (
        <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-fg-muted">{excerpt.heading}</div>
      )}
      {excerpt.numbered ? (
        <ol className="divide-y divide-line">
          {excerpt.lines.map((line, i) => (
            <li key={i} className="flex items-start gap-3 py-3">
              <RankBadge n={i + 1} />
              <span className="min-w-0 flex-1 text-[15px] text-fg break-words">{line}</span>
            </li>
          ))}
        </ol>
      ) : (
        excerpt.lines.map((line, i) => (
          <p key={i} className="py-0.5 text-[15px] text-fg-muted break-words">
            {line}
          </p>
        ))
      )}
      <Link
        href={`/jobs/runs/${encodeURIComponent(review.run_id)}`}
        className="mt-4 inline-block text-sm font-medium text-accent hover:underline"
      >
        Read the review →
      </Link>
    </>
  );
}

// ── Top three today ───────────────────────────────────────────────────────
// The three things to focus on today — the same pick, order and free slots
// as the morning brief (GET /today/top-three). Solo's main job, so it stays
// on Home as a short list under Needs you. Loads on its own after the
// briefing, since it may wait on the calendar (up to 4 s), and hides when
// there is nothing to pick or the viewer isn't the owner (the API returns
// null). `ownerName` turns a commitment's stored text into "you" wording.
export function TopThreeList({ ownerName, id }: { ownerName: string; id?: string }) {
  const [top, setTop] = useState<TopThreeToday | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getTopThree(controller.signal)
      .then(setTop)
      .catch(() => { /* no card is better than a broken one */ });
    return () => controller.abort();
  }, []);

  if (!top || top.items.length === 0) return null;
  return (
    <section id={id}>
      <div className="mb-3 flex items-center gap-1.5">
        <h2 className="text-lg font-semibold text-fg">Top three today</h2>
        <InfoTip align="left">
          The three things to focus on today, picked from what&apos;s overdue or due, goals that are slipping and
          your active projects — the same three your morning brief opens with. When the Executive can read your
          calendar, each gets a free slot in today&apos;s working hours.
        </InfoTip>
      </div>
      <ol className="divide-y divide-line rounded-2xl border border-line bg-surface-elevated px-4 sm:px-5">
        {top.items.map((item, i) => {
          const text =
            item.kind === "commitment" && ownerName
              ? loopText({ description: item.text, owner_name: ownerName })
              : item.text;
          const slot = topThreeSlot(item);
          return (
            <li key={item.key} className="flex items-start gap-3 py-3.5">
              <RankBadge n={i + 1} />
              <div className="min-w-0 flex-1">
                <div className="text-[15px] text-fg break-words">{text}</div>
                <div className="mt-0.5 text-sm text-fg-muted">{topThreeWhy(item)}</div>
              </div>
              {slot && (
                <span className={`mt-0.5 flex-shrink-0 text-sm tabular-nums ${item.slot ? "text-accent" : "text-fg-muted"}`}>
                  {slot}
                </span>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
