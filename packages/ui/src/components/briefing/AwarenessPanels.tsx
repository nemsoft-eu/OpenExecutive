"use client";

import Link from "next/link";
import { useState } from "react";

import Button, { buttonClass } from "@/components/ui/Button";
import OverflowMenu from "@/components/ui/OverflowMenu";
import type { ProposalItem } from "@/lib/api";
import { MEMORY_ACTIONS, briefingMemoryLine } from "@/lib/briefing-memory";
import { TellMe } from "./CardTalk";

import {
  MONITORING_DISMISS_OLDER_THAN_DAYS,
  PanelIntro,
  buildMonitoringSeed,
  olderThan,
  type ContinueHandler,
} from "./shared";

// The side panel behind Home's new-signals link: Monitoring, with every
// button it had on the page.

// One passive monitoring signal: headline + body excerpt, tap to discuss, and
// Dismiss signal in its ⋯ menu. The Discuss handoff reuses
// buildMonitoringSeed so it behaves exactly like the card does.
function MonitoringRow({
  proposal,
  onContinue,
  onDismiss,
}: {
  proposal: ProposalItem;
  onContinue?: ContinueHandler;
  onDismiss?: (p: ProposalItem) => void;
}) {
  // A watch's signal can be corrected right here ("that's our own repo"):
  // Correct it opens "Or tell me…" under the row, said on the card like any
  // other (POST /alerts/{id}/say); what the Executive said back shows there.
  const [steering, setSteering] = useState(false);
  const [reply, setReply] = useState<string | null>(null);
  const canSteer = Boolean(proposal.watch_slug && onDismiss);
  const seed = buildMonitoringSeed(proposal);
  const inner = (
    <>
      <div className="text-[15px] font-medium text-fg group-hover:text-accent transition-colors line-clamp-2" title={proposal.headline}>
        {proposal.headline}
      </div>
      {proposal.body && proposal.body !== proposal.headline && (
        <div className="mt-1 text-sm leading-snug text-fg-muted line-clamp-2">{proposal.body}</div>
      )}
      {proposal.view?.noticed && <div className="mt-1 text-xs text-fg-muted">{proposal.view.noticed}</div>}
    </>
  );
  return (
    <div id={`alert-${proposal.alert_id}`} className="py-3">
      <div className="group flex items-start gap-2">
        {onContinue ? (
          <button
            type="button"
            onClick={() => onContinue(seed, briefingMemoryLine(MEMORY_ACTIONS.monitoring, proposal.headline))}
            className="block min-w-0 flex-1 cursor-pointer rounded-lg text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
          >
            {inner}
          </button>
        ) : (
          <Link href="/watchlist" className="block min-w-0 flex-1">
            {inner}
          </Link>
        )}
        {canSteer && !steering && !reply && (
          <button
            type="button"
            onClick={() => setSteering(true)}
            className={buttonClass("ghost", "sm", "flex-shrink-0")}
          >
            Correct it
          </button>
        )}
        {onDismiss && (
          <OverflowMenu
            size="sm"
            label="More actions for this signal"
            items={[{ label: "Dismiss signal", onSelect: () => onDismiss(proposal) }]}
          />
        )}
      </div>
      {canSteer && steering && !reply && (
        <TellMe alertId={proposal.alert_id} autoFocus onSaid={(r) => setReply(r.reply)} />
      )}
      {reply && <p className="mt-2 text-sm leading-snug text-fg" aria-live="polite">{reply}</p>}
    </div>
  );
}

// Monitoring — passive signals (watchlist tickers, vendor status, external
// news) the Executive is tracking.
export function MonitoringPanelBody({
  proposals,
  onContinue,
  onDismiss,
  onBulkDismiss,
}: {
  proposals: ProposalItem[];
  onContinue?: ContinueHandler;
  onDismiss?: (p: ProposalItem) => void;
  onBulkDismiss?: (ids: number[]) => void;
}) {
  const staleIds = olderThan(proposals, MONITORING_DISMISS_OLDER_THAN_DAYS);
  return (
    <>
      <PanelIntro>
        Passive signals (watchlist tickers, vendor status, external news) the Executive is tracking. Nothing here
        needs a decision — tap one to talk it through.
      </PanelIntro>
      {proposals.length === 0 ? (
        <p className="py-3 text-[15px] text-fg-muted">No signals right now.</p>
      ) : (
        <div className="divide-y divide-line">
          {proposals.map((p) => (
            <MonitoringRow
              key={p.alert_id}
              proposal={p}
              onContinue={onContinue}
              onDismiss={onDismiss}
            />
          ))}
        </div>
      )}
      {onBulkDismiss && staleIds.length > 0 && (
        <Button variant="secondary" size="sm" className="mt-3" onClick={() => onBulkDismiss(staleIds)}>
          Dismiss {staleIds.length} older than {MONITORING_DISMISS_OLDER_THAN_DAYS} days
        </Button>
      )}
    </>
  );
}
