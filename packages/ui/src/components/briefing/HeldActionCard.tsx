"use client";

// A send or edit a team member asked for in chat that their department held
// (orchestrator.department_gate). Approve carries out exactly the call shown,
// on the server; it stays off until the whole call has loaded, since the
// card's headline keeps only the start of it.

import { CardDetailValue } from "@/components/CardDetailValue";
import { useEffect, useState } from "react";

import HeldFieldsEdit from "@/components/HeldFieldsEdit";
import {
  approveDecision,
  approveLeadCard,
  getHeldAction,
  rejectDecision,
  type HeldActionPayload,
  heldCardTop,
  type ProposalItem,
} from "@/lib/api";

import HeldCardTop from "./HeldCardTop";
import { parseOffers, useNextTimeRule } from "./NextTimeRule";
import StepTalk from "./StepTalk";

// The tag such a card carries (department_gate.DECISION_CLASS).
export const HELD_ACTION_TAG = "decision_class:department_action";

export default function HeldActionCard({
  proposal,
  onResolved,
  emphasized = false,
}: {
  proposal: ProposalItem;
  // Called once it's approved or declined, so the briefing can drop the card.
  onResolved: () => void;
  emphasized?: boolean;
}) {
  const decisionId = proposal.decision_instance_id;
  const [held, setHeld] = useState<HeldActionPayload | null>(null);
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nextTime = useNextTimeRule(parseOffers(held?.offers), onResolved, editing || Object.keys(held?.said ?? {}).length > 0);

  useEffect(() => {
    if (decisionId == null) return;
    const controller = new AbortController();
    getHeldAction(decisionId, controller.signal)
      .then(setHeld)
      .catch((err) => {
        if ((err as Error)?.name !== "AbortError") setError("Couldn't load what it would do.");
      });
    return () => controller.abort();
  }, [decisionId]);

  if (decisionId == null) return null;

  async function run(action: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await nextTime.withRule(action);
      nextTime.resolved();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Try again.");
      setBusy(false);
    }
  }

  // What was said on the card, and the step's fields as it shows them.
  const said = held?.said ?? {};
  // Its Do it sends what was said, so the step shows with it.
  const shownDetails = held && held.saidDetails.length > 0 ? held.saidDetails : held?.details ?? [];
  const seen = { ...Object.fromEntries((held?.fields ?? []).map((f) => [f.field, f.value])), ...said };

  return (
    <div
      id={`alert-${proposal.alert_id}`}
      className={`rounded-2xl border bg-surface-elevated p-4 sm:p-5 ${
        emphasized ? "border-accent/60 ring-1 ring-accent/25 shadow-sm" : "border-line"
      }`}
    >
      <HeldCardTop
        top={heldCardTop(proposal.view?.top) ?? held?.top ?? null}
        title={proposal.headline}
        at={proposal.created_at}
        fallbackWaits={held?.reason ? `It waits because ${held.reason}.` : "It waits for an OK."}
      />
      {editing && held ? (
        <HeldFieldsEdit
          id={decisionId}
          fields={held.fields}
          said={held.said}
          busy={busy}
          onCancel={() => setEditing(false)}
          onApprove={(changed) =>
            void run(() =>
              Object.keys(changed).length > 0 ? approveLeadCard(decisionId, { input: changed }) : approveDecision(decisionId),
            )
          }
        />
      ) : (
      <>
      {held && shownDetails.length > 0 && (
        <div className="mt-3">
          <div className="mb-0.5 text-xs font-semibold uppercase tracking-wide text-fg-subtle">What I&apos;ll do</div>
          <dl className="flex flex-col gap-2 rounded-xl border border-line bg-surface p-3">
            {shownDetails.map((d) => (
              <div key={d.field} className="min-w-0">
                <dt className="text-xs font-semibold uppercase tracking-wide text-fg-muted">{d.field}</dt>
                <dd className="whitespace-pre-wrap break-words text-[15px] leading-snug text-fg"><CardDetailValue detail={d} /></dd>
              </div>
            ))}
          </dl>
          {held.fields.length > 0 && (
            <button
              type="button"
              disabled={busy}
              onClick={() => setEditing(true)}
              className="mt-1 text-sm text-accent hover:underline disabled:opacity-50"
            >
              Change the wording
            </button>
          )}
        </div>
      )}
      {nextTime.box}
      <StepTalk
        decisionId={decisionId}
        disabled={busy || held == null}
        seen={seen}
        turns={proposal.view?.turns ?? []}
        canSay={Boolean(proposal.view?.tell_me) && Boolean(held?.mayChange) && (held?.fields.length ?? 0) > 0}
        options={[{
          label: "Do it",
          // What was said on the card goes as the edit.
          run: () =>
            nextTime.withRule(() =>
              Object.keys(said).length > 0 ? approveLeadCard(decisionId, { input: said }) : approveDecision(decisionId),
            ),
        }]}
        onSaid={() => {
          void getHeldAction(decisionId).then(setHeld).catch(() => setError("Couldn't load what it would do."));
        }}
        onResolved={nextTime.resolved}
        onSkip={() => rejectDecision(decisionId)}
      />
      {nextTime.done}
      </>
      )}
      {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
    </div>
  );
}
