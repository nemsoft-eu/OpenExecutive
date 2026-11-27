"use client";

// A Take the lead card: something the Executive wanted to do on its own that
// waited (in training, a rule, the Approvals level). Approve does it once; on
// a training card, with "Do this on
// its own next time" ticked it also lets it do this (the same action with the
// same person) from now on;
// Edit changes it first, and with "Do it like this next time" ticked keeps
// the edit as an example and allows it. Answered here, never through chat.

import { CardDetailValue } from "@/components/CardDetailValue";
import { useEffect, useState } from "react";

import GoOnItsOwnOffer from "@/components/briefing/GoOnItsOwnOffer";
import Button from "@/components/ui/Button";
import {
  approveLeadCard,
  getLeadCard,
  rejectDecision,
  type LeadCardPayload,
  heldCardTop,
  type ProposalItem,
} from "@/lib/api";

import HeldCardTop from "./HeldCardTop";
import { parseOffers, useNextTimeRule } from "./NextTimeRule";
import StepTalk from "./StepTalk";

// The tag a training card carries (orchestrator.take_the_lead.TRAINING_TAG).
export const LEAD_TRAINING_TAG = "take_the_lead:training";

export default function LeadTrainingCard({
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
  const [card, setCard] = useState<LeadCardPayload | null>(null);
  const [editing, setEditing] = useState(false);
  const [values, setValues] = useState<Record<string, string>>({});
  const [learn, setLearn] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nextTime = useNextTimeRule(parseOffers(card?.offers), onResolved, editing || Object.keys(card?.said ?? {}).length > 0);

  useEffect(() => {
    if (decisionId == null) return;
    const controller = new AbortController();
    getLeadCard(decisionId, controller.signal)
      .then((next) => {
        setCard(next);
        setValues({ ...Object.fromEntries(next.fields.map((f) => [f.field, f.value])), ...next.said });
      })
      .catch((err) => {
        if ((err as Error)?.name !== "AbortError") setError("Couldn't load the details of this card.");
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

  const allowLabel = card?.allow?.label ?? null;
  const fields = card?.fields ?? [];
  // What it would send or set, shown in full: the card's own fields, or a
  // connected app's email.
  // A connected app's email shows who it goes to too: its details, in full.
  const details = card && card.saidDetails.length > 0 ? card.saidDetails : card?.details ?? [];
  // Either way, with what was said on the card: what Approve sends.
  const shownFields =
    details.length > 0 ? details : fields.map((f) => ({ ...f, value: card?.said[f.field] ?? f.value }));
  const training = card ? card.kind === "training" : (proposal.topic_tags ?? []).includes(LEAD_TRAINING_TAG);
  // Older cards saved only the summary, which quotes the start of the
  // message: with the message shown below, the title keeps just who.
  const headline =
    card?.headline ||
    (card?.summary && shownFields.length > 0 ? card.summary.split(": “")[0] : card?.summary) ||
    proposal.headline;
  const changed = Object.fromEntries(
    fields.filter((f) => (values[f.field] ?? "") !== f.value).map((f) => [f.field, values[f.field] ?? ""]),
  );

  // What Approve sends: the fields as changed (by hand or by what was said),
  // and, on the option that says so, to do this on its own from now on.
  const edits = (allow: boolean) => ({
    ...(Object.keys(changed).length > 0 ? { input: changed } : {}),
    ...(allowLabel && allow ? { allow: true } : {}),
  });

  return (
    <div
      id={`alert-${proposal.alert_id}`}
      className={`rounded-2xl border bg-surface-elevated p-4 sm:p-5 ${
        emphasized ? "border-accent/60 ring-1 ring-accent/25 shadow-sm" : "border-line"
      }`}
    >
      <HeldCardTop
        top={heldCardTop(proposal.view?.top) ?? card?.top ?? null}
        title={headline}
        at={proposal.created_at}
        tag={training ? "In training" : undefined}
        fallbackWaits={
          training
            ? "It’s a new kind of step for me, so it waits for an OK once."
            : card?.reason
              ? `It waits because ${card.reason}.`
              : "It waits for an OK."
        }
      />
      {card?.where && <p className="mt-1 text-sm leading-snug text-fg-muted">{card.where}</p>}
      {!editing && shownFields.length > 0 && (
        <div className="mt-3">
          <div className="mb-0.5 text-xs font-semibold uppercase tracking-wide text-fg-subtle">What I&apos;ll do</div>
          <div className="flex flex-col gap-2">
            {shownFields.map((f) =>
              f.long ? (
                <div
                  key={f.field}
                  className="max-h-80 overflow-y-auto rounded-xl border-l-[3px] border-accent/40 bg-surface p-3 text-[15px] leading-relaxed text-fg whitespace-pre-wrap break-words"
                >
                  {shownFields.length > 1 && (
                    <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-fg-muted">{f.label}</span>
                  )}
                  <CardDetailValue detail={f} />
                </div>
              ) : (
                <div key={f.field} className="text-[15px] leading-snug text-fg break-words">
                  <span className="text-fg-muted">{f.label}: </span>
                  <CardDetailValue detail={f} />
                </div>
              ),
            )}
          </div>
          {fields.length > 0 && (
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

      {editing ? (
        <div className="mt-3 flex flex-col gap-3">
          {fields.map((f) => {
            const id = `lead-${decisionId}-${f.field}`;
            const common =
              "w-full rounded-xl border border-line bg-surface p-3 text-[15px] leading-snug text-fg focus:outline-none focus:ring-2 focus:ring-accent/40 disabled:opacity-50";
            return (
              <label key={f.field} htmlFor={id} className="flex flex-col gap-1">
                <span className="text-xs font-semibold uppercase tracking-wide text-fg-muted">{f.label}</span>
                {f.long ? (
                  <textarea
                    id={id}
                    rows={5}
                    value={values[f.field] ?? ""}
                    disabled={busy}
                    onChange={(e) => setValues({ ...values, [f.field]: e.target.value })}
                    className={`${common} whitespace-pre-wrap`}
                  />
                ) : (
                  <input
                    id={id}
                    value={values[f.field] ?? ""}
                    disabled={busy}
                    onChange={(e) => setValues({ ...values, [f.field]: e.target.value })}
                    className={common}
                  />
                )}
              </label>
            );
          })}
          {allowLabel && (
            <label className="flex cursor-pointer items-start gap-3 rounded-xl border border-accent/30 bg-accent/5 p-3">
              <input
                type="checkbox"
                checked={learn}
                disabled={busy}
                onChange={(e) => setLearn(e.target.checked)}
                className="mt-1 h-4 w-4 flex-shrink-0 accent-[rgb(var(--accent-strong))]"
              />
              <span>
                <span className="block text-[15px] font-semibold text-fg">Do it like this next time</span>
                <span className="mt-0.5 block text-sm leading-snug text-fg-muted">
                  It learns from your edit and does this on its own from now on: {allowLabel}.
                </span>
              </span>
            </label>
          )}
          <div className="flex flex-wrap justify-end gap-2">
            <Button variant="ghost" onClick={() => setEditing(false)} disabled={busy}>
              Cancel
            </Button>
            <Button
              variant="primary"
              disabled={busy}
              onClick={() =>
                void run(() =>
                  approveLeadCard(decisionId, {
                    input: Object.keys(changed).length > 0 ? changed : undefined,
                    allow: Boolean(allowLabel && learn),
                  }),
                )
              }
            >
              Do it
            </Button>
          </div>
        </div>
      ) : (
        <>
          {nextTime.box}
          <StepTalk
            decisionId={decisionId}
            disabled={busy || card === null}
            seen={values}
            turns={proposal.view?.turns ?? []}
            canSay={Boolean(proposal.view?.tell_me) && Boolean(card?.mayChange) && fields.length > 0}
            options={[
              { label: "Do it", run: () => nextTime.withRule(() => approveLeadCard(decisionId, edits(false))) },
              ...(allowLabel
                ? [{
                    label: "Do it, and do this on its own next time",
                    detail: `From now on: ${allowLabel}. You can undo this in As me.`,
                    run: () => approveLeadCard(decisionId, edits(true)),
                  }]
                : []),
            ]}
            onSaid={() => {
              void getLeadCard(decisionId)
                .then((next) => {
                  setCard(next);
                  setValues({ ...Object.fromEntries(next.fields.map((f) => [f.field, f.value])), ...next.said });
                })
                .catch(() => setError("Couldn't load the details of this card."));
            }}
            onResolved={nextTime.resolved}
            onSkip={() => rejectDecision(decisionId)}
          />
          {nextTime.done}
          <GoOnItsOwnOffer leadKey={card?.allow?.key ?? null} />
        </>
      )}
      {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
    </div>
  );
}
