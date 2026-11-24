"use client";

import { CardDetailValue } from "@/components/CardDetailValue";
import { cardDetails, type CardDetail } from "@/lib/cardDetails";
import { useEffect, useState } from "react";

import HeldFieldsEdit, { heldFields } from "@/components/HeldFieldsEdit";
import Button from "@/components/ui/Button";
import {
  approveDecision,
  approveLeadCard,
  getDecision,
  rejectDecision,
  type DecisionInstance,
  type LeadCardField,
} from "@/lib/api";

// The chat: a step the Executive held for your OK (orchestrator/levels.py,
// or a teammate's step their department held), shown under its reply.
// Approve carries out exactly the call shown, on the server; Not now drops
// it. The same card is on Today and in your chat app. Under a reply in the
// chat (``onApprove``), Approve goes through the chat itself: the server
// approves it as this button does and the Executive carries on with what
// you asked (decisions.approve_in_chat).

import { parseOffers, useNextTimeRule } from "@/components/briefing/NextTimeRule";

// What the chat shows you said when you tapped Approve on a card there.
export const APPROVED_MESSAGE = "Approved";

interface HeldPayload {
  summary: string;
  reason: string;
  details: CardDetail[];
  fields: LeadCardField[];
  offers: unknown;
}

function readPayload(instance: DecisionInstance): HeldPayload | null {
  try {
    const raw = JSON.parse(instance.proposed_payload_json || "{}");
    if (typeof raw.summary !== "string") return null;
    const details = Array.isArray(raw.details) ? raw.details : [];
    return {
      summary: raw.summary,
      reason: typeof raw.reason === "string" ? raw.reason : "",
      details: cardDetails(details),
      fields: heldFields(raw.fields),
      offers: raw.offers,
    };
  } catch {
    return null;
  }
}

const DONE = new Set(["approved_unchanged", "approved_with_edit", "executed"]);

export default function ChatHeldActionCard({
  id,
  onApprove,
  approveDisabled = false,
}: {
  id: number;
  // Resolves once the chat's turn has ended (or nothing was sent).
  onApprove?: (id: number, input?: Record<string, string>) => Promise<void> | void;
  approveDisabled?: boolean;
}) {
  const [instance, setInstance] = useState<DecisionInstance | null>(null);
  // Approve went to the chat, which carries on from it.
  const [handed, setHanded] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const shownPayload = instance ? readPayload(instance) : null;
  // The card stays in the chat once done, so nothing to drop.
  const nextTime = useNextTimeRule(parseOffers(shownPayload?.offers), () => {}, editing);

  useEffect(() => {
    const controller = new AbortController();
    getDecision(id, controller.signal)
      .then(setInstance)
      .catch(() => { /* not this person's to see, or gone: the chip stays */ });
    return () => controller.abort();
  }, [id]);

  if (!instance || instance.decision_class !== "department_action") return null;
  const payload = readPayload(instance);
  if (!payload) return null;
  const done = DONE.has(instance.status) || handed;
  // Approve through the chat: "Approved" while its turn runs, then the
  // card as the server has it now (done, or still waiting when it was
  // refused or nothing was sent), so it never stays "Approved" on a guess.
  const viaChat = async (input?: Record<string, string>) => {
    if (!onApprove || approveDisabled) return;
    setHanded(true);
    setEditing(false);
    let now: DecisionInstance | null = null;
    try {
      await nextTime.withRule(
        async () => {
          try {
            await onApprove(instance.id, input);
          } finally {
            try {
              now = await getDecision(instance.id);
              setInstance(now);
            } catch {
              /* gone or not this person's: what the reply says stands */
            }
          }
        },
        () => now !== null && DONE.has(now.status),
      );
    } finally {
      setHanded(false);
    }
  };
  const waiting = instance.status === "proposed";
  if (!done && !waiting) return null;

  const act = async (label: string, call: () => Promise<DecisionInstance>, approving = true) => {
    setBusy(label);
    setError(null);
    try {
      if (approving) await nextTime.withRule(async () => setInstance(await call()));
      else setInstance(await call());
    } catch (err) {
      setError(err instanceof Error ? err.message : "That didn't go through.");
    } finally {
      setBusy(null);
    }
  };

  return (
    <article className="mt-3 rounded-2xl border border-line bg-surface-elevated p-4 sm:p-5">
      <div className="text-xs font-semibold uppercase tracking-wide text-accent">Waiting for your OK</div>
      <div className="mt-1 text-base sm:text-[17px] font-semibold leading-snug text-fg break-words whitespace-pre-line">
        {payload.summary}
      </div>
      {payload.reason && !done && (
        <p className="mt-1.5 text-sm text-fg-muted">I waited because {payload.reason}.</p>
      )}
      {editing && !done ? (
        <HeldFieldsEdit
          id={instance.id}
          fields={payload.fields}
          busy={busy !== null || (onApprove !== undefined && approveDisabled)}
          onCancel={() => setEditing(false)}
          onApprove={(changed) => {
            const input = Object.keys(changed).length > 0 ? changed : undefined;
            if (onApprove) {
              void viaChat(input);
              return;
            }
            void act("Doing it…", () =>
              input ? approveLeadCard(instance.id, { input }) : approveDecision(instance.id),
            );
          }}
        />
      ) : (
      <>
      {payload.details.length > 0 && !done && (
        <dl className="mt-3 flex flex-col gap-2 rounded-xl border border-line bg-surface p-3">
          {payload.details.map((d) => (
            <div key={d.field} className="min-w-0">
              <dt className="text-xs font-semibold uppercase tracking-wide text-fg-muted">{d.field}</dt>
              <dd className="whitespace-pre-wrap break-words text-[15px] leading-snug text-fg"><CardDetailValue detail={d} /></dd>
            </div>
          ))}
        </dl>
      )}
      {!done && nextTime.box}
      {done ? (
        <p className="mt-3 text-sm font-semibold text-emerald-700 dark:text-emerald-300">{DONE.has(instance.status) ? "✓ Done." : "✓ Approved."}</p>
      ) : (
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <Button
            variant="primary"
            onClick={() => {
              if (onApprove) {
                void viaChat();
                return;
              }
              void act("Doing it…", () => approveDecision(instance.id));
            }}
            disabled={busy !== null || (onApprove !== undefined && approveDisabled)}
          >
            {busy ?? "Approve"}
          </Button>
          {payload.fields.length > 0 && (
            <Button variant="secondary" onClick={() => setEditing(true)} disabled={busy !== null}>
              Edit
            </Button>
          )}
          <Button
            variant="ghost"
            className="ml-auto"
            onClick={() => void act("Declining…", () => rejectDecision(instance.id), false)}
            disabled={busy !== null}
          >
            Not now
          </Button>
        </div>
      )}
      </>
      )}
      {nextTime.done}
      {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
    </article>
  );
}
