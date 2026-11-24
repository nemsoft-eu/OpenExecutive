"use client";

// The part a held step's card shares (Take the lead, a teammate's held send):
// the options (the first is the Executive's pick), what was said on the card,
// "Or tell me…" and Skip. What the person says goes to POST
// /decisions/{id}/say, which rewrites the step's fields from it (Approve then
// sends them as the edit) and, on a go, approves it the same as a tap.

import { useState } from "react";

import { type LeadCardField, type StepSayResult, sayOnStep } from "@/lib/api";
import { buttonClass } from "@/components/ui/Button";

import { SayBox, Turns } from "./CardTalk";

export interface StepOption {
  label: string;
  // What it does beyond the step itself ("From now on: …").
  detail?: string;
  run: () => Promise<unknown>;
}

export default function StepTalk({
  decisionId,
  options,
  seen,
  turns: saved = [],
  canSay,
  disabled = false,
  onSaid,
  onResolved,
  onSkip,
}: {
  decisionId: number;
  options: StepOption[];
  // The step's fields as the card shows them, so a go is judged against them.
  seen: Record<string, string>;
  turns?: { who: "you" | "me"; text: string }[];
  canSay: boolean;
  disabled?: boolean;
  // Words were taken: the step's fields as they read now.
  onSaid: (fields: LeadCardField[]) => void;
  onResolved: () => void;
  onSkip: () => Promise<unknown>;
}) {
  const [turns, setTurns] = useState(saved);
  const [sending, setSending] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // A go that was tried and failed: what happened stays on the card until
  // they've read it.
  const [tried, setTried] = useState(false);

  async function run(action: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await action();
      onResolved();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Try again.");
      setBusy(false);
    }
  }

  const off = disabled || busy || sending !== null || tried;

  return (
    <div>
      <Turns turns={turns} sending={sending} />
      {options.length > 0 && (
        <div className="mt-3 flex flex-col gap-2" role="group" aria-label="Options">
          {options.map((o, i) => (
            <button
              key={o.label}
              type="button"
              disabled={off}
              onClick={() => void run(o.run)}
              className={`${buttonClass(i === 0 ? "primary" : "secondary", "md")} h-auto w-full flex-col items-start justify-start py-2 text-left`}
            >
              <span>{o.label}</span>
              {o.detail && <span className="mt-0.5 break-words text-sm font-normal opacity-80">{o.detail}</span>}
            </button>
          ))}
        </div>
      )}
      {tried && (
        <div className="mt-3">
          <button type="button" onClick={onResolved} className={buttonClass("secondary", "md")}>
            Got it
          </button>
        </div>
      )}
      {canSay && !tried && (
        <SayBox
          id={`step-${decisionId}`}
          disabled={off}
          onSending={setSending}
          onFailed={() => setSending(null)}
          say={async (text) => {
            const r: StepSayResult = await sayOnStep(decisionId, text, seen);
            setTurns((t) => [...t, { who: "you", text }, { who: "me", text: r.reply }]);
            setSending(null);
            if (r.approved?.status === "failed") setTried(true);
            else if (r.approved) onResolved();
            else onSaid(r.fields);
          }}
        />
      )}
      <div className="mt-2 flex justify-end">
        <button
          type="button"
          disabled={off}
          onClick={() => void run(onSkip)}
          className="text-sm text-fg-muted hover:text-fg disabled:opacity-50"
        >
          Skip
        </button>
      </div>
      {error && <p role="alert" className="mt-2 text-sm text-red-500">{error}</p>}
    </div>
  );
}
