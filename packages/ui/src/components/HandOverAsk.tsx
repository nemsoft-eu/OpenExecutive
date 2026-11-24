"use client";

import { useState } from "react";

import Button from "@/components/ui/Button";
import { answerHandOver } from "@/lib/api";
import type { HandOverAsk as Ask } from "@/lib/handOverAsk";

// The turn waits on a step in your browser only you can do (a password, a
// code from your phone), and asks it here with Done and Stop
// (lib/handOverAsk.ts). The same press as the bar at the top of the page; the
// Needs you card goes out only if this one goes unanswered.
export default function HandOverAsk({ ask, onAnswered }: { ask: Ask; onAnswered: () => void }) {
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);

  async function answer(outcome: "done" | "stop") {
    setBusy(true);
    setFailed(false);
    const ok = await answerHandOver(ask.id, outcome);
    setBusy(false);
    if (ok) onAnswered();
    else setFailed(true);
  }

  return (
    <article className="mt-3 max-w-xl rounded-2xl border border-line bg-surface-elevated p-4" aria-live="polite">
      <p className="text-base font-semibold leading-snug text-fg break-words">{ask.text}</p>
      {ask.detail ? <p className="mt-1 text-sm text-fg-muted">{ask.detail}</p> : null}
      {failed ? <p className="mt-1 text-sm text-fg-muted">That one already finished.</p> : null}
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Button variant="primary" onClick={() => void answer("done")} disabled={busy}>
          {ask.doneLabel}
        </Button>
        <Button variant="ghost" className="ml-auto" onClick={() => void answer("stop")} disabled={busy}>
          {ask.stopLabel}
        </Button>
      </div>
    </article>
  );
}
