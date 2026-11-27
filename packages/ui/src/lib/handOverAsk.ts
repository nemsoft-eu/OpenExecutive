// A step in the person's browser only they can do (a password, a code from
// their phone), asked in the chat while the turn waits, with Done and Stop
// (browser/tools.py `hand_over_asks`). Every word is the server's.
// components/HandOverAsk.tsx shows it; Done and Stop answer
// POST /browser/hand-over/{id}.
import type { HandOverAskEvent } from "./api";

export interface HandOverAsk {
  id: string;
  text: string;
  detail: string;
  doneLabel: string;
  stopLabel: string;
}

function words(value: unknown, cap: number): string {
  return typeof value === "string" ? value.replace(/\s+/g, " ").trim().slice(0, cap) : "";
}

// The ask as the chat shows it, or null when the event isn't one it can
// answer (an id the server didn't mint, or nothing to say).
export function toHandOverAsk(event: HandOverAskEvent): HandOverAsk | null {
  const id = typeof event.id === "string" && /^[0-9a-f]{32}$/.test(event.id) ? event.id : "";
  const text = words(event.text, 200);
  if (!id || !text) return null;
  return {
    id,
    text,
    detail: words(event.detail, 300),
    doneLabel: words(event.done_label, 20) || "Done",
    stopLabel: words(event.stop_label, 20) || "Stop",
  };
}
