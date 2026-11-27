// The Approvals level (Careful / Balanced / Hands-off) in a few words, for
// the sidebar's quick picker and every level picker (LevelPicker). Settings → Your Executive keeps the full card
// with examples (components/settings/OutsideContentCard.tsx); both save the
// same setting (/executive/outside-content, orchestrator/levels.py).

export type ApprovalsLevel = "ask" | "outsiders" | "trust";

export const APPROVALS_LEVELS: { value: ApprovalsLevel; title: string; text: string }[] = [
  { value: "ask", title: "Careful", text: "I ask before I send, book, save or run anything." },
  {
    value: "outsiders",
    title: "Balanced",
    text: "I go ahead with your team and People. I ask before anyone else, or when something I read tries to give me orders.",
  },
  { value: "trust", title: "Hands-off", text: "I just do it. I don't stop to ask." },
];

// Sent on window when either picker saves, carrying the saved setting, so
// the other shows the new level without loading it again.
export const APPROVALS_CHANGED_EVENT = "oe:approvals-changed";

export function announceApprovals<T>(saved: T): void {
  window.dispatchEvent(new CustomEvent<T>(APPROVALS_CHANGED_EVENT, { detail: saved }));
}

// Calls `fn` with each saved setting; returns the unsubscribe.
export function onApprovalsChanged<T>(fn: (saved: T) => void): () => void {
  const listener = (e: Event) => fn((e as CustomEvent<T>).detail);
  window.addEventListener(APPROVALS_CHANGED_EVENT, listener);
  return () => window.removeEventListener(APPROVALS_CHANGED_EVENT, listener);
}

export function approvalsTitle(level: string): string {
  return APPROVALS_LEVELS.find((l) => l.value === level)?.title ?? "Careful";
}

// The one source for the level names and their order, for every picker:
// the company's level, a teammate's, a department's, onboarding's.
export const LEVEL_ORDER: ApprovalsLevel[] = APPROVALS_LEVELS.map((l) => l.value);
export const LEVEL_NAMES = Object.fromEntries(APPROVALS_LEVELS.map((l) => [l.value, l.title])) as Record<
  ApprovalsLevel,
  string
>;

// Whether `level` is no further than `limit` (Careful < Balanced < Hands-off).
export function atMost(level: ApprovalsLevel, limit: ApprovalsLevel): boolean {
  return LEVEL_ORDER.indexOf(level) <= LEVEL_ORDER.indexOf(limit);
}

// A department's stored authority_level ("How it acts") under the level
// names, for onboarding's draft: escalate is Careful, propose_only (the
// default) Balanced, auto_execute Hands-off.
export const AUTHORITY_LEVELS: { value: "escalate" | "propose_only" | "auto_execute"; label: string }[] = [
  { value: "escalate", label: LEVEL_NAMES.ask },
  { value: "propose_only", label: LEVEL_NAMES.outsiders },
  { value: "auto_execute", label: LEVEL_NAMES.trust },
];
