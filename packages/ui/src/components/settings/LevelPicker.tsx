"use client";

import { LEVEL_NAMES, LEVEL_ORDER, type ApprovalsLevel } from "@/lib/approvalsLevel";

// The one Careful / Balanced / Hands-off picker, a row of three, for every
// level set on Delegate → Approvals. `allowed` greys out the levels above
// a limit. `value` null shows none picked. The caller saves; this only reports the pick.
export default function LevelPicker({
  value,
  onPick,
  label,
  labelledBy,
  disabled = false,
  allowed,
  small = false,
  recommended,
}: {
  value: ApprovalsLevel | null;
  onPick: (level: ApprovalsLevel) => void;
  /** What it picks, for a screen reader, when no visible text names it. */
  label?: string;
  labelledBy?: string;
  disabled?: boolean;
  allowed?: (level: ApprovalsLevel) => boolean;
  small?: boolean;
  recommended?: ApprovalsLevel;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={labelledBy ? undefined : label}
      aria-labelledby={labelledBy}
      className="inline-flex max-w-full flex-wrap overflow-hidden rounded-xl border border-line"
    >
      {LEVEL_ORDER.map((level, i) => {
        const on = level === value;
        const can = !allowed || allowed(level);
        return (
          <button
            key={level}
            type="button"
            role="radio"
            aria-checked={on}
            disabled={disabled || !can}
            onClick={() => {
              if (!on) onPick(level);
            }}
            className={`${small ? "min-h-10 px-3" : "min-h-touch px-4"} text-sm transition-colors cursor-pointer disabled:cursor-not-allowed ${
              i > 0 ? "border-l border-line" : ""
            } ${on ? "bg-accent/10 font-semibold text-accent-strong" : "text-fg-muted hover:bg-surface-overlay/60"} ${
              can ? "disabled:opacity-60" : "opacity-40"
            }`}
          >
            {LEVEL_NAMES[level]}
            {recommended === level && <span className="ml-1.5 hidden text-xs font-normal text-fg-subtle sm:inline">Recommended</span>}
          </button>
        );
      })}
    </div>
  );
}
