// Who an email goes from, on every email card: you, or the Executive with
// you on cc. The reply card and the chat's email card both use it, so the
// two always look and work the same.
export default function FromSwitch({
  fromExecutive,
  executiveName,
  disabled,
  onPick,
}: {
  fromExecutive: boolean;
  executiveName: string;
  disabled: boolean;
  onPick: (executive: boolean) => void;
}) {
  return (
    <div className="mt-2 flex items-center gap-2">
      <span className="w-9 text-xs text-fg-subtle">From</span>
      <div role="radiogroup" aria-label="From" className="inline-flex overflow-hidden rounded-full border border-line-strong">
        {([false, true] as const).map((executive) => {
          const on = executive === fromExecutive;
          return (
            <button
              key={String(executive)}
              type="button"
              role="radio"
              aria-checked={on}
              disabled={disabled}
              onClick={() => {
                if (!on) onPick(executive);
              }}
              className={`min-h-9 px-3 text-[13px] disabled:opacity-60 ${
                on ? "bg-accent/20 font-semibold text-fg" : "text-fg-muted hover:bg-surface-overlay/60"
              }`}
            >
              {executive ? executiveName : "You"}
            </button>
          );
        })}
      </div>
    </div>
  );
}
