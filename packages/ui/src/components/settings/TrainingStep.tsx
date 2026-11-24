"use client";

import Link from "next/link";
import { useState } from "react";

import { useReachSave } from "@/components/settings/ReachCard";
import SettingsCard from "@/components/settings/SettingsCard";
import { LearnedSection } from "@/components/settings/TrainingCard";
import Switch from "@/components/Switch";
import {
  removeLeadLearned,
  setTakeTheLead,
  setTraining,
  type DelegationSettings,
  type TakeTheLead,
} from "@/lib/api";
import { APPROVALS_HREF } from "@/lib/reach";

// Delegate → Approvals, check 3, In training: the last check, both switches
// side by side. As you (PUT /delegation/training {training}): nothing goes
// out as you on its own, on any level, until you tick "Send to them on its
// own next time" on a card. As me (PUT /take-the-lead {training}): everything
// it would do comes to you as a card first, until you tick "Do this on its
// own next time". Under each, what you allowed so far, with Remove. The As
// you and As me tabs show only the state and link here (TrainingLine).
export default function TrainingStep({
  delegation,
  onDelegation,
  lead,
  leadState,
  setLead,
}: {
  delegation: DelegationSettings | null;
  onDelegation: (next: DelegationSettings) => void;
  lead: TakeTheLead | null;
  leadState: "loading" | "hidden" | "ready" | "error";
  setLead: (next: TakeTheLead) => void;
}) {
  const training = delegation?.training ?? null;
  if (!training && leadState !== "ready") return null;
  return (
    <SettingsCard
      step={3}
      title="In training"
      description="On: everything becomes a card first, whatever the level and rules say."
    >
      <div className="grid gap-4 sm:grid-cols-2 sm:items-start">
        {training && delegation && <AsYouSide settings={delegation} onSettings={onDelegation} />}
        {leadState === "ready" && lead && <AsMeSide lead={lead} setLead={setLead} />}
      </div>
    </SettingsCard>
  );
}

function SideSwitch({
  id,
  tag,
  text,
  checked,
  busy,
  onChange,
}: {
  id: string;
  tag: string;
  text: string;
  checked: boolean;
  busy: boolean;
  onChange: (next: boolean) => void;
}) {
  return (
    <div className="flex min-h-touch items-center gap-3 rounded-xl border border-line px-4 py-3">
      <span className="flex-none rounded-full bg-accent/10 px-2 py-0.5 text-xs font-semibold text-accent">{tag}</span>
      <span id={id} className="min-w-0 flex-1 text-[15px] text-fg">
        {text}
      </span>
      <Switch checked={checked} disabled={busy} labelledBy={id} onChange={onChange} />
    </div>
  );
}

function AsYouSide({
  settings,
  onSettings,
}: {
  settings: DelegationSettings;
  onSettings: (next: DelegationSettings) => void;
}) {
  const { busy, error, save } = useReachSave(onSettings);
  const training = settings.training;
  if (!training) return null;
  return (
    <div className="flex min-w-0 flex-col gap-3">
      <SideSwitch
        id="as-you-training-label"
        tag="As you"
        text="Hold everything it sends as you"
        checked={Boolean(training.training)}
        busy={busy}
        onChange={(next) => void save(() => setTraining({ training: next }))}
      />
      <LearnedSection
        training={training}
        onSettings={onSettings}
        setting="training"
        title="Goes on its own"
        empty="Nobody yet. Tick “on its own next time” when you send one."
      />
      {error && <p className="text-sm text-red-500">{error}</p>}
    </div>
  );
}

function AsMeSide({ lead, setLead }: { lead: TakeTheLead; setLead: (next: TakeTheLead) => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = async (change: () => Promise<TakeTheLead>) => {
    setBusy(true);
    setError(null);
    try {
      setLead(await change());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save the setting.");
    } finally {
      setBusy(false);
    }
  };
  // What it was allowed in its own name. The same table holds As you's
  // allowances and writing examples (feature act_as_me), listed under As you.
  const learned = lead.learned.filter((item) => item.feature !== "act_as_me");
  return (
    <div className="flex min-w-0 flex-col gap-3">
      <SideSwitch
        id="as-me-training-label"
        tag="As me"
        text="Hold everything it does in its own name"
        checked={lead.training}
        busy={busy}
        onChange={(next) => void run(() => setTakeTheLead({ training: next }))}
      />
      {(lead.training || learned.length > 0) && (
        <div className="border-t border-line pt-5">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-fg-subtle">Goes on its own</h3>
          {learned.length === 0 ? (
            <p className="mt-3 text-sm text-fg-muted">Nothing yet. Tick “Do this on its own next time” when you approve one.</p>
          ) : (
            <ul className="mt-3 flex flex-col divide-y divide-line">
              {learned.map((item) => {
                const example = Object.values(item.example).join(" · ");
                return (
                  <li key={item.id} className="flex min-h-touch items-center justify-between gap-3 py-1.5">
                    <span className="min-w-0 text-[15px] leading-snug">
                      <span className="font-semibold break-words">{item.label}</span>
                      <span className="text-fg-muted">
                        {" · "}allowed{" "}
                        {new Date(item.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                        {item.uses > 0 ? `, done ${item.uses} ${item.uses === 1 ? "time" : "times"} since` : ""}
                        {example ? `, like “${example}”` : ""}
                      </span>
                    </span>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => void run(() => removeLeadLearned(item.id))}
                      className="flex-shrink-0 min-h-touch text-sm font-semibold text-accent hover:underline disabled:opacity-60"
                      aria-label={`Remove ${item.label}`}
                    >
                      Remove
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
      {error && <p className="text-sm text-red-500">{error}</p>}
    </div>
  );
}

// The As you and As me tabs' In training: where it stands in one line, and
// a link to its switch on Approvals.
export function TrainingLine({ side, on }: { side: "you" | "me"; on: boolean }) {
  const what = side === "you" ? "it would send as you" : "it would do in its own name";
  return (
    <SettingsCard>
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
        <p className="min-w-0 flex-1 text-[15px] text-fg">
          <span className="font-semibold">In training: {on ? "On" : "Off"}.</span>{" "}
          <span className="text-fg-muted">
            {on ? `Everything ${what} comes to you as a card first.` : "Approvals alone decides what waits for you."}
          </span>
        </p>
        <Link href={APPROVALS_HREF} className="flex min-h-touch items-center text-sm font-semibold text-accent hover:underline">
          Change on Approvals
        </Link>
      </div>
    </SettingsCard>
  );
}
