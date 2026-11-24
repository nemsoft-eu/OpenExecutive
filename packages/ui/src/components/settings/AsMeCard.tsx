"use client";

import Link from "next/link";

import ScheduleLink from "@/components/schedule/ScheduleLink";
import { useEffect, useState, type FormEvent } from "react";

import ExecutiveRunSwitch from "@/components/executive/ExecutiveRunSwitch";
import SettingsCard from "@/components/settings/SettingsCard";
import { TrainingLine } from "@/components/settings/TrainingStep";
import {
  addLeadJob,
  getTakeTheLead,
  stopLeadJob,
  type OutsideContentMode,
  type TakeTheLead,
} from "@/lib/api";
import { APPROVALS_LEADS, APPROVALS_SENDS } from "@/lib/reach";
import { PageSpinner } from "@/components/ui/Spinner";

// Delegate → As me (Take the lead, GET/PUT /take-the-lead): the settings for
// what the Executive does in its own name without being asked. What it did,
// what it left for you and what it is watching are on Home; this tab says so
// in one line. Standing jobs are what you asked it to keep doing; When I look
// lists what makes it look. It has no switch of its own: Pause everything
// (the same switch as Settings) stops it with everything else, and Approvals
// decides each step. In training is the last check: everything it would do
// comes to you as a card on Home, with "Do this on its own next time"; its
// switch and what you allowed are on Delegate → Approvals, and this tab
// shows its state (TrainingLine). What you approved three times unchanged is
// offered on Delegate → Approvals (Could go on its own) and on its next
// card. The owner's alone: anyone else sees who
// can change it.

// What makes it look, besides its daily look before your brief
// (orchestrator.take_the_lead: wake and watch).
const LOOKS = ["New mail", "A chat message", "A goal slips", "A workflow fails", "A meeting ends", "A task goes overdue"];

function things(n: number): string {
  return `${n} ${n === 1 ? "thing" : "things"}`;
}

export function useTakeTheLead(): {
  lead: TakeTheLead | null;
  state: "loading" | "hidden" | "ready" | "error";
  setLead: (next: TakeTheLead) => void;
} {
  const [lead, setLead] = useState<TakeTheLead | null>(null);
  const [state, setState] = useState<"loading" | "hidden" | "ready" | "error">("loading");

  useEffect(() => {
    const controller = new AbortController();
    getTakeTheLead(controller.signal)
      .then((next) => {
        setLead(next);
        setState(next ? "ready" : "hidden");
      })
      .catch((err) => {
        if ((err as Error)?.name !== "AbortError") setState("error");
      });
    return () => controller.abort();
  }, []);

  return { lead, state, setLead };
}

export default function AsMeCard({
  lead,
  state,
  setLead,
  level,
}: {
  lead: TakeTheLead | null;
  state: "loading" | "hidden" | "ready" | "error";
  setLead: (next: TakeTheLead) => void;
  level: OutsideContentMode | null;
}) {
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

  const levelLine = level ? `Approvals is on ${APPROVALS_SENDS[level].label}. ${APPROVALS_LEADS[level]}` : "Approvals decides what waits for your yes.";

  const [job, setJob] = useState("");
  const addJob = (event: FormEvent) => {
    event.preventDefault();
    const text = job.trim();
    if (!text) return;
    void run(async () => {
      const next = await addLeadJob(text);
      setJob("");
      return next;
    });
  };
  const jobs = lead?.jobs ?? [];

  return (
    <>
      <div className="flex min-w-0 flex-col gap-4">
        <SettingsCard>
          <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
            <div className="min-w-0 flex-1">
              {state === "ready" && lead && (
                <p className="text-[15px] font-semibold text-fg">
                  {things(lead.done_today ?? 0)} done today
                </p>
              )}
              <p className="mt-1 text-sm text-fg-muted">
                What I did, what I left for you and what I am watching are on Home.
              </p>
            </div>
            <Link href="/" className="flex min-h-touch items-center text-sm font-semibold text-accent hover:underline">
              Open Home
            </Link>
          </div>
        </SettingsCard>

        {state === "ready" && lead && (
          <SettingsCard
            title="Standing work"
            description="Things you asked me to keep doing. Say it in chat or add one here. Each step still goes through Approvals."
            action={<ScheduleLink kind="standing" />}
          >
            {jobs.length > 0 && (
              <ul className="mb-4 flex flex-col divide-y divide-line rounded-xl border border-line">
                {jobs.map((item) => (
                  <li key={item.id} className="flex min-h-touch items-center justify-between gap-3 px-4 py-2">
                    <span className="min-w-0">
                      <span className="block text-[15px] font-medium break-words">{item.text}</span>
                      <span className="mt-0.5 block text-[13px] leading-snug text-fg-muted">
                        Since {new Date(item.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                        {item.this_week ? ` · ${item.this_week} this week` : " · nothing yet this week"}
                      </span>
                    </span>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => void run(() => stopLeadJob(item.id))}
                      className="flex-shrink-0 text-sm font-semibold text-accent hover:underline disabled:opacity-60"
                      aria-label={`Stop ${item.text}`}
                    >
                      Stop
                    </button>
                  </li>
                ))}
              </ul>
            )}
            <form onSubmit={addJob} className="flex gap-2">
              <input
                type="text"
                value={job}
                onChange={(event) => setJob(event.target.value)}
                maxLength={200}
                placeholder="Give me standing work…"
                aria-label="Give me standing work"
                className="min-h-touch min-w-0 flex-1 rounded-full border border-line bg-surface px-4 text-[15px] text-fg placeholder:text-fg-subtle focus:border-accent focus:outline-none"
              />
              {job.trim() && (
                <button
                  type="submit"
                  disabled={busy}
                  className="flex-none rounded-full bg-accent px-4 text-sm font-semibold text-white disabled:opacity-60"
                >
                  Add
                </button>
              )}
            </form>
            {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
          </SettingsCard>
        )}

        <SettingsCard title="When I look" description="Every day before your brief, and when any of these happen.">
          <ul className="flex flex-wrap gap-2">
            {LOOKS.map((look) => (
              <li key={look} className="rounded-lg bg-surface-overlay px-3 py-1.5 text-[15px] text-fg">
                {look}
              </li>
            ))}
          </ul>
        </SettingsCard>
      </div>

      <div className="flex min-w-0 flex-col gap-4">
        {state === "loading" && <PageSpinner />}
        {state === "ready" && lead && <TrainingLine side="me" on={lead.training} />}

        <SettingsCard
          title="Pause everything"
          description="This is the same switch as Settings → Your Executive. Paused, the Executive only answers when asked: no briefs, nudges, monitoring or work in its own name."
        >
          <ExecutiveRunSwitch />
          <p className="mt-4 border-t border-line pt-4 text-[15px] text-fg">
            <span className="text-fg-muted">Now: </span>
            {lead?.training ? "In training: everything it would do comes to you first, until you allow it." : levelLine}
          </p>
        </SettingsCard>
      </div>
    </>
  );
}
