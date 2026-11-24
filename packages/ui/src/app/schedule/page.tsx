"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useId, useMemo, useState } from "react";

import SectionTabs, { sectionPanelProps } from "@/components/ui/SectionTabs";
import { PageSpinner } from "@/components/ui/Spinner";
import { getSchedule, stopWatching, type ScheduleItem, type ScheduleKind, type ScheduleResponse } from "@/lib/api";
import {
  FILTER_LABEL,
  KIND_TAG,
  SCHEDULE_FILTERS,
  filterFor,
  filterItems,
  scheduleHref,
  scheduleLine,
  type ScheduleFilter,
} from "@/lib/schedule";

// Work → Schedule: everything the Executive does without being asked, in one
// list, soonest first (GET /schedule): follow-ups, standing work, workflows
// on a schedule, watches, and what it always does. A chip per kind filters
// it (`?show=watch`, how Home, Delegate, Workflows and the Watch list link
// here). Follow-ups and standing work stop here; the rest open where they
// are managed. Pause everything (Settings → How it runs) stops all of it.

const FIRST_ROWS = 7;

const TAG_TONE: Record<ScheduleKind, string> = {
  follow_up: "bg-orange-500/15 text-orange-600 dark:text-orange-400",
  standing: "bg-amber-500/15 text-amber-700 dark:text-amber-400",
  workflow: "bg-accent/15 text-accent",
  watch: "bg-surface-overlay text-fg-muted",
  always: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400",
};

export default function SchedulePage() {
  return (
    <main className="flex-1 min-h-0 overflow-y-auto">
      {/* useSearchParams (for ?show=) needs a Suspense boundary. */}
      <Suspense>
        <Schedule />
      </Suspense>
    </main>
  );
}

function Schedule() {
  const router = useRouter();
  const params = useSearchParams();
  const filter = filterFor(params.get("show"));
  const idBase = useId();
  const [data, setData] = useState<ScheduleResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stopped, setStopped] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState<string | null>(null);
  const [stopError, setStopError] = useState<string | null>(null);
  const [all, setAll] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    getSchedule(controller.signal)
      .then(setData)
      .catch((err: unknown) => {
        if ((err as Error)?.name !== "AbortError") setError("Couldn't load the schedule.");
      });
    return () => controller.abort();
  }, []);

  const items = useMemo(() => (data?.items ?? []).filter((i) => !stopped.has(i.key)), [data, stopped]);
  const counts = useMemo(() => {
    const c: Record<ScheduleFilter, number> = { all: items.length, follow_up: 0, standing: 0, workflow: 0, watch: 0, always: 0 };
    for (const i of items) c[i.kind] += 1;
    return c;
  }, [items]);
  const shown = filterItems(items, filter);
  const visible = all ? shown : shown.slice(0, FIRST_ROWS);

  const pick = (id: ScheduleFilter) => {
    setAll(false);
    router.replace(id === "all" ? scheduleHref() : scheduleHref(id), { scroll: false });
  };

  const stop = useCallback(async (item: ScheduleItem) => {
    if (!item.stop_key) return;
    setBusy(item.key);
    setStopError(null);
    try {
      await stopWatching(item.stop_key);
      setStopped((prev) => new Set(prev).add(item.key));
    } catch (err) {
      setStopError(err instanceof Error ? err.message : "Couldn't stop it.");
    } finally {
      setBusy(null);
    }
  }, []);

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6 sm:py-8 space-y-5">
      <header>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-fg">On my schedule</h1>
        <p className="text-[15px] text-fg-muted mt-1.5 max-w-2xl">
          Everything I do without being asked, soonest first.{" "}
          <Link href="/settings/executive" className="font-medium text-accent hover:underline">
            Pause everything
          </Link>{" "}
          stops all of it.
        </p>
      </header>

      {error && <p className="text-[15px] text-red-500">{error}</p>}
      {!data && !error && <PageSpinner />}

      {data && (
        <>
          <SectionTabs<ScheduleFilter>
            tabs={SCHEDULE_FILTERS.map((id) => ({ id, label: FILTER_LABEL[id], count: counts[id] }))}
            active={filter}
            onChange={pick}
            label="Show"
            idBase={idBase}
          />
          <div {...sectionPanelProps(idBase, filter)}>
            {shown.length === 0 ? (
              <p className="rounded-2xl border border-line bg-surface-elevated px-5 py-8 text-[15px] text-fg-muted">
                {filter === "all" ? "Nothing on my schedule right now." : `No ${FILTER_LABEL[filter].toLowerCase()} right now.`}
              </p>
            ) : (
              <div className="rounded-2xl border border-line bg-surface-elevated">
                <ul>
                  {visible.map((item) => (
                    <Row key={item.key} item={item} busy={busy === item.key} onStop={() => void stop(item)} />
                  ))}
                </ul>
                {shown.length > visible.length && (
                  <button
                    type="button"
                    onClick={() => setAll(true)}
                    className="flex min-h-touch w-full items-center border-t border-line px-4 sm:px-5 text-[15px] font-semibold text-accent hover:underline"
                  >
                    Show {shown.length - visible.length} more
                  </button>
                )}
              </div>
            )}
            {stopError && <p className="mt-2 text-sm text-red-500">{stopError}</p>}
          </div>
        </>
      )}
    </div>
  );
}

function Row({ item, busy, onStop }: { item: ScheduleItem; busy: boolean; onStop: () => void }) {
  const line = scheduleLine(item);
  return (
    <li className="flex items-center gap-3 border-t border-line px-4 py-3 first:border-t-0 sm:px-5">
      <span
        className={`w-[5.5rem] sm:w-24 flex-none rounded-full px-2 py-0.5 text-center text-xs font-semibold ${TAG_TONE[item.kind]}`}
      >
        {KIND_TAG[item.kind]}
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-[15px] font-medium text-fg break-words">{item.title}</p>
        {line && <p className="text-sm text-fg-muted break-words">{line}</p>}
      </div>
      {item.stop_key ? (
        <button
          type="button"
          disabled={busy}
          onClick={onStop}
          className="flex min-h-touch flex-none items-center text-sm font-semibold text-accent hover:underline disabled:opacity-60"
          aria-label={`Stop: ${item.title}`}
        >
          Stop
        </button>
      ) : item.href ? (
        <Link
          href={item.href}
          className="flex min-h-touch flex-none items-center text-sm font-semibold text-accent hover:underline"
          aria-label={`Open: ${item.title}`}
        >
          Open
        </Link>
      ) : null}
    </li>
  );
}
