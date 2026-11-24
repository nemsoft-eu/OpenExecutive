"use client";

// The part every Needs you card shares (briefing/card_view.py): what I'll
// do, the options to pick from before anything happens (the first is the
// Executive's own pick), what was said on the card, and "Or tell me…". What
// the person says goes to POST /alerts/{id}/say, which takes it the card's
// own way (an answer, a new plan, a correction to a watch, or a go) and
// hands back the card as it reads now. While that runs, their words show on
// the card with "On it…"; when it started the work, the card follows it to
// the end.

import Link from "next/link";
import { useEffect, useState } from "react";

import { type CardOption, type CardView, type ProposalItem, type SayResult, getAlertApproval, sayOnCard } from "@/lib/api";
import { APPROVAL_POLL_MAX_MS, APPROVAL_POLL_MS, type AlertApproval, chatHref } from "@/lib/approvals";
import { isImeComposing } from "@/lib/imeComposing";
import { buttonClass } from "@/components/ui/Button";

const FIELD =
  "min-w-0 flex-1 resize-none [field-sizing:content] max-h-40 bg-transparent px-1 py-1 text-[15px] leading-snug text-fg placeholder:text-fg-subtle focus:outline-none disabled:opacity-50";

export function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mt-3">
      <div className="mb-0.5 text-xs font-semibold uppercase tracking-wide text-fg-subtle">{title}</div>
      <div className="text-[15px] leading-snug text-fg whitespace-pre-wrap break-words">{children}</div>
    </div>
  );
}

// "Or tell me…": a line said on a card. ``say`` takes it where it goes and
// throws with a message to show when it couldn't.
export function SayBox({
  id,
  disabled = false,
  autoFocus = false,
  onSending,
  say,
  onFailed,
}: {
  id: string;
  disabled?: boolean;
  autoFocus?: boolean;
  onSending?: (text: string) => void;
  say: (text: string) => Promise<void>;
  onFailed?: () => void;
}) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fieldId = `card-say-${id}`;

  const send = async () => {
    const value = text.trim();
    if (!value || busy) return;
    setBusy(true);
    setError(null);
    onSending?.(value);
    try {
      await say(value);
      setText("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't send that. Try again.");
      onFailed?.();
    } finally {
      setBusy(false);
    }
  };

  return (
    <form
      className="mt-3"
      onSubmit={(e) => {
        e.preventDefault();
        void send();
      }}
    >
      <label htmlFor={fieldId} className="sr-only">
        Or tell me
      </label>
      <div className="flex items-end gap-2 rounded-xl border border-line bg-surface px-2.5 py-2 focus-within:ring-2 focus-within:ring-accent/40">
        <textarea
          id={fieldId}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !isImeComposing(e) && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
          rows={1}
          maxLength={500}
          autoFocus={autoFocus}
          disabled={busy || disabled}
          placeholder="Or tell me…"
          className={FIELD}
        />
        <button
          type="submit"
          disabled={busy || disabled || !text.trim()}
          aria-label="Send"
          className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-accent text-white disabled:opacity-40"
        >
          {busy ? (
            <span aria-hidden="true" className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/40 border-t-white" />
          ) : (
            <span aria-hidden="true">↑</span>
          )}
        </button>
      </div>
      {error && (
        <p role="alert" className="mt-1 text-sm text-red-500">
          {error}
        </p>
      )}
    </form>
  );
}

// "Or tell me…" on a Needs you card (POST /alerts/{id}/say). Also the whole
// of a watch row's "Correct it" (AwarenessPanels).
export function TellMe({
  alertId,
  seenPlan = "",
  disabled = false,
  autoFocus = false,
  onSending,
  onSaid,
  onFailed,
}: {
  alertId: number;
  // The plan the card shows as they speak.
  seenPlan?: string;
  disabled?: boolean;
  autoFocus?: boolean;
  onSending?: (text: string) => void;
  onSaid: (result: SayResult, said: string) => void;
  onFailed?: () => void;
}) {
  return (
    <SayBox
      id={String(alertId)}
      disabled={disabled}
      autoFocus={autoFocus}
      onSending={onSending}
      onFailed={onFailed}
      say={async (text) => onSaid(await sayOnCard(alertId, text, seenPlan), text)}
    />
  );
}

// What was said on a card, and what the Executive said back; their words
// just sent show with "On it…" until the answer comes.
export function Turns({
  turns,
  sending = null,
  children,
}: {
  turns: { who: "you" | "me"; text: string }[];
  sending?: string | null;
  children?: React.ReactNode;
}) {
  if (turns.length === 0 && !sending && !children) return null;
  return (
    <div className="mt-3 space-y-2 border-t border-line/60 pt-3" aria-live="polite">
      {turns.map((t, i) =>
        t.who === "you" ? (
          <div key={i} className="ml-8 rounded-xl bg-accent/10 px-3 py-2 text-[15px] leading-snug text-fg break-words">
            {t.text}
          </div>
        ) : (
          <p key={i} className="text-[15px] leading-snug text-fg break-words">
            {t.text}
          </p>
        ),
      )}
      {sending && (
        <>
          <div className="ml-8 rounded-xl bg-accent/10 px-3 py-2 text-[15px] leading-snug text-fg break-words">{sending}</div>
          <p className="text-[15px] text-fg-muted">On it…</p>
        </>
      )}
      {children}
    </div>
  );
}

// Follows work started from the card until it ends, and says how it went.
function Working({ approval }: { approval: AlertApproval }) {
  const [now, setNow] = useState(approval);
  useEffect(() => {
    if (now.status !== "running") return;
    const started = Date.now();
    const timer = setInterval(() => {
      if (Date.now() - started > APPROVAL_POLL_MAX_MS) {
        clearInterval(timer);
        return;
      }
      getAlertApproval(approval.id)
        .then((a) => {
          setNow(a);
          if (a.status !== "running") clearInterval(timer);
        })
        .catch(() => { /* try again on the next tick */ });
    }, APPROVAL_POLL_MS);
    return () => clearInterval(timer);
  }, [approval.id, now.status]);
  const href = chatHref(now.session_id);
  const text =
    now.status === "running" ? "Working on it…" : now.status === "done" ? `Done. ${now.outcome === "Done." ? "" : now.outcome}`.trim() : `It didn't finish. ${now.outcome}`.trim();
  return (
    <p className={`text-[15px] leading-snug ${now.status === "failed" ? "text-red-500" : "text-fg"}`} aria-live="polite">
      {now.status === "running" && (
        <span aria-hidden="true" className="mr-2 inline-block h-3 w-3 animate-spin rounded-full border-2 border-fg-subtle/40 border-t-fg-subtle align-[-1px]" />
      )}
      {text}
      {href && (
        <Link href={href} className="ml-2 text-sm text-accent hover:underline">
          {now.status === "running" ? "Watch it" : "See what I did"}
        </Link>
      )}
    </p>
  );
}

export default function CardTalk({
  proposal,
  view,
  disabled = false,
  onPick,
  onSaid,
}: {
  proposal: ProposalItem;
  view: CardView;
  disabled?: boolean;
  // A tap on an option: Approve with its plan, or send it as the answer.
  onPick?: (option: CardOption) => void;
  // What was said on the card was taken: the card as it reads now.
  onSaid?: (result: SayResult) => void;
}) {
  // What they just said, shown with "On it…" until the card comes back.
  const [sending, setSending] = useState<string | null>(null);
  const [approval, setApproval] = useState<AlertApproval | null>(null);
  // What was said when it closed the card: no card comes back to carry it.
  const [last, setLast] = useState<{ you: string; me: string } | null>(null);
  const turns = [
    ...(view.turns ?? []),
    ...(last ? [{ who: "you" as const, text: last.you }, { who: "me" as const, text: last.me }] : []),
  ];
  const options = view.options ?? [];
  const going = approval !== null;

  return (
    <div>
      {view.plan && view.plan_title && <Section title={view.plan_title}>{view.plan}</Section>}

      <Turns turns={turns} sending={sending}>
        {approval && <Working approval={approval} />}
      </Turns>

      {options.length > 0 && onPick && !going && (
        <div className="mt-3 flex flex-col gap-2" role="group" aria-label="Options">
          {options.map((o, i) => (
            <button
              key={`${i}-${o.label}`}
              type="button"
              disabled={disabled || sending !== null}
              onClick={() => onPick(o)}
              className={`${buttonClass(i === 0 ? "primary" : "secondary", "md")} h-auto w-full flex-col items-start justify-start py-2 text-left`}
            >
              <span>{o.label}</span>
              {/* What another option does, in full: the first one's is
                  under What I'll do. */}
              {o.action === "approve" && i > 0 && o.plan && (
                <span className="mt-0.5 whitespace-pre-wrap break-words text-sm font-normal opacity-80">{o.plan}</span>
              )}
            </button>
          ))}
        </div>
      )}

      {view.tell_me && !going && (
        <TellMe
          alertId={proposal.alert_id}
          seenPlan={view.plan_title && view.plan ? view.plan : ""}
          disabled={disabled}
          onSending={setSending}
          onFailed={() => setSending(null)}
          onSaid={(r, said) => {
            if (r.card === null) setLast({ you: said, me: r.reply });
            setSending(null);
            if (r.approval) setApproval(r.approval);
            onSaid?.(r);
          }}
        />
      )}
    </div>
  );
}
