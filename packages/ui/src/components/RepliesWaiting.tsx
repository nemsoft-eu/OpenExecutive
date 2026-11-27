"use client";

import { useCallback, useEffect, useState, type ReactNode } from "react";

import Button, { buttonClass } from "@/components/ui/Button";
import {
  answerReplyQuestions,
  changeReply,
  dismissReplyCard,
  editReplyText,
  setReplyFrom,
  setReplyRecipients,
  getReplyCards,
  ReplySendError,
  sendReplyCard,
  stopReplyWork,
  undoReplyChange,
  type ReplyCard,
  type ReplyTurn,
} from "@/lib/api";
import { Spinner } from "@/components/ui/Spinner";
import { formatRelativeTime } from "@/lib/relativeTime";
import {
  cardGist,
  questionsLeft,
  replyFlagLines,
  replyToChoice,
  mailboxName,
  safeGmailLink,
  sendLeftNothing,
  sendButtonLabel,
  sendQuestion,
  senderLine,
  recipientNames,
  firstNames,
} from "@/lib/replyCards";
import EmailWords from "@/components/EmailWords";
import FromSwitch from "@/components/FromSwitch";
import FeatureName from "@/components/FeatureName";
import { SayBox, Section } from "@/components/briefing/CardTalk";
import GoOnItsOwnOffer from "@/components/briefing/GoOnItsOwnOffer";
import NextTimeTick from "@/components/NextTimeTick";

// Home: the replies the Executive wrote as you for mail that needs you
// (Delegate → Act as me → Draft replies to my inbox), shown as cards in
// "Needs you". Each card starts closed, the way every Needs you card reads:
// what happened (the Executive's line on what they asked), what I'll do
// with "See reply to <name>" on its own line, the options (Send from me, or from the Executive
// with you on cc; with something left to decide, its answers to tap), "Or
// tell me…" and Skip. A go said in the box ("send it") brings up the same
// confirm a tap on Send does. Open, top to bottom: what they wrote and the
// Executive's word on the reply with what's left to decide (short answers to
// tap, or "Something else…" to type one) and what was decided (Change to pick
// again); the reply itself, the one version that goes, with From and Reply
// to; Send and Skip; then the talk, newest last: any newer email the
// Executive brought the reply up to date with, what you asked it to change
// and what it did, quoting the lines it changed, with Undo (POST
// /delegation/replies/{id}/undo), "Earlier" for the rest; and "Tell me what
// to change…" last (POST /delegation/replies/{id}/change), so an answer
// lands next to where you typed. Each
// answer or change rewrites the draft (POST /delegation/replies/{id}/answer)
// and lightly marks the new sentences. "Edit the words myself" changes them
// right on the card (and in your Drafts when it's there: PUT
// /delegation/replies/{id}/text). The reply waits on the card alone unless
// you chose to keep replies in your Drafts too. Send waits until each
// question with answers to tap is answered, or you changed the words
// yourself, and sends it from your own mailbox after you confirm who it goes
// to; Open in Gmail (when it's in your Drafts) and Skip sit beside it.
// GET /delegation/replies answers only the owner, so nothing shows for
// everyone else, on a backend without it, and when nothing is waiting.

// While a send Gmail hasn't confirmed is being settled (the card is
// "executing"), how often the cards are read again.
const SETTLE_POLL_MS = 20_000;
// While the Executive works on a card, how often it is read again, so its
// steps show as they happen.
const WORK_POLL_MS = 3_000;
// What shows while "Reply to" is being changed.
const RECIPIENTS_RUNNING = "Changing who it goes to…";
// What shows while From is being changed (the Executive rewrites it).
const FROM_RUNNING = "Switching who it's from…";
const STOPPING = "Stopping…";
const UNDOING = "Undoing…";
// Turns the card's talk shows before "Earlier": your latest words and my answer.
const TALK_SHOWN = 2;

// The cards and their refresh, for the Home "Needs you" list, which shows
// each one as a card among the proposals. `cards` stays null for anyone but
// the owner, on a backend without the route, and until the first read.
export function useReplyCards() {
  const [cards, setCards] = useState<ReplyCard[] | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getReplyCards(controller.signal)
      .then(setCards)
      .catch(() => { /* no card is better than a broken one */ });
    return () => controller.abort();
  }, []);

  const refresh = useCallback(async () => {
    try {
      const next = await getReplyCards();
      if (next) setCards(next);
    } catch {
      // Keep what's shown; the next look may work.
    }
  }, []);

  const settling = (cards ?? []).some((c) => c.status === "executing" || c.working);
  const working = (cards ?? []).some((c) => c.working);
  useEffect(() => {
    if (!settling) return;
    const timer = setInterval(() => void refresh(), working ? WORK_POLL_MS : SETTLE_POLL_MS);
    return () => clearInterval(timer);
  }, [settling, working, refresh]);

  const gone = useCallback((decisionId: number, note?: string) => {
    setCards((prev) => (prev ?? []).filter((c) => c.decision_id !== decisionId));
    setNotice(note ?? null);
  }, []);

  // A card changed in place (its words edited).
  const replace = useCallback((card: ReplyCard) => {
    setCards((prev) => (prev ?? []).map((c) => (c.decision_id === card.decision_id ? card : c)));
  }, []);

  return { cards: cards ?? [], notice, gone, refresh, replace };
}

// What the Replies-waiting cards are, for the Needs you header's tip.
export const REPLIES_WAITING_TIP =
  "Mail that needs you, with a first reply the Executive wrote in your voice. Tap an answer " +
  "under Decide before sending to write it in, or tap Edit to change it yourself. Nothing is " +
  "sent until you tap Send. If you keep replies " +
  "in your Drafts too, Skip deletes that draft unless you changed it in your mailbox. " +
  "Only you see these.";

// What the row is doing: idle, asking before sending (first or second
// time), or waiting on the backend.
// `allow`: Send with the "on its own next time" tick, In training (everything
// sent as you to them, or to a follow-up's people, goes on its own after).
type Step =
  | { kind: "idle" }
  | { kind: "ask"; recipients: string[]; allow: boolean }
  | { kind: "confirm"; message: string; recipients: string[]; threadMovedOn: boolean; allow: boolean }
  | { kind: "busy"; label: string };

// The Executive's turn in a card's conversation; `status` says how its
// work stands ("working", "done", "stopped", "stuck").
function ExecBubble({ children, status = "", dim = false }: { children: ReactNode; status?: string; dim?: boolean }) {
  return (
    <div
      className={`max-w-[92%] self-start rounded-2xl rounded-tl-md bg-accent/10 px-3 py-2 text-sm text-fg break-words ${
        dim ? "opacity-70" : ""
      }`}
    >
      <div className="text-xs font-semibold text-accent">
        Executive{status ? ` · ${status}` : ""}
      </div>
      <div className="mt-0.5">{children}</div>
    </div>
  );
}

// What a run of the work did, one line each: done steps ticked, the one it
// is on now with the spinner, a stopped or stuck end marked.
function WorkSteps({ steps, now = "", end = "" }: { steps: string[]; now?: string; end?: string }) {
  if (steps.length === 0 && !now) return null;
  return (
    <ul className="mt-1.5 space-y-0.5 text-[13.5px]" aria-label="Steps">
      {steps.map((text, i) => {
        const last = i === steps.length - 1 && !now;
        const mark = last && end === "stopped" ? "■" : last && end === "stuck" ? "!" : "✓";
        return (
          <li key={i} className="flex items-start gap-2">
            <span
              aria-hidden="true"
              className={`w-3.5 flex-shrink-0 font-bold ${
                mark === "✓" ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600 dark:text-amber-400"
              }`}
            >
              {mark}
            </span>
            <span>{text}</span>
          </li>
        );
      })}
      {now && (
        <li className="flex items-start gap-2" aria-live="polite">
          <span className="mt-0.5 w-3.5 flex-shrink-0">
            <Spinner label="Working" />
          </span>
          <span>{now}</span>
        </li>
      )}
    </ul>
  );
}

// How a run of the work ended, as the bubble's header says it.
const WORK_STATUS: Record<string, string> = { done: "done", stopped: "stopped", stuck: "couldn't finish" };

// What a newer email in the conversation carried: its subject when it is
// its own, and its files.
function NewsExtras({ turn, subject }: { turn: ReplyTurn; subject: string }) {
  const own = (turn.subject ?? "").trim();
  const files = turn.files ?? [];
  return (
    <>
      {own && own.replace(/^(re|fwd?):\s*/i, "") !== subject.replace(/^(re|fwd?):\s*/i, "") && (
        <span> · &ldquo;{own}&rdquo;</span>
      )}
      {files.length > 0 && (
        <ul className="mt-1 flex flex-wrap gap-1.5" aria-label="Files">
          {files.map((name, i) => (
            <li key={`${i}-${name}`} className="rounded-lg border border-line-strong bg-surface px-2 py-0.5 text-[13px] text-fg">
              {name}
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

export function ReplyCardItem({
  card,
  onGone,
  onRefresh,
  onChanged,
  emphasized = false,
  defaultOpen = false,
}: {
  card: ReplyCard;
  emphasized?: boolean;
  // Start open (Right now's View email shows the reply straight away).
  defaultOpen?: boolean;
  onGone: (id: number, note?: string) => void;
  onRefresh: () => Promise<void>;
  onChanged?: (card: ReplyCard) => void;
}) {
  const [step, setStep] = useState<Step>({ kind: "idle" });
  const [error, setError] = useState<string | null>(null);
  // Closed until tapped: the summary, what's left to decide, one main button.
  const [open, setOpen] = useState(defaultOpen);
  // Edit: the draft's words in a box on the card, until Save or Cancel.
  const [editing, setEditing] = useState<string | null>(null);
  // Learn from my edits: keep what you sent as an example when you changed it first.
  const [keepExample, setKeepExample] = useState(true);
  // "Something else…": which question's answer box is open, and its words.
  const [typing, setTyping] = useState<{ index: number; text: string } | null>(null);
  // "Tell me what to change…": the words in the box under the reply.
  const [ask, setAsk] = useState("");
  // The card's talk shows its latest exchange; "Earlier" shows the rest.
  const [showEarlier, setShowEarlier] = useState(false);
  // Answered questions opened again with Change, to tap another answer.
  const [reopened, setReopened] = useState<number[]>([]);
  const questions = card.questions ?? [];
  const left = questionsLeft(card);
  const changed = card.changed_sentences ?? [];
  const learnsStyle = Boolean(card.learns_style);
  // Being settled: Gmail didn't confirm a send. The server says so, and the
  // section reads the cards again until it's settled.
  const unconfirmed = card.status === "executing";
  const choice = replyToChoice(card);
  // The Reply to choice says who else was on it, so the warning doesn't.
  const warnings = replyFlagLines(choice ? card.flags.filter((f) => f !== "others_on_thread") : card.flags);
  const received = formatRelativeTime(card.received_at);
  const gmailLink = safeGmailLink(card.gmail_link);
  const mailbox = card.mailbox || mailboxName(card.gmail_link);
  // Kept in their Drafts too, or on this card alone until Send.
  const inMailbox = card.in_mailbox !== false;
  const who = card.from_name.trim() || card.from_email;
  const followUp = card.source === "follow_up";
  const firstName = card.from_name.trim().split(/\s+/)[0] || card.from_email;
  const [nextTime, setNextTime] = useState(false);

  const dismiss = async () => {
    setStep({ kind: "busy", label: "Removing…" });
    setError(null);
    try {
      await dismissReplyCard(card.decision_id);
      onGone(card.decision_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't dismiss that reply.");
      setStep({ kind: "idle" });
    }
  };

  const saveEdit = async () => {
    if (editing === null) return;
    setStep({ kind: "busy", label: "Saving…" });
    setError(null);
    try {
      const updated = await editReplyText(card.decision_id, editing);
      onChanged?.(updated);
      setEditing(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't save your change.");
    } finally {
      setStep({ kind: "idle" });
    }
  };

  const pickRecipients = async (everyone: boolean) => {
    if (!choice || everyone === choice.toEveryone) return;
    setStep({ kind: "busy", label: RECIPIENTS_RUNNING });
    setError(null);
    try {
      onChanged?.(await setReplyRecipients(card.decision_id, everyone));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't change who it goes to.");
    } finally {
      setStep({ kind: "idle" });
    }
  };

  const pickFrom = async (executive: boolean) => {
    if (executive === Boolean(card.from_executive)) return;
    setStep({ kind: "busy", label: FROM_RUNNING });
    setError(null);
    try {
      onChanged?.(await setReplyFrom(card.decision_id, executive));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't change who it's from.");
    } finally {
      setStep({ kind: "idle" });
    }
  };

  const answer = async (index: number, text: string) => {
    const words = text.trim();
    if (!words) return;
    setStep({ kind: "busy", label: "Writing it in…" });
    setError(null);
    try {
      const updated = await answerReplyQuestions(card.decision_id, [
        { question_index: index, answer_text: words },
      ]);
      onChanged?.(updated);
      setTyping(null);
      setReopened((r) => r.filter((x) => x !== index));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't add that to your reply.");
    } finally {
      setStep({ kind: "idle" });
    }
  };

  // Words said on the card ("make it shorter", "send it"): the card as it
  // reads now. A go brings up the send's confirm, as a tap on Send does.
  // Throws with a message to show when it couldn't.
  const say = async (words: string, work = false) => {
    setStep({ kind: "busy", label: "Changing it…" });
    setError(null);
    let next: Step = { kind: "idle" };
    try {
      const updated = await changeReply(card.decision_id, words, { work });
      onChanged?.(updated);
      if (updated.send_now) {
        setOpen(true);
        next = { kind: "ask", recipients: updated.draft_to, allow: Boolean(updated.can_allow) && nextTime };
      }
    } finally {
      setStep(next);
    }
  };

  const askChange = async (text = ask, work = false) => {
    const words = text.trim();
    if (!words) return;
    try {
      await say(words, work);
      if (!work) setAsk("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't change your reply.");
    }
  };

  // Undo: the words as they were before the last change you asked for.
  const undo = async () => {
    setStep({ kind: "busy", label: UNDOING });
    setError(null);
    try {
      onChanged?.(await undoReplyChange(card.decision_id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't undo that.");
    } finally {
      setStep({ kind: "idle" });
    }
  };

  // Stop what it is doing: the card keeps what it did so far.
  const stopWork = async () => {
    setStep({ kind: "busy", label: STOPPING });
    setError(null);
    try {
      onChanged?.(await stopReplyWork(card.decision_id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't stop it.");
    } finally {
      setStep({ kind: "idle" });
    }
  };
  const send = async (confirm: { recipients: string[]; thread_moved_on?: boolean }, allow: boolean) => {
    setStep({ kind: "busy", label: "Sending…" });
    setError(null);
    try {
      const result = await sendReplyCard(
        card.decision_id,
        {
          ...confirm,
          ...(card.caught_up_at ? { seen_caught_up_at: card.caught_up_at } : {}),
          ...(allow ? { allow: true } : {}),
          ...(learnsStyle && keepExample ? { example: true } : {}),
        },
      );
      if (result.status === "sent") {
        onGone(
          card.decision_id,
          card.from_executive
            ? `${card.executive_name || "Executive"} sent it to ${who}, with you on cc.`
            : followUp
            ? allow
              ? `Sent your follow-up to ${who}. Follow-ups to them go on their own from now on.`
              : `Sent your follow-up to ${who}.`
            : allow
              ? `Sent your reply to ${who}. Replies to them go on their own from now on.`
              : `Sent your reply to ${who}.`,
        );
        return;
      }
      // The reply may have been brought up to date with newer mail since it
      // was shown: show it as it is now, beside the second question.
      // Or changed from chat since it was shown.
      if (result.reasons.includes("thread_moved_on") || result.reasons.includes("changed_in_chat")) void onRefresh();
      setStep({
        kind: "confirm",
        message: result.message,
        recipients: result.recipients,
        threadMovedOn: result.reasons.includes("thread_moved_on"),
        allow,
      });
    } catch (err) {
      const code = err instanceof ReplySendError ? err.code : "error";
      const message = err instanceof Error ? err.message : "Couldn't send that reply.";
      if (sendLeftNothing(code)) {
        onGone(card.decision_id, message);
        return;
      }
      setError(message);
      setStep({ kind: "idle" });
      if (code === "send_unconfirmed") void onRefresh();
    }
  };

  const busy = step.kind === "busy";
  // Each button shows only its own running words, so Skip or Save never
  // reads as "Sending…" on the Send button.
  const running = (label: string) => step.kind === "busy" && step.label === label;
  const confirming = step.kind === "ask" || step.kind === "confirm";
  const label = "text-xs font-semibold uppercase tracking-wide text-fg-muted";
  // Closed, the card is its summary, what's left to decide and one main
  // button; it opens on a tap, and stays open while you edit or confirm.
  const expanded = open || confirming || editing !== null || typing !== null || ask.trim() !== "";
  const gist = cardGist(card);
  // Who the reply goes to, by name; a follow-up says nobody answered yet.
  const toList = card.draft_to?.length ? card.draft_to : [who];
  const toNames = recipientNames(toList);
  const toLine = followUp ? `Nobody answered your email to ${toNames}` : `To ${toNames}`;
  // The closed card's way in to the reply, named for who it goes to.
  const seeReply = followUp ? "See your follow-up" : `See reply to ${firstNames(toList)}`;
  const tappable = questions.some((q) => q.choices.length > 0);
  // Open, everything shows at once, as a conversation: what they wrote, the
  // Executive's word on the reply with what's left to decide, anything that
  // came in or changed since, then the reply itself and a box to ask for a
  // change. What's left to decide sits with the Executive's latest word.
  const openQuestions = questions
    .map((q, i) => ({ q, i }))
    .filter(({ q, i }) => !(q.choices.length > 0 && q.answer) || reopened.includes(i));
  const answeredQuestions = questions
    .map((q, i) => ({ q, i }))
    .filter(({ q, i }) => q.choices.length > 0 && q.answer && !reopened.includes(i));
  const turns = card.conversation ?? [];
  const lastNote = turns.map((t) => t.kind).lastIndexOf("note");
  // The latest exchange shows; the rest waits behind "Earlier".
  // The last note (its questions, its Undo) always shows.
  const shownFrom = showEarlier ? 0 : Math.max(0, Math.min(turns.length - TALK_SHOWN, lastNote < 0 ? turns.length : lastNote));
  const nextTimeTick = (className: string) => card.can_allow ? (
    <NextTimeTick
      className={className}
      checked={nextTime}
      disabled={busy}
      onChange={setNextTime}
      label={followUp ? "Send to them on its own next time" : `Send to ${firstName} on its own next time`}
      help="Everything sent as you to them goes on its own from then on, where Approvals lets it. Undo it on Delegate."
    />
  ) : null;
  const sendLabel = sendButtonLabel(card, "Send");
  const executiveName = card.executive_name || "Executive";
  const askToSend = () => {
    setOpen(true);
    setStep({ kind: "ask", recipients: card.draft_to, allow: Boolean(card.can_allow) && nextTime });
  };
  // Asking before Send shows the "next time" tick too (a closed card's Send
  // comes straight here), so what it says is read as it is now.
  const allowNow = step.kind === "ask" ? Boolean(card.can_allow) && nextTime : confirming && step.allow;
  const dismissButton = (
    // Skip is its own button, set apart on the right, so it's never hunted
    // for in a menu next to Send.
    <Button variant="ghost" className="ml-auto" onClick={() => void dismiss()} disabled={busy}>
      {running("Removing…") ? "Removing…" : "Skip"}
    </Button>
  );
  // Send from me, or from the Executive with you on cc: who it goes from
  // switches first (the Executive rewrites it), then the send's confirm.
  const sendFrom = async (executive: boolean) => {
    if (executive !== Boolean(card.from_executive)) {
      setStep({ kind: "busy", label: FROM_RUNNING });
      setError(null);
      try {
        const updated = await setReplyFrom(card.decision_id, executive);
        onChanged?.(updated);
        setOpen(true);
        setStep({ kind: "ask", recipients: updated.draft_to, allow: Boolean(updated.can_allow) && nextTime });
      } catch (err) {
        setError(err instanceof Error ? err.message : "Couldn't change who it's from.");
        setStep({ kind: "idle" });
      }
      return;
    }
    askToSend();
  };
  const sendBase = "Send";
  // The options on the closed card. The first is the Executive's pick: as
  // the reply is set to go now.
  const firstOpen = openQuestions[0];
  const fromMe = { label: `${sendBase} from me`, run: () => void sendFrom(false) };
  const fromExecutive = { label: `${sendBase} from ${executiveName}, cc me`, run: () => void sendFrom(true) };
  const options: { label: string; run: () => void }[] = unconfirmed || card.working
    ? []
    : left > 0
      ? firstOpen && firstOpen.q.choices.length > 0
        ? firstOpen.q.choices.map((c) => ({ label: c, run: () => void answer(firstOpen.i, c) }))
        : [{ label: "Answer it", run: () => setOpen(true) }]
      : followUp
        ? [fromMe]
        : card.from_executive
          ? [fromExecutive, fromMe]
          : [fromMe, fromExecutive];
  const willDo = left > 0
    ? firstOpen
      ? `${followUp ? "Send your follow-up" : `Send ${toNames} your reply`} once you answer: ${firstOpen.q.text}`
      : `${followUp ? "Send your follow-up" : `Send ${toNames} your reply`} once you decide.`
    : card.working
      ? "Finish what you asked, then show you the reply."
      : followUp
        ? `Send ${toNames} your follow-up.`
        : `Send ${toNames} your reply.`;

  const renderQuestion = ({ q, i }: { q: (typeof questions)[number]; i: number }) => (
    <li key={i} className="text-sm text-fg">
      <div className="break-words">
        {q.answer && <span className="mr-1 text-accent" aria-label="Answered">✓</span>}
        {q.text}
      </div>
      {q.source && <div className="mt-0.5 text-xs text-fg-subtle break-words">{q.source}</div>}
      {q.answer && (
        <div className="mt-0.5 text-xs text-fg-muted break-words">Your answer: {q.answer}</div>
      )}
      {q.choices.length > 0 && editing === null && !unconfirmed && (
        <div className="mt-1.5 flex flex-wrap gap-2">
          {q.choices.map((c) => (
            <button
              key={c}
              type="button"
              disabled={busy}
              aria-pressed={q.answer === c}
              onClick={() => void answer(i, c)}
              className={`min-h-10 max-w-full rounded-full border px-3.5 py-1.5 text-left text-sm break-words transition-colors cursor-pointer disabled:opacity-50 ${
                q.answer === c
                  ? "border-accent bg-accent/10 text-fg"
                  : "border-line bg-surface text-fg-muted hover:border-line-strong hover:text-fg"
              }`}
            >
              {c}
            </button>
          ))}
          <button
            type="button"
            disabled={busy}
            aria-expanded={typing?.index === i}
            onClick={() => setTyping(typing?.index === i ? null : { index: i, text: "" })}
            className="min-h-10 rounded-full border border-dashed border-line px-3.5 py-1.5 text-sm text-fg-muted hover:border-line-strong hover:text-fg transition-colors cursor-pointer disabled:opacity-50"
          >
            Something else…
          </button>
        </div>
      )}
      {typing?.index === i && (
        <form
          className="mt-2 flex flex-wrap items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void answer(i, typing.text);
          }}
        >
          <input
            type="text"
            aria-label={`Your answer: ${q.text}`}
            value={typing.text}
            maxLength={500}
            autoFocus
            disabled={busy}
            onChange={(e) => setTyping({ index: i, text: e.target.value })}
            className="min-h-10 min-w-0 flex-1 basis-48 rounded-xl border border-line-strong bg-surface px-3 text-[15px] text-fg focus:outline-none focus:ring-2 focus:ring-accent/40 disabled:opacity-50"
          />
          <Button type="submit" variant="secondary" disabled={busy || !typing.text.trim()}>
            Use this
          </Button>
        </form>
      )}
    </li>
  );

  const hasOpen = tappable ? openQuestions.length > 0 : card.open_questions.length > 0;
  const decideBlock = hasOpen ? (
    tappable ? (
      <ul className="mt-2 space-y-3">{openQuestions.map(renderQuestion)}</ul>
    ) : (
      <ul className="mt-1 list-disc space-y-0.5 pl-5">
        {card.open_questions.map((q, i) => (
          <li key={i} className="break-words">{q}</li>
        ))}
      </ul>
    )
  ) : null;
  const answeredLines = answeredQuestions.length > 0 ? (
    <ul className="mt-2 space-y-1">
      {answeredQuestions.map(({ q, i }) => (
        <li key={i} className="text-sm text-fg-muted break-words">
          <span className="mr-1 text-accent" aria-label="Answered">✓</span>
          {q.text} <span className="font-semibold text-fg">{q.answer}</span>
          {!unconfirmed && editing === null && (
            <button
              type="button"
              disabled={busy}
              onClick={() => setReopened((r) => [...r, i])}
              className="ml-2 cursor-pointer text-accent disabled:opacity-50"
            >
              Change
            </button>
          )}
        </li>
      ))}
    </ul>
  ) : null;
  const wrote = followUp ? "I wrote a follow-up for you." : "I wrote your reply.";
  const intro = left > 1
    ? `${wrote} ${left} things to decide:`
    : left === 1
      ? `${wrote} One thing to decide:`
      : hasOpen
        ? `${wrote} Check these before sending:`
        : `${wrote} Read it over and send it when it looks right.`;
  return (
    <article
      className={`rounded-2xl border bg-surface-elevated p-4 sm:p-5 ${
        emphasized || expanded ? "border-accent/60 ring-1 ring-accent/25 shadow-sm" : "border-line"
      }`}
    >
      <button
        type="button"
        aria-expanded={expanded}
        disabled={busy || (expanded && (confirming || editing !== null || typing !== null))}
        onClick={() => setOpen((v) => !v)}
        className="flex w-full cursor-pointer items-start gap-3 text-left disabled:cursor-default"
      >
        <span className="block min-w-0 flex-1">
          <span className="flex items-baseline gap-2">
            <FeatureName feature="act_as_me" className="shrink-0 text-[12px]" />
            <span className="min-w-0 flex-1 text-base sm:text-[17px] font-semibold leading-snug text-fg break-words">
              {card.subject || "(no subject)"}
            </span>
            {received && !expanded && <span className="shrink-0 text-sm text-fg-subtle tabular-nums">{received}</span>}
          </span>
          <span className="mt-1 block text-sm text-fg-muted break-words">
            {toLine}
            {card.sender_verified ? "" : " · Address not verified"}
          </span>
        </span>
        {expanded && <span className="shrink-0 pt-0.5 text-sm font-medium text-accent">Close</span>}
      </button>

      {!expanded ? (
        <>
          {warnings.length > 0 && !unconfirmed && (
            <div className="mt-2.5 flex flex-wrap gap-2">
              <span className="rounded-full border border-amber-500/50 px-2.5 py-0.5 text-[13px] text-amber-700 dark:text-amber-300">
                ⚠ Check before sending
              </span>
            </div>
          )}
          {unconfirmed ? (
            <p className="mt-3 text-sm text-fg-muted">
              {mailbox} hasn&apos;t confirmed this was sent. Check your Sent folder in {mailbox}; this card
              updates on its own within a few minutes.
            </p>
          ) : (
            <>
              {/* Plain text: the summary is written from a stranger's email too. */}
              {gist && <Section title="What happened">{gist}</Section>}
              <Section title="What I'll do">{willDo}</Section>
              {/* On its own line, well clear of Send, so a tap to look never sends. */}
              <button
                type="button"
                onClick={() => setOpen(true)}
                disabled={busy}
                className="mt-2 block cursor-pointer text-left text-[15px] font-medium text-accent underline underline-offset-2 disabled:opacity-50"
              >
                {seeReply}
              </button>
              {options.length > 0 && (
                <div className="mt-5 flex flex-col gap-2" role="group" aria-label="Options">
                  {options.map((o, i) => (
                    <button
                      key={o.label}
                      type="button"
                      disabled={busy}
                      onClick={o.run}
                      className={`${buttonClass(i === 0 ? "primary" : "secondary", "md")} h-auto w-full justify-start py-2 text-left`}
                    >
                      {o.label}
                    </button>
                  ))}
                </div>
              )}
              {step.kind === "busy" && (
                <p className="mt-2 text-sm text-fg-muted" aria-live="polite">{step.label}</p>
              )}
              <SayBox
                id={`reply-${card.decision_id}`}
                disabled={busy}
                say={async (text) => {
                  await say(text);
                  setOpen(true);
                }}
              />
              <div className="mt-2 flex justify-end">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void dismiss()}
                  className="text-sm text-fg-muted hover:text-fg disabled:opacity-50"
                >
                  {running("Removing…") ? "Removing…" : "Skip"}
                </button>
              </div>
            </>
          )}
        </>
      ) : (
        <>
      {card.waited_because && (
        <p className="mt-1 text-xs text-fg-subtle break-words">Waited for you: {card.waited_because}</p>
      )}
      <div className="mt-3 flex flex-col gap-2.5">
        {card.they_wrote && (
          <div className="max-w-[92%] self-start rounded-2xl rounded-tl-md bg-surface-overlay px-3 py-2">
            <div className="text-xs text-fg-subtle break-words">
              {followUp ? "You wrote" : senderLine(card)}
              {received ? ` · ${received}` : ""}
            </div>
            {/* Plain text: what a stranger wrote is never rendered as markup. */}
            <p className="mt-0.5 whitespace-pre-wrap break-words text-sm text-fg">{card.they_wrote}</p>
          </div>
        )}
        {turns.length === 0 ? (
          <ExecBubble>
            <p>{intro}</p>
            {decideBlock}
            {answeredLines}
          </ExecBubble>
        ) : (
          answeredQuestions.length > 0 && (
            <ExecBubble>
              <p>What you decided:</p>
              {answeredLines}
            </ExecBubble>
          )
        )}
      </div>
      <div
        className={`mt-3 rounded-xl border border-line-strong bg-surface px-3 py-2.5 ${card.working ? "opacity-60" : ""}`}
      >
        <div className="flex flex-wrap items-center gap-2">
          <span className={label}>{card.working ? "Your reply · being rewritten" : "Your reply"}</span>
          {choice && !unconfirmed && (
            <div
              role="radiogroup"
              aria-label="Reply to"
              className="ml-auto inline-flex overflow-hidden rounded-full border border-line-strong"
            >
              {([false, true] as const).map((everyone) => {
                const on = everyone === choice.toEveryone;
                return (
                  <button
                    key={String(everyone)}
                    type="button"
                    role="radio"
                    aria-checked={on}
                    disabled={busy || confirming || editing !== null}
                    onClick={() => void pickRecipients(everyone)}
                    className={`min-h-9 px-3 text-[13px] disabled:opacity-60 ${
                      on ? "bg-accent/20 font-semibold text-fg" : "text-fg-muted hover:bg-surface-overlay/60"
                    }`}
                  >
                    {everyone ? choice.everyone : choice.sender}
                  </button>
                );
              })}
            </div>
          )}
        </div>
        {!followUp && !unconfirmed && (
          <FromSwitch
            fromExecutive={Boolean(card.from_executive)}
            executiveName={executiveName}
            disabled={busy || confirming || editing !== null || Boolean(card.working)}
            onPick={(executive) => void pickFrom(executive)}
          />
        )}
        {running(FROM_RUNNING) && (
          <p className="mt-1 text-xs text-fg-subtle" aria-live="polite">{FROM_RUNNING}</p>
        )}
        <div className="mt-1 text-xs text-fg-subtle break-words" title={card.draft_to.join(", ")}>To {recipientNames(card.draft_to)}</div>
        {card.from_executive && <div className="mt-0.5 text-xs text-fg-subtle">Cc you</div>}
        {running(RECIPIENTS_RUNNING) && (
          <p className="mt-1 text-xs text-fg-subtle" aria-live="polite">{RECIPIENTS_RUNNING}</p>
        )}
        {editing !== null ? (
          <div className="mt-1.5">
            <textarea
              aria-label="Your reply"
              rows={9}
              value={editing}
              disabled={busy}
              autoFocus
              onChange={(e) => setEditing(e.target.value)}
              className="w-full resize-y whitespace-pre-wrap rounded-xl border border-line-strong bg-surface p-3 text-[15px] leading-relaxed text-fg focus:outline-none focus:ring-2 focus:ring-accent/40"
            />
            <p className="mt-1 px-1 text-xs text-fg-subtle">
              {inMailbox
                ? `Saving changes the draft in your ${mailbox} too. Nothing is sent until you tap Send.`
                : "Nothing is sent until you tap Send."}
            </p>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Button variant="primary" onClick={() => void saveEdit()} disabled={busy || !editing.trim()}>
                {running("Saving…") ? "Saving…" : "Save"}
              </Button>
              <Button variant="ghost" onClick={() => setEditing(null)} disabled={busy}>
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          <>
            <EmailWords text={card.draft_body} changed={changed} className="mt-1.5 text-[15px]" />
            {(card.files ?? []).length > 0 && (
              <ul className="mt-2 flex flex-wrap gap-2" aria-label="Attached">
                {(card.files ?? []).map((name, i) => (
                  <li
                    key={`${i}-${name}`}
                    className="max-w-full truncate rounded-lg border border-line-strong bg-surface-overlay px-2.5 py-1 text-[13px] text-fg"
                  >
                    {name}
                  </li>
                ))}
              </ul>
            )}
            {!unconfirmed && !confirming && (
              <button
                type="button"
                disabled={busy}
                onClick={() => {
                  setError(null);
                  setEditing(card.draft_body);
                }}
                className="mt-1 min-h-10 cursor-pointer text-sm font-medium text-accent disabled:opacity-50"
              >
                Edit the words myself
              </button>
            )}
          </>
        )}
      </div>
      {warnings.length > 0 && (
        <ul className="mt-3 space-y-0.5 text-sm text-amber-700 dark:text-amber-300">
          {warnings.map((w) => (
            <li key={w} className="break-words">⚠ {w}</li>
          ))}
        </ul>
      )}

      {unconfirmed ? (
        <p className="mt-3 text-sm text-fg-muted">
          {mailbox} hasn&apos;t confirmed this was sent. Check your Sent folder in {mailbox}; this card
          updates on its own within a few minutes.
        </p>
      ) : step.kind === "ask" || step.kind === "confirm" ? (
        <div className="mt-4 rounded-xl border border-accent/30 bg-accent/5 px-3.5 py-3">
          {step.kind === "confirm" && (
            <p className="text-sm text-amber-700 dark:text-amber-300">{step.message}</p>
          )}
          <p className="text-sm text-fg">{sendQuestion(step.recipients, mailbox, inMailbox, card.from_executive ? executiveName : "")}</p>
          {step.kind === "ask" && nextTimeTick("mt-2")}
          {allowNow && (
            <p className="mt-2 text-sm text-fg">
              From now on, replies and follow-ups to {who} go on their own, where Approvals lets them. Careful still
              holds everything, and your rules always wait.
            </p>
          )}
          {learnsStyle && (
            <label className="mt-2 flex min-h-touch items-center gap-2 text-sm text-fg">
              <input
                type="checkbox"
                checked={keepExample}
                onChange={(e) => setKeepExample(e.target.checked)}
                className="h-4 w-4 accent-accent"
              />
              Do it like this next time (if you changed it)
            </label>
          )}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Button
              variant="primary"
              onClick={() =>
                void send(
                  step.kind === "confirm"
                    ? { recipients: step.recipients, thread_moved_on: step.threadMovedOn || undefined }
                    : { recipients: step.recipients },
                  allowNow,
                )
              }
            >
              {step.kind === "confirm" ? "Send anyway" : "Send now"}
            </Button>
            <Button variant="ghost" onClick={() => setStep({ kind: "idle" })}>
              Cancel
            </Button>
          </div>
        </div>
      ) : editing !== null ? null : (
        <>
        {left > 0 && (
          <p className="mt-4 text-xs text-fg-subtle">
            {left === 1
              ? "Answer the question above to send, or edit the words yourself."
              : `Answer the ${left} questions above to send, or edit the words yourself.`}
          </p>
        )}
        {nextTimeTick(left > 0 ? "mt-2" : "mt-4")}
        <div className={`${left > 0 || card.can_allow ? "mt-2" : "mt-4"} flex flex-wrap items-center gap-2`}>
          <Button
            variant="primary"
            onClick={askToSend}
            disabled={busy || left > 0 || card.working}
          >
            {running("Sending…") ? "Sending…" : sendLabel}
          </Button>
          {gmailLink && (
            <a
              href={gmailLink}
              target="_blank"
              rel="noopener noreferrer"
              className={buttonClass("ghost")}
            >
              Open in {mailbox}
            </a>
          )}
          {dismissButton}
        </div>
        </>
      )}
      {/* The talk: below the reply and Send, newest last, with the box
          under it, so an answer lands next to where you typed. */}
      {(turns.length > 0 || card.working || busy) && (
        <div className="mt-4 flex flex-col gap-2.5 border-t border-line pt-3">
          {shownFrom > 0 && (
            <button
              type="button"
              onClick={() => setShowEarlier(true)}
              className="self-start min-h-8 cursor-pointer text-[13px] text-fg-muted hover:text-fg"
            >
              Earlier ({shownFrom})
            </button>
          )}
          {turns.map((t, i) =>
            i < shownFrom ? null : t.kind === "news" ? (
              <div
                key={i}
                className="max-w-[92%] self-start rounded-2xl rounded-tl-md border border-emerald-500/60 bg-surface-overlay px-3 py-2"
              >
                <div className="text-xs text-fg-subtle break-words">
                  <span className="mr-1.5 font-semibold uppercase tracking-wide text-emerald-600 dark:text-emerald-400">New</span>
                  {(t.from_name || t.from_email || "").trim()}
                  {t.at && formatRelativeTime(t.at) ? ` · ${formatRelativeTime(t.at)}` : ""}
                  <NewsExtras turn={t} subject={card.subject} />
                </div>
                {t.text && <p className="mt-0.5 whitespace-pre-wrap break-words text-sm text-fg">{t.text}</p>}
              </div>
            ) : t.kind === "asked" ? (
              <div key={i} className="max-w-[85%] self-end rounded-2xl rounded-tr-md bg-accent/20 px-3 py-2 text-sm text-fg break-words">
                {t.text}
              </div>
            ) : (
              <ExecBubble
                key={i}
                status={WORK_STATUS[t.state ?? ""] ?? ""}
                dim={t.state === "stopped" && (i < turns.length - 1 || card.working)}
              >
                <p>{t.text}</p>
                {(t.changed ?? []).length > 0 && (
                  <ul className="mt-1.5 space-y-1" aria-label="What changed">
                    {(t.changed ?? []).slice(0, 4).map((line, k) => (
                      <li key={k} className="line-clamp-3 border-l-2 border-accent/50 pl-2 text-fg break-words">
                        {line}
                      </li>
                    ))}
                  </ul>
                )}
                {i === lastNote && card.can_undo && !card.working && !unconfirmed && (
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void undo()}
                    className="mt-1.5 min-h-8 cursor-pointer text-[13px] font-semibold text-accent disabled:opacity-50"
                  >
                    {running(UNDOING) ? UNDOING : "Undo"}
                  </button>
                )}
                <WorkSteps steps={t.steps ?? []} end={t.state ?? ""} />
                {(t.choices ?? []).length > 0 && i === turns.length - 1 && !card.working && (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {(t.choices ?? []).map((choiceText) => (
                      <button
                        key={choiceText}
                        type="button"
                        disabled={busy}
                        onClick={() => void askChange(choiceText, true)}
                        className="min-h-9 cursor-pointer rounded-full border border-accent/40 bg-surface px-3 text-[13px] font-semibold text-accent hover:bg-accent/10 disabled:opacity-50"
                      >
                        {choiceText}
                      </button>
                    ))}
                  </div>
                )}
                {i === lastNote && hasOpen && decideBlock}
              </ExecBubble>
            ),
          )}
          {card.working && (
            <ExecBubble status="working">
              <p>Working on it. You can tell me to do it differently below.</p>
              <WorkSteps steps={card.work_steps ?? []} now={card.work_now || "Starting…"} />
              <div className="mt-1.5 flex justify-end">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void stopWork()}
                  className="min-h-8 cursor-pointer px-1 text-[13px] font-semibold text-accent disabled:opacity-50"
                >
                  {running(STOPPING) ? STOPPING : "Stop"}
                </button>
              </div>
            </ExecBubble>
          )}
          {busy && (step.label === "Writing it in…" || step.label === "Changing it…") && (
            <p className="text-xs text-fg-subtle" aria-live="polite">Writing it into your reply…</p>
          )}
        </div>
      )}
      {!unconfirmed && !confirming && editing === null && (
        <form
          className="mt-3 flex items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void askChange();
          }}
        >
          <input
            type="text"
            aria-label={card.working ? "Tell me to do it differently" : "Tell me what to change"}
            placeholder={card.working ? "Tell me to do it differently…" : "Tell me what to change…"}
            value={ask}
            maxLength={500}
            disabled={busy}
            onChange={(e) => setAsk(e.target.value)}
            className="min-h-11 min-w-0 flex-1 rounded-full border border-line-strong bg-surface px-4 text-[15px] text-fg placeholder:text-fg-subtle focus:outline-none focus:ring-2 focus:ring-accent/40"
          />
          {ask.trim() && (
            <Button type="submit" variant="secondary" disabled={busy}>
              Go
            </Button>
          )}
        </form>
      )}
        </>
      )}
      {!unconfirmed && editing === null && (
        <GoOnItsOwnOffer addresses={card.draft_to?.length ? card.draft_to : [card.from_email]} />
      )}
      {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
    </article>
  );
}
