"use client";

// People → Needs a look (lib/peopleBoard.ts): the top of People, only when
// something is there. People the Executive added (Undo) or suggests (Add /
// Not now), the details it filled in (one line that opens to each, with
// Undo), and two people with the same address. Got it clears what the
// Executive changed. Also the row every list on People is made of.

import Link from "next/link";
import { type ReactNode, useState } from "react";

import Button from "@/components/ui/Button";
import {
  decideContactChange,
  markContactChangesSeen,
  type ContactChange,
  type ContactChanges,
  type Person,
} from "@/lib/api";
import { changeLine } from "@/lib/contactChanges";
import { detailsLine, hasSeenable, splitChanges, type RowLine } from "@/lib/peopleBoard";

/** One person: initial, name with its mark (Owner, Admin, Added), and a second line only when it says something. */
export function PersonRow({
  name,
  href,
  badge,
  line,
  contact = false,
  right,
}: {
  name: string;
  href?: string;
  badge?: string | null;
  line?: RowLine | null;
  contact?: boolean;
  right?: ReactNode;
}) {
  const body = (
    <>
      <span
        aria-hidden="true"
        className={`w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 text-sm font-semibold ${
          contact ? "bg-surface-hover text-fg-muted" : "bg-accent/10 text-accent"
        }`}
      >
        {name.trim().charAt(0).toUpperCase() || "?"}
      </span>
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-2 min-w-0">
          <span className="text-[15px] font-semibold text-fg truncate group-hover:text-accent transition-colors">{name}</span>
          {badge && (
            <span
              className={`inline-block flex-shrink-0 px-2 py-0.5 rounded-full text-xs font-medium ${
                badge === "Owner" ? "bg-accent/10 text-accent" : "border border-line text-fg-muted"
              }`}
            >
              {badge}
            </span>
          )}
        </span>
        {line && (
          <span className={`block mt-0.5 text-sm truncate ${line.warn ? "text-amber-600 dark:text-amber-400" : "text-fg-muted"}`}>
            {line.text}
          </span>
        )}
      </span>
    </>
  );
  const cls = "flex items-center gap-3 sm:gap-4 px-4 sm:px-5 py-3 min-w-0 flex-1";
  return (
    <li className="flex items-center border-t border-line first:border-t-0">
      {href ? (
        <Link
          href={href}
          className={`${cls} hover:bg-surface-hover transition-colors group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent/60`}
        >
          {body}
          {!right && <span aria-hidden="true" className="text-fg-subtle text-lg">›</span>}
        </Link>
      ) : (
        <div className={cls}>{body}</div>
      )}
      {right && <div className="flex flex-shrink-0 items-center gap-2 pr-3 sm:pr-4">{right}</div>}
    </li>
  );
}

/** A section heading on People: "Team · 4", with something small on the right. */
export function SectionHead({ title, note, right }: { title: string; note?: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 mb-2 px-1">
      <h2 className="text-[13px] font-bold uppercase tracking-wider text-fg-muted">
        {title}
        {note && <span className="ml-2 normal-case tracking-normal font-medium text-fg-subtle">{note}</span>}
      </h2>
      {right}
    </div>
  );
}

export function NeedsALook({
  changes,
  pairs,
  onChanged,
}: {
  changes: ContactChanges | null;
  pairs: { person: Person; other: Person }[];
  /** The new state when the server sent one back; otherwise read it again. */
  onChanged: (next?: ContactChanges) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const split = changes ? splitChanges(changes.changes) : { people: [], details: [] };
  const count = split.people.length + (split.details.length > 0 ? 1 : 0) + pairs.length;
  if (count === 0) return null;

  const run = async (work: () => Promise<ContactChanges | ContactChange>, keep: boolean) => {
    setBusy(true);
    setError(null);
    try {
      const next = await work();
      onChanged(keep ? (next as ContactChanges) : undefined);
    } catch {
      setError("Couldn't save that. Try again.");
      onChanged();
    } finally {
      setBusy(false);
    }
  };
  const decide = (change: ContactChange, verb: "undo" | "accept" | "dismiss") =>
    void run(() => decideContactChange(change.id, verb), false);

  return (
    <section className="mb-6" aria-label="Needs a look">
      <SectionHead
        title={`Needs a look · ${count}`}
        right={
          changes && hasSeenable(changes.changes) ? (
            <Button size="sm" variant="secondary" disabled={busy} onClick={() => void run(markContactChangesSeen, true)}>
              Got it
            </Button>
          ) : undefined
        }
      />
      {error && <p className="text-sm text-rose-500 mb-2 px-1">{error}</p>}
      <ul className="rounded-2xl border border-amber-500/40 bg-surface-elevated overflow-hidden">
        {split.people.map((change) => {
          const line = changeLine(change);
          const canOpen = change.person_id !== null && change.status === "applied";
          return (
            <PersonRow
              key={`c${change.id}`}
              contact
              name={line.title}
              href={canOpen ? `/people/${change.person_id}` : undefined}
              badge={line.tag === "Suggested" ? "Suggested" : "Added"}
              line={{ text: line.detail, warn: false }}
              right={
                change.status === "suggested" ? (
                  <>
                    <Button size="sm" variant="ghost" disabled={busy} onClick={() => decide(change, "dismiss")}>
                      Not now
                    </Button>
                    <Button size="sm" variant="primary" disabled={busy} onClick={() => decide(change, "accept")}>
                      Add
                    </Button>
                  </>
                ) : (
                  <Button size="sm" variant="secondary" disabled={busy} onClick={() => decide(change, "undo")}>
                    Undo
                  </Button>
                )
              }
            />
          );
        })}
        {split.details.length > 0 && (
          <li className="border-t border-line first:border-t-0">
            <button
              type="button"
              aria-expanded={showDetails}
              onClick={() => setShowDetails((v) => !v)}
              className="flex w-full items-center justify-between gap-3 px-4 sm:px-5 py-3.5 text-left hover:bg-surface-hover"
            >
              <span className="text-[15px] font-semibold text-fg">{detailsLine(split.details.length)}</span>
              <span className="text-sm font-semibold text-accent">{showDetails ? "Hide" : "Show"}</span>
            </button>
            {showDetails && (
              <ul>
                {split.details.map((change) => {
                  const line = changeLine(change);
                  return (
                    <li key={`d${change.id}`} className="flex items-start gap-3 px-4 sm:px-5 py-3 border-t border-line">
                      <div className="min-w-0 flex-1">
                        {change.person_id !== null ? (
                          <Link href={`/people/${change.person_id}`} className="text-[15px] font-semibold text-fg hover:underline">
                            {line.title}
                          </Link>
                        ) : (
                          <span className="text-[15px] font-semibold text-fg">{line.title}</span>
                        )}
                        <p className="text-sm text-fg-muted mt-0.5 break-words">{line.detail}</p>
                      </div>
                      <Button size="sm" variant="secondary" disabled={busy} onClick={() => decide(change, "undo")}>
                        Undo
                      </Button>
                    </li>
                  );
                })}
              </ul>
            )}
          </li>
        )}
        {pairs.map(({ person, other }) => (
          <PersonRow
            key={`s${person.id}-${other.id}`}
            contact
            name={person.full_name}
            href={`/people/${person.id}`}
            line={{ text: `Same email as ${other.full_name}`, warn: false }}
            right={
              <Link href={`/people/${person.id}`} className="text-sm font-semibold text-accent hover:underline">
                Look
              </Link>
            }
          />
        ))}
      </ul>
    </section>
  );
}
