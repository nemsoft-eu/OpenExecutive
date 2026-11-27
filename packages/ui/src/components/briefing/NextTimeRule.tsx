"use client";

// A held card's "Next time" box: the Goes rules the server checked would let
// this exact step go (take_the_lead.rule_offers), plus Keep asking me, which
// is the default. Picking one adds that rule to Your rules once Do it has
// gone through; the card then says so, with Undo. A Goes rule is the
// account owner's, so only the owner sees the box.

import { useEffect, useRef, useState } from "react";

import { addCompanyLeadRule, deleteCompanyLeadRule, getPeopleViewer, type LeadRuleKind } from "@/lib/api";
import { describeRule } from "@/lib/leadRules";

export interface RuleOffer {
  kind: LeadRuleKind;
  value: string;
}

const OFFER_KINDS: LeadRuleKind[] = ["topic", "person", "domain", "amount"];

/** The offers a card's payload carries, or none. */
export function parseOffers(raw: unknown): RuleOffer[] {
  if (!Array.isArray(raw)) return [];
  return raw.filter(
    (o): o is RuleOffer =>
      Boolean(o) &&
      typeof o === "object" &&
      OFFER_KINDS.includes((o as RuleOffer).kind) &&
      typeof (o as RuleOffer).value === "string" &&
      (o as RuleOffer).value !== "",
  );
}

/** "Anything about money goes on its own" — the line for an offer. */
export function offerLine(offer: RuleOffer): string {
  return describeRule({ ...offer, effect: "go" });
}

/**
 * The box, its pick, and what Do it runs: `withRule(approve)` approves, then
 * adds the picked rule. While a rule just added (or why it wasn't) shows,
 * `resolved` waits for Close (or Undo) instead of dropping the card at once.
 * Once the step is changed (being edited, or changed by what was said on
 * the card) the box is hidden and no rule is added: the offers were worked
 * out for the step as it was.
 */
export function useNextTimeRule(offers: RuleOffer[], onResolved: () => void, changed = false) {
  const [owner, setOwner] = useState(false);
  const offered = offers.length > 0;
  useEffect(() => {
    if (!offered) return;
    let live = true;
    getPeopleViewer()
      .then((viewer) => {
        if (live) setOwner(viewer.is_principal);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [offered]);
  const shown = offered && owner && !changed;
  const [pick, setPick] = useState<number>(-1);
  const [added, setAdded] = useState<{ id: number | null; line: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const addedRef = useRef(false);
  const [closed, setClosed] = useState(false);

  // Close (or a done Undo): the note goes, whether or not the card does.
  function dismiss() {
    setClosed(true);
    onResolved();
  }

  // `went`: whether the step was carried out (an Approve through the chat
  // can come back still waiting); no rule is added when it wasn't.
  async function withRule(
    approve: () => Promise<unknown>,
    went: () => Promise<boolean> | boolean = () => true,
  ): Promise<unknown> {
    const out = await approve();
    const offer = shown && pick >= 0 ? offers[pick] : null;
    if (offer && (await went())) {
      addedRef.current = true;
      try {
        const lead = await addCompanyLeadRule(offer.kind, offer.value, "go");
        const rule = lead.rules.find((r) => r.kind === offer.kind && r.value === offer.value && r.effect === "go");
        setAdded({ id: rule?.id ?? null, line: offerLine(offer) });
      } catch (err) {
        setError(err instanceof Error ? `It was done, but the rule wasn't added: ${err.message}` : "It was done, but the rule wasn't added.");
      }
    }
    return out;
  }

  // The card stays while the new rule shows.
  const resolved = () => {
    if (!addedRef.current) onResolved();
  };

  async function undo() {
    if (added?.id == null) return;
    try {
      await deleteCompanyLeadRule(added.id);
      dismiss();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't remove the rule.");
    }
  }

  const box = shown && !added && !error && !closed ? (
    <fieldset className="mt-3 rounded-xl border border-line bg-surface p-3">
      <legend className="px-1 text-xs font-semibold uppercase tracking-wide text-fg-subtle">Next time</legend>
      <div className="flex flex-col gap-1.5">
        {[{ label: "Keep asking me", i: -1 }, ...offers.map((o, i) => ({ label: `${offerLine(o)} goes on its own`, i }))].map(
          ({ label, i }) => (
            <label key={i} className="flex min-h-touch cursor-pointer items-center gap-3 text-[15px] text-fg">
              <input
                type="radio"
                name="next-time"
                checked={pick === i}
                onChange={() => setPick(i)}
                className="h-4 w-4 flex-shrink-0 accent-[rgb(var(--accent-strong))]"
              />
              <span className="min-w-0 break-words">{label}</span>
            </label>
          ),
        )}
      </div>
      {pick >= 0 && (
        <p className="mt-1 text-sm text-fg-muted">Added to Your rules when you tap Do it. You can remove it there.</p>
      )}
    </fieldset>
  ) : null;

  const done = closed ? null : added ? (
    <div className="mt-3 rounded-xl bg-accent/10 px-4 py-3">
      <p className="text-[15px] text-fg break-words">
        Done. New rule: {added.line} goes on its own.
      </p>
      <div className="mt-2 flex gap-3">
        {added.id != null && (
          <button type="button" onClick={() => void undo()} className="min-h-touch text-sm text-accent hover:underline">
            Undo rule
          </button>
        )}
        <button type="button" onClick={dismiss} className="min-h-touch text-sm text-fg-muted hover:underline">
          Close
        </button>
      </div>
      {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
    </div>
  ) : error ? (
    <div className="mt-3 rounded-xl bg-surface px-4 py-3">
      <p className="text-[15px] text-fg break-words">Done.</p>
      <p className="mt-1 text-sm text-red-500">{error}</p>
      <button type="button" onClick={dismiss} className="mt-2 min-h-touch text-sm text-fg-muted hover:underline">
        Close
      </button>
    </div>
  ) : null;

  return { box, done, withRule, resolved };
}
