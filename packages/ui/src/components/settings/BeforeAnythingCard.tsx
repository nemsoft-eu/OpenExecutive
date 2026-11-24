"use client";

import Link from "next/link";
import { useState, type ReactNode } from "react";

import AiLineCard from "@/components/settings/AiLineCard";
import ApprovalsWeek from "@/components/settings/ApprovalsWeek";
import GoOnItsOwnCard from "@/components/settings/GoOnItsOwnCard";
import LeadRulesEditor from "@/components/settings/LeadRulesEditor";
import OutsideContentCard from "@/components/settings/OutsideContentCard";
import SettingsCard from "@/components/settings/SettingsCard";
import TrainingStep from "@/components/settings/TrainingStep";
import { useApprovalsSummary } from "@/components/settings/useApprovalsSummary";
import {
  addCompanyLeadRule,
  deleteCompanyLeadRule,
  type DelegationSettings,
  type OutsideContentSetting,
  type TakeTheLead,
} from "@/lib/api";
import { rulesChangeLine, rulesSummary } from "@/lib/leadRules";
import { LEVEL_ORDER } from "@/lib/approvalsLevel";
import { APPROVALS_LEADS, APPROVALS_SENDS } from "@/lib/reach";

// Delegate → Approvals: every check in one place, numbered in the order
// they run. 1 Level (the owner's; everyone else sees it): what goes on its
// own and what comes as an Approve card on Home, with what it did in the last
// 7 days (ApprovalsWeek). 2 Your rules (the owner's, POST/DELETE
// /take-the-lead/rules): a Waits rule always waits, a Goes rule lets
// something go on its own; with no rules the level alone decides. 3 In
// training, the last check, both switches side by side (TrainingStep). Then
// what you approved three times unchanged that Could go on its own, and Say
// it was written with AI, with an As me row when the install has one
// (`aiAsMe`). The sidebar's chip, Settings → Your Executive and the As you and
// As me tabs show where these stand and link here.

export default function BeforeAnythingCard({
  approvals,
  lead,
  leadState,
  setLead,
  delegation,
  onDelegation,
  aiAsMe,
}: {
  approvals: OutsideContentSetting | null;
  lead: TakeTheLead | null;
  leadState: "loading" | "hidden" | "ready" | "error";
  setLead: (next: TakeTheLead) => void;
  delegation: DelegationSettings | null;
  onDelegation: (next: DelegationSettings) => void;
  aiAsMe?: ReactNode;
}) {
  const [adding, setAdding] = useState(false);
  const summary = useApprovalsSummary();
  const week = summary && <ApprovalsWeek week={summary.week} />;
  const rules = lead?.rules ?? [];
  const changed = rulesChangeLine(rules);
  const readOnly = approvals && !approvals.can_change && (
    <SettingsCard
      step={1}
      title="Level"
      description={`On ${APPROVALS_SENDS[approvals.mode].label}. The account owner sets it, the same for everyone and on every channel.`}
    >
      <p className="text-sm text-fg-muted leading-relaxed">
        As you: {APPROVALS_SENDS[approvals.mode].line} As me: {APPROVALS_LEADS[approvals.mode]}
      </p>
      {week}
    </SettingsCard>
  );

  return (
    <>
      <OutsideContentCard
        after={week}
        note={
          changed && (
            <p className="mt-3 rounded-lg bg-surface-overlay px-3 py-2 text-sm text-fg-muted">
              <span className="font-semibold text-fg">Your rules change this:</span> {changed}{" "}
              <a href="#rules-label" className="font-medium text-accent hover:underline">
                See rules
              </a>
            </p>
          )
        }
      >
        {approvals?.can_change && (
          <ul className="grid gap-3 sm:grid-cols-3">
            {LEVEL_ORDER.map((mode) => (
              <li
                key={mode}
                className={`rounded-xl border p-4 ${mode === approvals.mode ? "border-accent/60 bg-accent/10" : "border-line"}`}
              >
                <span className="block text-[15px] font-semibold text-fg">{APPROVALS_SENDS[mode].label}</span>
                <span className="mt-2 block text-[13px] leading-relaxed text-fg-muted">
                  <span className="font-semibold text-fg">As you:</span> {APPROVALS_SENDS[mode].line}
                </span>
                <span className="mt-2 block text-[13px] leading-relaxed text-fg-muted">
                  <span className="font-semibold text-fg">As me:</span> {APPROVALS_LEADS[mode]}
                </span>
              </li>
            ))}
          </ul>
        )}
      </OutsideContentCard>
      {readOnly}
      {lead && (
        <SettingsCard
          step={2}
          title="Your rules"
          titleId="rules-label"
          description={`${rulesSummary(rules)} “Always waits” beats “Goes on its own”.`}
          action={
            !adding && (
              <button
                type="button"
                aria-expanded={adding}
                aria-controls="rules-editor"
                onClick={() => setAdding(true)}
                className="text-sm font-medium text-accent hover:underline cursor-pointer"
              >
                Add a rule
              </button>
            )
          }
        >
          {(rules.length > 0 || adding) && (
            <LeadRulesEditor
              rules={rules}
              adding={adding}
              onDone={() => setAdding(false)}
              onAdd={async (kind, value, effect) => setLead(await addCompanyLeadRule(kind, value, effect))}
              onDelete={async (id) => setLead(await deleteCompanyLeadRule(id))}
            />
          )}
        </SettingsCard>
      )}
      <TrainingStep
        delegation={delegation}
        onDelegation={onDelegation}
        lead={lead}
        leadState={leadState}
        setLead={setLead}
      />
      {summary && <GoOnItsOwnCard offers={summary.offers} />}
      <AiLineCard disclosure={delegation?.disclosure ?? null} onSettings={onDelegation} asMe={aiAsMe} />
      <p className="text-[15px] text-fg-muted">
        Pause, what it always does without asking, and the voice it answers in are under{" "}
        <Link href="/settings/executive" className="font-medium text-accent underline-offset-2 hover:underline">
          Settings → Your Executive
        </Link>
        .
      </p>
    </>
  );
}
