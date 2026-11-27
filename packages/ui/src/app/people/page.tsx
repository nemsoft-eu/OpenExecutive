"use client";

import { type ReactNode, useCallback, useEffect, useRef, useState } from "react";

import { useAskOEFormContext } from "@/components/askoe/AskOEContext";
import { ChatAppPickers } from "@/components/people/ChatAppPickers";
import { ContactKeeperSwitch } from "@/components/people/ContactChanges";
import { NeedsALook, PersonRow, SectionHead } from "@/components/people/PeopleNeeds";
import { TeamModeSwitch } from "@/components/people/TeamModeSwitch";
import Button from "@/components/ui/Button";
import { useChatDirectory, useSlackEmailMatch } from "@/components/ui/ChatPicker";
import SidePanel from "@/components/ui/SidePanel";
import { useWorkspace } from "@/components/workspace/WorkspaceContext";
import {
  createPerson,
  getContactChanges,
  getLastContacts,
  getPeopleViewer,
  listPeople,
  type ContactChanges,
  type PageFormField,
  type Person,
  type PersonKind,
} from "@/lib/api";
import {
  canClaimPrincipal,
  effectiveKind,
  hiddenTeamCount,
  isContact,
  peopleForTab,
  shouldSwitchToTeam,
} from "@/lib/peopleKinds";
import {
  addFormKind,
  CONTACTS_PREVIEW,
  contactLine,
  contactsByRecent,
  roleBadge,
  sameEmailPairs,
  teamInOrder,
  teamLine,
} from "@/lib/peopleBoard";
import { lastContactLabel, matchesSearch, type LastContact } from "@/lib/peopleList";
import { PageSpinner } from "@/components/ui/Spinner";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ALL_SCOPES = [
  { value: "spend_lt_2k", label: "Spend <$2K", hint: "Receives proposals for any spend under $2K." },
  { value: "spend_lt_10k", label: "Spend <$10K", hint: "Receives proposals for spend under $10K." },
  { value: "spend_gt_10k", label: "Spend >$10K", hint: "Receives proposals for spend over $10K." },
  { value: "hiring_signoff", label: "Hiring", hint: "Receives proposals related to hiring decisions." },
  { value: "vendor_onboarding", label: "Vendors", hint: "Receives proposals for vendor contracts." },
  { value: "customer_credit", label: "Credit", hint: "Receives proposals involving credit or debt." },
  { value: "legal_sign", label: "Legal", hint: "Receives proposals with legal implications." },
  { value: "board_comms", label: "Board", hint: "Receives proposals before board communications." },
  { value: "meeting_scheduling", label: "Meetings", hint: "Receives meetings the Executive wants to book." },
  { value: "wildcard", label: "All (wildcard)", hint: "Receives anything no one else is scoped for, usually the owner." },
];

const CHANNELS = ["any", "slack", "discord", "telegram", "email"];
const KINDS: PersonKind[] = ["team", "contact"];

// ---------------------------------------------------------------------------
// Add Person modal
// ---------------------------------------------------------------------------

interface AddPersonModalProps {
  initialKind: PersonKind;
  /** Contacts are the principal's alone: nobody else is offered the choice. */
  canAddContacts: boolean;
  /** Only while nobody is the principal yet (canClaimPrincipal). */
  canBePrincipal: boolean;
  onCreated: (p: Person) => void;
  onClose: () => void;
}

const BLANK_FORM = {
  full_name: "",
  role: "",
  kind: "team" as PersonKind,
  is_principal: false,
  email: "",
  slack_user_id: "",
  telegram_chat_id: "",
  discord_user_id: "",
  preferred_channel: "any",
  response_sla_hours: "24",
  authority_scope: [] as string[],
};

const INPUT_CLS =
  "w-full h-11 px-3 rounded-xl bg-surface-input/60 border border-line text-[15px] text-fg placeholder-fg-subtle focus:outline-none focus:border-accent";
const LABEL_CLS = "text-sm text-fg-muted flex flex-col gap-1.5";
const HINT_CLS = "text-[13px] text-fg-subtle mt-1";

function DisclosureSection({
  label,
  open,
  onToggle,
  children,
}: {
  label: string;
  open: boolean;
  onToggle: () => void;
  children: ReactNode;
}) {
  return (
    <div className="border-t border-line pt-2">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="flex items-center justify-between w-full min-h-11 py-2 text-[15px] font-medium text-fg hover:text-accent transition-colors"
      >
        <span>{label}</span>
        <span aria-hidden="true" className="text-fg-subtle text-xs">{open ? "▲" : "▼"}</span>
      </button>
      {open && <div className="pt-2 pb-1 space-y-4">{children}</div>}
    </div>
  );
}

// The approval scopes as toggles, each with what it routes to the person.
function ScopePicker({ selected, onToggle }: { selected: string[]; onToggle: (value: string) => void }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
      {ALL_SCOPES.map(({ value, label, hint }) => {
        const active = selected.includes(value);
        return (
          <button
            key={value}
            type="button"
            aria-pressed={active}
            onClick={() => onToggle(value)}
            className={`px-3.5 py-2.5 rounded-xl border text-left transition-colors ${
              active
                ? "bg-accent/10 border-accent/60 text-fg"
                : "bg-surface-elevated border-line text-fg-muted hover:border-line-strong"
            }`}
          >
            <div className="text-[15px] font-medium">{label}</div>
            <div className="text-[13px] leading-snug mt-0.5 text-fg-muted">{hint}</div>
          </button>
        );
      })}
    </div>
  );
}

const SCOPE_VALUES = ALL_SCOPES.map((s) => s.value);

function AddPersonModal({ initialKind, canAddContacts, canBePrincipal, onCreated, onClose }: AddPersonModalProps) {
  const [form, setForm] = useState({ ...BLANK_FORM, kind: canAddContacts ? initialKind : "team" });
  const kind = effectiveKind(form.kind, form.is_principal);
  const contact = kind === "contact";
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [showContact, setShowContact] = useState(false);
  const [showAuthority, setShowAuthority] = useState(false);
  const nameRef = useRef<HTMLInputElement>(null);

  // Slack, Discord and Telegram by name. Slack suggests whoever has the
  // email typed above, and opens the fold to show it until it is toggled.
  const chatDirectory = useChatDirectory("people");
  const slackMatch = useSlackEmailMatch(form.email, !!chatDirectory?.slack.configured);
  const [contactToggled, setContactToggled] = useState(false);
  const contactOpen = showContact || (!contactToggled && !!slackMatch);
  const setChatId = useCallback(
    (field: "slack_user_id" | "discord_user_id" | "telegram_chat_id", id: string) =>
      setForm((f) => (f[field] === id ? f : { ...f, [field]: id })),
    [],
  );

  useEffect(() => {
    // After the panel has taken focus on open, so it remembers the button
    // that opened it (and returns focus there on close).
    const id = requestAnimationFrame(() => nameRef.current?.focus());
    return () => cancelAnimationFrame(id);
  }, []);

  // Registered with Ask OE for the modal's lifetime — closing the modal
  // unregisters automatically (the hook's cleanup runs on unmount).
  const { suggestedCls, clearSuggested } = useAskOEFormContext({
    formId: "add_person",
    title: "Add person",
    description:
      "Adds a human the Executive coordinates with — a team member (can sign in, approve and be chased) or a contact (someone outside the team the Executive emails only when you ask). Authority scopes determine which proposals route to a team member for approval.",
    getFields: (): PageFormField[] => [
      { name: "full_name", label: "Full name", type: "text", value: form.full_name, required: true },
      ...(canAddContacts
        ? [{
            name: "kind",
            label: "Team member or contact",
            type: "select" as const,
            options: KINDS,
            value: form.kind,
            description: "team: works with you. contact: a client, contractor or advisor outside the team, private to you.",
          }]
        : []),
      { name: "role", label: contact ? "Role and company" : "Role", type: "text", value: form.role },
      ...(canBePrincipal
        ? [{
            name: "is_principal",
            label: "This is me — Primary",
            type: "boolean" as const,
            value: form.is_principal,
            description: "Marks the person as the primary decision-maker. Only for the user themselves.",
          }]
        : []),
      { name: "preferred_channel", label: "Preferred channel", type: "select", options: CHANNELS, value: form.preferred_channel },
      { name: "response_sla_hours", label: "Expected reply within (hours)", type: "number", value: Number(form.response_sla_hours) || 24 },
      { name: "email", label: "Email", type: "text", value: form.email },
      { name: "slack_user_id", label: "Slack user ID", type: "text", value: form.slack_user_id },
      { name: "discord_user_id", label: "Discord user ID", type: "text", value: form.discord_user_id },
      { name: "telegram_chat_id", label: "Telegram chat ID", type: "text", value: form.telegram_chat_id },
      {
        name: "authority_scope",
        label: "Approval authority",
        type: "json",
        value: form.authority_scope,
        description: `JSON array of scope tokens, each one of: ${SCOPE_VALUES.join(", ")}.`,
      },
    ],
    applyPatch: (values) => {
      const prior = { form, showContact, showAuthority };
      const applied: string[] = [];
      const skipped: string[] = [];
      const next = { ...form };
      for (const [key, raw] of Object.entries(values)) {
        switch (key) {
          case "full_name":
          case "role":
          case "email":
          case "slack_user_id":
          case "discord_user_id":
          case "telegram_chat_id":
            if (typeof raw !== "string") skipped.push(key);
            else { next[key] = raw; applied.push(key); }
            break;
          case "is_principal":
            if (!canBePrincipal || typeof raw !== "boolean") skipped.push(key);
            else { next.is_principal = raw; applied.push(key); }
            break;
          case "kind":
            if (canAddContacts && (raw === "team" || raw === "contact")) { next.kind = raw; applied.push(key); }
            else skipped.push(key);
            break;
          case "preferred_channel":
            if (typeof raw === "string" && CHANNELS.includes(raw)) {
              next.preferred_channel = raw;
              applied.push(key);
            } else skipped.push(key);
            break;
          case "response_sla_hours": {
            const n = Number(raw);
            if (Number.isFinite(n) && n >= 1) {
              next.response_sla_hours = String(Math.round(n));
              applied.push(key);
            } else skipped.push(key);
            break;
          }
          case "authority_scope": {
            const arr = Array.isArray(raw)
              ? raw.filter((s): s is string => typeof s === "string" && SCOPE_VALUES.includes(s))
              : null;
            // Empty after filtering means no proposed scope was recognized —
            // skip rather than silently wiping every existing scope.
            if (arr !== null && arr.length > 0) { next.authority_scope = arr; applied.push(key); }
            else skipped.push(key);
            break;
          }
          default:
            skipped.push(key);
        }
      }
      setForm(next);
      // Open the disclosures so the suggested values are visible to review.
      if (applied.some((k) => ["slack_user_id", "discord_user_id", "telegram_chat_id", "preferred_channel", "response_sla_hours"].includes(k))) {
        setShowContact(true);
      }
      if (applied.includes("authority_scope")) setShowAuthority(true);
      return {
        applied,
        skipped,
        undo: () => {
          setForm(prior.form);
          setShowContact(prior.showContact);
          setShowAuthority(prior.showAuthority);
        },
      };
    },
  });

  function toggleScope(val: string) {
    setForm((f) => ({
      ...f,
      authority_scope: f.authority_scope.includes(val)
        ? f.authority_scope.filter((s) => s !== val)
        : [...f.authority_scope, val],
    }));
  }

  async function submit() {
    setSaving(true);
    setErr(null);
    try {
      const person = await createPerson({
        full_name: form.full_name.trim(),
        role: form.role.trim(),
        kind,
        is_principal: form.is_principal,
        email: form.email.trim() || null,
        slack_user_id: form.slack_user_id.trim() || null,
        telegram_chat_id: form.telegram_chat_id.trim() || null,
        discord_user_id: form.discord_user_id.trim() || null,
        preferred_channel: form.preferred_channel,
        response_sla_hours: Number(form.response_sla_hours) || 24,
        // A contact approves nothing; don't send scopes picked before switching.
        authority_scope: contact ? [] : form.authority_scope,
      });
      onCreated(person);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Create failed");
    } finally {
      setSaving(false);
    }
  }

  const title = contact ? "Add contact" : "Add person";
  return (
    <SidePanel
      open
      onClose={saving ? () => {} : onClose}
      title={title}
      footer={
        <div className="flex gap-2">
          <Button
            variant="primary"
            disabled={saving || !form.full_name.trim()}
            onClick={submit}
            className="flex-1"
          >
            {saving ? "Creating…" : title}
          </Button>
          <Button disabled={saving} onClick={onClose}>
            Cancel
          </Button>
        </div>
      }
    >
        <div className="space-y-4">
          {canAddContacts && !form.is_principal && (
            <div role="radiogroup" aria-label="Team member or contact" className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {KINDS.map((k) => (
                <button
                  key={k}
                  type="button"
                  role="radio"
                  aria-checked={form.kind === k}
                  onClick={() => { setForm((f) => ({ ...f, kind: k })); clearSuggested("kind"); }}
                  className={`px-4 py-3 rounded-xl border text-left transition-colors ${
                    form.kind === k
                      ? "bg-accent/10 border-accent/60 text-fg"
                      : "bg-surface-elevated border-line text-fg-muted hover:border-line-strong"
                  } ${suggestedCls("kind")}`}
                >
                  <div className="text-[15px] font-semibold">{k === "team" ? "Team member" : "Contact"}</div>
                  <div className="text-sm leading-snug mt-1 text-fg-muted">
                    {k === "team"
                      ? "Works with you: can sign in, message the Executive and approve."
                      : "Outside the team and private to you: emailed or invited only when you ask."}
                  </div>
                </button>
              ))}
            </div>
          )}

          {/* Always-visible: the 10-second path */}
          <label className={LABEL_CLS}>
            Full name *
            <input
              ref={nameRef}
              value={form.full_name}
              onChange={(e) => { const v = e.target.value; setForm((f) => ({ ...f, full_name: v })); clearSuggested("full_name"); }}
              className={`${INPUT_CLS} ${suggestedCls("full_name")}`}
              placeholder="Sarah Chen"
            />
          </label>

          <label className={LABEL_CLS}>
            {contact ? "Role and company" : "Role"}
            <input
              value={form.role}
              onChange={(e) => { const v = e.target.value; setForm((f) => ({ ...f, role: v })); clearSuggested("role"); }}
              className={`${INPUT_CLS} ${suggestedCls("role")}`}
              placeholder={contact ? "Head of Procurement, Acme" : "CFO (fractional)"}
            />
          </label>

          <label className={LABEL_CLS}>
            Email
            <input
              type="email"
              value={form.email}
              onChange={(e) => { const v = e.target.value; setForm((f) => ({ ...f, email: v })); clearSuggested("email"); }}
              className={`${INPUT_CLS} ${suggestedCls("email")}`}
              placeholder={contact ? "jordan@acme.example" : "sarah@example.com"}
            />
          </label>

          {!contact && canBePrincipal && (
          <label className="flex items-start gap-3 cursor-pointer select-none rounded-xl border border-line px-4 py-3">
            <input
              type="checkbox"
              checked={form.is_principal}
              onChange={(e) => {
                const checked = e.target.checked;
                setForm((f) => ({ ...f, is_principal: checked }));
                if (checked) setShowAuthority(true);
              }}
              className="mt-0.5 w-5 h-5 rounded accent-indigo-500 flex-shrink-0"
            />
            <span>
              <span className="block text-[15px] font-medium text-fg">This is me — Primary</span>
              <span className="block text-sm text-fg-muted">Marks you as the primary decision-maker.</span>
            </span>
          </label>
          )}

          {/* Contact & routing */}
          <DisclosureSection
            label={contact ? "Chat apps" : "Contact & routing"}
            open={contactOpen}
            onToggle={() => {
              setContactToggled(true);
              setShowContact(!contactOpen);
            }}
          >
            {!contact && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className={LABEL_CLS}>
                  Preferred channel
                  <select
                    value={form.preferred_channel}
                    onChange={(e) => setForm((f) => ({ ...f, preferred_channel: e.target.value }))}
                    className={INPUT_CLS}
                  >
                    {CHANNELS.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </label>
                <p className={HINT_CLS}>
                  Proposals routed to this person are sent via {form.preferred_channel === "any" ? "the app they last talked in, then any other" : form.preferred_channel}.
                </p>
              </div>
              <div>
                <label className={LABEL_CLS}>
                  Expected reply within
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      min={1}
                      value={form.response_sla_hours}
                      onChange={(e) => setForm((f) => ({ ...f, response_sla_hours: e.target.value }))}
                      className={`${INPUT_CLS} flex-1 min-w-0`}
                    />
                    <span className="text-sm text-fg-muted flex-shrink-0">hours</span>
                  </div>
                </label>
                <p className={HINT_CLS}>
                  Items show as overdue on Home after {form.response_sla_hours || 24}h with no reply.
                </p>
              </div>
            </div>
            )}

            <ChatAppPickers
              ids={form}
              onChange={setChatId}
              directory={chatDirectory}
              slackMatch={slackMatch}
            />
          </DisclosureSection>

          {/* Approval authority — a contact approves nothing */}
          {!contact && (
          <DisclosureSection
            label="Approval authority"
            open={showAuthority}
            onToggle={() => setShowAuthority((v) => !v)}
          >
            <div className="text-sm text-fg-muted">What this person approves</div>
            <ScopePicker selected={form.authority_scope} onToggle={toggleScope} />
          </DisclosureSection>
          )}

        </div>

        {err && <p className="text-sm text-rose-500 mt-4">{err}</p>}
    </SidePanel>
  );
}

// ---------------------------------------------------------------------------
// Page — one page (lib/peopleBoard.ts): search and Add, Needs a look when
// something is there, the team, the contacts (the three you were last in
// touch with, then All), and the "Keep my contacts up to date" switch.
// ---------------------------------------------------------------------------

export default function PeoplePage() {
  const { mode } = useWorkspace();
  const [notice, setNotice] = useState<string | null>(null);
  const [people, setPeople] = useState<Person[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAdd, setShowAdd] = useState(false);
  const [switchedFor, setSwitchedFor] = useState<string | null>(null);
  // Contacts are private to the principal. Until the viewer is known (or if
  // the check fails) nobody is offered them; the API enforces it regardless.
  const [viewerIsPrincipal, setViewerIsPrincipal] = useState(false);
  const [query, setQuery] = useState("");
  const [allContacts, setAllContacts] = useState(false);
  const [lastContacts, setLastContacts] = useState<Map<number, LastContact>>(new Map());
  // What the Executive changed in the contacts: the principal's alone.
  const [contactChanges, setContactChanges] = useState<ContactChanges | null>(null);

  function refresh() {
    setLoading(true);
    listPeople({ includeContacts: true })
      .then(setPeople)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"))
      .finally(() => setLoading(false));
  }

  const loadContactChanges = useCallback((next?: ContactChanges) => {
    if (next) {
      setContactChanges(next);
      return;
    }
    getContactChanges().then(setContactChanges, () => setContactChanges(null));
    listPeople({ includeContacts: true }).then(setPeople, () => {});
  }, []);

  useEffect(() => {
    if (!viewerIsPrincipal) return;
    getContactChanges().then(setContactChanges, () => setContactChanges(null));
    getLastContacts().then(
      (rows) => setLastContacts(new Map(rows.map((r) => [r.person_id, r]))),
      () => setLastContacts(new Map()),
    );
  }, [viewerIsPrincipal]);

  useEffect(() => {
    refresh();
    getPeopleViewer()
      .then((v) => setViewerIsPrincipal(v.is_principal))
      .catch(() => setViewerIsPrincipal(false));
  }, []);

  const team = teamInOrder(peopleForTab(people, "team", mode));
  const contacts = viewerIsPrincipal ? contactsByRecent(peopleForTab(people, "contacts", mode), lastContacts) : [];
  const pairs = sameEmailPairs([...team, ...contacts]);
  const hidden = hiddenTeamCount(people, mode);
  const searching = query.trim().length > 0;
  const found = searching ? [...team, ...contacts].filter((p) => matchesSearch(p, query)) : [];
  const shownContacts = allContacts ? contacts : contacts.slice(0, CONTACTS_PREVIEW);
  // Local date (YYYY-MM-DD) for "on leave until", sampled once per mount.
  const [today] = useState(() => new Date().toLocaleDateString("en-CA"));

  const row = (person: Person) =>
    isContact(person) ? (
      <PersonRow
        key={person.id}
        contact
        name={person.full_name}
        href={`/people/${person.id}`}
        line={contactLine(person, lastContactLabel(lastContacts.get(person.id)))}
      />
    ) : (
      <PersonRow
        key={person.id}
        name={person.full_name}
        href={`/people/${person.id}`}
        badge={roleBadge(person)}
        line={teamLine(person, today)}
      />
    );

  return (
    <div className="flex flex-col h-full bg-surface">
      {showAdd && (
        <AddPersonModal
          initialKind={addFormKind({ mode, seesContacts: viewerIsPrincipal })}
          canAddContacts={viewerIsPrincipal}
          canBePrincipal={canClaimPrincipal(people, loading || error !== null)}
          onCreated={(p) => {
            setPeople((prev) => [...prev, p]);
            setShowAdd(false);
            setNotice(`${p.full_name} added.`);
            if (shouldSwitchToTeam(mode, p.kind ?? "team", p.is_principal)) setSwitchedFor(p.full_name);
          }}
          onClose={() => setShowAdd(false)}
        />
      )}
      {switchedFor && <TeamModeSwitch name={switchedFor} onDone={() => setSwitchedFor(null)} />}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
          <div className="flex items-center justify-between gap-4 mb-5">
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-fg">People</h1>
            <Button variant="primary" onClick={() => setShowAdd(true)} className="flex-shrink-0">
              Add
            </Button>
          </div>

          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search people"
            aria-label="Search people"
            className="w-full h-11 mb-5 px-4 rounded-xl bg-surface-input/60 border border-line text-[15px] text-fg placeholder-fg-subtle focus:outline-none focus:border-accent"
          />

          {notice && (
            <div
              role="status"
              className="p-4 rounded-xl border text-[15px] mb-5 bg-emerald-500/10 border-emerald-500/30 text-emerald-600 dark:text-emerald-300"
            >
              {notice}
            </div>
          )}
          {loading && <PageSpinner />}
          {error && (
            <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-500 text-[15px] mb-4">
              {error}
            </div>
          )}

          {!loading && !error && searching && (
            found.length === 0 ? (
              <p className="rounded-2xl border border-line bg-surface-elevated p-6 text-center text-[15px] text-fg-muted">
                No one matches.
              </p>
            ) : (
              <ul className="rounded-2xl border border-line bg-surface-elevated overflow-hidden">{found.map(row)}</ul>
            )
          )}

          {!loading && !error && !searching && (
            <>
              <NeedsALook
                changes={viewerIsPrincipal ? contactChanges : null}
                pairs={pairs}
                onChanged={loadContactChanges}
              />

              <section className="mb-6" aria-label="Team">
                <SectionHead title={`Team · ${team.length}`} />
                {team.length > 0 ? (
                  <ul className="rounded-2xl border border-line bg-surface-elevated overflow-hidden">{team.map(row)}</ul>
                ) : (
                  <p className="rounded-2xl border border-line bg-surface-elevated p-6 text-center text-[15px] text-fg-muted">
                    No team members yet.
                  </p>
                )}
                {hidden > 0 && (
                  <p className="text-sm text-fg-muted mt-3 px-1">
                    {hidden === 1 ? "1 other team member is" : `${hidden} other team members are`} hidden while you
                    use Open Executive just for yourself. Switch to team mode in Settings to see them.
                  </p>
                )}
              </section>

              {viewerIsPrincipal && (
                <section className="mb-6" aria-label="Contacts">
                  <SectionHead
                    title={`Contacts · ${contacts.length}`}
                    right={
                      contacts.length > CONTACTS_PREVIEW ? (
                        <button
                          type="button"
                          onClick={() => setAllContacts((v) => !v)}
                          className="text-sm font-semibold text-accent hover:underline"
                        >
                          {allContacts ? "Fewer" : `All ${contacts.length} ›`}
                        </button>
                      ) : undefined
                    }
                  />
                  {contacts.length > 0 ? (
                    <ul className="rounded-2xl border border-line bg-surface-elevated overflow-hidden">{shownContacts.map(row)}</ul>
                  ) : (
                    <p className="rounded-2xl border border-line bg-surface-elevated p-6 text-center text-[15px] text-fg-muted">
                      No contacts yet. Clients, contractors and advisors go here, private to you.
                    </p>
                  )}
                </section>
              )}

              {viewerIsPrincipal && contactChanges && (
                <ContactKeeperSwitch data={contactChanges} onChanged={loadContactChanges} />
              )}
            </>
          )}
        </div>
      </main>
    </div>
  );
}
