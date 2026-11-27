// The People page splits the roster into the team and the principal's
// contacts. A contact is someone outside the team (a client, a contractor):
// they cannot sign in or talk to the bot, and the Executive emails or invites
// them only when the principal asks directly. Contacts are private to the
// principal — the API never returns one to anyone else — so the Contacts tab
// is offered to the principal only.
//
// Type-only imports, so `npm test` can exercise this under
// `node --experimental-strip-types` (see scripts/peopleKinds.test.mjs).

import type { Person, PersonKind, WorkspaceMode } from "@/lib/api";

export type PeopleTab = "team" | "contacts";

type KindFields = Pick<Person, "kind" | "is_principal">;

/** A row from an older server has no `kind`: it is a team member. */
export function isContact(person: Pick<Person, "kind">): boolean {
  return person.kind === "contact";
}

/**
 * The people a tab lists. In solo mode the Team tab is just the principal:
 * there is no team to show.
 */
export function peopleForTab<P extends KindFields>(people: P[], tab: PeopleTab, mode: WorkspaceMode): P[] {
  if (tab === "contacts") return people.filter(isContact);
  const team = people.filter((p) => !isContact(p));
  return mode === "solo" ? team.filter((p) => p.is_principal) : team;
}

/** Team members the solo Team tab leaves out (everyone but the principal). */
export function hiddenTeamCount(people: KindFields[], mode: WorkspaceMode): number {
  if (mode !== "solo") return 0;
  return people.filter((p) => !isContact(p) && !p.is_principal).length;
}

/**
 * "This is me — Primary" is only for the first setup: once someone on the
 * list is the principal, nobody else can be, so the add form doesn't offer
 * it (not even to the principal, who is already on the list). Not offered
 * while the list is still loading either.
 */
export function canClaimPrincipal(people: Pick<Person, "is_principal">[], loading: boolean): boolean {
  return !loading && !people.some((p) => p.is_principal);
}

/** The principal is always on the team, whatever the form says. */
export function effectiveKind(kind: PersonKind, isPrincipal: boolean): PersonKind {
  return isPrincipal ? "team" : kind;
}

/**
 * Adding someone to the team while using Open Executive just for yourself
 * means there is a team now: the workspace switches to team mode on its own,
 * without asking. Never for the principal's own entry or for a contact.
 */
export function shouldSwitchToTeam(mode: WorkspaceMode, kind: PersonKind, isPrincipal: boolean): boolean {
  return mode === "solo" && effectiveKind(kind, isPrincipal) === "team" && !isPrincipal;
}
