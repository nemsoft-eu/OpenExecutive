// What comes back when you return to your own company after a demo fixture or
// a client. The return restores from the snapshot, which is taken on the first
// switch and reused after that, and which carries only these parts
// (cli/fixture_loader.py, snapshot_user_state): Markdown docs only, and team
// members only (no contacts, archived people, contact details or reporting
// lines). Every other per-company table is wiped on the swap
// (memory/company_tables.py), so it comes back empty.

const RESTORED =
  "your profile, Markdown documents, decisions, initiatives, advice, team roster, departments and workspace settings";
const EMPTY =
  "chat history, uploaded PDF, Word, Excel, CSV and text files, contacts, learned facts, department notes, and Act as me, Take the lead and Always in the loop settings";

/** Names what `action` (e.g. "Unload") restores and what comes back empty. */
export function returnNote(action: string): string {
  return `${action} restores ${RESTORED} from the snapshot. Everything else comes back empty, such as ${EMPTY}.`;
}

/** Typed to confirm a load that replaces your own company. */
export const LOAD_CONFIRM_TOKEN = "LOAD";

/**
 * Whether loading a fixture should ask first: only when no fixture is live, so
 * what is replaced may be your own company, which unload brings back only in
 * part. Over another fixture there is nothing of yours to lose.
 */
export function loadNeedsConfirm(activeFixture: string | null): boolean {
  return !activeFixture;
}
