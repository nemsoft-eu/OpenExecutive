// What comes back when you return to your own company after a demo fixture or
// a client. The return restores from the snapshot taken before the switch
// (cli/fixture_loader.py, snapshot_user_state): the whole company database
// (state.db, every table), the profile, and the docs folder's Markdown files
// only. Uploaded files in any other format are not in it, so they come back
// empty.

const RESTORED =
  "your profile, Markdown documents and the whole company database (chat history, decisions, team and contacts, learned facts, departments and every setting)";
const EMPTY =
  "uploaded PDF, Word, Excel, CSV and text files";

/** Names what `action` (e.g. "Unload") restores and what comes back empty. */
export function returnNote(action: string): string {
  return `${action} restores ${RESTORED} from the snapshot. Everything else comes back empty, such as ${EMPTY}.`;
}

/** Typed to confirm a load that replaces your own company. */
export const LOAD_CONFIRM_TOKEN = "LOAD";

/**
 * Whether loading a fixture should ask first: only when no fixture is live, so
 * what is replaced may be your own company, whose non-Markdown uploads unload
 * does not bring back. Over another fixture there is nothing of yours to lose.
 */
export function loadNeedsConfirm(activeFixture: string | null): boolean {
  return !activeFixture;
}
