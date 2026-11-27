// One wording for a connection that stopped working, everywhere it shows:
// Home's Needs you row (GET /today `waiting_elsewhere`, worded on the server
// in briefing/needs_reconnecting.py) and Delegate → As you. Tested without a browser
// (scripts/reconnect.test.mjs).

export const NEEDS_RECONNECTING = "Needs reconnecting";
export const RECONNECT = "Reconnect";
export const MAILBOX_BROKEN = "Your mailbox needs reconnecting";
export const MAILBOX_BROKEN_DETAIL = "I can't read or answer your email, or see your calendar, until you do.";

/** Whether the person's own mailbox (As you) was connected and needs
 *  connecting again: its saved sign-in is refused, or now opens another
 *  mailbox. */
export function mailboxNeedsReconnecting(status: string | null | undefined): boolean {
  return status === "needs_reconnect" || status === "mismatch";
}
