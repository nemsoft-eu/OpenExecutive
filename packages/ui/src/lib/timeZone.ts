/** This device's own time zone ("Europe/London"), or null when the browser
 * can't tell. The one copy: setup sends it as the company's zone, and the
 * Settings cards offer it. */
export function deviceTimeZone(): string | null {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || null;
  } catch {
    return null;
  }
}
