// How every held step's card opens (Take the lead, a teammate's held step),
// from the server's one place (``card_details.top``): whose OK it waits for,
// who started it, its title and why it waits.

import type { HeldCardTop as Top } from "@/lib/api";

import { Section } from "./CardTalk";
import CardTopRow from "./CardTopRow";

export default function HeldCardTop({
  top,
  title,
  at,
  tag,
  fallbackWaits,
}: {
  // Null until the card's details load.
  top: Top | null;
  title: string;
  at?: string;
  tag?: string;
  // Why it waits until the details load.
  fallbackWaits: string;
}) {
  return (
    <>
      <CardTopRow
        need={top?.waitingOn ? `Waiting on ${top.waitingOn}` : "Needs your OK"}
        source={top?.started || undefined}
        at={at}
        tag={tag}
      />
      <div className="min-w-0 text-base sm:text-[17px] font-semibold leading-snug text-fg break-words">
        {top?.title || title}
      </div>
      <Section title="Why it waits">{top?.waits || fallbackWaits}</Section>
    </>
  );
}
