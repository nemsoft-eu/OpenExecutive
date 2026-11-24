"use client";

// Hand me something: on a day when nothing needs you, things to give the
// Executive. Each starts a chat with that ask. The asks are the ones an
// empty chat suggests (GET /chat/suggested-prompts); nothing shows when
// there are none.
import { useEffect, useState } from "react";
import { getSuggestedPrompts } from "@/lib/api";
import { handAsks } from "@/lib/briefingSummary";
import Icon from "@/components/Icon";

export default function HandMeSomething({ onPick }: { onPick: (text: string) => void }) {
  const [asks, setAsks] = useState<string[]>([]);
  useEffect(() => {
    const ctrl = new AbortController();
    getSuggestedPrompts(ctrl.signal)
      .then((r) => setAsks(handAsks(r.prompts)))
      .catch(() => undefined);
    return () => ctrl.abort();
  }, []);
  if (asks.length === 0) return null;
  return (
    <section aria-labelledby="sec-hand-me" className="mb-8">
      <h2 id="sec-hand-me" className="mb-3 text-lg font-semibold text-fg">
        Hand me something
      </h2>
      <ul className="rounded-2xl border border-line bg-surface-elevated">
        {asks.map((text) => (
          <li key={text} className="border-t border-line first:border-t-0">
            <button
              type="button"
              onClick={() => onPick(text)}
              className="flex min-h-touch w-full items-center gap-3 px-4 py-3 text-left text-[15px] font-medium text-fg hover:bg-surface sm:px-5"
            >
              <span className="min-w-0 flex-1 break-words">{text}</span>
              <Icon name="chevron-right" size="w-4 h-4" />
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
