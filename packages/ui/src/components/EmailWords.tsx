import { emailBlocks } from "@/lib/emailBlocks";
import { highlightPieces } from "@/lib/replyCards";

/** An email's words on a card, the way the email will read: text as
 * written (line breaks kept), its pipe tables as tables, and the sentences
 * a change just wrote (`changed`) lightly marked. Every email card shows
 * its words through this. */
export default function EmailWords({
  text,
  changed = [],
  className = "text-sm",
  after,
}: {
  text: string;
  changed?: readonly string[];
  className?: string;
  // Shown right after the words (a "…" for words cut short).
  after?: React.ReactNode;
}) {
  const marked = (words: string) =>
    changed.length > 0
      ? highlightPieces(words, [...changed]).map((piece, i) =>
          piece.changed ? (
            <mark key={i} className="rounded bg-accent/15 px-0.5 text-fg">
              {piece.text}
            </mark>
          ) : (
            <span key={i}>{piece.text}</span>
          ),
        )
      : words;
  const blocks = emailBlocks(text);
  return (
    <div className={`whitespace-pre-wrap break-words text-fg ${className}`}>
      {blocks.map((block, b) =>
        block.kind === "text" ? (
          <span key={b}>{marked(block.text)}{b < blocks.length - 1 ? "\n" : ""}</span>
        ) : (
          <div key={b} className="my-2 max-w-full overflow-x-auto whitespace-normal">
            <table className="border-collapse text-left">
              <thead>
                <tr className="bg-surface-overlay">
                  {block.header.map((cell, c) => (
                    <th key={c} style={{ textAlign: block.align[c] }} className="border border-line px-2.5 py-1.5 font-semibold">
                      {marked(cell)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {block.rows.map((row, r) => (
                  <tr key={r}>
                    {row.map((cell, c) => (
                      <td key={c} style={{ textAlign: block.align[c] }} className="border border-line px-2.5 py-1.5 align-top">
                        {marked(cell)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ),
      )}
      {after}
    </div>
  );
}
