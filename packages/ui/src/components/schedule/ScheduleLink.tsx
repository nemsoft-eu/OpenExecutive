import Link from "next/link";

import Icon from "@/components/Icon";
import type { ScheduleKind } from "@/lib/api";
import { scheduleHref } from "@/lib/schedule";

/** "On my schedule ›": opens Work → Schedule with this kind's filter on.
 *  Home, Delegate, Workflows and the Watch list keep their own short lists
 *  and link to the one full schedule with it. */
export default function ScheduleLink({ kind, className = "" }: { kind: ScheduleKind; className?: string }) {
  return (
    <Link
      href={scheduleHref(kind)}
      className={`inline-flex min-h-touch flex-none items-center gap-1 text-sm font-semibold text-accent hover:underline ${className}`}
    >
      On my schedule
      <Icon name="chevron-right" size="w-4 h-4" />
    </Link>
  );
}
