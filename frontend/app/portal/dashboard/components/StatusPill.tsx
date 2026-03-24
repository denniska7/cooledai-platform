"use client";

import type { SystemStatus } from "@/lib/types/dashboard";

const STATUS_STYLES: Record<
  SystemStatus,
  { bg: string; text: string; border: string; dot: string }
> = {
  green: {
    bg: "bg-emerald-500/20",
    text: "text-emerald-400",
    border: "border-emerald-500/30",
    dot: "bg-emerald-400",
  },
  amber: {
    bg: "bg-amber-500/20",
    text: "text-amber-400",
    border: "border-amber-500/30",
    dot: "bg-amber-400",
  },
  red: {
    bg: "bg-red-500/20",
    text: "text-red-400",
    border: "border-red-500/30",
    dot: "bg-red-400",
  },
};

const DEFAULT_MESSAGES: Record<SystemStatus, string> = {
  green: "All systems optimized",
  amber: "1 item needs attention",
  red: "Action required",
};

export function StatusPill({
  status,
  message,
  onClick,
}: {
  status: SystemStatus;
  message?: string;
  onClick?: () => void;
}) {
  const styles = STATUS_STYLES[status];
  const displayMessage = message || DEFAULT_MESSAGES[status];

  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-medium transition-colors hover:opacity-90 ${styles.bg} ${styles.text} ${styles.border}`}
    >
      <span className="relative flex h-2 w-2">
        <span
          className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${styles.dot}`}
        />
        <span
          className={`relative inline-flex rounded-full h-2 w-2 ${styles.dot}`}
        />
      </span>
      {displayMessage}
    </button>
  );
}
