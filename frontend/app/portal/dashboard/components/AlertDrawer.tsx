"use client";

import { motion, AnimatePresence } from "framer-motion";
import type { Alert } from "@/lib/types/dashboard";

const CATEGORY_STYLES: Record<
  Alert["category"],
  { border: string; label: string; labelColor: string }
> = {
  action: {
    border: "border-l-red-400",
    label: "ACTION REQUIRED",
    labelColor: "text-red-400",
  },
  awareness: {
    border: "border-l-amber-400",
    label: "AWARENESS",
    labelColor: "text-amber-400",
  },
  achievement: {
    border: "border-l-emerald-400",
    label: "ACHIEVEMENT",
    labelColor: "text-emerald-400",
  },
};

function formatRelativeTime(ts: number): string {
  const seconds = Math.round((Date.now() - ts) / 1000);
  if (seconds < 60) return "Just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export function AlertDrawer({
  alerts,
  onDismiss,
  open,
  onClose,
}: {
  alerts: Alert[];
  onDismiss: (id: string) => void;
  open: boolean;
  onClose: () => void;
}) {
  const grouped = {
    action: alerts.filter((a) => a.category === "action"),
    awareness: alerts.filter((a) => a.category === "awareness"),
    achievement: alerts.filter((a) => a.category === "achievement"),
  };

  const hasAlerts =
    grouped.action.length + grouped.awareness.length + grouped.achievement.length > 0;

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
          />

          {/* Drawer */}
          <motion.aside
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
            className="fixed right-0 top-0 bottom-0 z-50 w-full sm:w-96 bg-[#0a0a0a] border-l border-white/10 flex flex-col"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-white/10">
              <h2 className="text-lg font-semibold text-white">
                Notifications
              </h2>
              <button
                type="button"
                onClick={onClose}
                className="p-1.5 rounded-lg text-white/50 hover:text-white hover:bg-white/10 transition-colors"
                aria-label="Close notifications"
              >
                <svg
                  width={18}
                  height={18}
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-auto p-4 space-y-6">
              {!hasAlerts ? (
                <div className="flex flex-col items-center justify-center h-full text-center py-12">
                  <svg
                    width={40}
                    height={40}
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="rgba(255,255,255,0.2)"
                    strokeWidth={1.5}
                  >
                    <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="text-sm text-white/40 mt-3">
                    No notifications &mdash; all systems running smoothly
                  </p>
                </div>
              ) : (
                (["action", "awareness", "achievement"] as const).map(
                  (category) => {
                    const items = grouped[category];
                    if (items.length === 0) return null;
                    const styles = CATEGORY_STYLES[category];
                    return (
                      <div key={category}>
                        <p
                          className={`text-xs font-semibold tracking-wider mb-3 ${styles.labelColor}`}
                        >
                          {styles.label}
                        </p>
                        <div className="space-y-2">
                          {items.map((alert) => (
                            <div
                              key={alert.id}
                              className={`rounded-lg border border-white/10 bg-[#141414] p-3 border-l-2 ${styles.border}`}
                            >
                              <div className="flex items-start justify-between gap-2">
                                <div className="flex-1 min-w-0">
                                  <p className="text-sm text-white">
                                    {alert.message}
                                  </p>
                                  {alert.detail && (
                                    <p className="text-xs text-white/40 mt-0.5">
                                      {alert.detail}
                                    </p>
                                  )}
                                  <p className="text-xs text-white/30 mt-1">
                                    {formatRelativeTime(alert.timestamp)}
                                  </p>
                                </div>
                                <button
                                  type="button"
                                  onClick={() => onDismiss(alert.id)}
                                  className="shrink-0 p-1 text-white/30 hover:text-white/60 transition-colors"
                                  aria-label="Dismiss"
                                >
                                  <svg
                                    width={14}
                                    height={14}
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    strokeWidth={2}
                                  >
                                    <path d="M18 6L6 18M6 6l12 12" />
                                  </svg>
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  }
                )
              )}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
