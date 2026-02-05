"use client";

import { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

type Props = {
  isOpen: boolean;
  onClose: () => void;
};

/**
 * Placeholder overlay for Lemon Squeezy Checkout.
 * Replace the inner content with your Lemon Squeezy embed or redirect when ready.
 * See: https://docs.lemonsqueezy.com/help/checkout/checkout-overlay
 */
export function LemonSqueezyCheckoutOverlay({ isOpen, onClose }: Props) {
  useEffect(() => {
    if (!isOpen) return;
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [isOpen, onClose]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-black/70 z-50"
            onClick={onClose}
            aria-hidden="true"
          />
          <motion.div
            role="dialog"
            aria-modal="true"
            aria-labelledby="lemonsqueezy-overlay-title"
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ duration: 0.2 }}
            className="fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-xl border border-white/10 bg-[#141414] p-6 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h2 id="lemonsqueezy-overlay-title" className="text-lg font-semibold text-white">
                Lemon Squeezy Checkout
              </h2>
              <button
                type="button"
                onClick={onClose}
                className="p-2 text-white/60 hover:text-white rounded-lg transition-colors"
                aria-label="Close"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>
            <p className="text-sm text-white/60 mb-6">
              This is a placeholder. Configure your Lemon Squeezy checkout URL or embed here to let customers update payment, change plans, or view invoices.
            </p>
            <div className="rounded-lg border border-[#22c55e]/30 bg-[#22c55e]/5 p-4 text-center">
              <p className="text-xs text-[#22c55e]/90">
                Add your Lemon Squeezy overlay URL or iframe in{" "}
                <code className="bg-white/10 px-1 rounded">LemonSqueezyCheckoutOverlay.tsx</code>
              </p>
            </div>
            <button
              type="button"
              onClick={onClose}
              className="mt-6 w-full rounded-lg border border-white/20 bg-white/5 py-2.5 text-sm font-medium text-white hover:bg-white/10 transition-colors"
            >
              Close
            </button>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
