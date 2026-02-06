"use client";

import { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { LeadForm } from "./LeadForm";

type Props = {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
};

export function LeadFormModal({ isOpen, onClose, title }: Props) {
  useEffect(() => {
    if (!isOpen) return;
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleEscape);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handleEscape);
      document.body.style.overflow = "";
    };
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
            className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm"
            onClick={onClose}
            aria-hidden
          />
          <div className="fixed inset-0 z-50 flex items-start justify-center p-4 pt-12 sm:pt-20 overflow-y-auto pointer-events-none">
            <motion.div
              initial={{ opacity: 0, scale: 0.98, y: 8 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.98, y: 8 }}
              transition={{ duration: 0.2 }}
              className="relative w-full max-w-lg rounded-xl border border-white/20 bg-[#0a0a0a] shadow-2xl pointer-events-auto"
              onClick={(e) => e.stopPropagation()}
              role="dialog"
              aria-modal="true"
              aria-labelledby={title ? "lead-form-modal-title" : undefined}
            >
              <div className="sticky top-0 flex items-center justify-between border-b border-white/10 bg-[#0a0a0a] px-6 py-4 rounded-t-xl z-10">
                {title && (
                  <h2 id="lead-form-modal-title" className="text-lg font-semibold text-white">
                    {title}
                  </h2>
                )}
                <button
                  type="button"
                  onClick={onClose}
                  className="ml-auto p-2 -mr-2 text-white/60 hover:text-white rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-[#22c55e]/50"
                  aria-label="Close"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M18 6L6 18M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <div className="p-6">
                <LeadForm compact onSuccess={onClose} />
              </div>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
}
