"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { usePathname } from "next/navigation";

const FORMSPREE_BETA = "https://formspree.io/f/xykpdzdd";

export function BetaSignupPopup() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);
  const [hasShown, setHasShown] = useState(false);
  const [form, setForm] = useState({ email: "", facilityScale: "" });
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (pathname?.startsWith("/portal")) return;
    if (hasShown) return;

    const timer = setTimeout(() => {
      setIsOpen(true);
      setHasShown(true);
    }, 30000);

    return () => clearTimeout(timer);
  }, [pathname, hasShown]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!form.email.trim()) {
      setError("Work email required");
      return;
    }
    setSubmitting(true);
    try {
      const res = await fetch(FORMSPREE_BETA, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: form.email,
          facilityScale: form.facilityScale,
        }),
      });
      if (!res.ok) throw new Error("Submission failed");
      setSubmitted(true);
    } catch {
      setError("Submission failed. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleClose = () => setIsOpen(false);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleClose}
            className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 20 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 px-4"
          >
            <div
              className="rounded border border-white/20 bg-black shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-start justify-between border-b border-white/10 px-6 py-4">
                <h3 className="text-lg font-medium tracking-tight text-white">
                  Early Access
                </h3>
                <button
                  onClick={handleClose}
                  className="text-white/50 transition-colors hover:text-white"
                  aria-label="Close"
                >
                  <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <div className="px-6 py-6">
                {submitted ? (
                  <p className="text-center text-sm text-white/90 leading-relaxed">
                    Request Logged. Dennis will reach out via email shortly to
                    coordinate your 7-day Shadow Mode audit.
                  </p>
                ) : (
                  <>
                    <p className="mb-6 text-sm text-white/80 leading-relaxed">
                      Join the CooledAI Private Beta. We are accepting 5 more
                      high-density facilities for Q1 2026.
                    </p>
                    <form onSubmit={handleSubmit} className="space-y-4">
                      <input
                        type="email"
                        placeholder="Work Email"
                        value={form.email}
                        onChange={(e) =>
                          setForm((f) => ({ ...f, email: e.target.value }))
                        }
                        className="w-full bg-transparent border border-white/30 px-4 py-3 text-white placeholder-white/40 focus:border-white focus:outline-none transition-colors"
                        required
                      />
                      <input
                        type="text"
                        placeholder="Facility Scale (MW)"
                        value={form.facilityScale}
                        onChange={(e) =>
                          setForm((f) => ({ ...f, facilityScale: e.target.value }))
                        }
                        className="w-full bg-transparent border border-white/30 px-4 py-3 text-white placeholder-white/40 focus:border-white focus:outline-none transition-colors"
                      />
                      {error && (
                        <p className="text-xs text-red-400">{error}</p>
                      )}
                      <button
                        type="submit"
                        disabled={submitting}
                        className="w-full rounded border border-white bg-white px-6 py-3 text-sm font-medium tracking-tight text-black transition-opacity hover:opacity-90 disabled:opacity-50"
                      >
                        {submitting ? "Submitting…" : "Request Beta Access"}
                      </button>
                    </form>
                  </>
                )}
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
