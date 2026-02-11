"use client";

import { useState } from "react";
import { motion } from "framer-motion";

const FORMSPREE_LEAD = "https://formspree.io/f/xqeldapd";

const COOLING_CHALLENGES = [
  { value: "", label: "Select primary challenge…" },
  { value: "overheating", label: "Overheating" },
  { value: "high_energy_bills", label: "High Energy Bills" },
  { value: "humidity_control", label: "Humidity Control" },
  { value: "other", label: "Other" },
] as const;

type LeadFormProps = {
  onSuccess?: () => void;
  /** If true, render without section wrapper (e.g. for modal) */
  compact?: boolean;
};

export function LeadForm({ onSuccess, compact = false }: LeadFormProps) {
  const [form, setForm] = useState({
    fullName: "",
    corporateEmail: "",
    companyName: "",
    facilitySqFt: "",
    currentPue: "",
    primaryChallenge: "",
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitted, setSubmitted] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const newErrors: Record<string, string> = {};

    if (!form.fullName.trim()) newErrors.fullName = "Required";
    if (!form.corporateEmail.trim()) newErrors.corporateEmail = "Required";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.corporateEmail))
      newErrors.corporateEmail = "Please enter a valid email";
    if (!form.companyName.trim()) newErrors.companyName = "Required";
    if (!form.facilitySqFt.trim()) newErrors.facilitySqFt = "Required";
    if (!form.primaryChallenge)
      newErrors.primaryChallenge = "Please select a challenge";

    setErrors(newErrors);
    setSubmitError(null);
    if (Object.keys(newErrors).length > 0) return;

    setSubmitting(true);
    try {
      const res = await fetch(FORMSPREE_LEAD, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fullName: form.fullName,
          corporateEmail: form.corporateEmail,
          companyName: form.companyName,
          facilitySqFt: form.facilitySqFt,
          currentPue: form.currentPue || undefined,
          primaryChallenge: form.primaryChallenge,
        }),
      });
      if (!res.ok) throw new Error("Submission failed");
      setSubmitted(true);
      if (onSuccess) {
        setTimeout(onSuccess, 1500);
      }
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Submission failed");
    } finally {
      setSubmitting(false);
    }
  };

  const inputClass =
    "w-full rounded-lg border border-white/20 bg-[#0a0a0a] px-4 py-3.5 text-white placeholder-white/40 focus:border-[#22c55e] focus:outline-none focus:ring-1 focus:ring-[#22c55e]/50 transition-colors text-sm tracking-tight";
  const labelClass = "block text-xs font-medium text-white/70 mb-1.5 tracking-tight";
  const errorClass = "mt-1 text-xs text-red-400";

  const content = (
    <div className={compact ? "max-w-xl" : "mx-auto max-w-xl px-4 sm:px-6"}>
        <h2 className="text-2xl font-semibold tracking-tight text-white md:text-3xl">
          Get Your Savings Roadmap
        </h2>
        <p className="mt-2 text-sm text-white/60 tracking-tight">
          For data center managers. No sales pitch—a data-driven view of reclaimable capacity and savings.
        </p>

        {submitted ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
            className="mt-10 rounded-xl border border-[#22c55e]/30 bg-[#22c55e]/5 p-8 text-center"
          >
            <p className="text-lg font-semibold text-[#22c55e]">Thank You</p>
            <p className="mt-2 text-sm text-white/80">
              We&apos;ll be in touch shortly with your preliminary thermal ROI report.
            </p>
          </motion.div>
        ) : (
          <form onSubmit={handleSubmit} className="mt-10 space-y-5">
            <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
              <div>
                <label htmlFor="lead-fullName" className={labelClass}>
                  Full Name
                </label>
                <input
                  id="lead-fullName"
                  type="text"
                  placeholder="Jane Smith"
                  value={form.fullName}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, fullName: e.target.value }))
                  }
                  className={inputClass}
                />
                {errors.fullName && (
                  <p className={errorClass}>{errors.fullName}</p>
                )}
              </div>
              <div>
                <label htmlFor="lead-email" className={labelClass}>
                  Corporate Email
                </label>
                <input
                  id="lead-email"
                  type="email"
                  placeholder="jane@company.com"
                  value={form.corporateEmail}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, corporateEmail: e.target.value }))
                  }
                  className={inputClass}
                />
                {errors.corporateEmail && (
                  <p className={errorClass}>{errors.corporateEmail}</p>
                )}
              </div>
            </div>

            <div>
              <label htmlFor="lead-company" className={labelClass}>
                Company Name
              </label>
              <input
                id="lead-company"
                type="text"
                placeholder="Acme Data Centers"
                value={form.companyName}
                onChange={(e) =>
                  setForm((f) => ({ ...f, companyName: e.target.value }))
                }
                className={inputClass}
              />
              {errors.companyName && (
                <p className={errorClass}>{errors.companyName}</p>
              )}
            </div>

            <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
              <div>
                <label htmlFor="lead-sqft" className={labelClass}>
                  Facility Square Footage (Approx)
                </label>
                <input
                  id="lead-sqft"
                  type="text"
                  placeholder="e.g. 50,000"
                  value={form.facilitySqFt}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, facilitySqFt: e.target.value }))
                  }
                  className={inputClass}
                />
                {errors.facilitySqFt && (
                  <p className={errorClass}>{errors.facilitySqFt}</p>
                )}
              </div>
              <div>
                <label htmlFor="lead-pue" className={labelClass}>
                  Current PUE <span className="text-white/40">(if known)</span>
                </label>
                <input
                  id="lead-pue"
                  type="text"
                  placeholder="e.g. 1.5"
                  value={form.currentPue}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, currentPue: e.target.value }))
                  }
                  className={inputClass}
                />
              </div>
            </div>

            <div>
              <label htmlFor="lead-challenge" className={labelClass}>
                Primary Cooling Challenge
              </label>
              <select
                id="lead-challenge"
                value={form.primaryChallenge}
                onChange={(e) =>
                  setForm((f) => ({ ...f, primaryChallenge: e.target.value }))
                }
                className={inputClass + " appearance-none cursor-pointer pr-10 bg-[#0a0a0a]"}
                style={{
                  backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='rgba(255,255,255,0.5)'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'/%3E%3C/svg%3E")`,
                  backgroundRepeat: "no-repeat",
                  backgroundPosition: "right 12px center",
                  backgroundSize: "18px",
                }}
              >
                {COOLING_CHALLENGES.map((opt) => (
                  <option key={opt.value || "empty"} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
              {errors.primaryChallenge && (
                <p className={errorClass}>{errors.primaryChallenge}</p>
              )}
            </div>

            {submitError && (
              <p className="text-sm text-red-400">{submitError}</p>
            )}

            <button
              type="submit"
              disabled={submitting}
              className="w-full min-h-[48px] rounded-lg border-2 border-[#22c55e] bg-[#22c55e] px-6 py-4 text-sm font-semibold tracking-tight text-black shadow-[0_0_20px_rgba(34,197,94,0.3)] transition-all hover:bg-[#22c55e]/90 hover:shadow-[0_0_24px_rgba(34,197,94,0.4)] focus:outline-none focus:ring-2 focus:ring-[#22c55e]/50 focus:ring-offset-2 focus:ring-offset-[#0a0a0a] disabled:pointer-events-none disabled:opacity-50"
            >
              {submitting ? "Submitting…" : "Get My Savings Roadmap"}
            </button>
          </form>
        )}
    </div>
  );

  if (compact) {
    return content;
  }

  return (
    <motion.section
      id="request-audit"
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="border-t border-white/20 py-24"
    >
      {content}
    </motion.section>
  );
}
