"use client";

import { useState } from "react";
import { motion } from "framer-motion";

const FORMSPREE_LEAD = "https://formspree.io/f/xqeldapd";

const CONSUMER_DOMAINS = [
  "gmail.com",
  "yahoo.com",
  "hotmail.com",
  "outlook.com",
  "icloud.com",
  "aol.com",
  "mail.com",
  "protonmail.com",
  "live.com",
  "msn.com",
];

function isBusinessEmail(email: string): boolean {
  const domain = email.split("@")[1]?.toLowerCase();
  if (!domain) return false;
  return !CONSUMER_DOMAINS.includes(domain);
}

function companyFromEmail(email: string): string {
  const domain = email.split("@")[1];
  if (!domain) return "";
  const name = domain.split(".")[0];
  return name ? name.charAt(0).toUpperCase() + name.slice(1) : "";
}

export function LeadForm() {
  const [form, setForm] = useState({
    fullName: "",
    businessEmail: "",
    phone: "",
    dataCenterScale: "",
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitted, setSubmitted] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const newErrors: Record<string, string> = {};

    if (!form.fullName.trim()) newErrors.fullName = "Required";
    if (!form.businessEmail.trim()) newErrors.businessEmail = "Required";
    else if (!isBusinessEmail(form.businessEmail))
      newErrors.businessEmail = "Please use a business email address";
    if (!form.phone.trim()) newErrors.phone = "Required";
    if (!form.dataCenterScale.trim()) newErrors.dataCenterScale = "Required";

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
          businessEmail: form.businessEmail,
          phone: form.phone,
          dataCenterScale: form.dataCenterScale,
        }),
      });
      if (!res.ok) throw new Error("Submission failed");
      setSubmitted(true);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Submission failed");
    } finally {
      setSubmitting(false);
    }
  };

  const inputClass =
    "w-full bg-transparent border border-white px-4 py-4 text-white placeholder-white/40 focus:border-accent-cyan focus:outline-none transition-colors tracking-wide";
  const errorClass = "mt-1 text-xs text-red-400";

  return (
    <motion.section
      id="request-audit"
      initial={{ opacity: 0, x: -80 }}
      whileInView={{ opacity: 1, x: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className="border-t border-white py-32"
    >
      <div className="mx-auto max-w-xl px-6">
        <h2 className="mb-4 text-3xl font-medium tracking-tight text-white">
          Get Your Savings Roadmap
        </h2>
        <p className="mb-12 text-sm text-white/70 tracking-wide">
          No sales pitch. A data-driven roadmap of your reclaimable capacity and savings.
        </p>

        {submitted ? (
          <div className="rounded border border-[rgba(255,255,255,0.1)] bg-white/5 p-8 text-center">
            <p className="font-medium text-white tracking-wide">
              Infrastructure Profile Received.
            </p>
            <p className="mt-3 text-sm text-white/60 tracking-wide">
              A Preliminary Thermal ROI Report is being generated for{" "}
              <span className="text-white">
                {companyFromEmail(form.businessEmail) || form.fullName}
              </span>
              . Check your dashboard in 4 hours.
            </p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <input
                type="text"
                placeholder="Name"
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
              <input
                type="email"
                placeholder="Business Email"
                value={form.businessEmail}
                onChange={(e) =>
                  setForm((f) => ({ ...f, businessEmail: e.target.value }))
                }
                className={inputClass}
              />
              {errors.businessEmail && (
                <p className={errorClass}>{errors.businessEmail}</p>
              )}
            </div>
            <div>
              <input
                type="tel"
                placeholder="Phone"
                value={form.phone}
                onChange={(e) =>
                  setForm((f) => ({ ...f, phone: e.target.value }))
                }
                className={inputClass}
              />
              {errors.phone && <p className={errorClass}>{errors.phone}</p>}
            </div>
            <div>
              <input
                type="text"
                placeholder="Data Center Scale (MW)"
                value={form.dataCenterScale}
                onChange={(e) =>
                  setForm((f) => ({ ...f, dataCenterScale: e.target.value }))
                }
                className={inputClass}
              />
              {errors.dataCenterScale && (
                <p className={errorClass}>{errors.dataCenterScale}</p>
              )}
            </div>
            {submitError && (
              <p className="text-sm text-red-400">{submitError}</p>
            )}
            <button
              type="submit"
              disabled={submitting}
              className="w-full rounded border border-white bg-white px-6 py-4 font-medium tracking-tight text-black transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              {submitting ? "Submitting…" : "Get My Savings Roadmap"}
            </button>
          </form>
        )}
      </div>
    </motion.section>
  );
}
