"use client";

import { useState } from "react";

const FORMSPREE_AUDIT = "https://formspree.io/f/mqkrpvwz";

const CONSUMER_DOMAINS = [
  "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
  "aol.com", "mail.com", "protonmail.com", "live.com", "msn.com",
];

function isCorporateEmail(email: string): boolean {
  const domain = email.split("@")[1]?.toLowerCase();
  if (!domain) return false;
  return !CONSUMER_DOMAINS.includes(domain);
}

const RACK_COUNT_OPTIONS = [
  { value: "", label: "Select range…" },
  { value: "1-50", label: "1–50" },
  { value: "50-200", label: "50–200" },
  { value: "200+", label: "200+" },
];

export function AuditRequestForm() {
  const [form, setForm] = useState({
    fullName: "",
    workEmail: "",
    companyName: "",
    facilityLocation: "",
    rackCount: "",
    message: "",
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const newErrors: Record<string, string> = {};

    if (!form.fullName.trim()) newErrors.fullName = "Required";
    if (!form.workEmail.trim()) newErrors.workEmail = "Required";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.workEmail))
      newErrors.workEmail = "Please enter a valid email";
    else if (!isCorporateEmail(form.workEmail))
      newErrors.workEmail = "Please use a work or corporate email address";

    setErrors(newErrors);
    setSubmitError(null);
    if (Object.keys(newErrors).length > 0) return;

    setSubmitting(true);
    try {
      const res = await fetch(FORMSPREE_AUDIT, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({
          _replyto: form.workEmail,
          fullName: form.fullName,
          workEmail: form.workEmail,
          companyName: form.companyName || undefined,
          facilityLocation: form.facilityLocation || undefined,
          rackCount: form.rackCount || undefined,
          message: form.message || undefined,
        }),
      });
      if (!res.ok) throw new Error("Submission failed");
      window.location.href = "/audit-success";
      return;
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

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div>
        <label htmlFor="audit-fullName" className={labelClass}>Full Name *</label>
        <input
          id="audit-fullName"
          type="text"
          required
          placeholder="Jane Smith"
          value={form.fullName}
          onChange={(e) => setForm((f) => ({ ...f, fullName: e.target.value }))}
          className={inputClass}
        />
        {errors.fullName && <p className={errorClass}>{errors.fullName}</p>}
      </div>

      <div>
        <label htmlFor="audit-workEmail" className={labelClass}>Work Email *</label>
        <input
          id="audit-workEmail"
          type="email"
          required
          placeholder="jane@company.com"
          value={form.workEmail}
          onChange={(e) => setForm((f) => ({ ...f, workEmail: e.target.value }))}
          className={inputClass}
        />
        {errors.workEmail && <p className={errorClass}>{errors.workEmail}</p>}
      </div>

      <div>
        <label htmlFor="audit-company" className={labelClass}>Company Name</label>
        <input
          id="audit-company"
          type="text"
          placeholder="Acme Data Centers"
          value={form.companyName}
          onChange={(e) => setForm((f) => ({ ...f, companyName: e.target.value }))}
          className={inputClass}
        />
      </div>

      <div>
        <label htmlFor="audit-location" className={labelClass}>Facility Location</label>
        <input
          id="audit-location"
          type="text"
          placeholder="e.g., Rocklin, Sacramento"
          value={form.facilityLocation}
          onChange={(e) => setForm((f) => ({ ...f, facilityLocation: e.target.value }))}
          className={inputClass}
        />
      </div>

      <div>
        <label htmlFor="audit-rackCount" className={labelClass}>Estimated Rack Count</label>
        <select
          id="audit-rackCount"
          value={form.rackCount}
          onChange={(e) => setForm((f) => ({ ...f, rackCount: e.target.value }))}
          className={inputClass + " appearance-none cursor-pointer pr-10"}
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='rgba(255,255,255,0.5)'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'/%3E%3C/svg%3E")`,
            backgroundRepeat: "no-repeat",
            backgroundPosition: "right 12px center",
            backgroundSize: "18px",
          }}
        >
          {RACK_COUNT_OPTIONS.map((opt) => (
            <option key={opt.value || "empty"} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>

      <div>
        <label htmlFor="audit-message" className={labelClass}>Message</label>
        <textarea
          id="audit-message"
          rows={4}
          placeholder="Tell us about your cooling challenges..."
          value={form.message}
          onChange={(e) => setForm((f) => ({ ...f, message: e.target.value }))}
          className={inputClass + " resize-y min-h-[100px]"}
        />
      </div>

      {submitError && <p className="text-sm text-red-400">{submitError}</p>}

      <button
        type="submit"
        disabled={submitting}
        className="w-full min-h-[48px] rounded-lg border-2 border-[#22c55e] bg-[#22c55e] px-6 py-4 text-sm font-semibold tracking-tight text-black shadow-[0_0_20px_rgba(34,197,94,0.3)] transition-all hover:bg-[#22c55e]/90 hover:shadow-[0_0_24px_rgba(34,197,94,0.4)] focus:outline-none focus:ring-2 focus:ring-[#22c55e]/50 focus:ring-offset-2 focus:ring-offset-[#0a0a0a] disabled:pointer-events-none disabled:opacity-50"
      >
        {submitting ? "Submitting…" : "Request Shadow Audit"}
      </button>
    </form>
  );
}
