"use client";

import { useState } from "react";

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

export function LeadForm() {
  const [form, setForm] = useState({
    fullName: "",
    workEmail: "",
    phone: "",
    powerUsage: "",
    hardware: "",
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const newErrors: Record<string, string> = {};

    if (!form.fullName.trim()) newErrors.fullName = "Required";
    if (!form.workEmail.trim()) newErrors.workEmail = "Required";
    else if (!isBusinessEmail(form.workEmail))
      newErrors.workEmail = "Please use a business email address";
    if (!form.phone.trim()) newErrors.phone = "Required";
    if (!form.powerUsage.trim()) newErrors.powerUsage = "Required";
    if (!form.hardware.trim()) newErrors.hardware = "Required";

    setErrors(newErrors);
    if (Object.keys(newErrors).length > 0) return;

    setSubmitted(true);
    // TODO: POST to backend / lead capture
  };

  const inputClass =
    "w-full bg-transparent border border-white/20 px-4 py-3 text-white placeholder-white/40 focus:border-white focus:outline-none transition-colors rounded";
  const errorClass = "mt-1 text-xs text-red-400";

  return (
    <section id="request-audit" className="border-t border-white/20 py-24">
      <div className="mx-auto max-w-xl px-6">
        <h2 className="mb-2 text-2xl font-medium tracking-tight text-white">
          Request a Custom Efficiency Blueprint
        </h2>
        <p className="mb-4 text-sm text-white/70">
          Because no two data centers are identical, we do not offer flat
          pricing. We provide custom integration plans based on your thermal
          load.
        </p>
        <p className="mb-10 text-xs text-white/50 italic">
          No sales pitch. Just a data-driven blueprint of your potential savings.
        </p>

        {submitted ? (
          <div className="rounded border border-white/20 bg-white/5 p-6 text-center">
            <p className="font-medium text-white">
              Thank you. We&apos;ll be in touch within 24 hours.
            </p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <input
                type="text"
                placeholder="Full Name"
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
                placeholder="Work Email"
                value={form.workEmail}
                onChange={(e) =>
                  setForm((f) => ({ ...f, workEmail: e.target.value }))
                }
                className={inputClass}
              />
              {errors.workEmail && (
                <p className={errorClass}>{errors.workEmail}</p>
              )}
            </div>
            <div>
              <input
                type="tel"
                placeholder="Phone Number"
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
                placeholder="Current Annual Power Usage (MW)"
                value={form.powerUsage}
                onChange={(e) =>
                  setForm((f) => ({ ...f, powerUsage: e.target.value }))
                }
                className={inputClass}
              />
              {errors.powerUsage && (
                <p className={errorClass}>{errors.powerUsage}</p>
              )}
            </div>
            <div>
              <input
                type="text"
                placeholder="Primary GPU/CPU Architecture (e.g., NVIDIA H100, AMD EPYC)"
                value={form.hardware}
                onChange={(e) =>
                  setForm((f) => ({ ...f, hardware: e.target.value }))
                }
                className={inputClass}
              />
              {errors.hardware && (
                <p className={errorClass}>{errors.hardware}</p>
              )}
            </div>
            <button
              type="submit"
              className="w-full rounded border border-white bg-white px-6 py-4 font-medium tracking-tight text-black transition-opacity hover:opacity-90"
            >
              Request Audit
            </button>
          </form>
        )}
      </div>
    </section>
  );
}
