"use client";

import { useState } from "react";
import { motion } from "framer-motion";

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
    businessEmail: "",
    phone: "",
    dataCenterScale: "",
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const newErrors: Record<string, string> = {};

    if (!form.fullName.trim()) newErrors.fullName = "Required";
    if (!form.businessEmail.trim()) newErrors.businessEmail = "Required";
    else if (!isBusinessEmail(form.businessEmail))
      newErrors.businessEmail = "Please use a business email address";
    if (!form.phone.trim()) newErrors.phone = "Required";
    if (!form.dataCenterScale.trim()) newErrors.dataCenterScale = "Required";

    setErrors(newErrors);
    if (Object.keys(newErrors).length > 0) return;

    setSubmitted(true);
    // TODO: POST to backend / lead capture
  };

  const inputClass =
    "w-full bg-transparent border border-white px-4 py-4 text-white placeholder-white/40 focus:border-accent-cyan focus:outline-none transition-colors tracking-wide";
  const errorClass = "mt-1 text-xs text-red-400";

  return (
    <motion.section
      id="request-audit"
      initial={{ opacity: 0, y: 30, filter: "blur(10px)" }}
      whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className="border-t border-white py-32"
    >
      <div className="mx-auto max-w-xl px-6">
        <h2 className="mb-4 text-3xl font-medium tracking-tight text-white">
          Request Your 2026 Efficiency Blueprint
        </h2>
        <p className="mb-12 text-sm text-white/70 tracking-wide">
          No sales pitch. Just a data-driven blueprint of your potential savings.
        </p>

        {submitted ? (
          <div className="rounded border border-white bg-white/5 p-8 text-center">
            <p className="font-medium text-white tracking-wide">
              Thank you. We&apos;ll be in touch within 24 hours.
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
            <button
              type="submit"
              className="w-full rounded border border-white bg-white px-6 py-4 font-medium tracking-tight text-black transition-opacity hover:opacity-90"
            >
              Request Blueprint
            </button>
          </form>
        )}
      </div>
    </motion.section>
  );
}
