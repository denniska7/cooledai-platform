import Link from "next/link";

export const metadata = {
  title: "Privacy Policy | CooledAI",
  description:
    "Privacy Policy for CooledAI. Data collection, AI usage, encryption, no sale of data. CCPA 2026 rights. Contact legal@cooledai.com.",
};

export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-[#0a0a0a]">
      <main className="mx-auto max-w-3xl px-6 py-16 sm:py-24">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-sm text-[#22c55e] hover:text-[#22c55e]/80 transition-colors mb-12"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5M12 19l-7-7 7-7" />
          </svg>
          Back to Home
        </Link>

        <h1 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
          Privacy Policy
        </h1>
        <p className="mt-3 text-sm text-white/60">
          Effective Date: February 4, 2026 · CooledAI LLC
        </p>

        <div className="mt-14 space-y-12 text-[#e5e5e5] leading-relaxed">
          <section>
            <h2 className="text-lg font-bold text-white mb-4">
              <span className="text-[#22c55e]">1.</span> Data Collection
            </h2>
            <p>
              We collect account info (email/billing via Lemon Squeezy and any account sign-in when enabled) and technical logs (chip temps, fan speeds, power draw).
            </p>
          </section>

          <section>
            <h2 className="text-lg font-bold text-white mb-4">
              <span className="text-[#22c55e]">2.</span> AI Usage
            </h2>
            <p>
              We use Automated Decision-Making Technology (ADMT) to predict and manage heat in your facility. You may opt-out, but this will disable autonomous features.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-bold text-white mb-4">
              <span className="text-[#22c55e]">3.</span> Data Security
            </h2>
            <p>
              All data is encrypted with AES-256 at rest and TLS 1.3 in transit.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-bold text-white mb-4">
              <span className="text-[#22c55e]">4.</span> No Sale of Data
            </h2>
            <p>
              We never sell your personal or facility data to third parties.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-bold text-white mb-4">
              <span className="text-[#22c55e]">5.</span> Your Rights
            </h2>
            <p>
              Under CCPA 2026, you have the right to access, delete, or correct your data. Contact{" "}
              <a href="mailto:legal@cooledai.com" className="text-[#22c55e] hover:underline">
                legal@cooledai.com
              </a>{" "}
              to exercise these rights.
            </p>
          </section>
        </div>

        <p className="mt-16 text-sm text-white/50">
          For privacy requests or questions, email{" "}
          <a href="mailto:legal@cooledai.com" className="text-[#22c55e] hover:underline">
            legal@cooledai.com
          </a>
          .
        </p>
      </main>
    </div>
  );
}
