import Link from "next/link";

export const metadata = {
  title: "Terms of Service | CooledAI",
  description:
    "Terms of Service for CooledAI. Autonomous thermal optimization software. Billing via Lemon Squeezy. Hardware safety, liability, jurisdiction Placer County, CA.",
};

export default function TermsPage() {
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
          Terms of Service
        </h1>
        <p className="mt-3 text-sm text-white/60">
          Effective Date: February 4, 2026 · CooledAI LLC
        </p>

        <div className="mt-14 space-y-12 text-[#e5e5e5] leading-relaxed">
          <section>
            <h2 className="text-lg font-bold text-white mb-4">
              <span className="text-[#22c55e]">1.</span> Agreement to Terms
            </h2>
            <p>
              By using CooledAI, you agree to these terms. We provide an autonomous software layer for thermal optimization.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-bold text-white mb-4">
              <span className="text-[#22c55e]">2.</span> Merchant of Record
            </h2>
            <p>
              All billing is handled by Lemon Squeezy. Their terms also apply to your financial transactions.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-bold text-white mb-4">
              <span className="text-[#22c55e]">3.</span> Hardware Safety
            </h2>
            <p>
              CooledAI is a software-only optimization layer. You are responsible for maintaining physical safety overrides and hardware-level thermal cutoffs. CooledAI LLC is not liable for hardware failure, hashrate loss, or physical damage.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-bold text-white mb-4">
              <span className="text-[#22c55e]">4.</span> Liability
            </h2>
            <p>
              To the extent permitted by California law, our liability is limited to the amount you paid for the service in the last 12 months.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-bold text-white mb-4">
              <span className="text-[#22c55e]">5.</span> Jurisdiction
            </h2>
            <p>
              Any legal disputes will be handled in Placer County, CA.
            </p>
          </section>
        </div>

        <p className="mt-16 text-sm text-white/50">
          Questions? Contact{" "}
          <a href="mailto:legal@cooledai.com" className="text-[#22c55e] hover:underline">
            legal@cooledai.com
          </a>
          .
        </p>
      </main>
    </div>
  );
}
