import Link from "next/link";
import { NavBar } from "../../components/NavBar";
import { LeadForm } from "../../components/LeadForm";

export const metadata = {
  title: "Contact | CooledAI",
  description: "Request a shadow audit or get in touch with CooledAI.",
};

const whyBullets = [
  "7-Day Non-Invasive Shadow Audit",
  "24/7 Thermal Drift Monitoring",
  "SOC2 Compliant Identity Management",
];

export default function ContactPage() {
  return (
    <div className="min-h-screen bg-black text-white">
      <NavBar />
      <main className="mx-auto max-w-6xl px-4 sm:px-6 pt-20 sm:pt-24 pb-16">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-sm text-[#22c55e] hover:text-[#22c55e]/80 transition-colors mb-10"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5M12 19l-7-7 7-7" />
          </svg>
          Back to Home
        </Link>

        <div className="flex flex-col lg:flex-row lg:gap-12 lg:items-start">
          {/* Form column */}
          <div className="flex-1 min-w-0 max-w-xl">
            <h1 className="text-2xl font-semibold tracking-tight text-white md:text-3xl">
              Contact
            </h1>
            <p className="mt-2 text-sm text-white/60">
              Request a 7-day shadow audit or discuss Enterprise and multi-site options.
            </p>
            <div className="mt-10">
              <LeadForm />
            </div>
          </div>

          {/* Why CooledAI? sidebar */}
          <aside className="mt-12 lg:mt-0 lg:w-80 lg:shrink-0">
            <div className="rounded-xl border border-white/10 bg-[#0a0a0a] p-5 sm:p-6 sticky top-24">
              <h2 className="text-lg font-semibold tracking-tight text-white">
                Why CooledAI?
              </h2>
              <ul className="mt-4 space-y-4">
                {whyBullets.map((item, i) => (
                  <li key={i} className="flex items-start gap-3 text-sm text-white/80">
                    <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-[#22c55e]" aria-hidden />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}
