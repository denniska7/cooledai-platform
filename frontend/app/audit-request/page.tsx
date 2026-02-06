import Link from "next/link";
import { NavBar } from "../../components/NavBar";
import { AuditRequestForm } from "../../components/AuditRequestForm";

export const metadata = {
  title: "Request Shadow Audit | CooledAI",
  description: "Request a 7-day non-invasive shadow audit and ROI projection.",
};

export default function AuditRequestPage() {
  return (
    <div className="min-h-screen bg-black text-white">
      <NavBar />
      <main className="mx-auto max-w-xl px-6 pt-24 pb-16">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-sm text-[#22c55e] hover:text-[#22c55e]/80 transition-colors mb-10"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5M12 19l-7-7 7-7" />
          </svg>
          Back to Home
        </Link>
        <h1 className="text-2xl font-semibold tracking-tight text-white md:text-3xl">
          Request Shadow Audit
        </h1>
        <p className="mt-2 text-sm text-white/60">
          Get a 7-day non-invasive audit and a custom ROI projection. Dennis will respond within 4 hours.
        </p>
        <div className="mt-10">
          <AuditRequestForm />
        </div>
      </main>
    </div>
  );
}
