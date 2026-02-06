import Link from "next/link";
import { NavBar } from "../../components/NavBar";

export const metadata = {
  title: "Success | CooledAI",
  description: "Your audit request has been received.",
};

export default function AuditSuccessPage() {
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
        <div className="rounded-xl border border-[#22c55e]/30 bg-[#22c55e]/10 p-8 text-center">
          <h1 className="text-xl font-semibold text-[#22c55e]">Success</h1>
          <p className="mt-3 text-white/90">
            Your request has been prioritized. Dennis is reviewing your facility specs now.
          </p>
          <Link
            href="/"
            className="mt-8 inline-block rounded-lg border-2 border-[#22c55e] bg-[#22c55e]/10 px-6 py-3 text-sm font-medium text-[#22c55e] hover:bg-[#22c55e]/20 transition-colors"
          >
            Return to Home
          </Link>
        </div>
      </main>
    </div>
  );
}
