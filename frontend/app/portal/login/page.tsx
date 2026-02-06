import Link from "next/link";

export default function PortalLoginPage() {
  return (
    <div className="min-h-screen bg-transparent flex flex-col items-center justify-center px-6">
      <div className="w-full max-w-sm text-center">
        <p className="rounded-full border border-[#22c55e]/40 bg-[#22c55e]/10 px-3 py-1 text-xs font-medium text-[#22c55e] inline-block mb-4">
          Demo Mode
        </p>
        <h1 className="text-2xl font-medium tracking-tight text-white mb-2">
          CooledAI Portal
        </h1>
        <p className="text-sm text-white/50 mb-8">
          Authentication is disabled. Open the portal to explore the demo.
        </p>
        <div className="flex flex-col gap-3">
          <Link
            href="/portal"
            className="rounded-lg border-2 border-[#22c55e] bg-[#22c55e]/10 px-4 py-3 text-sm font-medium text-[#22c55e] hover:bg-[#22c55e]/20 transition-colors"
          >
            Open Portal
          </Link>
          <Link
            href="/"
            className="rounded-lg border border-white/20 px-4 py-3 text-sm font-medium text-white/90 hover:bg-white/5 transition-colors"
          >
            Back to Home
          </Link>
        </div>
      </div>
    </div>
  );
}
