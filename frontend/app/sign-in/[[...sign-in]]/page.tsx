import Link from "next/link";

export default function SignInPage() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-[#0a0a0a] px-4 py-12">
      <Link href="/" className="mb-8">
        <img src="/logo.png" alt="CooledAI Logo" style={{ height: "80px", width: "auto" }} className="block" />
      </Link>
      <div className="rounded-xl border border-white/10 bg-[#141414] p-8 max-w-sm w-full text-center">
        <p className="rounded-full border border-[#22c55e]/40 bg-[#22c55e]/10 px-3 py-1 text-xs font-medium text-[#22c55e] inline-block mb-4">
          Demo Mode
        </p>
        <h1 className="text-xl font-semibold text-white mb-2">Sign in</h1>
        <p className="text-sm text-white/60 mb-6">
          Authentication is disabled. Use the portal to explore the demo.
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
