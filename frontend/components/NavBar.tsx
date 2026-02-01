"use client";

import Link from "next/link";
import { BackendStatusDot } from "./BackendStatusDot";

export function NavBar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-white/20 bg-black">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link href="/" className="text-lg font-medium tracking-tight text-white">
          CooledAI
        </Link>
        <div className="flex items-center gap-8">
          <Link
            href="/"
            className="text-sm tracking-tight text-white/80 hover:text-white transition-colors"
          >
            Home
          </Link>
          <Link
            href="/why"
            className="text-sm tracking-tight text-white/80 hover:text-white transition-colors"
          >
            Why CooledAI?
          </Link>
          <Link
            href="/optimization"
            className="text-sm tracking-tight text-white/80 hover:text-white transition-colors"
          >
            Optimization
          </Link>
          <Link
            href="/#request-audit"
            className="rounded border border-white bg-white px-4 py-2 text-sm font-medium tracking-tight text-black transition-opacity hover:opacity-90"
          >
            Request Audit
          </Link>
          <div className="flex items-center gap-2">
            <BackendStatusDot />
          </div>
        </div>
      </div>
    </nav>
  );
}
