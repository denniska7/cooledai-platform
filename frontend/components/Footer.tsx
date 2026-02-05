"use client";

import Link from "next/link";

export function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer className="border-t border-white/20 bg-black">
      <div className="mx-auto max-w-6xl px-6 py-10">
        <div className="flex flex-col gap-8 sm:flex-row sm:items-center sm:justify-between">
          <div className="space-y-2">
            <Link href="/" className="inline-flex items-center gap-2">
              <img src="/logo.png" alt="CooledAI Logo" width={40} height={40} className="h-10 w-auto object-contain transition-all hover:opacity-80" />
              <p className="text-sm font-medium tracking-tight text-white">
                CooledAI
              </p>
            </Link>
            <p className="text-xs text-white/70 tracking-tight">
              Headquartered in Roseville, CA
            </p>
            <p className="text-xs text-white/50">
              © {year} CooledAI. All rights reserved.
            </p>
          </div>
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:gap-8">
            <div className="flex flex-wrap gap-6 text-sm">
              <Link
                href="/privacy"
                className="text-white/70 hover:text-white transition-colors tracking-tight"
              >
                Privacy Policy
              </Link>
              <Link
                href="/terms"
                className="text-white/70 hover:text-white transition-colors tracking-tight"
              >
                Terms of Service
              </Link>
              <Link
                href="/cookies"
                className="text-white/70 hover:text-white transition-colors tracking-tight"
              >
                Cookie Policy
              </Link>
              <Link
                href="/#request-audit"
                className="text-white/70 hover:text-white transition-colors tracking-tight"
              >
                Contact
              </Link>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
