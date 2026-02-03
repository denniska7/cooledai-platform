"use client";

import Link from "next/link";

const LINKEDIN_PLACEHOLDER = "https://www.linkedin.com/company/cooledai";

function LinkedInIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden
    >
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
    </svg>
  );
}

export function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer className="border-t border-white/20 bg-black">
      <div className="mx-auto max-w-6xl px-6 py-10">
        <div className="flex flex-col gap-8 sm:flex-row sm:items-center sm:justify-between">
          <div className="space-y-2">
            <p className="text-sm font-medium tracking-tight text-white">
              CooledAI LLC
            </p>
            <p className="text-xs text-white/60 tracking-tight">
              Roseville, CA
            </p>
            <p className="text-xs text-white/50">
              © {year} CooledAI LLC. All rights reserved.
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
            <a
              href={LINKEDIN_PLACEHOLDER}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-white/70 hover:text-white transition-colors"
              aria-label="CooledAI on LinkedIn"
            >
              <LinkedInIcon className="h-5 w-5" />
              <span className="text-sm tracking-tight">LinkedIn</span>
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
