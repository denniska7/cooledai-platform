"use client";

import Link from "next/link";
import { useState } from "react";
import { BackendStatusDot } from "./BackendStatusDot";

const navLinks = [
  { href: "/", label: "Home" },
  { href: "/why", label: "Why CooledAI" },
  { href: "/optimization", label: "Optimization" },
  { href: "/implementation", label: "Implementation" },
  { href: "/portal", label: "Portal" },
];

export function NavBar() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-white/20 bg-black pt-[env(safe-area-inset-top)]">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6 sm:py-4">
        <Link href="/" className="flex items-center gap-2 text-lg font-medium tracking-tight text-white">
          <img src="/logo.png" alt="CooledAI Logo" className="h-8 w-auto sm:h-10" />
          <span>CooledAI</span>
        </Link>

        {/* Desktop nav */}
        <div className="hidden lg:flex items-center gap-6 xl:gap-8">
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="text-sm tracking-tight text-white/80 hover:text-white transition-colors"
            >
              {link.label}
            </Link>
          ))}
          <Link
            href="/#request-audit"
            className="rounded border border-white bg-white px-4 py-2.5 text-sm font-medium tracking-tight text-black transition-opacity hover:opacity-90 min-h-[44px] inline-flex items-center"
          >
            Get My Savings Roadmap
          </Link>
          <div className="flex items-center gap-2">
            <BackendStatusDot />
          </div>
        </div>

        {/* Mobile: hamburger + CTA */}
        <div className="flex lg:hidden items-center gap-2">
          <BackendStatusDot />
          <Link
            href="/#request-audit"
            className="rounded border border-white bg-white px-3 py-2 text-sm font-medium tracking-tight text-black transition-opacity hover:opacity-90 min-h-[44px] inline-flex items-center"
          >
            Get Roadmap
          </Link>
          <button
            type="button"
            onClick={() => setMobileOpen(true)}
            className="p-2 -mr-2 text-white/80 hover:text-white min-h-[44px] min-w-[44px] flex items-center justify-center"
            aria-label="Open menu"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          </button>
        </div>
      </div>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/80 z-40 lg:hidden"
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Mobile drawer */}
      <div
        className={`fixed top-0 right-0 z-50 h-full w-full max-w-sm bg-[#0a0a0a] border-l border-white/20 shadow-xl transform transition-transform duration-200 ease-out lg:hidden ${
          mobileOpen ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between p-4 border-b border-white/10">
          <span className="text-sm font-medium text-white/70">Menu</span>
          <button
            type="button"
            onClick={() => setMobileOpen(false)}
            className="p-2 -mr-2 text-white/80 hover:text-white min-h-[44px] min-w-[44px] flex items-center justify-center"
            aria-label="Close menu"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        <nav className="flex flex-col p-4 gap-1">
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              onClick={() => setMobileOpen(false)}
              className="px-4 py-3.5 text-base font-medium text-white/90 hover:text-white hover:bg-white/5 rounded-lg transition-colors min-h-[48px] flex items-center"
            >
              {link.label}
            </Link>
          ))}
          <Link
            href="/#request-audit"
            onClick={() => setMobileOpen(false)}
            className="mt-4 rounded-lg border border-white bg-white px-4 py-3.5 text-sm font-medium tracking-tight text-black text-center transition-opacity hover:opacity-90 min-h-[48px] flex items-center justify-center"
          >
            Get My Savings Roadmap
          </Link>
        </nav>
      </div>
    </nav>
  );
}
