"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { UserButton } from "@clerk/nextjs";

const navItems = [
  { href: "/portal", label: "Overview" },
  { href: "/portal/facility-pulse", label: "Facility Pulse" },
  { href: "/portal/savings-roadmap", label: "Savings Roadmap" },
  { href: "/portal/billing", label: "Billing" },
];

export function PortalSidebar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  const sidebarContent = (
    <>
      <div className="p-6 border-b border-white/10">
        <Link
          href="/portal"
          className="flex items-center gap-2 text-lg font-semibold tracking-tight text-white"
          onClick={() => setMobileOpen(false)}
        >
          <img src="/logo.png" alt="CooledAI Logo" width="160" height="auto" className="h-10 w-auto object-contain transition-all hover:opacity-80" />
          CooledAI
        </Link>
        <span className="block text-xs text-white/50 mt-0.5">Customer Portal</span>
      </div>
      <nav className="flex-1 p-4 space-y-1 overflow-auto">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={() => setMobileOpen(false)}
              className={`block px-4 py-3 text-sm rounded-lg transition-colors ${
                isActive
                  ? "bg-[#22c55e]/15 text-[#22c55e] border border-[#22c55e]/30"
                  : "text-white/70 hover:text-white hover:bg-white/5 border border-transparent"
              }`}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="p-4 border-t border-white/10 flex items-center justify-between gap-2">
        <UserButton
          afterSignOutUrl="/"
          appearance={{
            elements: {
              avatarBox: "w-8 h-8",
            },
          }}
        />
        <span className="text-xs text-white/50 truncate">Account</span>
      </div>
    </>
  );

  return (
    <>
      <header className="md:hidden flex items-center justify-between px-4 py-4 border-b border-white/10 bg-[#0a0a0a] sticky top-0 z-20">
        <button
          onClick={() => setMobileOpen(true)}
          className="p-2 -ml-2 text-white/80 hover:text-white"
          aria-label="Open menu"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>
        <Link href="/portal" className="flex items-center gap-2 text-lg font-semibold tracking-tight text-white">
          <img src="/logo.png" alt="CooledAI Logo" style={{ height: "32px", width: "auto" }} className="block" />
          CooledAI
        </Link>
        <UserButton afterSignOutUrl="/" />
      </header>

      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/80 z-40 md:hidden"
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside
        className={`
          fixed md:relative inset-y-0 left-0 z-50
          w-64 min-w-[16rem] max-w-[85vw] md:w-56 md:min-w-[14rem]
          border-r border-white/10 bg-[#0a0a0a] flex flex-col
          transform transition-transform duration-200 ease-out
          md:transform-none
          pt-0 md:pt-0 pb-[env(safe-area-inset-bottom)]
          ${mobileOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"}
        `}
      >
        {sidebarContent}
      </aside>
    </>
  );
}
