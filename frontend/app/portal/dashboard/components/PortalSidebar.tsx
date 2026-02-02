"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "../../context/AuthContext";

const navItems = [
  { href: "/portal/dashboard", label: "Overview" },
  { href: "/portal/thermal-map", label: "Thermal Map" },
  { href: "/portal/efficiency-logs", label: "Efficiency Logs" },
  { href: "/portal/settings", label: "Safety Shield Settings" },
];

export function PortalSidebar() {
  const pathname = usePathname();
  const { logout } = useAuth();
  const [mobileOpen, setMobileOpen] = useState(false);

  const sidebarContent = (
    <>
      <div className="p-6 border-b border-white/20">
        <Link
          href="/portal/dashboard"
          className="text-lg font-medium tracking-tight text-white"
          onClick={() => setMobileOpen(false)}
        >
          CooledAI
        </Link>
        <span className="block text-xs text-white/50 mt-0.5">Portal</span>
      </div>
      <nav className="flex-1 p-4 space-y-1 overflow-auto">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={() => setMobileOpen(false)}
              className={`block px-4 py-3 text-sm rounded transition-colors ${
                isActive
                  ? "bg-white/10 text-white border border-white/20"
                  : "text-white/70 hover:text-white hover:bg-white/5"
              }`}
            >
              {item.label}
            </Link>
          );
        })}
        <div className="mt-4 pt-4 border-t border-white/10">
          <Link
            href="/#request-audit"
            onClick={() => setMobileOpen(false)}
            className="flex items-center gap-2 px-4 py-3 text-sm text-white/60 hover:text-white rounded transition-colors"
          >
            <span>Join Beta</span>
            <span className="text-xs text-white/40">Coming soon</span>
          </Link>
        </div>
      </nav>
      <div className="p-4 border-t border-white/20">
        <button
          onClick={() => {
            logout();
            setMobileOpen(false);
          }}
          className="w-full px-4 py-2 text-sm text-white/60 hover:text-white border border-white/20 rounded transition-colors"
        >
          Sign out
        </button>
      </div>
    </>
  );

  return (
    <>
      {/* Mobile header with hamburger */}
      <header className="md:hidden flex items-center justify-between px-4 py-4 border-b border-white/20 bg-black sticky top-0 z-20 safe-area-inset-top">
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
        <Link href="/portal/dashboard" className="text-lg font-medium tracking-tight text-white">
          CooledAI
        </Link>
        <div className="w-10" />
      </header>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/80 z-40 md:hidden"
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar - drawer on mobile, fixed on desktop */}
      <aside
        className={`
          fixed md:relative inset-y-0 left-0 z-50
          w-64 min-w-[16rem] max-w-[85vw] md:w-56 md:min-w-[14rem]
          border-r border-white/20 bg-black flex flex-col
          transform transition-transform duration-200 ease-out
          md:transform-none
          pt-[env(safe-area-inset-top)] pb-[env(safe-area-inset-bottom)]
          ${mobileOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"}
        `}
      >
        {sidebarContent}
      </aside>
    </>
  );
}
