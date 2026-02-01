"use client";

import { BackendStatusDot } from "./BackendStatusDot";

export function NavBar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-white/10 bg-black/80 backdrop-blur-sm">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <a href="#" className="text-lg font-medium tracking-tight">
          CooledAI
        </a>
        <div className="flex items-center gap-3">
          <span className="text-xs text-white/50">System</span>
          <BackendStatusDot />
        </div>
      </div>
    </nav>
  );
}
