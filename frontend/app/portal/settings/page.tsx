"use client";

import { useState } from "react";
import { PortalSidebar } from "../dashboard/components/PortalSidebar";
import { SafetyShieldToggle } from "../dashboard/components/SafetyShieldToggle";

export default function SafetyShieldSettingsPage() {
  const [safetyShield, setSafetyShield] = useState(true);

  return (
    <div
      className={`min-h-screen bg-transparent flex flex-col md:flex-row transition-all duration-500 min-h-[100dvh] ${
        safetyShield
          ? "ring-2 ring-[#00FFCC] ring-inset animate-shield-pulse"
          : "ring-2 ring-white ring-inset"
      }`}
    >
      <PortalSidebar />
      <main className="flex-1 overflow-auto p-6 md:p-8">
        <h1 className="text-2xl font-medium tracking-tight text-white mb-6">
          Safety Shield Settings
        </h1>
        <div className="max-w-xl space-y-6">
          <div className="rounded border border-white/20 bg-black/50 p-6">
            <SafetyShieldToggle enabled={safetyShield} onToggle={setSafetyShield} />
            <p className="mt-4 text-sm text-white/50 leading-relaxed">
              When enabled, the AI monitors thermal conditions and can pre-cool or
              ramp cooling before heat spikes. The cyan border indicates active
              protection.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
