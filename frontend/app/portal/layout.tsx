"use client";

import { PortalSidebar } from "./dashboard/components/PortalSidebar";

export default function PortalLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-[#0a0a0a] flex flex-col md:flex-row">
      <PortalSidebar />
      <main className="flex-1 overflow-auto">{children}</main>
    </div>
  );
}
