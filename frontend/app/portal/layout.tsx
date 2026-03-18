"use client";

import { PortalSidebar } from "./dashboard/components/PortalSidebar";
import { useAuth } from "@clerk/nextjs";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function PortalLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { isLoaded, userId } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoaded) return;
    if (!userId) {
      // Debounce to prevent a brief `userId === null` flicker right after sign-in.
      const t = setTimeout(() => router.replace("/sign-in"), 400);
      return () => clearTimeout(t);
    }
  }, [isLoaded, userId, router]);

  if (!isLoaded) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
        <div className="animate-spin h-8 w-8 border-2 border-white/20 border-t-[#22c55e] rounded-full" />
      </div>
    );
  }

  if (!userId) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center text-white/70">
        Sign in required.
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] flex flex-col md:flex-row">
      <PortalSidebar />
      <main className="flex-1 overflow-auto">{children}</main>
    </div>
  );
}
