"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function PortalDashboardRedirect() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/portal");
  }, [router]);
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0a0a0a]">
      <div className="h-6 w-6 rounded-full border-2 border-[#22c55e]/30 border-t-[#22c55e] animate-spin" />
    </div>
  );
}
