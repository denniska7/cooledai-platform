"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "./context/AuthContext";

export default function PortalPage() {
  const router = useRouter();
  const { isAuthenticated } = useAuth();

  useEffect(() => {
    if (isAuthenticated) {
      router.replace("/portal/dashboard");
    } else {
      router.replace("/portal/login");
    }
  }, [isAuthenticated, router]);

  return (
    <div className="min-h-screen bg-transparent flex items-center justify-center">
      <div className="h-6 w-6 rounded-full border-2 border-white/30 border-t-white animate-spin" />
    </div>
  );
}
