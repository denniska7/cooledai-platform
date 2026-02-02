"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { AuthProvider, useAuth } from "./context/AuthContext";

function PortalAuthGuard({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, hydrated } = useAuth();
  const pathname = usePathname();
  const router = useRouter();

  useEffect(() => {
    if (!hydrated) return;
    const isLoginPage = pathname === "/portal/login";
    if (!isAuthenticated && !isLoginPage) {
      router.replace("/portal/login");
    } else if (isAuthenticated && isLoginPage) {
      router.replace("/portal/dashboard");
    }
  }, [isAuthenticated, hydrated, pathname, router]);

  if (!hydrated) {
    return (
      <div className="min-h-screen bg-transparent flex items-center justify-center">
        <div className="h-6 w-6 rounded-full border-2 border-white/30 border-t-white animate-spin" />
      </div>
    );
  }
  return <>{children}</>;
}

export default function PortalLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <AuthProvider>
      <PortalAuthGuard>{children}</PortalAuthGuard>
    </AuthProvider>
  );
}
