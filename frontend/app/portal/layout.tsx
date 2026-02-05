"use client";

import { SignedIn, SignedOut, RedirectToSignIn } from "@clerk/nextjs";
import { PortalSidebar } from "./dashboard/components/PortalSidebar";

const hasClerk = typeof process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY === "string" && process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY.length > 0;

export default function PortalLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  if (!hasClerk) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center p-6">
        <div className="rounded-xl border border-white/10 bg-[#141414] p-8 max-w-md text-center">
          <h1 className="text-lg font-semibold text-white">Customer Portal</h1>
          <p className="text-sm text-white/60 mt-2">
            Configure Clerk to enable sign-in. Add <code className="bg-white/10 px-1 rounded text-[#22c55e]">NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY</code> and <code className="bg-white/10 px-1 rounded text-[#22c55e]">CLERK_SECRET_KEY</code> to <code className="bg-white/10 px-1 rounded">.env.local</code>.
          </p>
          <p className="text-xs text-white/40 mt-4">See <code>.env.example</code> in the frontend folder.</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <SignedOut>
        <RedirectToSignIn />
      </SignedOut>
      <SignedIn>
        <div className="min-h-screen bg-[#0a0a0a] flex flex-col md:flex-row">
          <PortalSidebar />
          <main className="flex-1 overflow-auto">{children}</main>
        </div>
      </SignedIn>
    </>
  );
}
