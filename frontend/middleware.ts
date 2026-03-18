import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

function passthrough(_req: NextRequest) {
  return NextResponse.next();
}

// NOTE: Clerk middleware removed to bypass auth entirely.
// This is intentionally a total passthrough so we can confirm
// the rest of the app (including the dashboard data flow).
export default function middleware(_req: NextRequest) {
  return passthrough(_req);
}

export const config = {
  matcher: [
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ico|woff2?|map)).*)",
    "/(api|trpc)(.*)",
  ],
};
