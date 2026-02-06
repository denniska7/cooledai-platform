import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

// Only these routes are public; everything else requires auth.
const isPublicRoute = createRouteMatcher([
  "/",
  "/why",
  "/optimization",
  "/implementation",
  "/contact",
  "/audit-request",
  "/privacy",
  "/terms",
  "/cookies",
  "/sign-in(.*)",
  "/sign-up(.*)",
  "/portal/login(.*)",
]);

export default clerkMiddleware(
  async (auth, req) => {
    if (!isPublicRoute(req)) {
      await auth.protect();
    }
  },
  { signInUrl: "/portal/login" }
);

export const config = {
  matcher: [
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ico|woff2?|map)).*)",
    "/(api|trpc)(.*)",
  ],
};
