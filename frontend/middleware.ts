import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

export const config = {
  matcher: [
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ico|woff2?|map)).*)",
    "/(api|trpc)(.*)",
  ],
};

// Protect all `/portal*` routes. Unauthenticated users get redirected to Clerk sign-in.
const isPortalRoute = createRouteMatcher(["/portal", "/portal/(.*)"]);

export default clerkMiddleware(async (auth, req) => {
  if (isPortalRoute(req)) {
    await auth.protect();
  }
}, { signInUrl: "/sign-in" });
