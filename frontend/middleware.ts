import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

const isProtectedRoute = createRouteMatcher([
  "/portal(.*)",
  "/admin(.*)",
]);
const isPublicRoute = createRouteMatcher([
  "/",
  "/why",
  "/optimization",
  "/implementation",
  "/privacy",
  "/terms",
  "/cookies",
  "/sign-in(.*)",
  "/sign-up(.*)",
  "/portal/login",
]);

export default clerkMiddleware(async (auth, req) => {
  if (isProtectedRoute(req) && !isPublicRoute(req)) {
    await auth.protect();
  }
});

export const config = {
  matcher: [
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ico|woff2?|map)).*)",
    "/(api|trpc)(.*)",
  ],
};
