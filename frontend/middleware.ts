import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

const isPortalRoute = createRouteMatcher(["/portal(.*)"]);
const isPublicRoute = createRouteMatcher(["/", "/why", "/optimization", "/implementation", "/privacy", "/terms", "/cookies", "/sign-in(.*)", "/sign-up(.*)"]);

export default clerkMiddleware(async (auth, req) => {
  const hasClerk = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;
  if (hasClerk && isPortalRoute(req) && !isPublicRoute(req)) {
    await auth.protect();
  }
});

export const config = {
  matcher: [
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ico|woff2?|map)).*)",
    "/(api|trpc)(.*)",
  ],
};
