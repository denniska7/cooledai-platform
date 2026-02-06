# Vercel deployment

- **Root Directory:** In Vercel project settings, set **Root Directory** to `frontend` (or leave empty if this repo is only the frontend).
- **Dependencies:** `package.json` must include `@clerk/nextjs` so `npm install` installs it. If the build fails with "Can't resolve '@clerk/nextjs'", ensure the committed `package.json` lists it and redeploy.
