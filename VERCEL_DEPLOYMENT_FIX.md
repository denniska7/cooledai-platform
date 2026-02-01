# Vercel Deployment Not Updating – Troubleshooting Guide

Your git push succeeded (commit `498e5fb`). If cooledai.com still shows old content, follow these steps.

## 1. Set Root Directory (Most Likely Fix)

Your Next.js app lives in `frontend/`, but Vercel may be building from the repo root (where there’s no `package.json`).

**Fix:**
1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Open your **cooledai** (or cooledai-platform) project
3. **Settings** → **General** → **Build & Development Settings**
4. Find **Root Directory**
5. Click **Edit** and set it to: `frontend`
6. Click **Save**
7. **Deployments** → **Redeploy** the latest deployment (or push a small change to trigger a new build)

## 2. Check Build Status

1. **Deployments** tab in Vercel
2. Open the latest deployment
3. If it’s **Failed**, open the build logs
4. Common causes:
   - Root Directory wrong → build can’t find `package.json`
   - Missing env vars (e.g. `NEXT_PUBLIC_API_URL`)
   - Node/npm version issues

## 3. Production vs Preview

- **Production**: `cooledai.com` (from `main`)
- **Preview**: `*.vercel.app` URLs for non‑main branches

Confirm you’re checking the production URL and that the latest deployment is assigned to production.

## 4. Cache

- Hard refresh: `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Windows)
- Or open in an incognito/private window

## 5. Production Branch

In **Settings** → **Git** → **Production Branch**, ensure it’s `main` (or the branch you push to).

---

**Summary:** Set **Root Directory** to `frontend` in Vercel, then redeploy. That usually fixes “pushed but site not updating” for monorepos.
