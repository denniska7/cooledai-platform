#!/usr/bin/env bash
# Push Clerk env vars from .env.local to Vercel Production.
# Usage: ./push-keys.sh [CLERK_SECRET_KEY]
# If omitted, CLERK_SECRET_KEY is read from .env.local.

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ ! -f .env.local ]; then
  echo "Error: .env.local not found in $DIR"
  exit 1
fi

get_val() { grep -E "^${1}=" .env.local | cut -d= -f2- | head -1; }

SECRET="${1:-$(get_val CLERK_SECRET_KEY)}"
if [ -z "$SECRET" ] || [ "$SECRET" = "sk_test_PLACEHOLDER" ]; then
  echo "Error: Run: ./push-keys.sh sk_test_YOUR_REAL_KEY"
  echo "Or set CLERK_SECRET_KEY in .env.local (from https://dashboard.clerk.com)"
  exit 1
fi

echo "Pushing Clerk env vars to Vercel Production..."
echo "$(get_val NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY)" | vercel env add NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY production
echo "$SECRET" | vercel env add CLERK_SECRET_KEY production
echo "$(get_val NEXT_PUBLIC_CLERK_SIGN_IN_URL)" | vercel env add NEXT_PUBLIC_CLERK_SIGN_IN_URL production
echo "$(get_val NEXT_PUBLIC_CLERK_SIGN_UP_URL)" | vercel env add NEXT_PUBLIC_CLERK_SIGN_UP_URL production
echo "Done. Redeploy for changes to take effect."
