#!/bin/bash
# Vercel 404 Rescue - Force Next.js framework and frontend root directory

set -e

git add vercel.json frontend/package.json
git commit -m "Fix: Force Next.js framework and frontend root directory"
git push origin main
vercel --prod --force
