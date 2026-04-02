#!/bin/bash
# Run from your Mac (same network as nodes). Deploys to .100 and .101.
# Passwords: both nodes use <SSH_PASSWORD> (set in the .exp files).
set -e
cd "$(dirname "$0")/.."
echo "=== Deploying to node .100 (Pilot) ==="
/usr/bin/expect scripts/deploy_to_100.exp
echo ""
echo "=== Deploying to node .101 (Control) ==="
/usr/bin/expect scripts/deploy_to_101.exp
echo ""
echo "Done. Both nodes updated."
