#!/bin/bash
# CooledAI Edge Gateway — Health check for keepalived
#
# Returns 0 (healthy) or 1 (unhealthy) for keepalived VRRP tracking.
# If 3 consecutive checks fail, keepalived demotes this gateway.
#
# DEFERRED: Not wired into gateway startup. Deploy manually when HA is needed.

curl -sf http://localhost:8080/health > /dev/null 2>&1
exit $?
