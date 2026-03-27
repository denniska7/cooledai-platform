#!/usr/bin/env bash
# test_fan_2800rpm.sh — IPMI fan control test for Pilot ST550
#
# Sets fans to 40% duty (~2800 RPM) and monitors GPU temps for 2 minutes.
# Aborts and reverts to auto if GPU exceeds 75°C.
#
# Usage: ssh cooledaiadmin@100.92.29.44 'bash -s' < scripts/test_fan_2800rpm.sh

set -euo pipefail

DUTY_HEX="0x28"   # 40% = 0x28
MAX_GPU_TEMP=75
MONITOR_INTERVAL=10
MONITOR_DURATION=120  # 2 minutes

revert_to_auto() {
    echo "[$(date +%H:%M:%S)] Reverting fans to automatic control..."
    ipmitool raw 0x3a 0x07 0x00 0x00
    echo "[$(date +%H:%M:%S)] Done — fans back to BMC auto."
}

trap revert_to_auto EXIT

echo "=== CooledAI Fan Test: 40% duty (~2800 RPM) ==="
echo "[$(date +%H:%M:%S)] Enabling manual fan control..."
ipmitool raw 0x3a 0x07 0x01 0x00

echo "[$(date +%H:%M:%S)] Setting fans to 40% duty (${DUTY_HEX})..."
ipmitool raw 0x3a 0x07 0x01 "$DUTY_HEX"

echo "[$(date +%H:%M:%S)] Monitoring GPU temps every ${MONITOR_INTERVAL}s for ${MONITOR_DURATION}s..."
echo ""

elapsed=0
while [ "$elapsed" -lt "$MONITOR_DURATION" ]; do
    gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    fan_rpm=$(ipmitool sdr type Fan 2>/dev/null | head -1 | awk '{print $NF}' || echo "?")

    echo "[$(date +%H:%M:%S)] GPU: ${gpu_temp}°C | Fan: ${fan_rpm} RPM | Elapsed: ${elapsed}s"

    if [ -n "$gpu_temp" ] && [ "$gpu_temp" -gt "$MAX_GPU_TEMP" ] 2>/dev/null; then
        echo ""
        echo "*** ABORT: GPU temp ${gpu_temp}°C exceeds ${MAX_GPU_TEMP}°C safety limit ***"
        exit 1  # trap will revert to auto
    fi

    sleep "$MONITOR_INTERVAL"
    elapsed=$((elapsed + MONITOR_INTERVAL))
done

echo ""
echo "[$(date +%H:%M:%S)] Test complete — GPU stayed below ${MAX_GPU_TEMP}°C."
echo "Fans will revert to auto on exit."
