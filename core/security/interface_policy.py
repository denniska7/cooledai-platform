"""CooledAI Hardware Interface Whitelist Enforcement.

This module enforces the architectural privacy guarantee that CooledAI
operates EXCLUSIVELY at the BMC/IPMI out-of-band management layer and
never touches the host OS, hypervisor, filesystem, network data plane,
or running processes.

Every external command issued by CooledAI must pass through
``validate_command()`` before execution. Commands outside the strict
whitelist raise ``PrivacyBoundaryViolation``, which halts the current
agent tick and writes to the security audit log.

The whitelist is intentionally narrow and cannot be widened at runtime.
Any change requires a code review and a new software release.
"""

from __future__ import annotations

import ipaddress
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, List, Optional, Sequence, Tuple

logger = logging.getLogger("cooledai.security.interface_policy")


# ── Exceptions ──────────────────────────────────────────────────
class PrivacyBoundaryViolation(Exception):
    """Raised when CooledAI attempts to access a disallowed interface.

    This exception is **fatal** for the current agent tick.  The caller
    must catch it, log a ``[SECURITY]`` line, and skip the tick entirely.
    """

    def __init__(self, command: str, reason: str) -> None:
        self.command = command
        self.reason = reason
        super().__init__(
            f"[SECURITY] privacy_boundary_violation: "
            f"command={command!r} reason={reason}"
        )


# ── Command types ───────────────────────────────────────────────
class CommandType(str, Enum):
    IPMI = "ipmi"
    NVIDIA_SMI = "nvidia_smi"
    ROCM_SMI = "rocm_smi"
    REDFISH = "redfish"
    SYSFS_READ = "sysfs_read"
    NETWORK = "network"


# ── Whitelist definitions ───────────────────────────────────────

# nvidia-smi flags that are SAFE (thermal/power telemetry only)
_NVIDIA_SAFE_FLAGS: FrozenSet[str] = frozenset({
    "-q", "--query",
    "-pl", "--power-limit",
    "--query-gpu",
    "--query-supported-clocks",
    "-ac", "--applications-clocks",
    "-rac", "--reset-applications-clocks",
    "-L", "--list-gpus",
    "--format",
})

# nvidia-smi --query-gpu fields that are SAFE
_NVIDIA_SAFE_QUERY_FIELDS: FrozenSet[str] = frozenset({
    "temperature.gpu",
    "temperature.memory",
    "power.draw",
    "power.limit",
    "power.min_limit",
    "power.max_limit",
    "power.default_limit",
    "utilization.gpu",
    "utilization.memory",
    "clocks.current.graphics",
    "clocks.current.memory",
    "clocks.max.graphics",
    "clocks.max.memory",
    "fan.speed",
    "gpu_name",
    "gpu_bus_id",
    "index",
    "count",
    "memory.total",
    "memory.used",
    "memory.free",
    "gpu_uuid",
})

# nvidia-smi flags that are BLOCKED (expose workload data)
_NVIDIA_BLOCKED_FLAGS: FrozenSet[str] = frozenset({
    "--query-compute-apps",
    "--query-accounted-apps",
    "-c", "--compute-mode",
    "--query-retired-pages",
    "-pm",  # persistence mode — OS-level
})

# Redfish endpoints that are SAFE
_REDFISH_SAFE_PATHS: Tuple[str, ...] = (
    "/redfish/v1/Systems/",      # → Thermal, Power subresources
    "/redfish/v1/Chassis/",      # → Sensors, Thermal, Power
    "/redfish/v1/Managers/",     # → Fan control, BMC status
    "/redfish/v1/UpdateService",  # → firmware version check only
)

# Redfish subresources that are SAFE
_REDFISH_SAFE_SUBRESOURCES: FrozenSet[str] = frozenset({
    "Thermal", "Power", "Sensors", "FanZones", "Fans",
    "PowerSupplies", "Temperatures", "NetworkProtocol",
    # Lenovo OEM thermal
    "Oem", "Lenovo",
})

# sysfs paths that are SAFE (hardware monitoring only)
_SYSFS_SAFE_PREFIXES: Tuple[str, ...] = (
    "/sys/class/hwmon/",
    "/sys/class/thermal/",
    "/sys/devices/system/cpu/cpu",   # cpuidle, cpufreq
    "/dev/ipmi",
    "/dev/cpu_dma_latency",
)

# OS-level commands that are ALWAYS BLOCKED
_BLOCKED_COMMANDS: FrozenSet[str] = frozenset({
    "ps", "top", "htop", "lsof", "netstat", "ss", "strace",
    "ltrace", "tcpdump", "wireshark", "nmap", "arp",
    "cat", "head", "tail", "less", "more", "grep", "find",
    "docker", "podman", "kubectl", "crictl",
    "journalctl", "dmesg",
    "who", "w", "last", "lastlog",
    "mount", "df", "du", "lsblk",
})


# ── Validation functions ────────────────────────────────────────

def validate_command(
    command: Sequence[str],
    *,
    command_type: CommandType,
    target_host: Optional[str] = None,
    bmc_addresses: Optional[FrozenSet[str]] = None,
    gateway_address: Optional[str] = None,
) -> None:
    """Validate a command against the hardware interface whitelist.

    Args:
        command: The command as a list of strings (e.g. ["nvidia-smi", "-q"]).
        command_type: The category of command being issued.
        target_host: For network commands, the destination IP/hostname.
        bmc_addresses: Set of known BMC OOB management IPs.
        gateway_address: The CooledAI gateway IP.

    Raises:
        PrivacyBoundaryViolation: If the command is outside the whitelist.
    """
    cmd_str = " ".join(str(c) for c in command)
    binary = str(command[0]).split("/")[-1] if command else ""

    # ── Block known OS-level commands ──
    if binary in _BLOCKED_COMMANDS:
        raise PrivacyBoundaryViolation(
            cmd_str,
            f"OS-level command '{binary}' is never permitted. "
            "CooledAI operates at the BMC layer only.",
        )

    # ── /proc access is always blocked ──
    for arg in command:
        arg_s = str(arg)
        if "/proc/" in arg_s or "/etc/" in arg_s:
            raise PrivacyBoundaryViolation(
                cmd_str,
                f"Filesystem path '{arg_s}' is outside the permitted "
                "hardware monitoring scope.",
            )

    if command_type == CommandType.NVIDIA_SMI:
        _validate_nvidia_smi(command, cmd_str)
    elif command_type == CommandType.REDFISH:
        _validate_redfish(cmd_str, target_host, bmc_addresses)
    elif command_type == CommandType.IPMI:
        _validate_ipmi(command, cmd_str, target_host, bmc_addresses)
    elif command_type == CommandType.SYSFS_READ:
        _validate_sysfs(command, cmd_str)
    elif command_type == CommandType.NETWORK:
        _validate_network(
            cmd_str, target_host, bmc_addresses, gateway_address
        )


def _validate_nvidia_smi(command: Sequence[str], cmd_str: str) -> None:
    """Validate nvidia-smi commands — thermal/power only, no process lists."""
    args = [str(a) for a in command[1:]]  # skip "nvidia-smi" or "sudo"

    # Skip "sudo" prefix
    if args and args[0] == "sudo":
        args = args[1:]
    if args and "nvidia-smi" in args[0]:
        args = args[1:]

    for arg in args:
        # Check blocked flags
        for blocked in _NVIDIA_BLOCKED_FLAGS:
            if arg == blocked or arg.startswith(blocked + "="):
                raise PrivacyBoundaryViolation(
                    cmd_str,
                    f"nvidia-smi flag '{arg}' exposes workload/process data. "
                    "Only thermal and power query flags are permitted.",
                )

        # Validate --query-gpu fields
        if arg.startswith("--query-gpu=") or (
            "query-gpu" in " ".join(str(a) for a in command)
            and "," in arg
            and not arg.startswith("--")
            and not arg.startswith("-")
        ):
            fields_str = arg.replace("--query-gpu=", "")
            fields = [f.strip() for f in fields_str.split(",")]
            for f in fields:
                if f and f not in _NVIDIA_SAFE_QUERY_FIELDS and f not in (
                    "csv", "noheader", "nounits",
                ):
                    raise PrivacyBoundaryViolation(
                        cmd_str,
                        f"nvidia-smi query field '{f}' is not in the "
                        "permitted telemetry field list.",
                    )


def _validate_redfish(
    cmd_str: str,
    target_host: Optional[str],
    bmc_addresses: Optional[FrozenSet[str]],
) -> None:
    """Validate Redfish calls — thermal/power endpoints on BMC only."""
    if bmc_addresses and target_host:
        if target_host not in bmc_addresses:
            raise PrivacyBoundaryViolation(
                cmd_str,
                f"Redfish target '{target_host}' is not a known BMC address. "
                "CooledAI only connects to out-of-band management interfaces.",
            )


def _validate_ipmi(
    command: Sequence[str],
    cmd_str: str,
    target_host: Optional[str],
    bmc_addresses: Optional[FrozenSet[str]],
) -> None:
    """Validate IPMI commands — BMC management only."""
    if bmc_addresses and target_host:
        if target_host not in bmc_addresses:
            raise PrivacyBoundaryViolation(
                cmd_str,
                f"IPMI target '{target_host}' is not a known BMC address.",
            )


def _validate_sysfs(command: Sequence[str], cmd_str: str) -> None:
    """Validate sysfs reads — hwmon, thermal, cpuidle only."""
    for arg in command:
        arg_s = str(arg)
        if arg_s.startswith("/sys/") or arg_s.startswith("/dev/"):
            if not any(arg_s.startswith(p) for p in _SYSFS_SAFE_PREFIXES):
                raise PrivacyBoundaryViolation(
                    cmd_str,
                    f"sysfs path '{arg_s}' is outside the permitted "
                    "hardware monitoring directories.",
                )


def _validate_network(
    cmd_str: str,
    target_host: Optional[str],
    bmc_addresses: Optional[FrozenSet[str]],
    gateway_address: Optional[str],
) -> None:
    """Validate network connections — BMC and gateway only."""
    if not target_host:
        return

    allowed: set[str] = set()
    if bmc_addresses:
        allowed.update(bmc_addresses)
    if gateway_address:
        allowed.add(gateway_address)

    # Allow link-local (BMC default)
    try:
        addr = ipaddress.ip_address(target_host)
        if addr.is_link_local:
            return
    except ValueError:
        pass

    if target_host not in allowed:
        raise PrivacyBoundaryViolation(
            cmd_str,
            f"Network target '{target_host}' is not a permitted destination. "
            "CooledAI may only connect to the BMC management network and "
            "the CooledAI gateway.",
        )


# ── Convenience wrappers ────────────────────────────────────────

def validate_nvidia_smi_command(command: Sequence[str]) -> None:
    """Shorthand for validating an nvidia-smi command."""
    validate_command(command, command_type=CommandType.NVIDIA_SMI)


def validate_sysfs_path(path: str) -> None:
    """Shorthand for validating a sysfs read."""
    validate_command([path], command_type=CommandType.SYSFS_READ)


def get_permitted_nvidia_query_fields() -> FrozenSet[str]:
    """Return the set of nvidia-smi --query-gpu fields CooledAI may use."""
    return _NVIDIA_SAFE_QUERY_FIELDS


def get_blocked_commands() -> FrozenSet[str]:
    """Return the set of OS-level commands CooledAI will never execute."""
    return _BLOCKED_COMMANDS


def get_privacy_boundary_summary() -> dict:
    """Return a machine-readable summary of the privacy boundary.

    Used by the attestation endpoint to document what CooledAI can
    and cannot access.
    """
    return {
        "permitted_interfaces": {
            "ipmi": "BMC out-of-band management via /dev/ipmi0 or ipmitool to BMC addresses",
            "nvidia_smi": {
                "safe_flags": sorted(_NVIDIA_SAFE_FLAGS),
                "safe_query_fields": sorted(_NVIDIA_SAFE_QUERY_FIELDS),
                "blocked_flags": sorted(_NVIDIA_BLOCKED_FLAGS),
            },
            "redfish": {
                "safe_path_prefixes": list(_REDFISH_SAFE_PATHS),
                "safe_subresources": sorted(_REDFISH_SAFE_SUBRESOURCES),
            },
            "sysfs": {
                "safe_prefixes": list(_SYSFS_SAFE_PREFIXES),
            },
        },
        "blocked_interfaces": {
            "os_commands": sorted(_BLOCKED_COMMANDS),
            "filesystem": ["/proc/*", "/etc/*", "/home/*", "/var/log/* (except cooledai)"],
            "network": "Only BMC OOB management IP and CooledAI gateway permitted",
        },
        "data_plane_access": False,
        "statement": (
            "CooledAI operates exclusively via out-of-band BMC interfaces. "
            "No tenant workload data, process information, network traffic, "
            "or filesystem contents are accessible or logged by this system."
        ),
    }
