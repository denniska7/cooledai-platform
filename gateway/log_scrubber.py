"""
CooledAI Log Scrubber - Security

Scrubs sensitive IP addresses and credentials from logs
before transmission to cloud.
"""

import re
from typing import Optional

# Patterns to scrub
IP_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)
IPV6_PATTERN = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|"
    r"\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b|"
    r"\b(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}\b"
)
CREDENTIAL_PATTERNS = [
    (re.compile(r"password\s*=\s*['\"]?[^'\s\"]+['\"]?", re.I), "password=***"),
    (re.compile(r"auth_key\s*=\s*['\"]?[^'\s\"]+['\"]?", re.I), "auth_key=***"),
    (re.compile(r"priv_key\s*=\s*['\"]?[^'\s\"]+['\"]?", re.I), "priv_key=***"),
    (re.compile(r"community\s*=\s*['\"]?[^'\s\"]+['\"]?", re.I), "community=***"),
    (re.compile(r"api_key\s*=\s*['\"]?[^'\s\"]+['\"]?", re.I), "api_key=***"),
    (re.compile(r"token\s*=\s*['\"]?[^'\s\"]+['\"]?", re.I), "token=***"),
]


def scrub_log_message(msg: str) -> str:
    """
    Scrub IP addresses and credentials from log message.
    Call before transmitting logs to cloud.
    """
    if not msg:
        return msg
    out = msg
    out = IP_PATTERN.sub("[IP_REDACTED]", out)
    out = IPV6_PATTERN.sub("[IPV6_REDACTED]", out)
    for pat, repl in CREDENTIAL_PATTERNS:
        out = pat.sub(repl, out)
    return out
