#!/usr/bin/env python3
"""
ST550 Comparative Demo — Llama Workload Scheduler

Runs on BOTH servers (.100 Pilot and .101 Control) with IDENTICAL workload:
  - Wall-clock aligned: LIGHT at :00, :05, :10... HEAVIER at :00, :30 DIFFICULT at :00 (UTC)
  - Same prompt every run: cycle index derived from timestamp so both nodes pick the same prompt

Each tier runs 2 concurrent Ollama generate requests (assumes 2 GPUs / can
utilize both P2000s with parallel inference).

Writes to /tmp/cooledai_inference_load while inference is active:
  - "1.0" when any tier execution is running
  - "0.0" when idle

The predictive engine (run only on the pilot node) reads that file.

Ollama contract:
  POST {OLLAMA_URL}/api/generate
  JSON: { model, prompt, stream:false, options:{...} }
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Any, List, Optional

import psutil  # used only for "CPU pressure" logging / sanity fallback
import requests


# ----------------------------
# Hardcoded variables
# ----------------------------
LOAD_FILE = os.environ.get("COOLEDAI_LOAD_FILE", "/tmp/cooledai_inference_load")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
# When set (e.g. "11434,11435"), send one request per URL to spread load across GPUs.
# Requires two Ollama instances: CUDA_VISIBLE_DEVICES=0 ollama serve & CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 ollama serve
OLLAMA_SPREAD_URLS = os.environ.get("OLLAMA_SPREAD_URLS", "")

POLL_INTERVAL_SEC = float(os.environ.get("SCHEDULER_POLL_SEC", "5"))
THREADS_PER_BATCH = int(os.environ.get("THREADS_PER_BATCH", "2"))  # 2 GPUs

REQUEST_TIMEOUT_SEC = float(os.environ.get("OLLAMA_TIMEOUT_SEC", "180"))

LIGHT_EVERY_SEC = 5 * 60
HEAVIER_EVERY_SEC = 30 * 60
DIFFICULT_EVERY_SEC = 60 * 60


# ----------------------------
# Prompts (minimal + tier-specific)
# ----------------------------
LIGHT_PROMPTS = [
    "Define PUE in one paragraph. Keep it concise.",
    "Explain cache invalidation in one sentence.",
    "What is idempotency? Provide a practical example.",
    "Explain rate limiting briefly.",
]

HEAVIER_PROMPTS = [
    "Design a fault-tolerant database replication strategy for a multi-region deployment. Include consistency models and failure scenarios.",
    "Compare REST, GraphQL, and gRPC for building APIs. When would you choose each?",
    "Describe how Kubernetes handles pod scheduling, resource limits, and autoscaling. Include a practical example.",
]

DIFFICULT_PROMPTS = [
    "Design an audit logging system that is tamper-proof and queryable at scale. Include schema, indexing strategy, retention policy, and a failure-mode analysis.",
    "Explain the trade-offs between synchronous and asynchronous programming in distributed systems. Include concrete examples and failure scenarios.",
    "Design a system for real-time anomaly detection on sensor telemetry with strict latency targets. Provide a pipeline and evaluation strategy.",
]


def _write_load(val: float) -> None:
    try:
        with open(LOAD_FILE, "w") as f:
            f.write(str(float(val)))
    except OSError:
        # best-effort; telemetry should still upload even if we can't write load file
        pass


def _ollama_generate(
    session: requests.Session,
    prompt: str,
    *,
    num_predict: int,
    temperature: float,
    base_url: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Best-effort call to Ollama generate endpoint.
    base_url: override (e.g. http://localhost:11435 for second GPU).
    """
    url_base = (base_url or OLLAMA_URL).rstrip("/")
    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }

    url = f"{url_base}/api/generate"
    resp = session.post(url, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    if resp.status_code != 200:
        return None
    try:
        return resp.json()
    except Exception:
        return None


def _prompts_for_cycle(prompts: List[str], cycle: int, k: int) -> List[str]:
    """Deterministic prompt selection so both nodes run the same prompts. k = THREADS_PER_BATCH."""
    n = len(prompts)
    if not n or k <= 0:
        return []
    return [prompts[(cycle + i) % n] for i in range(min(k, n))]


def _spread_urls() -> List[str]:
    """Parse OLLAMA_SPREAD_URLS into list of base URLs for load distribution across GPUs."""
    if not OLLAMA_SPREAD_URLS.strip():
        return []
    urls = []
    for part in OLLAMA_SPREAD_URLS.split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            urls.append(f"http://localhost:{part}")
        else:
            urls.append(part)
    return urls


def _run_tier(
    session: requests.Session,
    tier_name: str,
    prompts: List[str],
    run_ts: float,
    interval_sec: float,
    *,
    num_predict: int,
    temperature: float,
) -> None:
    """
    Run one tier execution with deterministic prompts (same on both nodes).
    run_ts = wall-clock boundary time for this run; cycle = run_ts // interval_sec.
    """
    cycle = int(run_ts // interval_sec)
    chosen = _prompts_for_cycle(prompts, cycle, THREADS_PER_BATCH)
    if not chosen:
        return

    cpu_pct = psutil.cpu_percent(interval=0.1)
    print(
        f"[{tier_name}] starting at {datetime.utcnow().isoformat()}Z "
        f"cycle={cycle} threads={len(chosen)} model={OLLAMA_MODEL} cpu%~{cpu_pct:.0f}"
    )

    _write_load(1.0)
    try:
        spread = _spread_urls()
        # When spread URLs set, send one request per URL to distribute across GPUs (avoids GPU0-only load).
        base_urls = spread[: len(chosen)] if spread else [None] * len(chosen)

        results: list[Optional[dict[str, Any]]] = [None] * len(chosen)
        exc: list[Optional[Exception]] = [None] * len(chosen)

        def worker(i: int, prompt: str, base_url: Optional[str]) -> None:
            try:
                results[i] = _ollama_generate(
                    session,
                    prompt,
                    num_predict=num_predict,
                    temperature=temperature,
                    base_url=base_url,
                )
            except Exception as e:
                exc[i] = e

        threads: list[threading.Thread] = []
        for i, prompt in enumerate(chosen):
            t = threading.Thread(
                target=worker,
                args=(i, prompt, base_urls[i] if i < len(base_urls) else None),
                daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        ok = sum(1 for r in results if r is not None)
        spread_info = f" spread={len(spread)}" if spread else ""
        print(f"[{tier_name}] done. ok={ok}/{len(chosen)}{spread_info} load_file=1.0")
    finally:
        _write_load(0.0)


def _next_aligned(now: float, interval_sec: float) -> float:
    """Next wall-clock boundary (epoch-aligned) so both nodes run at the same time."""
    return (int(now // interval_sec) + 1) * interval_sec


def main() -> None:
    spread = _spread_urls()
    spread_info = f" OLLAMA_SPREAD_URLS={len(spread)} URLs" if spread else ""
    print(
        "[scheduler] starting (identical workload: wall-clock aligned + deterministic prompts). "
        f"OLLAMA_URL={OLLAMA_URL} OLLAMA_MODEL={OLLAMA_MODEL} "
        f"THREADS_PER_BATCH={THREADS_PER_BATCH}{spread_info} "
        f"intervals: light={LIGHT_EVERY_SEC}s heavier={HEAVIER_EVERY_SEC}s difficult={DIFFICULT_EVERY_SEC}s"
    )

    now = time.time()
    next_light = _next_aligned(now, LIGHT_EVERY_SEC)
    next_heavier = _next_aligned(now, HEAVIER_EVERY_SEC)
    next_difficult = _next_aligned(now, DIFFICULT_EVERY_SEC)

    with requests.Session() as session:
        while True:
            now = time.time()

            # Run tiers in priority order if multiple are due (same order on both nodes).
            if now >= next_difficult:
                _run_tier(
                    session,
                    "DIFFICULT",
                    DIFFICULT_PROMPTS,
                    next_difficult,
                    DIFFICULT_EVERY_SEC,
                    num_predict=450,
                    temperature=0.7,
                )
                next_difficult = _next_aligned(now, DIFFICULT_EVERY_SEC)
                continue

            if now >= next_heavier:
                _run_tier(
                    session,
                    "HEAVIER",
                    HEAVIER_PROMPTS,
                    next_heavier,
                    HEAVIER_EVERY_SEC,
                    num_predict=220,
                    temperature=0.55,
                )
                next_heavier = _next_aligned(now, HEAVIER_EVERY_SEC)
                continue

            if now >= next_light:
                _run_tier(
                    session,
                    "LIGHT",
                    LIGHT_PROMPTS,
                    next_light,
                    LIGHT_EVERY_SEC,
                    num_predict=80,
                    temperature=0.25,
                )
                next_light = _next_aligned(now, LIGHT_EVERY_SEC)
                continue

            time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()

