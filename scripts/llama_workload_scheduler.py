#!/usr/bin/env python3
"""
ST550 Comparative Demo — Llama Workload Scheduler

Runs on BOTH servers (.100 Pilot and .101 Control) and continuously applies
three workload tiers on a fixed cadence:
  - Light:     every 5 minutes
  - Heavier:   every 30 minutes
  - Difficult: every 60 minutes

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
import random
import threading
import time
from datetime import datetime
from typing import Any, Optional

import psutil  # used only for "CPU pressure" logging / sanity fallback
import requests


# ----------------------------
# Hardcoded variables
# ----------------------------
LOAD_FILE = os.environ.get("COOLEDAI_LOAD_FILE", "/tmp/cooledai_inference_load")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

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
) -> Optional[dict[str, Any]]:
    """
    Best-effort call to Ollama generate endpoint.
    """
    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }

    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    resp = session.post(url, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    if resp.status_code != 200:
        return None
    try:
        return resp.json()
    except Exception:
        return None


def _run_tier(
    session: requests.Session,
    tier_name: str,
    prompts: list[str],
    *,
    num_predict: int,
    temperature: float,
) -> None:
    """
    Run one tier execution:
      - choose prompts
      - launch THREADS_PER_BATCH concurrent generations
      - wait for all
    """
    chosen = random.sample(prompts, k=min(len(prompts), THREADS_PER_BATCH))
    if not chosen:
        return

    cpu_pct = psutil.cpu_percent(interval=0.1)
    print(
        f"[{tier_name}] starting at {datetime.utcnow().isoformat()}Z "
        f"threads={len(chosen)} model={OLLAMA_MODEL} cpu%~{cpu_pct:.0f}"
    )

    _write_load(1.0)
    try:
        results: list[Optional[dict[str, Any]]] = [None] * len(chosen)
        exc: list[Optional[Exception]] = [None] * len(chosen)

        def worker(i: int, prompt: str) -> None:
            try:
                results[i] = _ollama_generate(
                    session,
                    prompt,
                    num_predict=num_predict,
                    temperature=temperature,
                )
            except Exception as e:
                exc[i] = e

        threads: list[threading.Thread] = []
        for i, prompt in enumerate(chosen):
            t = threading.Thread(target=worker, args=(i, prompt), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        ok = sum(1 for r in results if r is not None)
        print(f"[{tier_name}] done. ok={ok}/{len(chosen)} load_file=1.0")
    finally:
        _write_load(0.0)


def main() -> None:
    print(
        "[scheduler] starting. "
        f"OLLAMA_URL={OLLAMA_URL} OLLAMA_MODEL={OLLAMA_MODEL} "
        f"THREADS_PER_BATCH={THREADS_PER_BATCH} "
        f"intervals: light={LIGHT_EVERY_SEC}s heavier={HEAVIER_EVERY_SEC}s difficult={DIFFICULT_EVERY_SEC}s"
    )

    next_light = time.time()
    next_heavier = time.time()
    next_difficult = time.time()

    # If you want an immediate "warm start" rather than waiting a full interval,
    # set next_* = time.time().

    with requests.Session() as session:
        while True:
            now = time.time()

            # Run tiers in priority order if multiple are due.
            if now >= next_difficult:
                _run_tier(
                    session,
                    "DIFFICULT",
                    DIFFICULT_PROMPTS,
                    num_predict=450,
                    temperature=0.7,
                )
                next_difficult = now + DIFFICULT_EVERY_SEC
                continue

            if now >= next_heavier:
                _run_tier(
                    session,
                    "HEAVIER",
                    HEAVIER_PROMPTS,
                    num_predict=220,
                    temperature=0.55,
                )
                next_heavier = now + HEAVIER_EVERY_SEC
                continue

            if now >= next_light:
                _run_tier(
                    session,
                    "LIGHT",
                    LIGHT_PROMPTS,
                    num_predict=80,
                    temperature=0.25,
                )
                next_light = now + LIGHT_EVERY_SEC
                continue

            time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()

