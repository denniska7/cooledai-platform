#!/usr/bin/env python3
"""
CooledAI Thermal Workload Benchmark
====================================
Generates a mixed workload via Ollama (Llama 3.2) that combines:

  PREDICTABLE phases  — lets the CooledAI model learn and pre-cool
  UNPREDICTABLE phases — tests whether the model generalises

Run identically on pilot (optimised) and control (traditional) nodes
to measure efficiency delta.

Every 3 cycles (9 hours) the phase order is shuffled and durations
re-randomized (+-30%), so the macro pattern is never the same twice.
The first 3 cycles use the default order below.

Phases (60-minute default cycle):
  1. WARM-UP        (5 min)  — steady light inference, predictable
  2. RAMP-UP        (5 min)  — linearly increasing concurrency, predictable
  3. SUSTAINED HIGH (8 min)  — constant heavy load, predictable
  4. CHAOS BURST    (7 min)  — random spikes & pauses, unpredictable
  5. COOL-DOWN      (3 min)  — idle / very light, predictable
  6. SAWTOOTH       (7 min)  — repeating 90s ramp + drop, predictable
  7. RANDOM WALK    (8 min)  — brownian-motion concurrency, unpredictable
  8. PULSE TRAIN    (5 min)  — 30s on / 30s off square wave, predictable
  9. STORM          (7 min)  — poisson-arrival heavy prompts, unpredictable
 10. TAPER          (5 min)  — exponential decay to idle, predictable

Metrics are logged to stdout as JSON lines for post-analysis.

Usage:
  python3 thermal_workload_benchmark.py [--cycles N] [--model MODEL] [--port PORT]
  # Runs indefinitely by default (cycles=0). Ctrl-C or kill to stop.

Requirements:
  - Ollama running (`ollama serve`)
  - Model pulled (`ollama pull llama3.2:3b`)
  - Python 3.8+ (stdlib only)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_PORTS = [11434, 11435]  # one Ollama instance per GPU
CYCLE_MINUTES = 60  # one full cycle

# Prompt pools — varying complexity drives varying GPU load
LIGHT_PROMPTS = [
    "What is 2+2?",
    "Name three colors.",
    "Say hello in French.",
    "What day comes after Monday?",
    "Is water wet?",
]

MEDIUM_PROMPTS = [
    "Explain how a heat exchanger works in 3 sentences.",
    "Write a Python function that checks if a number is prime.",
    "Summarize the water cycle in 50 words.",
    "What are the main differences between TCP and UDP?",
    "Describe the greenhouse effect briefly.",
]

HEAVY_PROMPTS = [
    "Write a detailed 200-word essay on the thermodynamics of data center cooling, "
    "covering conduction, convection, and radiation heat transfer mechanisms.",
    "Implement a complete binary search tree in Python with insert, delete, search, "
    "and in-order traversal methods. Include docstrings.",
    "Explain the physics behind fan affinity laws and how they relate to energy "
    "consumption in HVAC systems. Derive the cubic power relationship.",
    "Write a 150-word technical analysis of liquid cooling versus air cooling for "
    "high-density GPU racks, considering PUE, capex, and failure modes.",
    "Describe the architecture of a physics-informed neural network for thermal "
    "prediction, including loss function components and training strategy.",
]

# Longer context prompts that maximize GPU memory + compute
STRESS_PROMPTS = [
    "You are a data center thermal engineer. Write a comprehensive 500-word report "
    "analyzing the following scenario: A 42U rack containing 4 NVIDIA GB200 GPUs at "
    "1200W each has lost its rear-door heat exchanger. The ambient temperature is 35C. "
    "Calculate the expected temperature rise, time to thermal throttling, and recommend "
    "an emergency mitigation plan. Show your calculations step by step.",
    "Write a complete Python module (300+ lines) that implements a PID controller for "
    "data center cooling. Include: the PID class with anti-windup, a simulator for "
    "thermal dynamics, a logging system, and unit tests. Use type hints throughout.",
    "Analyze the energy efficiency tradeoffs of running AI inference workloads at "
    "different GPU utilization levels (25%, 50%, 75%, 100%). Consider: power draw "
    "curves, thermal throttling effects, fan power consumption (cubic law), and total "
    "cost of ownership. Write 400 words with specific numbers and formulas.",
]


# ---------------------------------------------------------------------------
# Ollama client (stdlib only)
# ---------------------------------------------------------------------------

def ollama_generate(
    prompt: str,
    model: str,
    port: int,
    max_tokens: int = 256,
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    """Fire a generate request and return timing + token stats."""
    url = f"http://localhost:{port}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.7},
    }).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode())
        elapsed = time.monotonic() - t0
        return {
            "ok": True,
            "elapsed_s": round(elapsed, 3),
            "eval_count": body.get("eval_count", 0),
            "eval_duration_ns": body.get("eval_duration", 0),
            "prompt_eval_count": body.get("prompt_eval_count", 0),
            "tokens_per_sec": round(
                body.get("eval_count", 0)
                / max(1e-9, body.get("eval_duration", 1) / 1e9),
                1,
            ),
        }
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_s": round(time.monotonic() - t0, 3),
            "error": str(exc)[:200],
        }


# ---------------------------------------------------------------------------
# Workload executor
# ---------------------------------------------------------------------------

_stats_lock = threading.Lock()
_stats: Dict[str, Any] = {
    "requests": 0,
    "failures": 0,
    "total_tokens": 0,
    "total_elapsed": 0.0,
}


def _run_one(prompt: str, model: str, port: int, max_tokens: int, phase: str) -> None:
    """Execute a single inference and record stats."""
    result = ollama_generate(prompt, model, port, max_tokens)
    with _stats_lock:
        _stats["requests"] += 1
        if result["ok"]:
            _stats["total_tokens"] += result.get("eval_count", 0)
            _stats["total_elapsed"] += result["elapsed_s"]
        else:
            _stats["failures"] += 1


def run_concurrent(
    prompts: List[str],
    concurrency: int,
    model: str,
    ports: List[int],
    max_tokens: int = 256,
    phase: str = "",
) -> None:
    """Fire `concurrency` prompts on EVERY GPU port in parallel.

    Each port gets the same prompts at the same concurrency so both GPUs
    (and both nodes) experience identical load.
    """
    threads = []
    for port in ports:
        for i in range(concurrency):
            prompt = prompts[i % len(prompts)]
            t = threading.Thread(
                target=_run_one,
                args=(prompt, model, port, max_tokens, phase),
                daemon=True,
            )
            threads.append(t)
            t.start()
    for t in threads:
        t.join(timeout=180)


def log_metric(phase: str, detail: str = "") -> None:
    """Emit a JSON-lines metric record."""
    with _stats_lock:
        snap = dict(_stats)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "detail": detail,
        "requests": snap["requests"],
        "failures": snap["failures"],
        "total_tokens": snap["total_tokens"],
        "avg_latency_s": round(
            snap["total_elapsed"] / max(1, snap["requests"] - snap["failures"]), 3
        ),
    }
    print(json.dumps(record), flush=True)


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def phase_warmup(model: str, ports: List[int], duration_s: float) -> None:
    """PREDICTABLE: Alternating 30s bursts and 30s idle — gradual warm-up."""
    end = time.monotonic() + duration_s
    on = True
    while time.monotonic() < end:
        seg_end = time.monotonic() + 30.0
        while time.monotonic() < min(seg_end, end):
            if on:
                run_concurrent(MEDIUM_PROMPTS, 4, model, ports, max_tokens=256, phase="warmup")
                log_metric("warmup", "4 medium burst")
            else:
                log_metric("warmup", "idle")
                time.sleep(min(10.0, max(0, min(seg_end, end) - time.monotonic())))
        on = not on


def phase_ramp_up(model: str, ports: List[int], duration_s: float) -> None:
    """PREDICTABLE: Staircase ramp 0->10 with 20s idle gaps between steps."""
    steps = 5
    step_s = duration_s / steps
    for i in range(steps):
        conc = 2 * (i + 1)  # 2, 4, 6, 8, 10
        prompts = HEAVY_PROMPTS if conc <= 6 else STRESS_PROMPTS
        # 70% of step is load, 30% is idle
        load_end = time.monotonic() + step_s * 0.7
        while time.monotonic() < load_end:
            run_concurrent(prompts, conc, model, ports, max_tokens=512, phase="ramp_up")
            log_metric("ramp_up", f"concurrency={conc}")
        # Idle gap
        idle_dur = step_s * 0.3
        log_metric("ramp_up", f"idle gap after conc={conc}")
        time.sleep(min(idle_dur, max(0, (time.monotonic() + idle_dur) - time.monotonic())))


def phase_sustained_high(model: str, ports: List[int], duration_s: float) -> None:
    """PREDICTABLE: Maximum blast — 10 concurrent stress, zero sleep."""
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        run_concurrent(STRESS_PROMPTS, 10, model, ports, max_tokens=1024, phase="sustained")
        log_metric("sustained_high", "10 stress MAX BLAST")
        # No sleep — back-to-back max load


def phase_chaos_burst(model: str, ports: List[int], duration_s: float) -> None:
    """UNPREDICTABLE: Violent spikes (2-12 conc) followed by total silence."""
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        # Random burst
        burst_size = random.randint(4, 12)
        pool = random.choice([HEAVY_PROMPTS, STRESS_PROMPTS, STRESS_PROMPTS])
        max_tok = random.choice([384, 512, 768, 1024])
        burst_duration = random.uniform(10.0, 40.0)
        burst_end = time.monotonic() + burst_duration
        while time.monotonic() < min(burst_end, end):
            run_concurrent(pool, burst_size, model, ports, max_tokens=max_tok, phase="chaos")
            log_metric("chaos_burst", f"burst={burst_size} tokens={max_tok}")
        # Then total silence
        silence = random.uniform(15.0, 45.0)
        log_metric("chaos_burst", f"SILENCE for {silence:.0f}s")
        remaining = end - time.monotonic()
        if remaining > 0:
            time.sleep(min(silence, remaining))


def phase_cooldown(model: str, ports: List[int], duration_s: float) -> None:
    """PREDICTABLE: Complete idle — zero GPU work, let temps drop."""
    end = time.monotonic() + duration_s
    log_metric("cooldown", "TOTAL IDLE — zero GPU load")
    while time.monotonic() < end:
        remaining = end - time.monotonic()
        if remaining > 0:
            time.sleep(min(15.0, remaining))
        log_metric("cooldown", "still idle")


def phase_sawtooth(model: str, ports: List[int], duration_s: float) -> None:
    """PREDICTABLE: 60s blast to max then 30s total idle, repeating."""
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        # 60s full blast
        blast_end = time.monotonic() + 60.0
        while time.monotonic() < min(blast_end, end):
            conc = 10
            run_concurrent(STRESS_PROMPTS, conc, model, ports, max_tokens=768, phase="sawtooth")
            log_metric("sawtooth", f"BLAST conc={conc}")
        # 30s total idle
        idle_end = time.monotonic() + 30.0
        log_metric("sawtooth", "IDLE drop")
        while time.monotonic() < min(idle_end, end):
            time.sleep(min(10.0, max(0, min(idle_end, end) - time.monotonic())))


def phase_random_walk(model: str, ports: List[int], duration_s: float) -> None:
    """UNPREDICTABLE: Wild swings between 0 and max load."""
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        # Coin flip: heavy burst or total idle
        if random.random() < 0.6:  # 60% chance of load
            conc = random.randint(4, 12)
            pool = random.choice([HEAVY_PROMPTS, STRESS_PROMPTS])
            max_tok = random.randint(384, 1024)
            burst_dur = random.uniform(15.0, 60.0)
            burst_end = time.monotonic() + burst_dur
            while time.monotonic() < min(burst_end, end):
                run_concurrent(pool, conc, model, ports, max_tokens=max_tok, phase="random_walk")
                log_metric("random_walk", f"BURST conc={conc}")
        else:  # 40% chance of idle
            idle_dur = random.uniform(20.0, 45.0)
            log_metric("random_walk", f"IDLE for {idle_dur:.0f}s")
            remaining = end - time.monotonic()
            if remaining > 0:
                time.sleep(min(idle_dur, remaining))


def phase_pulse_train(model: str, ports: List[int], duration_s: float) -> None:
    """PREDICTABLE: Sharp square wave — 45s MAX blast / 30s TOTAL idle."""
    end = time.monotonic() + duration_s
    on = True
    while time.monotonic() < end:
        if on:
            seg_end = time.monotonic() + 45.0
            while time.monotonic() < min(seg_end, end):
                run_concurrent(
                    STRESS_PROMPTS, 10, model, ports, max_tokens=768, phase="pulse_on"
                )
                log_metric("pulse_train", "ON — 10 stress MAX")
        else:
            idle_dur = 30.0
            log_metric("pulse_train", "OFF — TOTAL IDLE")
            remaining = end - time.monotonic()
            if remaining > 0:
                time.sleep(min(idle_dur, remaining))
        on = not on


def phase_storm(model: str, ports: List[int], duration_s: float) -> None:
    """UNPREDICTABLE: Violent random bursts with random idle gaps."""
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        # Random heavy burst
        conc = random.randint(6, 12)
        pool = random.choice([STRESS_PROMPTS, STRESS_PROMPTS, HEAVY_PROMPTS])
        max_tok = random.choice([512, 768, 1024])
        burst_dur = random.uniform(20.0, 50.0)
        burst_end = time.monotonic() + burst_dur
        while time.monotonic() < min(burst_end, end):
            run_concurrent(pool, conc, model, ports, max_tokens=max_tok, phase="storm")
            log_metric("storm", f"STORM conc={conc}")
        # Random idle gap
        gap = random.uniform(10.0, 40.0)
        log_metric("storm", f"eye-of-storm IDLE {gap:.0f}s")
        remaining = end - time.monotonic()
        if remaining > 0:
            time.sleep(min(gap, remaining))


def phase_taper(model: str, ports: List[int], duration_s: float) -> None:
    """PREDICTABLE: Step-down from max to zero — 4 steps."""
    end = time.monotonic() + duration_s
    steps = [(10, 768), (6, 512), (3, 256), (0, 0)]  # (concurrency, tokens)
    step_s = duration_s / len(steps)
    for conc, max_tok in steps:
        step_end = time.monotonic() + step_s
        if conc == 0:
            log_metric("taper", "IDLE — final cooldown")
            while time.monotonic() < min(step_end, end):
                time.sleep(min(10.0, max(0, min(step_end, end) - time.monotonic())))
        else:
            prompts = STRESS_PROMPTS if conc >= 6 else MEDIUM_PROMPTS
            while time.monotonic() < min(step_end, end):
                run_concurrent(prompts, conc, model, ports, max_tokens=max_tok, phase="taper")
                log_metric("taper", f"step conc={conc}")


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

PHASES = [
    ("warmup",          phase_warmup,          5),
    ("ramp_up",         phase_ramp_up,         5),
    ("sustained_high",  phase_sustained_high,  8),
    ("chaos_burst",     phase_chaos_burst,     7),
    ("cooldown",        phase_cooldown,        3),
    ("sawtooth",        phase_sawtooth,        7),
    ("random_walk",     phase_random_walk,     8),
    ("pulse_train",     phase_pulse_train,     5),
    ("storm",           phase_storm,           7),
    ("taper",           phase_taper,           5),
]

assert sum(m for _, _, m in PHASES) == CYCLE_MINUTES

# How many cycles before shuffling phase order + re-randomizing durations
SHUFFLE_EVERY_N_CYCLES = 3


def _shuffle_phases(rng: random.Random) -> List[tuple]:
    """Shuffle phase order and randomize durations (keeping total = 60 min).

    Each phase gets a base duration +-30% (clamped to 2-min minimum), then
    the residual is distributed evenly so the cycle still totals 60 minutes.
    """
    pool = list(PHASES)
    rng.shuffle(pool)

    # Randomize individual durations
    raw = []
    for name, fn, base_min in pool:
        jitter = rng.uniform(-0.30, 0.30)
        new_min = max(2, round(base_min * (1.0 + jitter)))
        raw.append((name, fn, new_min))

    # Adjust to hit exactly CYCLE_MINUTES
    total = sum(m for _, _, m in raw)
    diff = CYCLE_MINUTES - total
    # Spread the difference across the longest phases first
    indices = sorted(range(len(raw)), key=lambda i: raw[i][2], reverse=True)
    for idx in indices:
        if diff == 0:
            break
        step = 1 if diff > 0 else -1
        name, fn, m = raw[idx]
        new_m = m + step
        if new_m >= 2:
            raw[idx] = (name, fn, new_m)
            diff -= step

    return raw


def run_cycle(
    model: str,
    ports: List[int],
    cycle_num: int,
    phase_order: List[tuple],
) -> None:
    """Execute one full 60-minute benchmark cycle with the given phase order."""
    phase_names = [name for name, _, _ in phase_order]
    print(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "cycle_start",
        "cycle": cycle_num,
        "model": model,
        "hostname": os.uname().nodename,
        "gpu_count": len(ports),
        "phase_order": phase_names,
    }), flush=True)

    for name, fn, minutes in phase_order:
        print(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "phase_start",
            "phase": name,
            "duration_min": minutes,
            "cycle": cycle_num,
        }), flush=True)
        fn(model, ports, minutes * 60.0)
        print(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "phase_end",
            "phase": name,
            "cycle": cycle_num,
        }), flush=True)

    print(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "cycle_end",
        "cycle": cycle_num,
    }), flush=True)


def wait_for_ollama(ports: List[int], timeout_s: float = 120.0) -> bool:
    """Block until ALL Ollama ports are responsive."""
    for port in ports:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            try:
                req = urllib.request.Request(f"http://localhost:{port}/api/tags")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                pass
            time.sleep(2)
        else:
            return False  # this port never came up
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CooledAI Thermal Workload Benchmark — mixed predictable/unpredictable GPU load",
    )
    parser.add_argument(
        "--cycles", type=int, default=0,
        help="Number of 60-minute cycles to run (0 = infinite, default). "
        "Phase order shuffles every 3 cycles (9 hours).",
    )
    parser.add_argument(
        "--model", default=os.environ.get("BENCHMARK_MODEL", DEFAULT_MODEL),
        help=f"Ollama model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--ports", default=os.environ.get("OLLAMA_PORTS", "11434,11435"),
        help="Comma-separated Ollama API ports, one per GPU (default: 11434,11435).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic workload across nodes (default: 42). "
        "Both nodes MUST use the same seed for identical workloads.",
    )
    args = parser.parse_args()

    ports = [int(p.strip()) for p in args.ports.split(",")]
    random.seed(args.seed)

    UNPREDICTABLE_PHASES = {"chaos_burst", "random_walk", "storm"}

    print(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "benchmark_start",
        "hostname": os.uname().nodename,
        "model": args.model,
        "gpu_ports": ports,
        "seed": args.seed,
        "cycles": args.cycles if args.cycles > 0 else "infinite",
        "shuffle_every": SHUFFLE_EVERY_N_CYCLES,
        "phases": [
            {"name": n, "minutes": m, "type": "unpredictable" if n in UNPREDICTABLE_PHASES else "predictable"}
            for n, _, m in PHASES
        ],
    }), flush=True)

    # Wait for all Ollama instances
    print(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "waiting_for_ollama",
        "ports": ports,
    }), flush=True)

    if not wait_for_ollama(ports):
        print(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "error",
            "message": f"Ollama not responding on all ports {ports} after 120s",
        }), flush=True)
        sys.exit(1)

    # Warm the model into GPU memory on EVERY instance
    for port in ports:
        print(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "warming_model",
            "model": args.model,
            "port": port,
        }), flush=True)
        warmup_result = ollama_generate("Hi", args.model, port, max_tokens=4, timeout_s=300)
        if not warmup_result["ok"]:
            print(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "error",
                "message": f"Model warmup failed on port {port}: {warmup_result.get('error', 'unknown')}",
            }), flush=True)
            sys.exit(1)
        print(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "model_ready",
            "model": args.model,
            "port": port,
            "tokens_per_sec": warmup_result.get("tokens_per_sec", 0),
        }), flush=True)

    # Run cycles — shuffle phase order every SHUFFLE_EVERY_N_CYCLES
    # Use seeded RNG so both nodes shuffle identically
    shuffle_rng = random.Random(args.seed + 1_000_000)
    current_order = list(PHASES)  # first 3 cycles use the default order
    cycle = 0
    max_cycles = args.cycles if args.cycles > 0 else float("inf")

    try:
        while cycle < max_cycles:
            cycle += 1

            # Every N cycles: shuffle order + randomize durations
            if cycle > 1 and (cycle - 1) % SHUFFLE_EVERY_N_CYCLES == 0:
                current_order = _shuffle_phases(shuffle_rng)
                phase_names = [n for n, _, _ in current_order]
                phase_mins = [m for _, _, m in current_order]
                print(json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "shuffle",
                    "cycle": cycle,
                    "new_order": phase_names,
                    "new_durations": phase_mins,
                    "total_min": sum(phase_mins),
                }), flush=True)

            run_cycle(args.model, ports, cycle, current_order)
    except KeyboardInterrupt:
        pass

    # Final summary
    with _stats_lock:
        final = dict(_stats)
    print(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "benchmark_complete",
        "hostname": os.uname().nodename,
        "total_cycles": cycle,
        "total_requests": final["requests"],
        "total_failures": final["failures"],
        "total_tokens": final["total_tokens"],
        "avg_latency_s": round(
            final["total_elapsed"] / max(1, final["requests"] - final["failures"]), 3
        ),
    }), flush=True)


if __name__ == "__main__":
    main()
