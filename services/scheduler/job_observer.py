"""
CooledAI Job Observer - Upcoming High-Compute Load Prediction

Queries job schedulers (Slurm, Kubernetes) to identify racks about to receive
GPU training jobs. Used by OptimizationBrain for pre-emptive cooling.

Uses core/config/topology_map.yaml for node->rack and rack->cooling mappings.
Plug-and-play: update the YAML for any new facility—no code changes.
"""

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_logger = logging.getLogger("cooledai.job_observer")


@dataclass
class UpcomingJob:
    """A high-compute job about to start on a rack."""

    rack_id: str
    job_id: str
    seconds_until_start: float
    job_type: str = "gpu_training"  # e.g. gpu_training, inference
    node_id: Optional[str] = None
    partition: Optional[str] = None


class SlurmObserver:
    """
    Observes Slurm scheduler for pending jobs that will soon start on GPU nodes.

    Maps Slurm nodes to rack IDs via config or convention (e.g. node-rack-01 -> rack_01).
    Parses squeue/scontrol output for pending jobs with GPU reservations.
    """

    def __init__(
        self,
        node_to_rack_map: Optional[dict] = None,
        gpu_partitions: Optional[List[str]] = None,
    ):
        """
        Args:
            node_to_rack_map: Optional dict mapping node names to rack_id (e.g. {"node1": "rack_01"}).
                If None, uses heuristic: node name with "rack" -> rack_ID, else node name as rack.
            gpu_partitions: Partitions considered high-compute (e.g. ["gpu", "a100"]).
                If None, treats all pending jobs as potential GPU unless GRES contains gpu.
        """
        self.node_to_rack_map = node_to_rack_map or {}
        self.gpu_partitions = gpu_partitions or ["gpu", "a100", "h100", "a100x", "gpu-t4"]

    def _node_to_rack(self, node_name: str) -> str:
        """Map Slurm node name to rack ID."""
        if node_name in self.node_to_rack_map:
            return self.node_to_rack_map[node_name]
        # Heuristic: "node-rack-01" -> "rack_01", "gpu-node-3" -> "gpu-node-3"
        if "rack" in node_name.lower():
            parts = node_name.lower().split("-")
            for i, p in enumerate(parts):
                if p == "rack" and i + 1 < len(parts):
                    return f"rack_{parts[i + 1]}"
        return f"rack_{node_name}"

    def _is_gpu_job(self, partition: str, gres: Optional[str]) -> bool:
        """True if job is GPU/high-compute."""
        if partition and partition.lower() in [p.lower() for p in self.gpu_partitions]:
            return True
        if gres and "gpu" in (gres or "").lower():
            return True
        return False

    def get_upcoming_jobs(self, max_seconds: float = 60.0) -> List[UpcomingJob]:
        """
        Query Slurm for pending jobs that will start within max_seconds.

        Uses squeue for pending jobs and estimates start time from ST (scheduled time)
        or current time + estimated start. For simplicity, pending jobs with
        Reason=Resources or similar are assumed to start soon.

        Returns:
            List of UpcomingJob for racks about to receive GPU training jobs.
        """
        try:
            result = subprocess.run(
                [
                    "squeue",
                    "--states=PENDING",
                    "--format=%.18i %.9P %.20j %.8T %.10S %.20V %.N",
                    "--noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            _logger.debug("Slurm squeue not available: %s", e)
            return []

        if result.returncode != 0:
            return []

        jobs: List[UpcomingJob] = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            job_id = parts[0]
            partition = parts[1] if len(parts) > 1 else ""
            job_name = parts[2] if len(parts) > 2 else ""
            state = parts[3] if len(parts) > 3 else ""
            start_time_str = parts[4] if len(parts) > 4 else ""
            nodes_str = parts[6] if len(parts) > 6 else ""

            if not self._is_gpu_job(partition, None):
                # Could also check GRES via --format=%.b; for now use partition
                continue

            # Parse nodes: "node-rack-01" or "node[1-4]"
            node_names = []
            if nodes_str:
                if "[" in nodes_str:
                    # Expand node[1-4] -> node1, node2, ...
                    base, rest = nodes_str.split("[", 1)
                    range_part = rest.split("]", 1)[0]
                    try:
                        lo, hi = range_part.split("-")
                        for i in range(int(lo), int(hi) + 1):
                            node_names.append(f"{base}{i}")
                    except (ValueError, IndexError):
                        node_names = [nodes_str]
                else:
                    node_names = [n.strip() for n in nodes_str.split(",")]

            # Estimate seconds until start
            # N/A = not scheduled yet; use 30s as placeholder for "soon"
            if start_time_str in ("N/A", "") or not start_time_str:
                seconds_until_start = 30.0  # Assume soon
            else:
                try:
                    from datetime import datetime

                    st = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S")
                    seconds_until_start = (st - datetime.now()).total_seconds()
                except Exception:
                    seconds_until_start = 30.0

            if seconds_until_start > max_seconds or seconds_until_start < 0:
                continue

            for node in node_names or ["unknown"]:
                rack_id = self._node_to_rack(node)
                jobs.append(
                    UpcomingJob(
                        rack_id=rack_id,
                        job_id=job_id,
                        seconds_until_start=seconds_until_start,
                        job_type="gpu_training",
                        node_id=node,
                        partition=partition or None,
                    )
                )

        return jobs


class KubernetesObserver:
    """
    Placeholder for Kubernetes job/pod observation.

    Future: Watch for pending pods with GPU resource requests that are
    about to be scheduled onto nodes. Map nodes to racks via labels or topology.
    """

    def __init__(self, node_to_rack_map: Optional[dict] = None):
        self.node_to_rack_map = node_to_rack_map or {}

    def get_upcoming_jobs(self, max_seconds: float = 60.0) -> List[UpcomingJob]:
        """
        Placeholder: returns empty list.

        TODO: Implement via Kubernetes API or kubectl to list pending pods
        with nvidia.com/gpu resource requests, estimate scheduling time.
        """
        return []


def get_upcoming_load(
    observers: Optional[List] = None,
    max_seconds: float = 60.0,
    topology_path: Optional[Path] = None,
) -> List[UpcomingJob]:
    """
    Return list of racks about to receive a high-compute job (GPU training).

    Aggregates results from all registered observers (Slurm, Kubernetes, etc.).
    Only includes jobs that will start within max_seconds (default 60).

    Uses core/config/topology_map.yaml for node->rack mapping if observers is None.

    Args:
        observers: Optional list of observer instances (SlurmObserver, KubernetesObserver).
            If None, uses defaults loaded from topology_map.yaml.
        max_seconds: Only include jobs starting within this many seconds.
        topology_path: Optional path to topology YAML. Default: core/config/topology_map.yaml.

    Returns:
        List of UpcomingJob, one per rack+job combination.
    """
    if observers is None:
        node_to_rack: dict = {}
        gpu_partitions: Optional[List[str]] = None
        try:
            from core.config import get_node_to_rack_map, get_gpu_partitions
            node_to_rack = get_node_to_rack_map(topology_path)
            gpu_partitions = get_gpu_partitions(topology_path)
        except ImportError:
            pass
        observers = [
            SlurmObserver(node_to_rack_map=node_to_rack, gpu_partitions=gpu_partitions),
            KubernetesObserver(node_to_rack_map=node_to_rack),
        ]

    all_jobs: List[UpcomingJob] = []
    seen: set = set()

    for obs in observers:
        try:
            jobs = obs.get_upcoming_jobs(max_seconds=max_seconds)
            for j in jobs:
                key = (j.rack_id, j.job_id)
                if key not in seen:
                    seen.add(key)
                    all_jobs.append(j)
        except Exception as e:
            _logger.warning("Observer %s failed: %s", type(obs).__name__, e)

    return all_jobs
