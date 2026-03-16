"""
CooledAI Scheduler Services - Job observation and upcoming load prediction.

Integrates with Slurm, Kubernetes, and other schedulers to predict which racks
are about to receive high-compute jobs (GPU training). Enables pre-emptive
cooling before thermal spikes.
"""

from .job_observer import (
    SlurmObserver,
    KubernetesObserver,
    get_upcoming_load,
    UpcomingJob,
)

__all__ = [
    "SlurmObserver",
    "KubernetesObserver",
    "get_upcoming_load",
    "UpcomingJob",
]
