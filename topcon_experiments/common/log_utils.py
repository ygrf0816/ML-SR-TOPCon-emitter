"""Progress logging and known-warning suppression for long-running jobs."""

from __future__ import annotations

import sys
import warnings


def setup_runtime() -> None:
    """Suppress noisy but harmless third-party warnings."""
    warnings.filterwarnings(
        "ignore",
        message=".*insecure pickle module.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*pkg_resources is deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Unable to compare versions for packaging.*",
        category=UserWarning,
    )


def log(msg: str) -> None:
    print(f"[topcon] {msg}", flush=True)
    sys.stdout.flush()


def log_step(current: int, total: int, msg: str) -> None:
    log(f"[{current}/{total}] {msg}")
