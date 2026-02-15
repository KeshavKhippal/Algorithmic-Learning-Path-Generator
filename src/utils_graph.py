"""
Phase 3 utility helpers: deterministic seeding, timing, logging.
"""

import contextlib
import logging
import os
import time
from typing import Generator

import numpy as np

logger = logging.getLogger(__name__)

SEED = 42


def set_phase3_seed(seed: int = SEED) -> None:
    """Set numpy and env seeds for deterministic Phase 3 execution."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Phase 3 seed set to %d.", seed)


@contextlib.contextmanager
def timed(label: str) -> Generator[None, None, None]:
    """Context manager that logs elapsed wall-clock time for *label*."""
    t0 = time.monotonic()
    yield
    elapsed = time.monotonic() - t0
    logger.info("⏱  %s completed in %.2fs.", label, elapsed)
