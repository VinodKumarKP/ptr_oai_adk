"""Internal retry helpers — no external dependencies."""
import random


def compute_backoff(attempt: int, factor: float, cap: float, jitter: float) -> float:
    """Exponential backoff with full-jitter style scaling.

    ``jitter`` is a fraction in [0, 1]:
      * 0.0 -> deterministic ``min(cap, factor * 2**attempt)``
      * 1.0 -> full jitter, uniform in (0, exp]
      * mid -> shrink by ``(1 - jitter)`` and add a random slice up to
        ``jitter * exp``.

    The returned delay is always within ``[0, cap]``.
    """
    exp = min(cap, factor * (2 ** attempt))
    jitter = max(0.0, min(1.0, jitter))
    delay = exp * (1.0 - jitter + random.random() * jitter)
    if delay < 0:
        delay = 0.0
    if delay > cap:
        delay = cap
    return delay
