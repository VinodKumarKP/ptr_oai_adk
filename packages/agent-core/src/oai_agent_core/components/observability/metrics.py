"""Optional Prometheus metrics for agent-core (prometheus_client based).

agent-core records operation counts and durations **directly into
``prometheus_client``'s default registry** — the same registry the agent HTTP
server's ``/metrics`` route serves via ``generate_latest()``. This deliberately
does *not* go through OpenTelemetry's ``MeterProvider``: doing so caused provider
ordering conflicts (OpenLIT installs its own ``MeterProvider`` during agent
construction, which blocked the server's Prometheus reader). Recording straight
into the default registry sidesteps all of that.

**Prometheus is pull-based** — it scrapes an HTTP ``/metrics`` page:

- Inside the HTTP server, the server already exposes ``/metrics`` from the
  default registry, so agent-core's metrics appear there automatically.
- Standalone (no server), set ``PROMETHEUS_PORT`` and agent-core starts its own
  exposition endpoint for Prometheus to scrape.

Enablement: recording is a no-op unless ``PROMETHEUS_ENABLED`` is truthy (and
``prometheus_client`` is installed — ``pip install 'oai-agent-core[metrics]'``).
"""

import logging
import os

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, start_http_server

    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False

# Buckets tuned for agent operations: sub-millisecond guardrails up to multi-second
# LLM calls (default prometheus buckets top out at 10s, too coarse for LLM latency).
_DURATION_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
    10.0, 20.0, 30.0, 60.0, 120.0,
)

_configure_attempted = False
_metrics_enabled = False
_own_server_started = False
_instruments = {}


def _truthy(value) -> bool:
    return str(value).lower() in ("1", "true", "yes", "on")


def is_metrics_enabled() -> bool:
    """Return True if metric recording is active for this process."""
    return _metrics_enabled


def configure_metrics(default_service_name=None) -> bool:
    """Enable agent-core metrics. Runs at most once per process.

    - No-op unless ``PROMETHEUS_ENABLED`` is truthy and ``prometheus_client`` is
      installed.
    - Metrics register in the default registry, so a host that serves
      ``/metrics`` (the agent HTTP server) exposes them automatically.
    - If ``PROMETHEUS_PORT`` is set (standalone use), also start a Prometheus
      exposition HTTP server on that port. The HTTP server does *not* set
      ``PROMETHEUS_PORT`` (it serves ``/metrics`` on its own app port), so this
      won't create a duplicate endpoint under the server.

    Returns True if metrics are enabled afterwards.
    """
    global _configure_attempted, _metrics_enabled, _own_server_started
    if _configure_attempted or not _PROM_AVAILABLE:
        return _metrics_enabled
    _configure_attempted = True

    if not _truthy(os.environ.get("PROMETHEUS_ENABLED", "")):
        return False

    _metrics_enabled = True

    port = os.environ.get("PROMETHEUS_PORT")
    if port:
        try:
            start_http_server(int(port))
            _own_server_started = True
            logger.info(
                "agent-core Prometheus exposition started on :%s (scrape /)", port
            )
        except Exception as exc:
            logger.debug("Could not start Prometheus exposition server: %s", exc)

    logger.info("agent-core metrics enabled (prometheus_client default registry)")
    return True


def _get_instruments():
    if "count" not in _instruments:
        _instruments["count"] = Counter(
            "agent_operations_total",
            "Count of agent operations (ainvoke, astream, llm.invoke, guardrails, "
            "load_* etc.)",
            ["agent", "operation", "status"],
        )
        _instruments["duration"] = Histogram(
            "agent_operation_duration_seconds",
            "Duration of agent operations in seconds",
            ["agent", "operation", "status"],
            buckets=_DURATION_BUCKETS,
        )
    return _instruments


def record_operation(operation, agent=None, status="ok", seconds=None) -> None:
    """Record a count (and optional duration) for an agent operation.

    No-op unless metrics are enabled. Labels are low-cardinality:
    ``operation`` (bounded span name), ``status`` (ok/error), ``agent``.
    """
    if not _metrics_enabled or not _PROM_AVAILABLE:
        return
    try:
        labels = {"agent": agent or "", "operation": operation, "status": status}
        instruments = _get_instruments()
        instruments["count"].labels(**labels).inc()
        if seconds is not None:
            instruments["duration"].labels(**labels).observe(seconds)
    except Exception as exc:  # never let metrics break the request path
        logger.debug("Failed to record metric for %s: %s", operation, exc)
