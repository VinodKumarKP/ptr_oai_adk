"""Optional Prometheus / OpenTelemetry metrics for agent-core.

Metrics are recorded through the *global* OpenTelemetry ``MeterProvider``. As with
tracing, agent-core does not own the exporter: when the host process (e.g. the
agent HTTP server) has already installed a ``MeterProvider`` with a
``PrometheusMetricReader``, agent-core's metrics automatically appear on that
process's ``/metrics`` endpoint — no extra wiring.

**Prometheus is pull-based**: it scrapes an HTTP ``/metrics`` page; nothing pushes
to it. So:

- Inside the HTTP server, the server exposes ``/metrics`` and agent-core metrics
  ride along via the shared global MeterProvider.
- Standalone (no host provider), agent-core will stand up its own Prometheus
  exposition endpoint **only if enabled via env vars** (see ``configure_metrics``),
  which Prometheus then scrapes.

Enablement (recording is a no-op unless one of these holds):

- a host has already configured an SDK ``MeterProvider`` (server case); or
- ``PROMETHEUS_ENABLED`` is truthy *and* the optional deps are installed
  (``pip install 'oai-agent-core[metrics]'``), in which case agent-core starts a
  Prometheus exposition server on ``PROMETHEUS_PORT`` (default 9464).
"""

import logging
import os

logger = logging.getLogger(__name__)

try:
    from opentelemetry import metrics as _otel_metrics

    _OTEL_METRICS_AVAILABLE = True
except ImportError:
    _OTEL_METRICS_AVAILABLE = False

# One-time guard for standalone setup, and a cached "enabled" flag.
_configure_attempted = False
_metrics_enabled = False  # cached True once a real provider is detected/created
_own_provider_started = False

# Lazily-created instruments (created once a provider is in place).
_instruments = {}


def _truthy(value) -> bool:
    return str(value).lower() in ("1", "true", "yes", "on")


def _host_provider_present() -> bool:
    """True if a real SDK MeterProvider is currently installed (by the host)."""
    if not _OTEL_METRICS_AVAILABLE:
        return False
    try:
        from opentelemetry.sdk.metrics import MeterProvider as _SDKMeterProvider

        return isinstance(_otel_metrics.get_meter_provider(), _SDKMeterProvider)
    except Exception:
        return False


def is_metrics_enabled() -> bool:
    """Return True if metric recording is active.

    Dynamic on purpose: a host (e.g. the agent HTTP server) often installs its
    ``MeterProvider`` *after* agents are constructed, so we keep checking until a
    real provider appears, then cache the result. This lets agent-core metrics
    flow onto the host's ``/metrics`` without agent-core ever owning the provider.
    """
    global _metrics_enabled
    if _metrics_enabled:
        return True
    if _own_provider_started or _host_provider_present():
        _metrics_enabled = True
    return _metrics_enabled


def configure_metrics(default_service_name=None) -> bool:
    """Set up agent-core metrics. Runs its standalone setup at most once.

    Resolution:

    1. If a host already installed an SDK ``MeterProvider`` (e.g. the HTTP server
       with a ``PrometheusMetricReader`` exposing ``/metrics``), reuse it — nothing
       to set up; agent-core metrics flow onto that endpoint.
    2. Else, **standalone** mode: only if ``PROMETHEUS_PORT`` is set (and the deps
       are installed), create a ``MeterProvider`` + ``PrometheusMetricReader`` and
       start an exposition server on that port for Prometheus to scrape. We key on
       ``PROMETHEUS_PORT`` (not ``PROMETHEUS_ENABLED``) precisely so agent-core
       never races the HTTP server — the server enables Prometheus via
       ``PROMETHEUS_ENABLED`` but serves ``/metrics`` on its own app port and does
       not set ``PROMETHEUS_PORT``.
    3. Otherwise do nothing now. Recording still activates automatically if a host
       provider appears later (see ``is_metrics_enabled``).

    Returns True if metrics are enabled afterwards.
    """
    global _configure_attempted, _own_provider_started, _metrics_enabled
    if not _OTEL_METRICS_AVAILABLE:
        return False

    # Always (cheaply) pick up a host provider if one already exists.
    if _host_provider_present():
        _metrics_enabled = True
        return True

    if _configure_attempted:
        return is_metrics_enabled()
    _configure_attempted = True

    # Standalone exposition only when an explicit port is requested.
    port_env = os.environ.get("PROMETHEUS_PORT")
    if not port_env:
        # Not standalone — defer to a host provider that may appear later.
        return False

    try:
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource
        from prometheus_client import start_http_server
    except ImportError:
        logger.debug(
            "PROMETHEUS_PORT is set but metrics deps are missing; install "
            "'oai-agent-core[metrics]' to enable a Prometheus exposition endpoint."
        )
        return False

    try:
        service_name = (
            os.environ.get("OTEL_SERVICE_NAME")
            or default_service_name
            or "oai-agent-core"
        )
        reader = PrometheusMetricReader()
        provider = MeterProvider(
            resource=Resource.create({"service.name": service_name}),
            metric_readers=[reader],
        )
        _otel_metrics.set_meter_provider(provider)

        start_http_server(int(port_env))
        _own_provider_started = True
        _metrics_enabled = True
        logger.info(
            "Prometheus metrics exposition started on :%s (scrape http://<host>:%s/) "
            "service.name=%s",
            port_env,
            port_env,
            service_name,
        )
        return True
    except Exception as exc:
        logger.debug("Failed to configure Prometheus metrics: %s", exc)
        return False


def _get_instruments():
    if "count" not in _instruments:
        meter = _otel_metrics.get_meter("oai_agent_core")
        _instruments["count"] = meter.create_counter(
            "agent_operations_total",
            description="Count of agent operations (ainvoke, astream, guardrails, "
            "llm.invoke, load_* etc.)",
            unit="1",
        )
        _instruments["duration"] = meter.create_histogram(
            "agent_operation_duration_seconds",
            description="Duration of agent operations in seconds",
            unit="s",
        )
    return _instruments


def record_operation(operation, agent=None, status="ok", seconds=None) -> None:
    """Record a count (and optional duration) for an agent operation.

    No-op unless metrics are enabled. Labels are intentionally low-cardinality:
    ``operation`` (bounded span name), ``status`` (ok/error), and ``agent``.
    """
    if not _metrics_enabled or not _OTEL_METRICS_AVAILABLE:
        return
    try:
        attrs = {"operation": operation, "status": status}
        if agent:
            attrs["agent"] = agent
        instruments = _get_instruments()
        instruments["count"].add(1, attrs)
        if seconds is not None:
            instruments["duration"].record(seconds, attrs)
    except Exception as exc:  # never let metrics break the request path
        logger.debug("Failed to record metric for %s: %s", operation, exc)
