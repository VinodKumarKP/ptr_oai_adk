"""Optional OpenTelemetry span instrumentation for agent-core.

Spans are emitted through the *global* OpenTelemetry tracer. When the host
process (e.g. ``agent-server``) has already configured a ``TracerProvider`` with
an exporter (Jaeger / OTLP / Tempo), the spans produced here automatically join
that pipeline — no wiring is required inside agent-core.

**Auto-configuration**: if no ``TracerProvider`` has been configured by the host
*and* ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, agent-core lazily configures one
itself (OTLP exporter + batch processor) the first time a span is created. This
lets standalone scripts get tracing just by setting the env var, while never
overriding a provider the host already installed. It requires the optional
``opentelemetry-sdk`` / ``opentelemetry-exporter-otlp`` packages
(``pip install 'oai-agent-core[tracing]'``); if they're missing it stays a no-op.

Relevant environment variables:

- ``OTEL_EXPORTER_OTLP_ENDPOINT``: OTLP gRPC endpoint (e.g. ``http://localhost:4317``).
  Presence of this var is what enables auto-configuration.
- ``OTEL_SERVICE_NAME``: service name reported for the auto-configured provider
  (default ``oai-agent-core``).

When ``opentelemetry-api`` is not installed, every helper degrades to a no-op so
agent-core keeps zero hard dependency on OpenTelemetry.

Usage::

    from oai_agent_core.components.observability.tracing import trace_span

    with trace_span("agent.kb", agent_name=self.agent_name):
        await do_work()
"""

import functools
import inspect
import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    _OTEL_AVAILABLE = True
    # A no-op tracer is returned if no SDK/provider is configured, so this is
    # always safe to call at import time.
    _tracer = trace.get_tracer("oai_agent_core")
except ImportError:
    _OTEL_AVAILABLE = False
    _tracer = None

# Guards one-time, lazy auto-configuration of a TracerProvider from env.
_auto_init_attempted = False

# --- Trace levels (mirrors logging) ---------------------------------------
# A span is emitted only when its level is >= the active threshold, exactly like
# logging: at threshold INFO, INFO spans emit and DEBUG spans are skipped; at
# threshold DEBUG, everything emits. Set via the AGENT_TRACE_LEVEL env var
# (DEBUG / INFO / WARNING) or programmatically via set_trace_level().
DEBUG = 10
INFO = 20
WARNING = 30

_LEVEL_NAMES = {"debug": DEBUG, "info": INFO, "warning": WARNING}


def _coerce_level(level) -> int:
    if isinstance(level, int):
        return level
    return _LEVEL_NAMES.get(str(level).lower(), INFO)


_trace_level_threshold = _coerce_level(os.environ.get("AGENT_TRACE_LEVEL", "info"))


def set_trace_level(level) -> None:
    """Set the active trace level threshold (e.g. "debug", "info", or an int).

    Spans below this level become no-ops. Useful to flip verbosity at runtime,
    e.g. ``set_trace_level("debug")`` to capture detailed spans.
    """
    global _trace_level_threshold
    _trace_level_threshold = _coerce_level(level)


def get_trace_level() -> int:
    """Return the active trace level threshold as an int."""
    return _trace_level_threshold


def is_tracing_available() -> bool:
    """Return True if the OpenTelemetry API is importable."""
    return _OTEL_AVAILABLE


def configure_tracing(default_service_name=None) -> None:
    """Lazily configure an OTLP TracerProvider from environment, if appropriate.

    Runs at most once per process. Does nothing unless ALL of the following hold:

    - the OpenTelemetry API is importable;
    - ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set;
    - no real ``TracerProvider`` has been installed yet (so we never override a
      provider configured by the host process);
    - the OpenTelemetry SDK and OTLP exporter are importable.

    The exported ``service.name`` (what Jaeger shows as the service) is chosen
    with this precedence:

    1. ``OTEL_SERVICE_NAME`` environment variable (explicit ops override);
    2. ``default_service_name`` (e.g. the agent name passed by ``BaseAgent``);
    3. ``"oai-agent-core"`` fallback.

    Because ``service.name`` is a process-level resource attribute set once, the
    first caller to configure the provider wins. With one agent per process this
    means traces show up under that agent's name; if several agents share a
    process, the first one's name is used (set ``OTEL_SERVICE_NAME`` to override).

    Any failure degrades silently to the no-op tracer.
    """
    global _auto_init_attempted
    if _auto_init_attempted or not _OTEL_AVAILABLE:
        return
    _auto_init_attempted = True

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    # Never clobber a provider the host already configured. Before set, the API
    # returns a ProxyTracerProvider; a real provider means someone set it up.
    try:
        from opentelemetry.trace import ProxyTracerProvider

        if not isinstance(trace.get_tracer_provider(), ProxyTracerProvider):
            return
    except Exception:
        return

    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    except ImportError:
        logger.debug(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but the OpenTelemetry SDK/OTLP "
            "exporter are not installed; install 'oai-agent-core[tracing]' to "
            "enable auto-configured tracing."
        )
        return

    try:
        service_name = (
            os.environ.get("OTEL_SERVICE_NAME")
            or default_service_name
            or "oai-agent-core"
        )
        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        logger.info(
            "Auto-configured OpenTelemetry tracing -> %s (service.name=%s)",
            endpoint,
            service_name,
        )
        # Route OpenLIT GenAI auto-instrumentation into this same provider so its
        # LLM/tool spans land in the configured backend(s) and nest under the
        # agent spans. No-op if openlit isn't installed.
        instrument_openlit(application_name=service_name)
    except Exception as exc:
        logger.debug("Failed to auto-configure OpenTelemetry tracing: %s", exc)


# Backwards-compatible alias for the lazy, no-default call used by ``trace_span``.
def _ensure_provider_from_env() -> None:
    configure_tracing()


def _get_or_create_provider(service_name=None):
    """Return the active SDK ``TracerProvider``, creating+installing one if only a
    proxy (no provider) is present. Returns ``None`` if the SDK isn't available."""
    from opentelemetry.trace import ProxyTracerProvider

    current = trace.get_tracer_provider()
    if not isinstance(current, ProxyTracerProvider):
        return current
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
    except ImportError:
        return None
    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name or "oai-agent-core"})
    )
    trace.set_tracer_provider(provider)
    return provider


def add_otlp_exporter(endpoint, headers=None, protocol="grpc", service_name=None) -> bool:
    """Add an OTLP span exporter to the global ``TracerProvider``.

    Lets a single provider fan out to multiple backends (e.g. Jaeger *and*
    Langfuse) — every span is then sent to all configured exporters. Creates the
    provider if none exists yet. No-op (returns ``False``) when the OTel SDK /
    OTLP exporter isn't installed.

    Args:
        endpoint: OTLP endpoint URL.
        headers: Optional dict of headers (e.g. auth) for the exporter.
        protocol: ``"grpc"`` (default, e.g. Jaeger :4317) or ``"http"`` (e.g.
            Langfuse ``/api/public/otel``).
        service_name: ``service.name`` to use if a provider must be created.
    """
    if not _OTEL_AVAILABLE or not endpoint:
        return False
    try:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        if str(protocol).lower() == "http":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        else:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
    except ImportError:
        logger.debug("OTLP exporter/SDK not installed; cannot add exporter -> %s", endpoint)
        return False

    try:
        provider = _get_or_create_provider(service_name)
        if provider is None or not hasattr(provider, "add_span_processor"):
            return False
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
        )
        logger.info("Added OTLP span exporter (%s) -> %s", protocol, endpoint)
        return True
    except Exception as exc:
        logger.debug("Failed to add OTLP exporter -> %s: %s", endpoint, exc)
        return False


# Guards one-time OpenLIT instrumentation.
_openlit_instrumented = False


def instrument_openlit(application_name=None, environment=None, traces_only=True) -> bool:
    """Route OpenLIT GenAI auto-instrumentation into the global ``TracerProvider``.

    OpenLIT emits OTel spans for LLM / embedding / vector / tool calls. When
    ``openlit.init`` is called **without** an ``otlp_endpoint``, it attaches its
    instrumentation to the already-configured global ``TracerProvider`` — so its
    spans flow into whatever exporters that provider has (Jaeger and/or Langfuse)
    and nest under the surrounding agent spans, rather than OpenLIT managing its
    own separate export pipeline.

    Therefore a real provider must already be set (e.g. via ``configure_tracing``
    or ``add_otlp_exporter``) before calling this. Runs at most once per process.
    No-op if ``openlit`` isn't installed.

    Args:
        application_name: OpenLIT application name.
        environment: Deployment environment label.
        traces_only: When True (default), OpenLIT's metric and event/log export
            are disabled. This matters for trace-only backends like Jaeger, whose
            OTLP collector rejects non-trace signals (producing
            ``BadStatusLine`` / "failed to export logs/metrics" errors). LLM token
            and cost data still appear as **span attributes**, so nothing is lost
            in the trace view. Set False when the backend (e.g. Langfuse, an OTel
            Collector) also ingests OTLP metrics/logs.
    """
    global _openlit_instrumented
    if _openlit_instrumented or not _OTEL_AVAILABLE:
        return False
    try:
        import inspect

        import openlit
    except ImportError:
        logger.debug("openlit not installed; skipping GenAI auto-instrumentation")
        return False

    # Ensure a real provider is installed so OpenLIT emits into it (not its own).
    if _get_or_create_provider(application_name) is None:
        logger.debug("No tracer provider available; skipping OpenLIT init")
        return False

    # No otlp_endpoint -> OpenLIT uses the existing global provider/exporters.
    init_kwargs = {
        "application_name": application_name or "oai-agent-core",
        "environment": environment or os.environ.get("ENVIRONMENT", "production"),
        "disable_batch": True,
    }
    if traces_only:
        # Jaeger (and most trace backends) only accept OTLP traces. Disabling
        # metric/event export avoids noisy transient export failures when those
        # signals are pushed to a trace-only collector. The OTel SDK env knobs
        # are the reliable switch (OpenLIT's disable_* flags don't fully prevent
        # the exporters in all versions); we only set them if the user hasn't.
        os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
        os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")
        init_kwargs.update(
            disable_metrics=True,
            disable_events=True,
            evals_logs_export=False,
        )
    # Drop any kwargs this openlit version doesn't support, for forward/back-compat.
    supported = set(inspect.signature(openlit.init).parameters)
    init_kwargs = {k: v for k, v in init_kwargs.items() if k in supported}

    try:
        openlit.init(**init_kwargs)
        _openlit_instrumented = True
        logger.info(
            "OpenLIT GenAI instrumentation enabled (-> global tracer provider, "
            "traces_only=%s)",
            traces_only,
        )
        return True
    except Exception as exc:
        logger.debug("Failed to initialise OpenLIT: %s", exc)
        return False


@contextmanager
def trace_span(name, attributes=None, level=INFO, **attribute_kwargs):
    """Run a block inside an OpenTelemetry span set as the current span.

    The span becomes the active span for the duration of the block, so any spans
    started by callees (including across ``await`` boundaries and within
    ``asyncio.gather`` tasks scheduled inside the block) nest underneath it.

    No-op when OpenTelemetry is not installed, or when ``level`` is below the
    active threshold (see ``set_trace_level`` / ``AGENT_TRACE_LEVEL``). On
    exception the span records the error and is marked as failed before the
    exception propagates, so failures are visible in the trace.

    Args:
        name: Span name (e.g. ``"agent.ainvoke"``).
        attributes: Optional dict of span attributes.
        level: Span verbosity level (``DEBUG``/``INFO``/``WARNING`` or the
            equivalent string). The span is skipped when it is below the active
            threshold — DEBUG spans only appear when the threshold is DEBUG.
        **attribute_kwargs: Additional span attributes as keyword arguments.
            ``None`` values are skipped.

    Yields:
        The active span, or ``None`` when tracing is unavailable / suppressed.
    """
    if not _OTEL_AVAILABLE or _coerce_level(level) < _trace_level_threshold:
        yield None
        return

    # Lazily stand up a provider from env on first use (no-op if the host already
    # configured one, or if auto-config isn't applicable).
    _ensure_provider_from_env()

    attrs = dict(attributes or {})
    attrs.update(attribute_kwargs)

    with _tracer.start_as_current_span(name) as span:
        for key, value in attrs.items():
            if value is not None:
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


async def traced_stream(name, source, level=INFO, **attributes):
    """Wrap an async iterator in a span kept open for the whole stream.

    Use this to trace a streaming call (e.g. an LLM/graph ``astream``) where the
    work happens incrementally as items are pulled. The span stays active while
    the underlying source produces items, so spans created by it (including
    OpenLIT's per-LLM-call spans) nest underneath. No-op span when tracing is
    unavailable / below threshold — items still pass through.

    Usage::

        async for chunk in traced_stream("agent.llm.invoke", graph.astream(...),
                                         agent_name=self.agent_name):
            ...
    """
    with trace_span(name, level=level, **attributes):
        async for item in source:
            yield item


def traced(name, level=INFO):
    """Decorator that wraps a method in a span.

    Works on:

    - async generator methods (e.g. ``astream``) — the span stays open for the
      life of the stream;
    - coroutine methods (e.g. ``ainvoke``);
    - plain synchronous methods/functions.

    When applied to a method, the first positional argument is assumed to be
    ``self`` and its ``agent_name`` attribute (when present) is attached to the
    span. ``level`` controls verbosity (see ``trace_span``): mark detailed
    helpers ``level="debug"`` so they only appear when the threshold is DEBUG.
    No-op when OpenTelemetry is not installed or the level is below threshold.

    Usage::

        @traced("agent.ainvoke")                       # important -> INFO
        async def ainvoke(self, user_message, config=None):
            ...

        @traced("agent.build_prompt", level="debug")   # detailed -> DEBUG only
        def build_prompt(self, ...):
            ...
    """
    def _agent_name(args):
        return getattr(args[0], "agent_name", None) if args else None

    def decorator(func):
        if inspect.isasyncgenfunction(func):
            @functools.wraps(func)
            async def async_gen_wrapper(*args, **kwargs):
                with trace_span(name, level=level, agent_name=_agent_name(args)):
                    async for item in func(*args, **kwargs):
                        yield item
            return async_gen_wrapper

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with trace_span(name, level=level, agent_name=_agent_name(args)):
                    return await func(*args, **kwargs)
            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with trace_span(name, level=level, agent_name=_agent_name(args)):
                return func(*args, **kwargs)
        return sync_wrapper

    return decorator
