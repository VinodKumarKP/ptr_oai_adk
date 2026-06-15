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
    except Exception as exc:
        logger.debug("Failed to auto-configure OpenTelemetry tracing: %s", exc)


# Backwards-compatible alias for the lazy, no-default call used by ``trace_span``.
def _ensure_provider_from_env() -> None:
    configure_tracing()


@contextmanager
def trace_span(name, attributes=None, **attribute_kwargs):
    """Run a block inside an OpenTelemetry span set as the current span.

    The span becomes the active span for the duration of the block, so any spans
    started by callees (including across ``await`` boundaries and within
    ``asyncio.gather`` tasks scheduled inside the block) nest underneath it.

    No-op when OpenTelemetry is not installed. On exception the span records the
    error and is marked as failed before the exception propagates, so failures
    are visible in the trace.

    Args:
        name: Span name (e.g. ``"agent.ainvoke"``).
        attributes: Optional dict of span attributes.
        **attribute_kwargs: Additional span attributes as keyword arguments.
            ``None`` values are skipped.

    Yields:
        The active span, or ``None`` when tracing is unavailable.
    """
    if not _OTEL_AVAILABLE:
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


def traced(name):
    """Decorator that wraps a method in a span.

    Works on:

    - async generator methods (e.g. ``astream``) — the span stays open for the
      life of the stream;
    - coroutine methods (e.g. ``ainvoke``);
    - plain synchronous methods/functions.

    When applied to a method, the first positional argument is assumed to be
    ``self`` and its ``agent_name`` attribute (when present) is attached to the
    span. No-op when OpenTelemetry is not installed.

    Usage::

        @traced("agent.ainvoke")
        async def ainvoke(self, user_message, config=None):
            ...

        @traced("agent.build_prompt")
        def build_prompt(self, ...):
            ...
    """
    def _agent_name(args):
        return getattr(args[0], "agent_name", None) if args else None

    def decorator(func):
        if inspect.isasyncgenfunction(func):
            @functools.wraps(func)
            async def async_gen_wrapper(*args, **kwargs):
                with trace_span(name, agent_name=_agent_name(args)):
                    async for item in func(*args, **kwargs):
                        yield item
            return async_gen_wrapper

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with trace_span(name, agent_name=_agent_name(args)):
                    return await func(*args, **kwargs)
            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with trace_span(name, agent_name=_agent_name(args)):
                return func(*args, **kwargs)
        return sync_wrapper

    return decorator
