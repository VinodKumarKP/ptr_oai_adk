"""OpenTelemetry and Prometheus observability setup for the agent server.

Provides:
- Distributed tracing via OpenTelemetry
- Prometheus metrics (request latency, error rates, throughput)
- Structured logging integration
- Performance monitoring hooks

Metrics strategy
----------------
OTel's global MeterProvider is often set by oai_agent_core / openlit *before*
ObservabilityManager is constructed, and OTel silently refuses a second
set_meter_provider() call.  Even "reusing" that provider does not help because
it was created without a PrometheusMetricReader, so generate_latest() never
sees any HTTP metrics.

We therefore record HTTP / DB / streaming metrics via **native prometheus_client
instruments** (Counter, Histogram, Gauge) which always land in the default
prometheus_client.REGISTRY — the same registry that generate_latest() reads.
OTel tracing (spans) is kept unchanged.
"""

import logging
import time
import os
from contextlib import contextmanager
from typing import Optional, Dict, Any
from functools import wraps

try:
    from opentelemetry import trace, metrics
    # Try to import Jaeger first, fall back to OTLP
    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        _JAEGER_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        _JAEGER_AVAILABLE = False
        # Fall back to OTLP which is more universally available
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            _OTLP_AVAILABLE = True
        except (ImportError, ModuleNotFoundError):
            _OTLP_AVAILABLE = False

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    _JAEGER_AVAILABLE = False
    _OTLP_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge
    _PROM_CLIENT_AVAILABLE = True
except ImportError:
    _PROM_CLIENT_AVAILABLE = False


class ObservabilityConfig:
    """Configuration for observability stack.
    
    Reads from environment variables for flexibility:
    - OTEL_ENABLED: Enable/disable OpenTelemetry tracing
    - OTEL_EXPORTER_JAEGER_ENDPOINT: Jaeger gRPC endpoint (e.g., http://host:14250)
    - OTEL_EXPORTER_JAEGER_AGENT_HOST: Jaeger agent host (UDP, fallback if no endpoint)
    - OTEL_EXPORTER_JAEGER_AGENT_PORT: Jaeger agent port (UDP, fallback if no endpoint)
    - OTEL_SERVICE_NAME: Service name for traces
    - PROMETHEUS_ENABLED: Enable/disable Prometheus metrics
    """
    
    def __init__(self):
        """Initialize config from environment variables or defaults."""
        # OpenTelemetry
        self.ENABLE_TRACING = os.environ.get('OTEL_ENABLED', 'true').lower() == 'true'

        # Jaeger configuration - supports both UDP (agent) and gRPC (endpoint)
        self.JAEGER_ENDPOINT = os.environ.get('OTEL_EXPORTER_JAEGER_ENDPOINT', None)
        self.JAEGER_AGENT_HOST = os.environ.get('OTEL_EXPORTER_JAEGER_AGENT_HOST', 'localhost')
        self.JAEGER_AGENT_PORT = int(os.environ.get('OTEL_EXPORTER_JAEGER_AGENT_PORT', '6831'))

        self.SERVICE_NAME = os.environ.get('OTEL_SERVICE_NAME', 'oai-agent-server')
        self.SERVICE_VERSION = os.environ.get('OTEL_SERVICE_VERSION', '3.0.0')

        # Prometheus
        self.ENABLE_METRICS = os.environ.get('PROMETHEUS_ENABLED', 'true').lower() == 'true'
        self.METRICS_PORT = int(os.environ.get('PROMETHEUS_PORT', '8001'))
        
        # Structured logging
        self.ENABLE_STRUCTURED_LOGGING = os.environ.get('LOG_FORMAT', 'json').lower() == 'json'


class ObservabilityManager:
    """Central observability manager for the agent server.
    
    Coordinates:
    - Distributed tracing (OpenTelemetry/Jaeger)
    - Metrics collection (Prometheus)
    - Structured logging
    - Performance instrumentation
    """
    
    def __init__(self, config: Optional[ObservabilityConfig] = None, 
                 logger: Optional[logging.Logger] = None):
        """Initialize observability manager.
        
        Args:
            config: Observability configuration
            logger: Logger instance
        """
        self.config = config or ObservabilityConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.tracer: Optional[trace.Tracer] = None
        # Native prometheus_client instruments (bypass OTel MeterProvider conflicts)
        self._prom: Dict[str, Any] = {}
        # Keep self._metrics and self.meter as aliases/stubs for back-compat
        self._metrics: Dict[str, Any] = {}
        self.meter = None

        self._initialize_tracing()
        self._initialize_metrics()
        self._setup_instrumentation()
    
    def _initialize_tracing(self) -> None:
        """Initialize OpenTelemetry tracing.
        
        Supports:
        - Jaeger (if available) via gRPC endpoint or UDP agent
        - OTLP (if Jaeger not available) as fallback
        """
        if not _OTEL_AVAILABLE or not self.config.ENABLE_TRACING:
            self.logger.info("OpenTelemetry tracing disabled or not available")
            return
        
        try:
            span_exporter = None
            exporter_info = ""
            
            # Try to use Jaeger if available
            if _JAEGER_AVAILABLE:
                try:
                    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
                    if self.config.JAEGER_ENDPOINT:
                        # Use gRPC endpoint (e.g., http://host:14250)
                        span_exporter = JaegerExporter(
                            collector_endpoint=self.config.JAEGER_ENDPOINT,
                        )
                        exporter_info = f"Jaeger gRPC: {self.config.JAEGER_ENDPOINT}"
                    else:
                        # Use UDP agent (traditional)
                        span_exporter = JaegerExporter(
                            agent_host_name=self.config.JAEGER_AGENT_HOST,
                            agent_port=self.config.JAEGER_AGENT_PORT,
                        )
                        exporter_info = f"Jaeger agent: {self.config.JAEGER_AGENT_HOST}:{self.config.JAEGER_AGENT_PORT}"
                except Exception as jaeger_error:
                    self.logger.warning(f"Jaeger exporter failed: {jaeger_error}, falling back to OTLP")
                    span_exporter = None
            
            # Fall back to OTLP if Jaeger not available
            if not span_exporter and _OTLP_AVAILABLE:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                    # OTLP endpoint should be set via OTEL_EXPORTER_OTLP_ENDPOINT env var
                    # Default: http://localhost:4317
                    span_exporter = OTLPSpanExporter()
                    exporter_info = "OTLP gRPC (fallback)"
                except Exception as otlp_error:
                    self.logger.warning(f"OTLP exporter failed: {otlp_error}")
                    raise otlp_error
            
            if not span_exporter:
                raise RuntimeError("No span exporter available (Jaeger and OTLP both unavailable)")
            
            # Create TracerProvider with resource information
            resource = Resource.create({
                "service.name": self.config.SERVICE_NAME,
                "service.version": self.config.SERVICE_VERSION,
            })
            tracer_provider = TracerProvider(resource=resource)
            tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
            
            # Set global tracer provider
            trace.set_tracer_provider(tracer_provider)
            self.tracer = trace.get_tracer(__name__)
            
            self.logger.info(f"OpenTelemetry tracing initialized ({exporter_info})")
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenTelemetry tracing: {e}")
    
    def _initialize_metrics(self) -> None:
        """Initialize Prometheus metrics using native prometheus_client instruments.

        We intentionally bypass the OTel MeterProvider here.  oai_agent_core and
        openlit set the global MeterProvider before AgentHTTPServer is constructed
        and OTel refuses a second set_meter_provider() call.  Even if we "reuse"
        the existing provider it has no PrometheusMetricReader, so nothing appears
        in generate_latest() output.

        Native prometheus_client instruments always write to
        prometheus_client.REGISTRY — the registry that generate_latest() reads —
        regardless of who owns the OTel global provider.
        """
        if not self.config.ENABLE_METRICS:
            self.logger.info("Prometheus metrics disabled")
            return

        if not _PROM_CLIENT_AVAILABLE:
            self.logger.warning("prometheus_client not available — metrics disabled")
            return

        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Create native prometheus_client instruments and register them in REGISTRY."""
        if not _PROM_CLIENT_AVAILABLE:
            return

        try:
            # HTTP request metrics
            self._prom['request_duration'] = Histogram(
                "http_request_duration_seconds",
                "HTTP request duration in seconds",
                labelnames=["method", "path", "status_code"],
            )
            self._prom['request_count'] = Counter(
                "http_requests_total",
                "Total number of HTTP requests",
                labelnames=["method", "path", "status_code"],
            )
            self._prom['error_count'] = Counter(
                "http_errors_total",
                "Total number of HTTP errors (status >= 400)",
                labelnames=["status_code", "error_type"],
            )

            # Database metrics
            self._prom['db_query_duration'] = Histogram(
                "db_query_duration_seconds",
                "Database query duration in seconds",
                labelnames=["operation", "success"],
            )
            self._prom['db_pool_utilization'] = Gauge(
                "db_connection_pool_utilization_percent",
                "Database connection pool utilization percentage",
            )

            # LLM / streaming metrics
            self._prom['llm_judge_duration'] = Histogram(
                "llm_judge_duration_seconds",
                "LLM judge evaluation duration in seconds",
                labelnames=["success"],
            )
            self._prom['streaming_events'] = Counter(
                "chat_streaming_events_total",
                "Total streaming events sent",
                labelnames=["event_type", "success"],
            )

            # Keep _metrics as a shim so any code still referencing it doesn't crash.
            self._metrics = self._prom
            # Mark metrics as active (used by record_* guards).
            self.meter = True  # truthy sentinel; we no longer use OTel meters

            self.logger.info(
                "Prometheus metrics initialized via native prometheus_client "
                "(http_requests_total, http_request_duration_seconds, …)"
            )
        except ValueError as exc:
            # Duplicate registration is harmless — another instance or reload
            # already registered these metrics.
            if "Duplicated timeseries" in str(exc) or "already exist" in str(exc).lower():
                self.logger.debug("Prometheus metrics already registered, reusing existing.")
                self.meter = True
            else:
                self.logger.warning(f"Failed to setup Prometheus metrics: {exc}")
        except Exception as e:
            self.logger.warning(f"Failed to setup Prometheus metrics: {e}")
    
    def _setup_instrumentation(self) -> None:
        """Setup automatic instrumentation for FastAPI and other libraries."""
        if not _OTEL_AVAILABLE:
            return
        
        try:
            # Instrument FastAPI automatically
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor().instrument()
            self.logger.debug("FastAPI instrumentation enabled")
        except Exception as e:
            self.logger.warning(f"Failed to instrument FastAPI: {e}")
        
        try:
            # Instrument requests library
            from opentelemetry.instrumentation.requests import RequestsInstrumentor
            RequestsInstrumentor().instrument()
            self.logger.debug("Requests instrumentation enabled")
        except Exception as e:
            self.logger.warning(f"Failed to instrument Requests: {e}")
    
    @contextmanager
    def trace_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing an operation.
        
        Args:
            operation_name: Name of the operation to trace
            attributes: Additional span attributes
            
        Yields:
            Tracing span
        """
        if self.tracer is None:
            yield None
            return
        
        with self.tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    try:
                        span.set_attribute(key, value)
                    except Exception:
                        pass  # Skip attributes that can't be set
            yield span
    
    def record_request(self, method: str, path: str, status_code: int, duration: float):
        """Record HTTP request metrics.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            status_code: HTTP response status code
            duration: Request duration in seconds
        """
        if not self._prom:
            return

        try:
            labels = {"method": method, "path": path, "status_code": str(status_code)}
            self._prom['request_duration'].labels(**labels).observe(duration)
            self._prom['request_count'].labels(**labels).inc()

            if status_code >= 400:
                self._prom['error_count'].labels(
                    status_code=str(status_code),
                    error_type=f"HTTP_{status_code}",
                ).inc()
        except Exception as e:
            self.logger.debug(f"Failed to record request metrics: {e}")

    def record_db_query(self, query_type: str, duration: float, success: bool = True):
        """Record database query metrics.

        Args:
            query_type: Type of query (SELECT, INSERT, UPDATE, etc.)
            duration: Query duration in seconds
            success: Whether query succeeded
        """
        if not self._prom:
            return

        try:
            self._prom['db_query_duration'].labels(
                operation=query_type, success=str(success)
            ).observe(duration)
        except Exception as e:
            self.logger.debug(f"Failed to record DB metrics: {e}")

    def record_llm_judge_evaluation(self, duration: float, success: bool = True):
        """Record LLM judge evaluation metrics.

        Args:
            duration: Evaluation duration in seconds
            success: Whether evaluation succeeded
        """
        if not self._prom:
            return

        try:
            self._prom['llm_judge_duration'].labels(success=str(success)).observe(duration)
        except Exception as e:
            self.logger.debug(f"Failed to record LLM judge metrics: {e}")

    def record_streaming_event(self, event_type: str, success: bool = True):
        """Record streaming event metrics.

        Args:
            event_type: Type of streaming event
            success: Whether event was successfully sent
        """
        if not self._prom:
            return

        try:
            self._prom['streaming_events'].labels(
                event_type=event_type, success=str(success)
            ).inc()
        except Exception as e:
            self.logger.debug(f"Failed to record streaming metrics: {e}")

    def set_pool_utilization(self, utilization_percent: float):
        """Set database connection pool utilization gauge.

        Args:
            utilization_percent: Pool utilization percentage (0-100)
        """
        if not self._prom:
            return

        try:
            self._prom['db_pool_utilization'].set(utilization_percent)
        except Exception as e:
            self.logger.debug(f"Failed to record pool utilization: {e}")


def instrument_function(observability_manager: ObservabilityManager):
    """Decorator to instrument a function with observability.
    
    Automatically traces execution and records metrics.
    
    Args:
        observability_manager: ObservabilityManager instance
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                with observability_manager.trace_operation(
                    f"{func.__module__}.{func.__name__}",
                    attributes={"function.name": func.__name__}
                ):
                    result = func(*args, **kwargs)
                return result
            except Exception as e:
                observability_manager.logger.error(f"Error in {func.__name__}: {e}")
                raise
            finally:
                duration = time.time() - start_time
                observability_manager.logger.debug(
                    f"{func.__name__} completed in {duration:.3f}s"
                )
        return wrapper
    return decorator
