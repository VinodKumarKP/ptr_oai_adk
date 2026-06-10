"""OpenTelemetry and Prometheus observability setup for the agent server.

Provides:
- Distributed tracing via OpenTelemetry
- Prometheus metrics (request latency, error rates, throughput)
- Structured logging integration
- Performance monitoring hooks
"""

import logging
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any
from functools import wraps

try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from prometheus_client import Counter, Histogram, Gauge
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


class ObservabilityConfig:
    """Configuration for observability stack."""
    
    # OpenTelemetry
    ENABLE_TRACING = True
    JAEGER_AGENT_HOST = "localhost"
    JAEGER_AGENT_PORT = 6831
    SERVICE_NAME = "oai-agent-server"
    SERVICE_VERSION = "3.0.0"
    
    # Prometheus
    ENABLE_METRICS = True
    METRICS_PORT = 8001
    
    # Structured logging
    ENABLE_STRUCTURED_LOGGING = True


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
        self.meter: Optional[metrics.Meter] = None
        self._metrics: Dict[str, Any] = {}
        
        self._initialize_tracing()
        self._initialize_metrics()
    
    def _initialize_tracing(self) -> None:
        """Initialize OpenTelemetry tracing."""
        if not _OTEL_AVAILABLE or not self.config.ENABLE_TRACING:
            self.logger.info("OpenTelemetry tracing disabled or not available")
            return
        
        try:
            # Configure Jaeger exporter for distributed tracing
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.config.JAEGER_AGENT_HOST,
                agent_port=self.config.JAEGER_AGENT_PORT,
            )
            
            # Create TracerProvider with resource information
            resource = Resource.create({
                "service.name": self.config.SERVICE_NAME,
                "service.version": self.config.SERVICE_VERSION,
            })
            tracer_provider = TracerProvider(resource=resource)
            tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
            
            # Set global tracer provider
            trace.set_tracer_provider(tracer_provider)
            self.tracer = trace.get_tracer(__name__)
            
            self.logger.info(f"OpenTelemetry tracing initialized (Jaeger: {self.config.JAEGER_AGENT_HOST}:{self.config.JAEGER_AGENT_PORT})")
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenTelemetry tracing: {e}")
    
    def _initialize_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        if not _OTEL_AVAILABLE or not self.config.ENABLE_METRICS:
            self.logger.info("Prometheus metrics disabled or not available")
            return
        
        try:
            # Prometheus metrics reader
            prometheus_reader = PrometheusMetricReader()
            
            # Create MeterProvider
            resource = Resource.create({
                "service.name": self.config.SERVICE_NAME,
                "service.version": self.config.SERVICE_VERSION,
            })
            meter_provider = MeterProvider(resource=resource, metric_readers=[prometheus_reader])
            
            # Set global meter provider
            metrics.set_meter_provider(meter_provider)
            self.meter = metrics.get_meter(__name__)
            
            # Define metrics
            self._setup_metrics()
            
            self.logger.info(f"Prometheus metrics initialized (port {self.config.METRICS_PORT})")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Prometheus metrics: {e}")
    
    def _setup_metrics(self) -> None:
        """Setup Prometheus metrics."""
        if self.meter is None:
            return
        
        try:
            # Request metrics
            self._metrics['request_duration'] = self.meter.create_histogram(
                "request_duration_seconds",
                description="HTTP request duration in seconds",
                unit="s",
            )
            
            self._metrics['request_count'] = self.meter.create_counter(
                "requests_total",
                description="Total number of HTTP requests",
                unit="1",
            )
            
            self._metrics['error_count'] = self.meter.create_counter(
                "errors_total",
                description="Total number of errors",
                unit="1",
            )
            
            # Database metrics
            self._metrics['db_query_duration'] = self.meter.create_histogram(
                "db_query_duration_seconds",
                description="Database query duration in seconds",
                unit="s",
            )
            
            self._metrics['db_connection_pool_utilization'] = self.meter.create_gauge(
                "db_connection_pool_utilization",
                description="Database connection pool utilization percentage",
                unit="%",
            )
            
            # Chat/LLM metrics
            self._metrics['llm_judge_duration'] = self.meter.create_histogram(
                "llm_judge_duration_seconds",
                description="LLM judge evaluation duration in seconds",
                unit="s",
            )
            
            self._metrics['chat_streaming_events'] = self.meter.create_counter(
                "chat_streaming_events_total",
                description="Total number of streaming events sent",
                unit="1",
            )
            
            self.logger.debug("Prometheus metrics setup complete")
        except Exception as e:
            self.logger.warning(f"Failed to setup metrics: {e}")
    
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
        if self.meter is None:
            return
        
        try:
            attributes = {
                "http.method": method,
                "http.path": path,
                "http.status_code": status_code,
            }
            
            # Record duration histogram
            self._metrics['request_duration'].record(duration, attributes)
            
            # Record count counter
            self._metrics['request_count'].add(1, attributes)
            
            # Record errors
            if status_code >= 400:
                self._metrics['error_count'].add(1, {
                    "http.status_code": status_code,
                    "error_type": f"HTTP_{status_code}",
                })
        except Exception as e:
            self.logger.debug(f"Failed to record request metrics: {e}")
    
    def record_db_query(self, query_type: str, duration: float, success: bool = True):
        """Record database query metrics.
        
        Args:
            query_type: Type of query (SELECT, INSERT, UPDATE, etc.)
            duration: Query duration in seconds
            success: Whether query succeeded
        """
        if self.meter is None:
            return
        
        try:
            attributes = {
                "db.operation": query_type,
                "db.success": success,
            }
            
            self._metrics['db_query_duration'].record(duration, attributes)
            
            if not success:
                self._metrics['error_count'].add(1, {"error_type": f"DB_{query_type}"})
        except Exception as e:
            self.logger.debug(f"Failed to record DB metrics: {e}")
    
    def record_llm_judge_evaluation(self, duration: float, success: bool = True):
        """Record LLM judge evaluation metrics.
        
        Args:
            duration: Evaluation duration in seconds
            success: Whether evaluation succeeded
        """
        if self.meter is None:
            return
        
        try:
            attributes = {"llm_judge.success": success}
            self._metrics['llm_judge_duration'].record(duration, attributes)
            
            if not success:
                self._metrics['error_count'].add(1, {"error_type": "LLM_JUDGE"})
        except Exception as e:
            self.logger.debug(f"Failed to record LLM judge metrics: {e}")
    
    def record_streaming_event(self, event_type: str, success: bool = True):
        """Record streaming event metrics.
        
        Args:
            event_type: Type of streaming event
            success: Whether event was successfully sent
        """
        if self.meter is None:
            return
        
        try:
            attributes = {
                "stream.event_type": event_type,
                "stream.success": success,
            }
            self._metrics['chat_streaming_events'].add(1, attributes)
        except Exception as e:
            self.logger.debug(f"Failed to record streaming metrics: {e}")
    
    def set_pool_utilization(self, utilization_percent: float):
        """Set database connection pool utilization gauge.
        
        Args:
            utilization_percent: Pool utilization percentage (0-100)
        """
        if self.meter is None:
            return
        
        try:
            self._metrics['db_connection_pool_utilization'].record(utilization_percent)
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
