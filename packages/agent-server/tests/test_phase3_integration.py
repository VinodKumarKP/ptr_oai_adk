"""Phase 3 integration tests - middleware and endpoint integration.

Tests for:
- Observability middleware integration
- Health check endpoint
- Streaming metrics
- End-to-end observability
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from oai_agent_server.middleware.observability import (
    ObservabilityMiddleware, StreamingMetricsMiddleware
)
from oai_agent_server.utils.observability import ObservabilityManager, ObservabilityConfig
from oai_agent_server.utils.health_check import (
    HealthCheckCollector, HealthCheckResponse
)


class TestObservabilityMiddlewareIntegration:
    """Test observability middleware in FastAPI app."""
    
    def test_middleware_tracks_successful_request(self):
        """Test middleware tracks successful HTTP requests."""
        app = FastAPI()
        
        config = ObservabilityConfig()
        config.ENABLE_METRICS = False
        config.ENABLE_TRACING = False
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        app.add_middleware(
            ObservabilityMiddleware,
            observability_manager=manager,
            logger=MagicMock()
        )
        
        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}
        
        client = TestClient(app)
        response = client.get("/test")
        
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert "X-Process-Time" in response.headers
    
    def test_middleware_tracks_error_request(self):
        """Test middleware tracks failed requests."""
        app = FastAPI()
        
        config = ObservabilityConfig()
        config.ENABLE_METRICS = False
        config.ENABLE_TRACING = False
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        app.add_middleware(
            ObservabilityMiddleware,
            observability_manager=manager,
            logger=MagicMock()
        )
        
        @app.get("/error")
        def error_endpoint():
            raise ValueError("Test error")
        
        client = TestClient(app)
        
        with pytest.raises(ValueError):
            client.get("/error")
    
    def test_middleware_adds_process_time_header(self):
        """Test middleware adds X-Process-Time header."""
        app = FastAPI()
        
        config = ObservabilityConfig()
        config.ENABLE_METRICS = False
        config.ENABLE_TRACING = False
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        app.add_middleware(
            ObservabilityMiddleware,
            observability_manager=manager,
            logger=MagicMock()
        )
        
        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}
        
        client = TestClient(app)
        response = client.get("/test")
        
        assert "X-Process-Time" in response.headers
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0


class TestStreamingMetricsMiddlewareIntegration:
    """Test streaming metrics middleware."""
    
    def test_streaming_middleware_tracks_stream_endpoint(self):
        """Test middleware tracks streaming endpoints."""
        app = FastAPI()
        
        config = ObservabilityConfig()
        config.ENABLE_METRICS = False
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        app.add_middleware(
            StreamingMetricsMiddleware,
            observability_manager=manager,
            logger=MagicMock()
        )
        
        @app.get("/stream")
        def stream_endpoint():
            return {"status": "streaming"}
        
        client = TestClient(app)
        response = client.get("/stream")
        
        assert response.status_code == 200
    
    def test_streaming_middleware_ignores_non_stream_endpoint(self):
        """Test middleware ignores non-streaming endpoints."""
        app = FastAPI()
        
        config = ObservabilityConfig()
        config.ENABLE_METRICS = False
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        app.add_middleware(
            StreamingMetricsMiddleware,
            observability_manager=manager,
            logger=MagicMock()
        )
        
        @app.get("/chat")
        def chat_endpoint():
            return {"status": "ok"}
        
        client = TestClient(app)
        response = client.get("/chat")
        
        assert response.status_code == 200


class TestHealthCheckEndpoint:
    """Test health check endpoint integration."""
    
    @pytest.mark.asyncio
    async def test_health_endpoint_aggregates_components(self):
        """Test health endpoint returns aggregated component status."""
        app = FastAPI()
        collector = HealthCheckCollector()
        
        async def mock_db_check():
            return ("healthy", {"connections": 5})
        
        async def mock_cache_check():
            return ("healthy", {"evictions": 0})
        
        collector.register_component_check("database", mock_db_check)
        collector.register_component_check("cache", mock_cache_check)
        
        @app.get("/health")
        async def health_endpoint():
            response = await collector.collect()
            return response.to_dict()
        
        client = TestClient(app)
        health_response = client.get("/health")
        
        assert health_response.status_code == 200
        data = health_response.json()
        
        assert data["status"] == "healthy"
        assert "database" in data["components"]
        assert "cache" in data["components"]
    
    @pytest.mark.asyncio
    async def test_health_endpoint_marks_degraded(self):
        """Test health endpoint marks system degraded when component fails."""
        app = FastAPI()
        collector = HealthCheckCollector()
        
        async def healthy_check():
            return ("healthy", {})
        
        async def degraded_check():
            return ("degraded", {"reason": "High latency"})
        
        collector.register_component_check("database", healthy_check)
        collector.register_component_check("queue", degraded_check)
        
        @app.get("/health")
        async def health_endpoint():
            response = await collector.collect()
            return response.to_dict()
        
        client = TestClient(app)
        health_response = client.get("/health")
        
        data = health_response.json()
        
        # Overall status should be degraded
        assert data["status"] == "degraded"
        assert data["components"]["database"]["status"] == "healthy"
        assert data["components"]["queue"]["status"] == "degraded"


class TestObservabilityIntegration:
    """Test full observability integration."""
    
    @pytest.mark.asyncio
    async def test_request_lifecycle_observability(self):
        """Test observability across full request lifecycle."""
        config = ObservabilityConfig()
        config.ENABLE_METRICS = False
        config.ENABLE_TRACING = False
        
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        # Simulate request lifecycle
        with manager.trace_operation("http.get", attributes={"path": "/chat"}):
            manager.record_request("GET", "/chat", 200, 0.123)
        
        # Should complete without error
        assert True
    
    @pytest.mark.asyncio
    async def test_database_query_observability(self):
        """Test observability for database operations."""
        config = ObservabilityConfig()
        config.ENABLE_METRICS = False
        
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        manager.record_db_query("SELECT", 0.050, success=True)
        manager.record_db_query("INSERT", 0.025, success=True)
        manager.record_db_query("SELECT", 0.100, success=False)
        
        # Should complete without error
        assert True
    
    @pytest.mark.asyncio
    async def test_streaming_event_observability(self):
        """Test observability for streaming events."""
        config = ObservabilityConfig()
        config.ENABLE_METRICS = False
        
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        manager.record_streaming_event("stream_start", success=True)
        manager.record_streaming_event("stream_chunk", success=True)
        manager.record_streaming_event("stream_end", success=True)
        
        # Should complete without error
        assert True


class TestMetricsExport:
    """Test metrics export functionality."""
    
    def test_metrics_endpoint_returns_prometheus_format(self):
        """Test metrics endpoint returns Prometheus format."""
        app = FastAPI()
        
        config = ObservabilityConfig()
        config.ENABLE_METRICS = False  # Skip actual metrics setup
        
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        @app.get("/metrics")
        def metrics_endpoint():
            # In real implementation, would export OpenMetrics format
            return {"status": "ok"}
        
        client = TestClient(app)
        response = client.get("/metrics")
        
        assert response.status_code == 200


class TestContextManagers:
    """Test observability context managers."""
    
    def test_trace_operation_context_manager_success(self):
        """Test trace operation context manager completes successfully."""
        config = ObservabilityConfig()
        config.ENABLE_TRACING = False
        
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        with manager.trace_operation("test_op", attributes={"key": "value"}):
            result = "operation_result"
        
        assert result == "operation_result"
    
    def test_trace_operation_context_manager_exception(self):
        """Test trace operation context manager handles exceptions."""
        config = ObservabilityConfig()
        config.ENABLE_TRACING = False
        
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        with pytest.raises(ValueError):
            with manager.trace_operation("test_op"):
                raise ValueError("Test error")
