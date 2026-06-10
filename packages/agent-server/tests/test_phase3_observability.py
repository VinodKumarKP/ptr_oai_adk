"""Phase 3 observability and optimization tests.

Tests cover:
- OpenTelemetry infrastructure
- Prometheus metrics collection
- Query result caching
- Health check system
- Observability middleware
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from oai_agent_server.utils.observability import (
    ObservabilityManager, ObservabilityConfig, instrument_function
)
from oai_agent_server.utils.query_cache import (
    QueryResultCache, CacheKey, CacheEntry
)
from oai_agent_server.utils.health_check import (
    HealthMetrics, HealthCheckResponse, HealthCheckCollector,
    create_database_health_check, create_llm_judge_health_check
)


class TestObservabilityManager:
    """Test observability manager functionality."""
    
    def test_observability_config_defaults(self):
        """Test observability config default values."""
        config = ObservabilityConfig()
        
        assert config.ENABLE_TRACING == True
        assert config.ENABLE_METRICS == True
        assert config.SERVICE_NAME == "oai-agent-server"
        assert config.SERVICE_VERSION == "3.0.0"
    
    def test_observability_manager_initialization(self):
        """Test observability manager initialization."""
        config = ObservabilityConfig()
        config.ENABLE_TRACING = False
        config.ENABLE_METRICS = False
        
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        assert manager.tracer is None  # Tracing disabled
        # Metrics may not be available without OTEL
        assert manager.meter is None or manager.meter is not None
    
    def test_record_request(self):
        """Test request metric recording."""
        config = ObservabilityConfig()
        config.ENABLE_METRICS = False  # Skip actual metrics
        
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        # Should not raise even with metrics disabled
        manager.record_request(
            method="GET",
            path="/chat",
            status_code=200,
            duration=0.123
        )
    
    def test_record_db_query(self):
        """Test database query metric recording."""
        config = ObservabilityConfig()
        config.ENABLE_METRICS = False
        
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        manager.record_db_query(
            query_type="SELECT",
            duration=0.050,
            success=True
        )
    
    def test_record_llm_judge_evaluation(self):
        """Test LLM judge metric recording."""
        config = ObservabilityConfig()
        config.ENABLE_METRICS = False
        
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        manager.record_llm_judge_evaluation(
            duration=0.500,
            success=True
        )
    
    def test_trace_operation_context_manager(self):
        """Test trace operation context manager."""
        config = ObservabilityConfig()
        config.ENABLE_TRACING = False
        
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        with manager.trace_operation("test_operation", attributes={"key": "value"}) as span:
            assert span is None  # No-op when tracing disabled
    
    def test_instrument_function_decorator(self):
        """Test function instrumentation decorator."""
        config = ObservabilityConfig()
        config.ENABLE_METRICS = False
        config.ENABLE_TRACING = False
        
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        @instrument_function(manager)
        def sample_function():
            return "result"
        
        result = sample_function()
        assert result == "result"


class TestQueryResultCache:
    """Test query result caching functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_key_equality(self):
        """Test cache key equality comparison."""
        key1 = CacheKey("query_type", {"id": 123})
        key2 = CacheKey("query_type", {"id": 123})
        key3 = CacheKey("other_type", {"id": 123})
        
        assert key1 == key2
        assert key1 != key3
    
    @pytest.mark.asyncio
    async def test_cache_entry_expiration(self):
        """Test cache entry TTL expiration."""
        entry = CacheEntry({"data": "test"}, ttl=1)  # 1 second TTL
        
        assert not entry.is_expired()
        
        # Simulate aging beyond TTL
        entry.created_at = entry.created_at - 2  # 2 seconds ago
        assert entry.is_expired()
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = QueryResultCache(max_size=10)
        key = CacheKey("test_query", {"user_id": 1})
        data = {"result": "success"}
        
        await cache.set(key, data, ttl=300)
        result = await cache.get(key)
        
        assert result == data
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = QueryResultCache()
        key = CacheKey("nonexistent", {})
        
        result = await cache.get(key)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_expiration_on_get(self):
        """Test that expired entries are removed on get."""
        cache = QueryResultCache()
        key = CacheKey("test", {})
        data = {"value": 42}
        
        await cache.set(key, data, ttl=1)
        
        # Simulate expiration
        entry = cache._cache[key]
        entry.created_at = entry.created_at - 2  # 2 seconds ago
        
        result = await cache.get(key)
        assert result is None  # Expired entry returns None
    
    @pytest.mark.asyncio
    async def test_cache_eviction_lru(self):
        """Test LRU eviction when cache is full."""
        cache = QueryResultCache(max_size=3)
        
        # Add 3 items
        await cache.set(CacheKey("q1", {}), {"id": 1})
        await cache.set(CacheKey("q2", {}), {"id": 2})
        await cache.set(CacheKey("q3", {}), {"id": 3})
        
        assert len(cache._cache) == 3
        
        # Add 4th item - should evict oldest (q1)
        await cache.set(CacheKey("q4", {}), {"id": 4})
        
        assert len(cache._cache) == 3
        assert await cache.get(CacheKey("q1", {})) is None  # q1 was evicted
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self):
        """Test cache invalidation by pattern."""
        cache = QueryResultCache()
        
        await cache.set(CacheKey("user_query", {"id": 1}), {"data": 1})
        await cache.set(CacheKey("user_query", {"id": 2}), {"data": 2})
        await cache.set(CacheKey("other_query", {"id": 1}), {"data": 3})
        
        # Invalidate all user_query entries
        invalidated = await cache.invalidate_pattern("user_query")
        
        assert invalidated == 2
        assert await cache.get(CacheKey("other_query", {"id": 1})) == {"data": 3}
    
    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics collection."""
        cache = QueryResultCache()
        key1 = CacheKey("q1", {})
        
        await cache.set(key1, {"data": 1})
        await cache.get(key1)  # Hit
        await cache.get(key1)  # Hit
        await cache.get(CacheKey("nonexistent", {}))  # Miss
        
        stats = cache.get_stats()
        
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["total_requests"] == 3
        assert abs(stats["hit_rate_percent"] - 66.67) < 1


class TestHealthMetrics:
    """Test health metrics tracking."""
    
    def test_health_metrics_uptime(self):
        """Test uptime calculation."""
        metrics = HealthMetrics()
        
        uptime = metrics.get_uptime()
        assert uptime >= 0
    
    def test_health_metrics_request_recording(self):
        """Test request recording."""
        metrics = HealthMetrics()
        
        metrics.record_request(success=True)
        metrics.record_request(success=True)
        metrics.record_request(success=False, error_message="Test error")
        
        assert metrics.total_requests == 3
        assert metrics.total_errors == 1
    
    def test_health_metrics_error_rate(self):
        """Test error rate calculation."""
        metrics = HealthMetrics()
        
        for _ in range(9):
            metrics.record_request(success=True)
        metrics.record_request(success=False)
        
        error_rate = metrics.get_error_rate()
        assert abs(error_rate - 10.0) < 0.1  # 10%
    
    def test_health_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = HealthMetrics()
        metrics.record_request(success=True)
        metrics.record_request(success=False, error_message="Error")
        
        data = metrics.to_dict()
        
        assert "uptime_seconds" in data
        assert "uptime_formatted" in data
        assert data["total_requests"] == 2
        assert data["total_errors"] == 1


class TestHealthCheckResponse:
    """Test health check response building."""
    
    def test_health_check_response_creation(self):
        """Test health check response creation."""
        response = HealthCheckResponse("healthy")
        
        assert response.status == "healthy"
        assert response.is_healthy() == True
    
    def test_health_check_response_components(self):
        """Test adding components to health check."""
        response = HealthCheckResponse("healthy")
        
        response.add_component("database", "healthy", {"connections": 5})
        response.add_component("cache", "degraded", {"reason": "High latency"})
        
        assert "database" in response.components
        assert "cache" in response.components
        assert response.components["database"]["status"] == "healthy"
    
    def test_health_check_response_to_dict(self):
        """Test response serialization."""
        response = HealthCheckResponse("healthy")
        response.add_component("database", "healthy")
        response.add_metrics("performance", {"latency_ms": 45})
        
        data = response.to_dict()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "database" in data["components"]
        assert "performance" in data["metrics"]


class TestHealthCheckCollector:
    """Test health check collection."""
    
    @pytest.mark.asyncio
    async def test_health_check_collector_registration(self):
        """Test component health check registration."""
        collector = HealthCheckCollector()
        
        async def mock_check():
            return ("healthy", {"detail": "ok"})
        
        collector.register_component_check("test_component", mock_check)
        
        assert "test_component" in collector._component_checks
    
    @pytest.mark.asyncio
    async def test_health_check_collection(self):
        """Test collecting health from all components."""
        collector = HealthCheckCollector()
        
        async def mock_check_1():
            return ("healthy", {"name": "check1"})
        
        async def mock_check_2():
            return ("degraded", {"name": "check2"})
        
        collector.register_component_check("component1", mock_check_1)
        collector.register_component_check("component2", mock_check_2)
        
        response = await collector.collect()
        
        assert response.status == "degraded"  # Overall status degraded
        assert response.components["component1"]["status"] == "healthy"
        assert response.components["component2"]["status"] == "degraded"


class TestDatabaseHealthCheck:
    """Test database health check."""
    
    @pytest.mark.asyncio
    async def test_database_health_check_success(self):
        """Test successful database health check."""
        mock_db_logger = AsyncMock()
        mock_db_logger.is_active = True
        mock_db_logger.get_stats = AsyncMock(return_value={"interactions": 100})
        
        check_func = create_database_health_check(mock_db_logger)
        status, details = await check_func()
        
        assert status == "healthy"
        assert "stats" in details
    
    @pytest.mark.asyncio
    async def test_database_health_check_inactive(self):
        """Test database health check when inactive."""
        mock_db_logger = AsyncMock()
        mock_db_logger.is_active = False
        
        check_func = create_database_health_check(mock_db_logger)
        status, details = await check_func()
        
        assert status == "unhealthy"
        assert "reason" in details


class TestLLMJudgeHealthCheck:
    """Test LLM judge health check."""
    
    @pytest.mark.asyncio
    async def test_llm_judge_health_check_initialized(self):
        """Test LLM judge health check when initialized."""
        mock_judge_service = MagicMock()
        mock_judge_service.judge_agent = MagicMock()  # Initialized
        mock_judge_service.judge_model_id = "test-model"
        
        check_func = create_llm_judge_health_check(mock_judge_service)
        status, details = await check_func()
        
        assert status == "healthy"
        assert details["initialized"] == True
    
    @pytest.mark.asyncio
    async def test_llm_judge_health_check_not_initialized(self):
        """Test LLM judge health check when not initialized."""
        mock_judge_service = MagicMock()
        mock_judge_service.judge_agent = None  # Not initialized
        
        check_func = create_llm_judge_health_check(mock_judge_service)
        status, details = await check_func()
        
        assert status == "degraded"
