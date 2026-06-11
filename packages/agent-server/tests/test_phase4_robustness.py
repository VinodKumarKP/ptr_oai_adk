"""Phase 4 robustness tests - Circuit breaker, retry strategy, caching, versioning.

Tests cover:
- Circuit breaker state management
- Retry logic with exponential backoff
- Agent info and logs caching
- API versioning
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta

from oai_agent_server.utils.circuit_breaker import (
    CircuitBreaker, CircuitBreakerRegistry, CircuitState, CircuitBreakerError
)
from oai_agent_server.utils.retry_strategy import (
    Retry, RetryStrategy, RetryConfig, RetryRegistry, create_http_retry_config
)
from oai_agent_server.utils.agent_cache import (
    CachedAgent, CachedAgentLogs, AgentInfoCache, AgentLogsCache, AgentCacheManager
)
from oai_agent_server.utils.api_versioning import (
    APIVersion, VersionedEndpoint, APIVersionRegistry, VersionedRouter
)


# ============================================================================
# CIRCUIT BREAKER TESTS
# ============================================================================

class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts in CLOSED state."""
        breaker = CircuitBreaker("test_service")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_records_success(self):
        """Test circuit breaker records successful calls."""
        breaker = CircuitBreaker("test_service")
        
        async def success_func():
            return "success"
        
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after failure threshold."""
        breaker = CircuitBreaker(
            "test_service",
            failure_threshold=3
        )
        
        async def fail_func():
            raise ValueError("Service error")
        
        # First 3 failures
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fail_func)
        
        # Should be OPEN now
        assert breaker.state == CircuitState.OPEN
        
        # Next call should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            await breaker.call(fail_func)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_transition(self):
        """Test circuit breaker transitions to HALF_OPEN after timeout."""
        breaker = CircuitBreaker(
            "test_service",
            failure_threshold=2,
            timeout=0,  # Immediate timeout for testing
        )
        
        async def fail_func():
            raise ValueError("Service error")
        
        # Trigger OPEN state
        with pytest.raises(ValueError):
            await breaker.call(fail_func)
        with pytest.raises(ValueError):
            await breaker.call(fail_func)
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for timeout and try again
        await asyncio.sleep(0.1)
        
        async def success_func():
            return "recovered"
        
        result = await breaker.call(success_func)
        assert result == "recovered"
        assert breaker.state == CircuitState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closes_after_success_threshold(self):
        """Test circuit breaker closes after success threshold in HALF_OPEN."""
        breaker = CircuitBreaker(
            "test_service",
            failure_threshold=1,
            success_threshold=2,
            timeout=0,
        )
        
        async def fail_func():
            raise ValueError("Error")
        
        async def success_func():
            return "ok"
        
        # Trigger OPEN
        with pytest.raises(ValueError):
            await breaker.call(fail_func)
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait and transition to HALF_OPEN
        await asyncio.sleep(0.1)
        await breaker.call(success_func)
        assert breaker.state == CircuitState.HALF_OPEN
        
        # One more success should close
        await breaker.call(success_func)
        assert breaker.state == CircuitState.CLOSED
    
    def test_circuit_breaker_get_state(self):
        """Test getting circuit breaker state snapshot."""
        breaker = CircuitBreaker("test_service")
        state = breaker.get_state()
        
        assert state["name"] == "test_service"
        assert state["state"] == "closed"
        assert state["failure_count"] == 0


class TestCircuitBreakerRegistry:
    """Test circuit breaker registry."""
    
    @pytest.mark.asyncio
    async def test_registry_register_breaker(self):
        """Test registering circuit breaker."""
        registry = CircuitBreakerRegistry()
        
        breaker = await registry.register(
            "service1",
            failure_threshold=5
        )
        
        assert breaker.name == "service1"
        assert registry.get("service1") is breaker
    
    @pytest.mark.asyncio
    async def test_registry_call_with_protection(self):
        """Test registry call with protection."""
        registry = CircuitBreakerRegistry()
        await registry.register("service1", failure_threshold=1)
        
        async def success_func():
            return "result"
        
        result = await registry.call("service1", success_func)
        assert result == "result"
    
    @pytest.mark.asyncio
    async def test_registry_get_all_states(self):
        """Test getting all breaker states."""
        registry = CircuitBreakerRegistry()
        await registry.register("service1")
        await registry.register("service2")
        
        states = registry.get_all_states()
        
        assert "service1" in states
        assert "service2" in states
        assert states["service1"]["state"] == "closed"


# ============================================================================
# RETRY STRATEGY TESTS
# ============================================================================

class TestRetryStrategy:
    """Test retry strategy functionality."""
    
    def test_retry_config_delay_exponential(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            initial_delay=0.1,
            max_delay=10.0,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter=False,
        )
        
        assert config.get_delay(0) == pytest.approx(0.1)
        assert config.get_delay(1) == pytest.approx(0.2)
        assert config.get_delay(2) == pytest.approx(0.4)
        assert config.get_delay(3) == pytest.approx(0.8)
    
    def test_retry_config_delay_linear(self):
        """Test linear backoff delay calculation."""
        config = RetryConfig(
            initial_delay=0.1,
            strategy=RetryStrategy.LINEAR,
            jitter=False,
        )
        
        assert config.get_delay(0) == pytest.approx(0.1)
        assert config.get_delay(1) == pytest.approx(0.2)
        assert config.get_delay(2) == pytest.approx(0.3)
    
    def test_retry_config_delay_fixed(self):
        """Test fixed delay strategy."""
        config = RetryConfig(
            initial_delay=0.5,
            strategy=RetryStrategy.FIXED,
            jitter=False,
        )
        
        assert config.get_delay(0) == pytest.approx(0.5)
        assert config.get_delay(1) == pytest.approx(0.5)
        assert config.get_delay(2) == pytest.approx(0.5)
    
    def test_retry_config_max_delay(self):
        """Test max delay ceiling."""
        config = RetryConfig(
            initial_delay=1.0,
            max_delay=2.0,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter=False,
        )
        
        assert config.get_delay(5) == pytest.approx(2.0)  # Would be 32, capped at 2
    
    @pytest.mark.asyncio
    async def test_retry_execute_success(self):
        """Test retry executes successfully."""
        retry = Retry(max_attempts=3, initial_delay=0.01)
        
        async def success_func():
            return "success"
        
        result = await retry.execute(success_func)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_retry_execute_retries_then_succeeds(self):
        """Test retry succeeds after failures."""
        attempt_count = [0]
        retry = Retry(
            max_attempts=3,
            initial_delay=0.01,
            strategy=RetryStrategy.FIXED
        )
        
        async def sometimes_fails():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = await retry.execute(sometimes_fails)
        assert result == "success"
        assert attempt_count[0] == 3
    
    @pytest.mark.asyncio
    async def test_retry_execute_exhausts_attempts(self):
        """Test retry exhausts attempts and raises."""
        retry = Retry(max_attempts=2, initial_delay=0.01)
        
        async def always_fails():
            raise ValueError("Persistent error")
        
        with pytest.raises(ValueError, match="Persistent error"):
            await retry.execute(always_fails)
    
    def test_retry_config_is_retryable(self):
        """Test exception retryability check."""
        config = RetryConfig(
            retryable_exceptions=[TimeoutError, ConnectionError]
        )
        
        assert config.is_retryable(TimeoutError("timeout"))
        assert config.is_retryable(ConnectionError("connection"))
        assert not config.is_retryable(ValueError("value"))


class TestRetryRegistry:
    """Test retry registry."""
    
    @pytest.mark.asyncio
    async def test_registry_register_policy(self):
        """Test registering retry policy."""
        registry = RetryRegistry()
        config = RetryConfig(max_attempts=5)
        
        registry.register_policy("service1", config)
        
        assert registry.get_policy("service1") is config
    
    @pytest.mark.asyncio
    async def test_registry_execute_with_policy(self):
        """Test executing with registered policy."""
        registry = RetryRegistry()
        config = create_http_retry_config("service1")
        registry.register_policy("service1", config)
        
        async def success_func():
            return "result"
        
        result = await registry.execute(
            "service1",
            success_func,
            operation_name="test_op"
        )
        
        assert result == "result"


# ============================================================================
# AGENT CACHE TESTS
# ============================================================================

class TestAgentInfoCache:
    """Test agent info cache functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test caching and retrieving agent info."""
        cache = AgentInfoCache()
        
        agent = CachedAgent(
            agent_id="agent1",
            agent_name="Test Agent",
            capabilities=["chat", "analysis"],
            config={"version": "1.0"},
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        await cache.set("agent1", agent)
        result = await cache.get("agent1")
        
        assert result is not None
        assert result.agent_name == "Test Agent"
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = AgentInfoCache()
        
        result = await cache.get("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test cache entry expires."""
        cache = AgentInfoCache(default_ttl=1)
        
        agent = CachedAgent(
            agent_id="agent1",
            agent_name="Test",
            capabilities=[],
            config={},
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        await cache.set("agent1", agent)
        
        # Simulate expiration
        entry = cache._cache["agent1"]
        entry.cached_at = datetime.now() - timedelta(seconds=2)
        
        result = await cache.get("agent1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction when cache full."""
        cache = AgentInfoCache(max_agents=2)
        
        for i in range(3):
            agent = CachedAgent(
                agent_id=f"agent{i}",
                agent_name=f"Agent {i}",
                capabilities=[],
                config={},
                status="active",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            await cache.set(f"agent{i}", agent)
        
        # agent0 should be evicted (LRU)
        result = await cache.get("agent0")
        assert result is None
        
        # agent1 and agent2 should still exist
        assert await cache.get("agent1") is not None
        assert await cache.get("agent2") is not None
    
    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics."""
        cache = AgentInfoCache()
        
        agent = CachedAgent(
            agent_id="agent1",
            agent_name="Test",
            capabilities=[],
            config={},
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        await cache.set("agent1", agent)
        await cache.get("agent1")  # Hit
        await cache.get("agent1")  # Hit
        await cache.get("agent2")  # Miss
        
        stats = await cache.get_stats()
        
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["cached_agents"] == 1


class TestAgentLogsCache:
    """Test agent logs cache functionality."""
    
    @pytest.mark.asyncio
    async def test_logs_cache_set_and_get(self):
        """Test caching and retrieving logs."""
        cache = AgentLogsCache()
        
        logs = CachedAgentLogs(
            agent_id="agent1",
            logs=[{"id": 1, "message": "test"}],
            total_count=100,
            page=1,
            page_size=10,
        )
        
        await cache.set("agent1", logs)
        result = await cache.get("agent1", page=1)
        
        assert result is not None
        assert result.logs[0]["message"] == "test"
    
    @pytest.mark.asyncio
    async def test_logs_cache_pagination(self):
        """Test caching multiple pages."""
        cache = AgentLogsCache()
        
        for page in range(1, 4):
            logs = CachedAgentLogs(
                agent_id="agent1",
                logs=[{"page": page, "id": i} for i in range(10)],
                total_count=30,
                page=page,
                page_size=10,
            )
            await cache.set("agent1", logs)
        
        page1 = await cache.get("agent1", page=1)
        page2 = await cache.get("agent1", page=2)
        
        assert page1.logs[0]["page"] == 1
        assert page2.logs[0]["page"] == 2
    
    @pytest.mark.asyncio
    async def test_logs_cache_invalidate(self):
        """Test invalidating logs for agent."""
        cache = AgentLogsCache()
        
        logs = CachedAgentLogs(
            agent_id="agent1",
            logs=[{"id": 1}],
            total_count=1,
            page=1,
            page_size=10,
        )
        
        await cache.set("agent1", logs)
        await cache.invalidate("agent1")
        
        result = await cache.get("agent1", page=1)
        assert result is None


# ============================================================================
# API VERSIONING TESTS
# ============================================================================

class TestAPIVersioning:
    """Test API versioning functionality."""
    
    @pytest.mark.asyncio
    async def test_versioned_endpoint_execute(self):
        """Test executing versioned endpoint."""
        v1_handler = AsyncMock(return_value="v1_result")
        v2_handler = AsyncMock(return_value="v2_result")
        
        endpoint = VersionedEndpoint(
            path="/chat",
            versions={
                APIVersion.V1: v1_handler,
                APIVersion.V2: v2_handler,
            }
        )
        
        result_v1 = await endpoint.execute("v1")
        result_v2 = await endpoint.execute("v2")
        
        assert result_v1 == "v1_result"
        assert result_v2 == "v2_result"
    
    @pytest.mark.asyncio
    async def test_versioned_endpoint_default_to_latest(self):
        """Test endpoint defaults to latest version."""
        v1_handler = AsyncMock(return_value="v1_result")
        v2_handler = AsyncMock(return_value="v2_result")
        
        endpoint = VersionedEndpoint(
            path="/chat",
            versions={
                APIVersion.V1: v1_handler,
                APIVersion.V2: v2_handler,
            }
        )
        
        result = await endpoint.execute(None)  # No version specified
        
        assert result == "v2_result"  # Should use v2 (latest)
    
    @pytest.mark.asyncio
    async def test_versioned_endpoint_invalid_version(self):
        """Test invalid version raises error."""
        v1_handler = AsyncMock(return_value="v1_result")
        
        endpoint = VersionedEndpoint(
            path="/chat",
            versions={APIVersion.V1: v1_handler}
        )
        
        with pytest.raises(ValueError, match="Unsupported API version"):
            await endpoint.execute("v5")


class TestAPIVersionRegistry:
    """Test API version registry."""
    
    @pytest.mark.asyncio
    async def test_registry_register_endpoint(self):
        """Test registering versioned endpoint."""
        registry = APIVersionRegistry()
        
        v1_handler = AsyncMock(return_value="v1")
        v2_handler = AsyncMock(return_value="v2")
        
        registry.register(
            "/chat",
            {
                APIVersion.V1: v1_handler,
                APIVersion.V2: v2_handler,
            }
        )
        
        assert "/chat" in registry._endpoints
    
    @pytest.mark.asyncio
    async def test_registry_get_endpoint_info(self):
        """Test getting endpoint information."""
        registry = APIVersionRegistry()
        
        v1_handler = AsyncMock(return_value="v1")
        v2_handler = AsyncMock(return_value="v2")
        
        registry.register(
            "/chat",
            {
                APIVersion.V1: v1_handler,
                APIVersion.V2: v2_handler,
            }
        )
        
        info = registry.get_endpoint_info("/chat")
        
        assert info["path"] == "/chat"
        assert "v1" in info["versions"]
        assert "v2" in info["versions"]
        assert info["latest_version"] == "v2"
    
    @pytest.mark.asyncio
    async def test_registry_get_all_endpoints_info(self):
        """Test getting all endpoints info."""
        registry = APIVersionRegistry()
        
        v1 = AsyncMock(return_value="v1")
        v2 = AsyncMock(return_value="v2")
        
        registry.register("/chat", {APIVersion.V1: v1, APIVersion.V2: v2})
        registry.register("/agent", {APIVersion.V1: v1, APIVersion.V2: v2})
        
        all_info = registry.get_all_endpoints_info()
        
        assert "/chat" in all_info
        assert "/agent" in all_info


class TestAgentCacheManager:
    """Test agent cache manager."""
    
    @pytest.mark.asyncio
    async def test_manager_invalidate_agent(self):
        """Test invalidating all cache for agent."""
        manager = AgentCacheManager()
        
        agent = CachedAgent(
            agent_id="agent1",
            agent_name="Test",
            capabilities=[],
            config={},
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        await manager.info.set("agent1", agent)
        await manager.invalidate_agent("agent1")
        
        result = await manager.info.get("agent1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_manager_get_stats(self):
        """Test getting manager statistics."""
        manager = AgentCacheManager()
        
        stats = await manager.get_stats()
        
        assert "agent_info" in stats
        assert "agent_logs" in stats
        assert "hits" in stats["agent_info"]
        assert "misses" in stats["agent_logs"]
