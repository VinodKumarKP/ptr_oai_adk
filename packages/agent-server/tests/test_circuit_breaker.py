import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from oai_agent_server.utils.circuit_breaker import (
    CircuitState, CircuitBreakerError, CircuitBreaker, CircuitBreakerRegistry
)

def test_circuit_breaker_error():
    err = CircuitBreakerError("test-service", "test-reason")
    assert err.service_name == "test-service"
    assert err.reason == "test-reason"
    assert "test-service" in str(err)

@pytest.mark.asyncio
async def test_circuit_breaker_flow():
    breaker = CircuitBreaker(
        name="test-service",
        failure_threshold=2,
        success_threshold=2,
        timeout=10
    )
    
    assert breaker.state == CircuitState.CLOSED
    
    # Success scenario
    async def success_fn(val):
        return val * 2
        
    res = await breaker.call(success_fn, 5)
    assert res == 10
    assert breaker.failure_count == 0
    
    # Failure scenario
    async def failure_fn():
        raise ValueError("failed call")
        
    with pytest.raises(ValueError):
        await breaker.call(failure_fn)
    assert breaker.failure_count == 1
    assert breaker.state == CircuitState.CLOSED
    
    # Second failure triggers OPEN
    with pytest.raises(ValueError):
        await breaker.call(failure_fn)
    assert breaker.failure_count == 2
    assert breaker.state == CircuitState.OPEN
    assert breaker.opened_at is not None
    assert breaker._time_since_open() >= 0
    
    # Call when OPEN raises CircuitBreakerError
    with pytest.raises(CircuitBreakerError):
        await breaker.call(success_fn, 5)
        
    # Attempt recovery (simulate time pass)
    with patch('oai_agent_server.utils.circuit_breaker.datetime') as mock_datetime:
        # Move time forward by 15 seconds
        opened_time = breaker.opened_at
        mock_datetime.now.return_value = opened_time + timedelta(seconds=15)
        
        # Next call transitions to HALF_OPEN and attempts function execution
        res = await breaker.call(success_fn, 10)
        assert res == 20
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.success_count == 1
        
        # Second success in HALF_OPEN transitions to CLOSED
        res = await breaker.call(success_fn, 10)
        assert res == 20
        assert breaker.state == CircuitState.CLOSED
        assert breaker.success_count == 2
        assert breaker.failure_count == 0

@pytest.mark.asyncio
async def test_circuit_breaker_half_open_failure():
    breaker = CircuitBreaker(
        name="test-service",
        failure_threshold=2,
        success_threshold=2,
        timeout=10
    )
    
    async def failure_fn():
        raise ValueError("failed call")
        
    # Trip circuit to OPEN
    with pytest.raises(ValueError):
        await breaker.call(failure_fn)
    with pytest.raises(ValueError):
        await breaker.call(failure_fn)
    assert breaker.state == CircuitState.OPEN
    
    # Move time forward by 15 seconds to trigger HALF_OPEN
    with patch('oai_agent_server.utils.circuit_breaker.datetime') as mock_datetime:
        opened_time = breaker.opened_at
        mock_datetime.now.return_value = opened_time + timedelta(seconds=15)
        
        # Call fails in HALF_OPEN, should immediately transition to OPEN
        with pytest.raises(ValueError):
            await breaker.call(failure_fn)
        assert breaker.state == CircuitState.OPEN
        assert breaker.success_count == 0

@pytest.mark.asyncio
async def test_circuit_breaker_registry():
    registry = CircuitBreakerRegistry()
    
    # Register breaker
    breaker1 = await registry.register("service1", failure_threshold=3)
    assert breaker1.name == "service1"
    assert breaker1.failure_threshold == 3
    
    # Register duplicate returns cached
    breaker2 = await registry.register("service1")
    assert breaker1 is breaker2
    
    # Get breaker
    assert registry.get("service1") is breaker1
    assert registry.get("non-existent") is None
    
    # Call with breaker
    async def dummy_fn():
        return "ok"
    res = await registry.call("service1", dummy_fn)
    assert res == "ok"
    
    # Call without breaker (no protection)
    res_no_prot = await registry.call("non-existent", dummy_fn)
    assert res_no_prot == "ok"
    
    # Get all states
    states = registry.get_all_states()
    assert "service1" in states
    assert states["service1"]["state"] == "closed"
    
    # Reset all
    breaker1.state = CircuitState.OPEN
    breaker1.failure_count = 5
    await registry.reset_all()
    assert breaker1.state == CircuitState.CLOSED
    assert breaker1.failure_count == 0
