"""Circuit breaker pattern for resilience against downstream service failures.

Implements the circuit breaker pattern with states:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is failing, requests are rejected
- HALF_OPEN: Testing if service recovered
"""

import asyncio
import time
import logging
from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Any, Optional, Dict


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Service is failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, service_name: str, reason: str = "Circuit breaker is open"):
        self.service_name = service_name
        self.reason = reason
        super().__init__(f"[{service_name}] {reason}")


class CircuitBreaker:
    """Circuit breaker for handling service failures.
    
    Protects against cascading failures by:
    - Tracking failures and successes
    - Transitioning to OPEN state when threshold exceeded
    - Providing HALF_OPEN state for recovery testing
    - Auto-resetting after timeout
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: int = 60,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize circuit breaker.
        
        Args:
            name: Service name for logging/identification
            failure_threshold: Failures before opening circuit (default: 5)
            success_threshold: Successes in HALF_OPEN before closing (default: 2)
            timeout: Seconds before transitioning from OPEN to HALF_OPEN (default: 60)
            logger: Logger instance
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.opened_at: Optional[datetime] = None
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    self.logger.info(
                        f"[{self.name}] Circuit transitioned to HALF_OPEN. "
                        f"Testing recovery..."
                    )
                else:
                    raise CircuitBreakerError(
                        self.name,
                        f"Circuit is OPEN (opened {self._time_since_open()}s ago)"
                    )
        
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure(str(e))
            raise
    
    async def _record_success(self) -> None:
        """Record successful call."""
        async with self._lock:
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                self.logger.debug(
                    f"[{self.name}] Success in HALF_OPEN: {self.success_count}/{self.success_threshold}"
                )
                
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.logger.info(
                        f"[{self.name}] Circuit closed. Service recovered."
                    )
            elif self.state == CircuitState.CLOSED:
                self.logger.debug(f"[{self.name}] Success (failures reset)")
    
    async def _record_failure(self, error: str) -> None:
        """Record failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            self.logger.warning(
                f"[{self.name}] Failure {self.failure_count}/{self.failure_threshold}: {error}"
            )
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.opened_at = datetime.now()
                self.success_count = 0
                self.logger.error(
                    f"[{self.name}] Circuit reopened after HALF_OPEN failure. "
                    f"Will retry in {self.timeout}s"
                )
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = datetime.now()
                self.logger.error(
                    f"[{self.name}] Circuit opened after {self.failure_count} failures. "
                    f"Will retry in {self.timeout}s"
                )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.opened_at is None:
            return False
        
        time_since_open = (datetime.now() - self.opened_at).total_seconds()
        return time_since_open >= self.timeout
    
    def _time_since_open(self) -> float:
        """Get seconds since circuit was opened."""
        if self.opened_at is None:
            return 0
        return (datetime.now() - self.opened_at).total_seconds()
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state snapshot.
        
        Returns:
            Dict with state, failure_count, success_count, etc.
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "timeout_seconds": self.timeout,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize circuit breaker registry.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    async def register(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: int = 60,
    ) -> CircuitBreaker:
        """Register a new circuit breaker.
        
        Args:
            name: Service name
            failure_threshold: Failures before opening
            success_threshold: Successes in HALF_OPEN before closing
            timeout: Seconds before recovery attempt
            
        Returns:
            CircuitBreaker instance
        """
        async with self._lock:
            if name in self._breakers:
                return self._breakers[name]
            
            breaker = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                timeout=timeout,
                logger=self.logger,
            )
            self._breakers[name] = breaker
            self.logger.info(f"Registered circuit breaker: {name}")
            return breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name.
        
        Args:
            name: Service name
            
        Returns:
            CircuitBreaker instance or None
        """
        return self._breakers.get(name)
    
    async def call(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Call function with circuit breaker protection.
        
        Args:
            name: Service name (breaker must be registered)
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        breaker = self.get(name)
        if not breaker:
            self.logger.warning(
                f"No circuit breaker registered for '{name}'. "
                f"Executing without protection."
            )
            return await func(*args, **kwargs)
        
        return await breaker.call(func, *args, **kwargs)
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get state of all circuit breakers.
        
        Returns:
            Dict mapping breaker name to state snapshot
        """
        return {name: breaker.get_state() for name, breaker in self._breakers.items()}
    
    async def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED state."""
        async with self._lock:
            for breaker in self._breakers.values():
                async with breaker._lock:
                    breaker.state = CircuitState.CLOSED
                    breaker.failure_count = 0
                    breaker.success_count = 0
                    self.logger.info(f"Reset circuit breaker: {breaker.name}")
