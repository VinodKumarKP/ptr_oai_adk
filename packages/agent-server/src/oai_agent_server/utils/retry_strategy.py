"""Enhanced retry strategy with exponential backoff for upstream services.

Provides configurable retry policies:
- Exponential backoff with jitter
- Retryable vs non-retryable exceptions
- Per-service retry policies
"""

import asyncio
import logging
import random
from typing import Callable, Any, List, Optional, Type, Dict
from enum import Enum


class RetryStrategy(str, Enum):
    """Retry strategy types."""
    FIXED = "fixed"              # Fixed delay between retries
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"            # Linear backoff


class RetryConfig:
    """Retry configuration for a service/operation."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 0.1,
        max_delay: float = 10.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of attempts (default: 3)
            initial_delay: Initial delay in seconds (default: 0.1)
            max_delay: Maximum delay between retries (default: 10.0)
            strategy: Retry strategy (default: EXPONENTIAL)
            jitter: Add randomness to backoff (default: True)
            retryable_exceptions: List of exceptions to retry on (default: Exception)
            logger: Logger instance
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.strategy = strategy
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [Exception]
        self.logger = logger or logging.getLogger(__name__)
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number.
        
        Args:
            attempt: Attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        if self.strategy == RetryStrategy.FIXED:
            delay = self.initial_delay
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.initial_delay * (attempt + 1)
        else:  # EXPONENTIAL
            delay = self.initial_delay * (2 ** attempt)
        
        # Apply ceiling
        delay = min(delay, self.max_delay)
        
        # Add jitter (randomness)
        if self.jitter:
            jitter = random.uniform(0, delay * 0.1)  # Up to 10% jitter
            delay += jitter
        
        return delay
    
    def is_retryable(self, exception: Exception) -> bool:
        """Check if exception is retryable.
        
        Args:
            exception: Exception to check
            
        Returns:
            True if should retry, False otherwise
        """
        return isinstance(exception, tuple(self.retryable_exceptions))


class Retry:
    """Retry helper for executing operations with retry logic."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 0.1,
        max_delay: float = 10.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize retry helper.
        
        Args:
            max_attempts: Maximum number of attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
            strategy: Retry strategy
            jitter: Add randomness to backoff
            retryable_exceptions: List of exceptions to retry on
            logger: Logger instance
        """
        self.config = RetryConfig(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            strategy=strategy,
            jitter=jitter,
            retryable_exceptions=retryable_exceptions,
            logger=logger,
        )
        self.logger = logger or logging.getLogger(__name__)
    
    async def execute(
        self,
        func: Callable,
        *args,
        operation_name: str = "Operation",
        **kwargs
    ) -> Any:
        """Execute function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            operation_name: Name for logging
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                self.logger.debug(
                    f"{operation_name}: Attempt {attempt + 1}/{self.config.max_attempts}"
                )
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(
                        f"{operation_name}: Succeeded on attempt {attempt + 1}"
                    )
                
                return result
            
            except Exception as e:
                last_exception = e
                
                if not self.config.is_retryable(e):
                    self.logger.error(
                        f"{operation_name}: Non-retryable exception: {type(e).__name__}: {str(e)}"
                    )
                    raise
                
                if attempt == self.config.max_attempts - 1:
                    self.logger.error(
                        f"{operation_name}: All {self.config.max_attempts} attempts failed. "
                        f"Last error: {type(e).__name__}: {str(e)}"
                    )
                    raise
                
                delay = self.config.get_delay(attempt)
                self.logger.warning(
                    f"{operation_name}: Attempt {attempt + 1} failed: {type(e).__name__}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                await asyncio.sleep(delay)
        
        if last_exception:
            raise last_exception


class RetryRegistry:
    """Registry for managing retry policies per service."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize retry registry.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self._policies: Dict[str, RetryConfig] = {}
    
    def register_policy(
        self,
        service_name: str,
        config: RetryConfig,
    ) -> None:
        """Register retry policy for service.
        
        Args:
            service_name: Service name
            config: RetryConfig instance
        """
        self._policies[service_name] = config
        self.logger.info(
            f"Registered retry policy for {service_name}: "
            f"{config.max_attempts} attempts, "
            f"{config.strategy.value} strategy"
        )
    
    def get_policy(self, service_name: str) -> Optional[RetryConfig]:
        """Get retry policy for service.
        
        Args:
            service_name: Service name
            
        Returns:
            RetryConfig or None
        """
        return self._policies.get(service_name)
    
    async def execute(
        self,
        service_name: str,
        func: Callable,
        *args,
        operation_name: str = "Operation",
        **kwargs
    ) -> Any:
        """Execute function with service-specific retry policy.
        
        Args:
            service_name: Service name
            func: Async function to execute
            *args: Positional arguments
            operation_name: Name for logging
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception if all retries exhausted or no policy registered
        """
        config = self.get_policy(service_name)
        if not config:
            self.logger.warning(
                f"No retry policy registered for '{service_name}'. "
                f"Executing with default policy."
            )
            config = RetryConfig(logger=self.logger)
        
        retry = Retry(
            max_attempts=config.max_attempts,
            initial_delay=config.initial_delay,
            max_delay=config.max_delay,
            strategy=config.strategy,
            jitter=config.jitter,
            retryable_exceptions=config.retryable_exceptions,
            logger=self.logger,
        )
        
        return await retry.execute(
            func,
            *args,
            operation_name=operation_name,
            **kwargs
        )


# Common retryable exceptions
RETRYABLE_HTTP_ERRORS = [
    TimeoutError,
    ConnectionError,
    asyncio.TimeoutError,
]


def create_http_retry_config(
    service_name: str,
    logger: Optional[logging.Logger] = None,
) -> RetryConfig:
    """Create retry config for HTTP services.
    
    Args:
        service_name: Service name
        logger: Logger instance
        
    Returns:
        RetryConfig configured for HTTP retries
    """
    return RetryConfig(
        max_attempts=5,
        initial_delay=0.1,
        max_delay=10.0,
        strategy=RetryStrategy.EXPONENTIAL,
        jitter=True,
        retryable_exceptions=RETRYABLE_HTTP_ERRORS,
        logger=logger,
    )
