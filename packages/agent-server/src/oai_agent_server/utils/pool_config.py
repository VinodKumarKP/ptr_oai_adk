"""Database connection pool configuration and monitoring utilities."""

import asyncio
import logging
import os
from typing import Optional, Dict, Any


class PoolConfig:
    """Configuration for database connection pools.
    
    Supports both PostgreSQL (asyncpg) and SQLite (aiosqlite) backends
    with sensible defaults for production use.
    """
    
    # PostgreSQL defaults
    POSTGRES_MIN_SIZE = int(os.environ.get("DB_POOL_MIN_SIZE", "5"))
    POSTGRES_MAX_SIZE = int(os.environ.get("DB_POOL_MAX_SIZE", "20"))
    POSTGRES_MAX_CACHED_STMTS = int(os.environ.get("DB_MAX_CACHED_STMTS", "100"))
    POSTGRES_MAX_STMT_CACHE_TTL = int(os.environ.get("DB_MAX_STMT_CACHE_TTL", "300"))
    POSTGRES_COMMAND_TIMEOUT = float(os.environ.get("DB_COMMAND_TIMEOUT", "30.0"))
    
    # SQLite defaults (single connection model)
    SQLITE_TIMEOUT = float(os.environ.get("DB_SQLITE_TIMEOUT", "10.0"))
    
    # Connection pool health check
    HEALTH_CHECK_INTERVAL = int(os.environ.get("DB_HEALTH_CHECK_INTERVAL", "60"))
    
    @staticmethod
    def get_asyncpg_pool_kwargs() -> Dict[str, Any]:
        """Get asyncpg connection pool configuration.
        
        Returns:
            Dictionary of asyncpg.create_pool() keyword arguments
        """
        return {
            "min_size": PoolConfig.POSTGRES_MIN_SIZE,
            "max_size": PoolConfig.POSTGRES_MAX_SIZE,
            "max_cached_statement_lifetime": PoolConfig.POSTGRES_MAX_STMT_CACHE_TTL,
            "max_cacheable_statement_size": 1024 * 15,  # 15KB
            "command_timeout": PoolConfig.POSTGRES_COMMAND_TIMEOUT,
        }
    
    @staticmethod
    def get_sqlite_pool_kwargs() -> Dict[str, Any]:
        """Get aiosqlite connection configuration.
        
        Note: SQLite uses a single persistent connection model,
        not a traditional pool. This provides timeout configuration.
        
        Returns:
            Dictionary of SQLite connection parameters
        """
        return {
            "timeout": PoolConfig.SQLITE_TIMEOUT,
            "check_same_thread": False,  # Allow cross-thread access with async
        }


class PoolMonitor:
    """Monitor database connection pool health and utilization.
    
    Tracks pool metrics and periodically logs health status.
    """
    
    def __init__(self, pool: Any, backend_name: str = "database", 
                 logger: Optional[logging.Logger] = None):
        """Initialize pool monitor.
        
        Args:
            pool: asyncpg.Pool or sqlite connection
            backend_name: Human-readable backend name for logging
            logger: Logger instance
        """
        self.pool = pool
        self.backend_name = backend_name
        self.logger = logger or logging.getLogger(__name__)
        self._started = False
    
    async def start(self) -> None:
        """Start periodic health check loop."""
        if self._started:
            return
        
        self._started = True
        self.logger.debug(f"Starting {self.backend_name} pool monitor")
        
        # Run health check every interval seconds
        while self._started:
            try:
                await self._check_pool_health()
            except Exception as e:
                self.logger.error(f"Pool health check failed: {e}")
            
            await asyncio.sleep(PoolConfig.HEALTH_CHECK_INTERVAL)
    
    async def stop(self) -> None:
        """Stop periodic health check."""
        self._started = False
        self.logger.debug(f"Stopped {self.backend_name} pool monitor")
    
    async def _check_pool_health(self) -> None:
        """Check pool health and log metrics."""
        try:
            if hasattr(self.pool, '_holders'):
                # PostgreSQL asyncpg pool
                available = sum(1 for h in self.pool._holders if not h._in_use)
                in_use = sum(1 for h in self.pool._holders if h._in_use)
                total = len(self.pool._holders)
                
                self.logger.debug(
                    f"{self.backend_name} pool: {in_use}/{total} connections in use, "
                    f"{available} available"
                )
                
                # Warn if pool near exhaustion
                if in_use >= self.pool._maxsize * 0.8:
                    self.logger.warning(
                        f"{self.backend_name} pool near exhaustion: "
                        f"{in_use}/{self.pool._maxsize} connections in use"
                    )
            else:
                self.logger.debug(f"{self.backend_name} pool health check completed")
        except Exception as e:
            self.logger.debug(f"Could not retrieve detailed pool metrics: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current pool metrics.
        
        Returns:
            Dictionary with pool health metrics
        """
        metrics = {"backend": self.backend_name}
        
        try:
            if hasattr(self.pool, '_holders'):
                # PostgreSQL asyncpg pool
                available = sum(1 for h in self.pool._holders if not h._in_use)
                in_use = sum(1 for h in self.pool._holders if h._in_use)
                total = len(self.pool._holders)
                
                metrics.update({
                    "type": "connection_pool",
                    "total_connections": total,
                    "in_use": in_use,
                    "available": available,
                    "utilization_percent": (in_use / total * 100) if total > 0 else 0,
                })
        except Exception:
            pass
        
        return metrics
