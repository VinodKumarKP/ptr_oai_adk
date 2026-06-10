"""Health check and metrics export for the agent server.

Provides:
- System health status
- Database connection pool metrics
- Service uptime
- Performance statistics
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime


class HealthMetrics:
    """Tracks health metrics for the server."""
    
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.total_errors = 0
        self.last_error_time: Optional[float] = None
        self.last_error_message: Optional[str] = None
    
    def get_uptime(self) -> float:
        """Get server uptime in seconds."""
        return time.time() - self.start_time
    
    def record_request(self, success: bool = True, error_message: Optional[str] = None):
        """Record a request.
        
        Args:
            success: Whether request was successful
            error_message: Error message if request failed
        """
        self.total_requests += 1
        if not success:
            self.total_errors += 1
            self.last_error_time = time.time()
            self.last_error_message = error_message
    
    def get_error_rate(self) -> float:
        """Get error rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.total_errors / self.total_requests) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        uptime = self.get_uptime()
        return {
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_duration(uptime),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate_percent": self.get_error_rate(),
            "last_error_time": self.last_error_time,
            "last_error_message": self.last_error_message,
        }
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration (e.g., "2h 30m 45s")
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
        
        return " ".join(parts)


class HealthCheckResponse:
    """Structured health check response."""
    
    def __init__(self, status: str, timestamp: Optional[float] = None):
        """Initialize health check response.
        
        Args:
            status: Health status (healthy, degraded, unhealthy)
            timestamp: Response timestamp
        """
        self.status = status
        self.timestamp = timestamp or time.time()
        self.components: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, Any] = {}
    
    def add_component(self, name: str, status: str, details: Optional[Dict[str, Any]] = None):
        """Add component health status.
        
        Args:
            name: Component name
            status: Health status (healthy, degraded, unhealthy)
            details: Additional details
        """
        self.components[name] = {
            "status": status,
            "details": details or {},
        }
    
    def add_metrics(self, name: str, metrics: Dict[str, Any]):
        """Add metrics section.
        
        Args:
            name: Metrics section name
            metrics: Metrics dictionary
        """
        self.metrics[name] = metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "components": self.components,
            "metrics": self.metrics,
        }
    
    def is_healthy(self) -> bool:
        """Check if overall health is healthy."""
        return self.status == "healthy"


class HealthCheckCollector:
    """Collects health information from various components."""
    
    def __init__(self):
        self.metrics = HealthMetrics()
        self._component_checks: Dict[str, callable] = {}
    
    def register_component_check(self, name: str, check_func: callable):
        """Register a component health check function.
        
        Args:
            name: Component name
            check_func: Async function that returns (status, details) tuple
        """
        self._component_checks[name] = check_func
    
    async def collect(self) -> HealthCheckResponse:
        """Collect health information from all components.
        
        Returns:
            HealthCheckResponse with collected health info
        """
        response = HealthCheckResponse("healthy")
        
        # Add server metrics
        response.add_metrics("server", self.metrics.to_dict())
        
        # Collect component health
        all_healthy = True
        for component_name, check_func in self._component_checks.items():
            try:
                status, details = await check_func()
                response.add_component(component_name, status, details)
                
                if status != "healthy":
                    all_healthy = False
            except Exception as e:
                response.add_component(component_name, "unhealthy", {
                    "error": str(e)
                })
                all_healthy = False
        
        # Set overall status
        if not all_healthy:
            response.status = "degraded"
        
        return response


def create_database_health_check(db_logger):
    """Create a database health check function.
    
    Args:
        db_logger: DatabaseLogger instance
        
    Returns:
        Async function for health check
    """
    async def check_db_health():
        try:
            if not db_logger.is_active:
                return ("unhealthy", {"reason": "Database logger not initialized"})
            
            # Try a simple query to verify connectivity
            stats = await db_logger.get_stats()
            
            # Check pool utilization if available
            pool_details = {}
            if hasattr(db_logger, '_backend') and hasattr(db_logger._backend, '_pool'):
                pool = db_logger._backend._pool
                if hasattr(pool, '_holders'):
                    available = sum(1 for h in pool._holders if not h._in_use)
                    in_use = sum(1 for h in pool._holders if h._in_use)
                    total = len(pool._holders)
                    utilization = (in_use / total * 100) if total > 0 else 0
                    
                    pool_details = {
                        "total_connections": total,
                        "in_use": in_use,
                        "available": available,
                        "utilization_percent": utilization,
                    }
            
            return ("healthy", {
                "stats": stats,
                "pool": pool_details,
            })
        except Exception as e:
            return ("unhealthy", {"error": str(e)})
    
    return check_db_health


def create_llm_judge_health_check(llm_judge_service):
    """Create an LLM judge service health check function.
    
    Args:
        llm_judge_service: LLMJudgeService instance
        
    Returns:
        Async function for health check
    """
    async def check_llm_judge_health():
        try:
            if llm_judge_service.judge_agent is None:
                return ("degraded", {"reason": "Judge agent not initialized"})
            
            return ("healthy", {
                "initialized": True,
                "model_id": llm_judge_service.judge_model_id,
            })
        except Exception as e:
            return ("unhealthy", {"error": str(e)})
    
    return check_llm_judge_health
