import os
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from oai_agent_server.middleware.request_context import get_original_environ, request_env, sanitize_for_logging
from oai_agent_server.security.dependencies import verify_api_key


def _debug_mode_enabled() -> bool:
    return os.environ.get("DEBUG_MODE", "").lower() == "true"


def create_health_router(agent_name, server_state, enable_request_isolation, allowed_modes: Optional[List[str]] = None):
    """Create the health router with configured endpoints."""
    # Health router does NOT have the verify_api_key dependency
    router = APIRouter(tags=["health"], )

    if allowed_modes is None:
        allowed_modes = ["chat", "agent", "logs", "health"]

    @router.get("/")
    async def root():
        """Root endpoint returning server information and available endpoints."""
        original_environ = get_original_environ()
        auth_enabled = original_environ.get('AGENT_AUTH_ENABLED', '').lower() == 'true'

        endpoints = {}

        if "chat" in allowed_modes:
            endpoints.update({
                "POST /chat": "Send a message and get a response",
                "POST /chat/stream": "Send a message and get a streaming response"
            })

        if "health" in allowed_modes:
            endpoints.update({
                "GET /health": "Health check endpoint",
                "GET /ready": "Readiness probe (agent + DB checks)",
                "GET /status": "Get server status",
                "GET /debug/env": "Debug request environment (if enabled)"
            })

        if "agent" in allowed_modes:
            endpoints.update({
                "GET /info": "Get agent information"
            })

        return {
            "message": f"Agent HTTP Server for {agent_name}",
            "endpoints": endpoints,
            "auth_enabled": auth_enabled,
            "features": {
                "request_isolation": enable_request_isolation,
                "concurrent_requests": "supported" if enable_request_isolation else "not isolated"
            }
        }

    if "health" in allowed_modes:
        @router.get("/health")
        async def health_check():
            """Liveness probe — returns 200 as long as the process is up."""
            return {"status": "healthy", "agent": agent_name}

        @router.get("/ready")
        async def readiness_check(request: Request):
            """Readiness probe: agent initialised + DB reachable."""
            failures = {}

            srv_state = getattr(request.app.state, "server_state", server_state)
            if not getattr(srv_state, "is_agent_ready", True):
                failures["agent"] = "not ready"

            # DB check (best-effort)
            db_ok = True
            db_logger = getattr(request.app.state, "db_logger", None)
            if db_logger is None:
                # Fallback: many setups attach the logger via the AgentHTTPServer instance.
                pass
            else:
                try:
                    backend = getattr(db_logger, "_backend", None)
                    if backend is not None:
                        await backend.fetch_one("SELECT 1", ())
                    else:
                        db_ok = False
                        failures["database"] = "no active backend"
                except Exception as exc:
                    db_ok = False
                    failures["database"] = f"unreachable: {type(exc).__name__}"

            if failures:
                return JSONResponse(status_code=503, content={"status": "not_ready", "checks": failures})
            return {"status": "ready", "agent": agent_name, "database": "ok" if db_ok else "skipped"}

        @router.get("/status")
        async def server_status():
            """Get current server status including active requests"""
            return {
                "agent_name": agent_name,
                "active_requests": server_state.active_requests,
                "is_shutting_down": server_state.is_shutting_down,
                "uptime": time.time() - getattr(server_state, 'start_time', time.time()),
                "status": "shutting_down" if server_state.is_shutting_down else "running"
            }

        @router.get("/metrics")
        async def metrics(request: Request):
            """Prometheus metrics endpoint. Exposes all collected metrics.

            Uses the OpenTelemetry-managed Prometheus registry when available
            (where http_request_duration, requests_total etc. are recorded).
            Falls back to the default prometheus_client registry.
            """
            try:
                from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

                # The OpenTelemetry PrometheusMetricReader stores metrics in its
                # own internal registry, NOT in prometheus_client.REGISTRY.
                # We must collect from that registry to expose OTel HTTP metrics.
                try:
                    from opentelemetry.exporter.prometheus import _CustomCollector  # noqa: F401 – existence check
                    # OTel's PrometheusMetricReader installs a custom collector
                    # into prometheus_client.REGISTRY, so generate_latest() on
                    # the default registry will include OTel metrics.
                    metrics_output = generate_latest()
                except (ImportError, AttributeError):
                    # Older / unavailable OTel: fall back to default registry
                    metrics_output = generate_latest()

                return Response(content=metrics_output, media_type=CONTENT_TYPE_LATEST)
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Failed to generate metrics: {str(e)}"}
                )

        @router.get("/check-env", dependencies=[Depends(verify_api_key)])
        async def check_environment():
            """Return all HTTP_ environment variables. Gated behind DEBUG_MODE."""
            if not _debug_mode_enabled():
                return JSONResponse(status_code=404, content={"detail": "Not Found"})
            http_env_vars = {k: v for k, v in os.environ.items() if k.startswith('HTTP_')}
            return {"environment_variables": http_env_vars}

        @router.get("/debug/env", dependencies=[Depends(verify_api_key)])
        async def debug_env():
            """
            Debug endpoint to show current request environment. Gated behind DEBUG_MODE.
            Useful for testing request isolation.
            """
            if not _debug_mode_enabled():
                return JSONResponse(status_code=404, content={"detail": "Not Found"})

            req_env = request_env.get()

            # Sanitize sensitive values
            sanitized = sanitize_for_logging(req_env) if req_env is not None else {}

            return JSONResponse({
                "request_env_count": len(req_env) if req_env is not None else 0,
                "request_env": sanitized,
                "isolation_enabled": enable_request_isolation
            })

    return router
