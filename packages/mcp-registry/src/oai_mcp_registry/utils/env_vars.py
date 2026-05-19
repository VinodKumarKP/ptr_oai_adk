"""
MCP-registry deployment environment helper.

:func:`get_common_server_env` builds the full env-var dict required when
deploying an MCP server.  The 13 infrastructure fields that are shared
with the agent registry are assembled by
:func:`oai_platform_core.deployers.env_utils.build_deployment_env`;
MCP-specific keys are added on top.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from oai_platform_core.deployers.env_utils import build_deployment_env

__all__ = ["get_common_server_env"]


def get_common_server_env(
    server_name: str,
    port: int,
    base_url: str,
    local_registry_url: str,
    env_overrides: Optional[Dict[str, Any]] = None,
    deployment_mode: str = "docker",
) -> Dict[str, str]:
    """Return env-vars required to run an MCP server container or subprocess.

    Args:
        server_name:         Value for the ``MCP_SERVER_NAME`` env var.
        port:                Service port (``PORT``).
        base_url:            External base URL (``MCP_BASE_URL``).
        local_registry_url:  Registry URL reachable from inside Docker
                             (``MCP_REGISTRY_URL``).
        env_overrides:       Optional caller-supplied overrides applied last.
        deployment_mode:     ``"docker"`` or ``"local"`` — selects the
                             correct Redis / DB hostnames.

    Returns:
        ``Dict[str, str]`` suitable for passing to Docker Compose or a
        subprocess environment.
    """
    env = build_deployment_env(
        port=port,
        redis_host="mcp-valkey" if deployment_mode == "docker" else "localhost",
        logging_db_host="mcp_logs_db" if deployment_mode == "docker" else "localhost",
        logging_db_name="mcp_logs",
    )
    env.update({
        "MCP_SERVER_NAME": server_name,
        "MCP_AUTH_ENABLED": "true",
        "AUTH_ENABLED": "true",
        "MCP_BASE_URL": base_url,
        "MCP_REGISTRY_URL": local_registry_url,
        "FORCE_AUTH": "true",
    })
    for key, value in (env_overrides or {}).items():
        env[key] = str(value)
    return env
