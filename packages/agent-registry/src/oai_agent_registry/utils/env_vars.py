"""
Agent-registry deployment environment helper.

:func:`get_common_agent_env` builds the full env-var dict required when
deploying an agent.  The 13 infrastructure fields that are shared with
the MCP registry are assembled by
:func:`oai_platform_core.deployers.env_utils.build_deployment_env`;
agent-specific keys are added on top.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from oai_platform_core.deployers.env_utils import build_deployment_env

__all__ = ["get_common_agent_env"]


def get_common_agent_env(
    agent_name: str,
    port: int,
    base_url: str,
    local_registry_url: str,
    env_overrides: Optional[Dict[str, Any]] = None,
    deployment_mode: str = "docker",
) -> Dict[str, str]:
    """Return env-vars required to run an agent container or subprocess.

    Args:
        agent_name:          Value for the ``AGENT_NAME`` env var.
        port:                Service port (``PORT``).
        base_url:            External base URL (``AGENT_BASE_URL``).
        local_registry_url:  Registry URL reachable from inside Docker
                             (``AGENT_REGISTRY_URL``).
        env_overrides:       Optional caller-supplied overrides applied last.
        deployment_mode:     ``"docker"`` or ``"local"`` — selects the
                             correct Redis / DB hostnames.

    Returns:
        ``Dict[str, str]`` suitable for passing to Docker Compose or a
        subprocess environment.
    """
    env = build_deployment_env(
        port=port,
        redis_host="agent-valkey" if deployment_mode == "docker" else "localhost",
        logging_db_host="agent_logs_db" if deployment_mode == "docker" else "localhost",
        logging_db_name="agent_logs",
    )
    env.update({
        "AGENT_NAME": agent_name,
        "AGENT_AUTH_ENABLED": "true",
        "AGENT_BASE_URL": base_url,
        "AGENT_REGISTRY_URL": local_registry_url,
        "FORCE_AUTH": "false",
        "TRUSTED_CIDRS": "127.0.0.0/8,::1/128,172.16.0.0/12",
    })
    for key, value in (env_overrides or {}).items():
        env[key] = str(value)
    return env
