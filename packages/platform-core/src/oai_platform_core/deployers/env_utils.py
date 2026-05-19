"""
oai_platform_core.deployers.env_utils — shared deployment environment helpers.

:func:`build_deployment_env` assembles the ~12 env-var fields that are
identical across every OAI service deployment (DB pool settings, Redis
connection, AWS region, etc.).  Package-specific wrappers
(``get_common_agent_env``, ``get_common_server_env``) call this helper
and layer on their own naming conventions.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

__all__ = ["build_deployment_env"]


def build_deployment_env(
    port: int,
    redis_host: str,
    logging_db_host: str,
    logging_db_name: str,
    env_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Return the env-var fields shared by all OAI service deployments.

    Callers merge this dict with their package-specific keys and then
    apply any user-supplied *env_overrides* on top.

    Args:
        port:             The service's assigned port number.
        redis_host:       Redis hostname (differs per deployment mode).
        logging_db_host:  Postgres hostname for the logging DB.
        logging_db_name:  Postgres database name for the logging DB.
        env_overrides:    Optional caller-supplied key/value overrides
                          applied last so they take priority.

    Returns:
        ``Dict[str, str]`` ready to be passed to a subprocess or
        Docker Compose environment block.
    """
    env: Dict[str, str] = {
        "PORT": str(port),
        "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
        "REDIS_HOST": redis_host,
        "REDIS_PORT": "6379",
        "LOGGING_DB_HOST": logging_db_host,
        "LOGGING_DB_PORT": "5432",
        "LOGGING_DB_USER": "postgres",
        "LOGGING_DB_PASSWORD": "postgres",
        "LOGGING_DB_NAME": logging_db_name,
        "DB_LOGGING_ENABLED": "true",
        "DB_POOL_MAX_SIZE": "2",
        "DB_POOL_TIMEOUT": "60",
        "DB_POOL_MIN_SIZE": "1",
    }

    for key, value in (env_overrides or {}).items():
        env[key] = str(value)

    return env
