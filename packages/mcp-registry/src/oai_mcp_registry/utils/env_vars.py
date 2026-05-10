import os
from typing import Dict, Any

def get_common_server_env(
    server_name: str,
    port: int,
    base_url: str,
    local_registry_url: str,
    env_overrides: Dict[str, Any] = None,
    deployment_mode: str = "docker"
) -> Dict[str, str]:
    """
    Returns a common dictionary of environment variables required for an MCP server.
    This can be used by both Docker Compose and Python Subprocess deployers.
    """
    env_overrides = env_overrides or {}
    
    # Base environment
    env = {
        "MCP_SERVER_NAME": server_name,
        "PORT": str(port),
        "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
        "REDIS_HOST": "mcp-valkey" if deployment_mode == 'docker' else 'localhost',
        "REDIS_PORT": "6379",
        "MCP_AUTH_ENABLED": "true",
        "AUTH_ENABLED": "true",
        "LOGGING_DB_HOST": "mcp_logs_db" if deployment_mode == 'docker' else 'localhost',
        "LOGGING_DB_PORT": "5432",
        "LOGGING_DB_USER": "postgres",
        "LOGGING_DB_PASSWORD": "postgres",
        "LOGGING_DB_NAME": "mcp_logs",
        "DB_LOGGING_ENABLED": "true",
        "DB_POOL_MAX_SIZE": "2",
        "DB_POOL_TIMEOUT": "60",
        "DB_POOL_MIN_SIZE": "1",
        "MCP_BASE_URL": base_url,
        "MCP_REGISTRY_URL": local_registry_url,
        "FORCE_AUTH": "false"
    }

    # Apply overrides (user-supplied variables)
    for key, value in env_overrides.items():
        env[key] = str(value)

    return env
