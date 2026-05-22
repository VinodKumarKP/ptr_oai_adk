import argparse
import os
import sys
from urllib.parse import urlparse

import uvicorn
import logging

from oai_mcp_registry.app import app
from oai_mcp_registry.dependencies import registry_instance

logger = logging.getLogger("MCPRegistry")

def main():
    parser = argparse.ArgumentParser(description="Start MCP Proxy Server")
    parser.add_argument("--config", "-c", help="Path to config file")
    parser.add_argument("--host", help="Host address")
    parser.add_argument("--port", "-p", type=int, help="Port number")
    parser.add_argument("--enable-auto-discovery", action="store_true", default=False, help="Enable auto-discovery of MCP servers")
    parser.add_argument("--start-port", type=int, help="Start of port range for auto-discovery")
    parser.add_argument("--end-port", type=int, help="End of port range for auto-discovery")
    parser.add_argument(
        "--auto-start-infra",
        action="store_true",
        default=False,
        help="Run docker compose up on the infra compose file (postgres, valkey) before initialising the database",
    )
    parser.add_argument(
        "--infra-compose-file",
        help="Path to the infra docker-compose file (default: bundled docker-compose.yaml)",
    )
    parser.add_argument(
        "--infra-startup-timeout",
        type=int,
        help="Seconds to wait for Postgres to become ready after docker compose up (default: 60)",
    )
    args = parser.parse_args()

    if args.config:
        try:
            registry_instance.config_path = args.config
            registry_instance.load_configuration()
            registry_instance.initialize_proxies()
        except Exception as e:
            logger.critical(f"Failed to start with provided config: {e}")
            sys.exit(1)

    if args.enable_auto_discovery:
        registry_instance.registry_config.enable_auto_discovery = True
    if args.start_port:
        registry_instance.registry_config.start_port = args.start_port
    if args.end_port:
        registry_instance.registry_config.end_port = args.end_port
    if args.auto_start_infra:
        registry_instance.registry_config.auto_start_infra = True
    if args.infra_compose_file:
        registry_instance.registry_config.infra_compose_file = args.infra_compose_file
    if args.infra_startup_timeout:
        registry_instance.registry_config.infra_startup_timeout = args.infra_startup_timeout

    if os.environ.get('MCP_REGISTRY_URL', None):
        parsed = urlparse(os.environ.get('MCP_REGISTRY_URL'))
        port = parsed.port
        os.environ['MCP_BASE_URL'] = f"{parsed.scheme}://{parsed.netloc.split(':')[0]}"
        os.environ['MCP_BASE_URL_PORT'] = str(port)
    elif os.environ.get('MCP_BASE_URL', None):
        port = os.environ.get('MCP_BASE_URL_PORT')
        os.environ['MCP_REGISTRY_URL'] = f"{os.environ.get('MCP_BASE_URL')}:{os.environ.get('MCP_BASE_URL_PORT')}"
    else:
        port = args.port or registry_instance.registry_config.port

    host = args.host or registry_instance.registry_config.host

    registry_instance.registry_config.start_port = port
    registry_instance.registry_config.host = host

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_keep_alive=300,
        loop="uvloop"
    )

if __name__ == "__main__":
    main()
