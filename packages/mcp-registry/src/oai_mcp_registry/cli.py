import argparse
import os
import sys
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

    host = args.host or registry_instance.registry_config.host
    port = args.port or registry_instance.registry_config.port

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
