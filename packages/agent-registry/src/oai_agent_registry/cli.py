import argparse
import os
import platform
from urllib.parse import urlparse

import uvicorn
from oai_agent_registry.app import app, logger
from oai_agent_registry.dependencies import registry_instance

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Start Agent Registry")
    parser.add_argument("--config", "-c", help="Path to config file")
    parser.add_argument("--host", help="Host address")
    parser.add_argument("--port", "-p", type=int, help="Port number")
    parser.add_argument("--enable-auto-discovery", action="store_true", default=False, help="Enable auto-discovery of agents")
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
        registry_instance.config_path = args.config
        registry_instance.load_config()

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

    if os.environ.get('AGENT_REGISTRY_URL', None):
        parsed = urlparse(os.environ.get('AGENT_REGISTRY_URL'))
        port = parsed.port
        os.environ['AGENT_BASE_URL'] = f"{parsed.scheme}://{parsed.netloc.split(':')[0]}"
        os.environ['AGENT_BASE_URL_PORT'] = str(port)
    elif os.environ.get('AGENT_BASE_URL', None):
        port = os.environ.get('AGENT_BASE_URL_PORT')
        os.environ['AGENT_REGISTRY_URL'] = f"{os.environ.get('AGENT_BASE_URL')}:{port}"
    else:
        port = args.port or registry_instance.registry_config.port

    host = args.host or registry_instance.registry_config.host
    registry_instance.registry_config.host = host
    registry_instance.registry_config.port = port

    # Use uvloop only on non-Windows systems for better performance
    loop_type = "uvloop" if platform.system() != "Windows" else "asyncio"

    logger.info(f"Starting server on {host}:{port} using {loop_type} loop")
    uvicorn.run(
        app,
        host=host.replace('http://', ''),
        port=int(port),
        workers=1,
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_keep_alive=300,
        loop=loop_type,
    )

if __name__ == "__main__":
    main()
