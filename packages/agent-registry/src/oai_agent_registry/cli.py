import argparse
import platform
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

    host = args.host or registry_instance.registry_config.host
    port = args.port or registry_instance.registry_config.port

    if args.host:
        registry_instance.registry_config.host = host

    if args.port:
        registry_instance.registry_config.port = port

    # Use uvloop only on non-Windows systems for better performance
    loop_type = "uvloop" if platform.system() != "Windows" else "asyncio"

    logger.info(f"Starting server on {host}:{port} using {loop_type} loop")
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_keep_alive=300,
        loop=loop_type,
    )

if __name__ == "__main__":
    main()
