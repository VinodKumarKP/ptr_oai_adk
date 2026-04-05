import argparse
import uvicorn
from oai_agent_registry.app import app, logger
from oai_agent_registry.dependencies import registry_instance

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Start Agent Registry")
    parser.add_argument("--config", "-c", help="Path to config file")
    parser.add_argument("--host", help="Host address")
    parser.add_argument("--port", "-p", type=int, help="Port number")

    args = parser.parse_args()

    if args.config:
        registry_instance.config_path = args.config
        registry_instance.load_config()

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
        loop="uvloop",
    )

if __name__ == "__main__":
    main()