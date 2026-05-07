import argparse
import os
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

from oai_mcp_registry.services.deployers.docker_compose import DockerComposeManager

def main():
    parser = argparse.ArgumentParser(description="Generate Infrastructure Docker Compose File")
    parser.add_argument("--output", "-o", default=None, help="Output path for docker-compose.yaml")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress logs and only print the output path")
    args = parser.parse_args()

    if os.environ.get('MCP_REGISTRY_URL', None):
        parsed = urlparse(os.environ.get('MCP_REGISTRY_URL'))
        host = parsed.netloc.split(':')[0]
        port = parsed.port
        os.environ['MCP_BASE_URL'] = host
        os.environ['MCP_BASE_URL_PORT'] = str(port)
    elif os.environ.get('MCP_BASE_URL', None):
        host = os.environ.get('MCP_BASE_URL')
        port = os.environ.get('MCP_BASE_URL_PORT')
        os.environ['MCP_REGISTRY_URL'] = f"{host}:{port}"

    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr # Send logs to stderr so stdout is clean
    )
    logger = logging.getLogger(__name__)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.abspath(os.path.join(current_dir, 'resources', 'docker'))

    # Use specified output path or default to resources/docker/docker-compose.yaml
    output_path = args.output if args.output else os.path.join(build_dir, "docker-compose.yaml")

    if not args.quiet:
        logger.info(f"Generating infra compose file at: {output_path}")

    # Initialize a dummy deployer to leverage its infra-generation logic
    deployer = DockerComposeManager(
        seed_config={},
        compose_output_path=os.path.join(build_dir, "docker-compose.generated.yaml"),
        base_compose_path=output_path,
        agent_base_url=f"{os.environ.get('MCP_BASE_URL', 'localhost')}:{os.environ.get('MCP_BASE_URL_PORT', 8081)}",
        agent_local_registry_url=f"http://host.docker.internal:{os.environ.get('MCP_BASE_URL_PORT', 8081)}",
    )

    infra_dict = deployer._build_infra_compose_dict()
    deployer._write_yaml(infra_dict, Path(output_path))

    if args.quiet:
        # Print EXACTLY the path to stdout for easy bash capturing
        print(output_path)
    else:
        logger.info(f"Successfully generated infrastructure docker-compose file: {output_path}")
        logger.info(f"You can now run: docker compose -f {output_path} up -d")

if __name__ == "__main__":
    main()
