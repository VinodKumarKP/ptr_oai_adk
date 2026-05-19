"""
oai_platform_core.deployers.infra_cli — shared ``generate-infra-compose`` CLI.

Both the agent-registry and mcp-registry packages expose a CLI command
that writes a ``docker-compose.yaml`` for their infrastructure services.
The boilerplate is identical; only the env-var prefix and the concrete
``DockerComposeManager`` class differ.

Usage (from each package's ``infra_compose_generator_cli.py``)::

    from oai_platform_core.deployers.infra_cli import generate_infra_compose_main
    from oai_agent_registry.services.deployers.docker_compose import DockerComposeManager

    def main():
        generate_infra_compose_main("AGENT", DockerComposeManager)

    if __name__ == "__main__":
        main()
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Type
from urllib.parse import urlparse

__all__ = ["generate_infra_compose_main"]


def generate_infra_compose_main(
    env_prefix: str,
    deployer_class: Type,
    default_port: int = 8081,
) -> None:
    """Shared ``main()`` for infrastructure Docker Compose generation CLIs.

    Args:
        env_prefix:      Upper-case prefix for environment variables —
                         ``"AGENT"`` or ``"MCP"``.  Used to build variable
                         names such as ``AGENT_REGISTRY_URL``,
                         ``MCP_BASE_URL``, etc.
        deployer_class:  The concrete ``DockerComposeManager`` subclass
                         from the calling package.  Instantiated with the
                         same keyword arguments used by both registries.
        default_port:    Default service port when none is configured
                         (default ``8081``).
    """
    registry_url_var  = f"{env_prefix}_REGISTRY_URL"
    base_url_var      = f"{env_prefix}_BASE_URL"
    base_url_port_var = f"{env_prefix}_BASE_URL_PORT"

    parser = argparse.ArgumentParser(
        description="Generate Infrastructure Docker Compose File"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output path for docker-compose.yaml"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress logs and only print the output path"
    )
    args = parser.parse_args()

    # Derive BASE_URL / PORT from registry URL or vice-versa
    if os.environ.get(registry_url_var):
        parsed = urlparse(os.environ[registry_url_var])
        host = parsed.netloc.split(":")[0]
        port = parsed.port
        os.environ[base_url_var] = host
        os.environ[base_url_port_var] = str(port)
    elif os.environ.get(base_url_var):
        host = os.environ[base_url_var]
        port = os.environ.get(base_url_port_var, str(default_port))
        os.environ[registry_url_var] = f"{host}:{port}"

    # Configure logging — send to stderr so stdout stays clean for --quiet
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger(__name__)

    # Locate the resources/docker directory next to the calling package
    # The caller's package lives two levels above the CLI module, so we
    # resolve via the deployer_class's module file to stay portable.
    import inspect
    deployer_module_file = inspect.getfile(deployer_class)
    # …/services/deployers/docker_compose.py → …/resources/docker
    package_root = Path(deployer_module_file).parent.parent.parent
    build_dir = package_root / "resources" / "docker"

    output_path = args.output or str(build_dir / "docker-compose.yaml")

    if not args.quiet:
        logger.info("Generating infra compose file at: %s", output_path)

    deployer = deployer_class(
        seed_config={},
        compose_output_path=str(build_dir / "docker-compose.generated.yaml"),
        base_compose_path=output_path,
        agent_base_url=(
            f"{os.environ.get(base_url_var, 'localhost')}"
            f":{os.environ.get(base_url_port_var, str(default_port))}"
        ),
        agent_local_registry_url=(
            f"http://host.docker.internal"
            f":{os.environ.get(base_url_port_var, str(default_port))}"
        ),
    )

    infra_dict = deployer._build_infra_compose_dict()
    deployer._write_yaml(infra_dict, Path(output_path))

    if args.quiet:
        print(output_path)
    else:
        logger.info("Successfully generated: %s", output_path)
        logger.info("Run: docker compose -f %s up -d", output_path)
