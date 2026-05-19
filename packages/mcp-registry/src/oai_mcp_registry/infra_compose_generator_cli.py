"""
CLI entry-point: generate the MCP-registry infrastructure docker-compose file.

Delegates all logic to the shared
:func:`oai_platform_core.deployers.infra_cli.generate_infra_compose_main`
helper; only the env-var prefix and the concrete ``DockerComposeManager``
class are specific to this package.
"""
from oai_platform_core.deployers.infra_cli import generate_infra_compose_main
from oai_mcp_registry.services.deployers.docker_compose import DockerComposeManager


def main() -> None:
    generate_infra_compose_main("MCP", DockerComposeManager)


if __name__ == "__main__":
    main()
