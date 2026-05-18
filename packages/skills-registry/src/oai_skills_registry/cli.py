"""
Entry point for the Skills Registry server.

CLI flags override environment variables; environment variables are used as
defaults when flags are omitted.  All flags are optional — the server starts
with sensible defaults if none are provided.

Examples
--------
# Minimal start (no infra management):
    oai-skills-registry

# Auto-start postgres + valkey before DB init:
    oai-skills-registry --auto-start-infra

# Custom host/port + infra:
    oai-skills-registry --host 0.0.0.0 --port 8083 --auto-start-infra

# Point to a shared infra compose file (e.g., shared with agent-registry):
    oai-skills-registry --auto-start-infra \\
        --infra-compose-file /opt/infra/docker-compose.yaml \\
        --infra-startup-timeout 90
"""

import argparse
import os
import platform

import uvicorn

import oai_skills_registry.dependencies as _deps
from oai_skills_registry.main import app


def main():
    """Main entry point for the Skills Registry CLI."""
    parser = argparse.ArgumentParser(
        description="Start Skills Registry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server settings
    parser.add_argument("--host", help="Host address to bind to")
    parser.add_argument("--port", "-p", type=int, help="Port number to listen on")

    # Infra auto-start
    parser.add_argument(
        "--auto-start-infra",
        action="store_true",
        default=False,
        help=(
            "Run docker compose up on the infra compose file (postgres, valkey) "
            "before initialising the database"
        ),
    )
    parser.add_argument(
        "--infra-compose-file",
        help=(
            "Path to the infra docker-compose file "
            "(default: bundled docker-compose.yaml shipped with this package)"
        ),
    )
    parser.add_argument(
        "--infra-startup-timeout",
        type=int,
        help="Seconds to wait for Postgres to become ready after docker compose up (default: 60)",
    )

    args = parser.parse_args()

    # --- Wire flags into dependencies module-level config ---------------
    # Setting module-level variables directly (rather than only env vars)
    # means initialize_registry() sees the CLI values at the highest priority
    # regardless of what env vars are already set.
    # Env vars are also set as a fallback for any code that reads them directly.

    if args.auto_start_infra:
        _deps._auto_start_infra = True
        os.environ["AUTO_START_INFRA"] = "true"

    if args.infra_compose_file:
        _deps._infra_compose_file = args.infra_compose_file
        os.environ["INFRA_COMPOSE_FILE"] = args.infra_compose_file

    if args.infra_startup_timeout:
        _deps._infra_startup_timeout = args.infra_startup_timeout
        os.environ["INFRA_STARTUP_TIMEOUT"] = str(args.infra_startup_timeout)

    # Resolve host and port: CLI flag > env var > default
    host = args.host or os.environ.get("HOST", "0.0.0.0")
    port = args.port or int(os.environ.get("PORT", "8083"))

    # Propagate back so main.py __main__ block and any nested reads are consistent
    os.environ["HOST"] = host
    os.environ["PORT"] = str(port)

    # Use uvloop only on non-Windows systems for better performance
    loop_type = "uvloop" if platform.system() != "Windows" else "asyncio"

    import logging
    logging.getLogger("oai_skills_registry").info(
        "Starting Skills Registry on %s:%s using %s loop", host, port, loop_type
    )

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
