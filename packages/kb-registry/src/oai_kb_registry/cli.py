"""
Entry point for the Knowledge Base Registry server.

CLI flags override environment variables; environment variables are used as
defaults when flags are omitted.  All flags are optional — the server starts
with sensible defaults if none are provided.

Examples
--------
# Minimal start (no infra management):
    oai-kb-registry

# Auto-start postgres + valkey before DB init:
    oai-kb-registry --auto-start-infra

# Also start the builtin vector stores (pgvector + chromadb):
    oai-kb-registry --auto-start-infra \\
        --infra-services postgres valkey pgvector chromadb

# Custom host/port + infra:
    oai-kb-registry --host 0.0.0.0 --port 8085 --auto-start-infra

# Point to a shared infra compose file:
    oai-kb-registry --auto-start-infra \\
        --infra-compose-file /opt/infra/docker-compose.yaml \\
        --infra-startup-timeout 90

# Development mode (no auth, hot-reload):
    oai-kb-registry --no-auth --reload
"""

import argparse
import logging
import os
import platform

import uvicorn

import oai_kb_registry.dependencies as _deps
from oai_kb_registry.main import app


def main():
    """Main entry point for the KB Registry CLI."""
    parser = argparse.ArgumentParser(
        description="Start Knowledge Base Registry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server settings
    parser.add_argument("--host", help="Host address to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, help="Port number to listen on (default: 8085)")
    parser.add_argument("--reload", action="store_true", default=False,
                        help="Enable hot-reload (development mode)")
    parser.add_argument("--no-auth", action="store_true", default=False,
                        help="Disable authentication (sets KB_AUTH_ENABLED=false)")
    parser.add_argument("--log-level", default=None,
                        choices=["debug", "info", "warning", "error"],
                        help="Uvicorn log level (default: info)")
    parser.add_argument("--kb-data-dir", default=None,
                        help="Directory for builtin vector store data (default: /tmp/kb_data)")

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
    parser.add_argument(
        "--infra-services",
        nargs="+",
        metavar="SERVICE",
        help=(
            "Which Docker Compose services to start "
            "(default: postgres valkey). "
            "Add 'pgvector' and/or 'chromadb' to also spin up builtin vector stores."
        ),
    )

    args = parser.parse_args()

    # --- Wire flags into dependencies module-level config ---------------
    # Setting module-level variables directly (rather than only env vars)
    # means initialize_registry() sees the CLI values at the highest priority
    # regardless of what env vars are already set.

    if args.auto_start_infra:
        _deps._auto_start_infra = True
        os.environ["AUTO_START_INFRA"] = "true"

    if args.infra_compose_file:
        _deps._infra_compose_file = args.infra_compose_file
        os.environ["INFRA_COMPOSE_FILE"] = args.infra_compose_file

    if args.infra_startup_timeout:
        _deps._infra_startup_timeout = args.infra_startup_timeout
        os.environ["INFRA_STARTUP_TIMEOUT"] = str(args.infra_startup_timeout)

    # If the user specified explicit services, patch the InfraManager default
    # before initialize_registry() is called.
    if args.infra_services:
        try:
            from oai_kb_registry.services.infra_manager import InfraManager  # noqa: PLC0415
            import oai_kb_registry.services.infra_manager as _im  # noqa: PLC0415
            _im._DEFAULT_INFRA_SERVICES = list(args.infra_services)
        except ImportError:
            pass  # InfraManager not installed — auto_start_infra will fail gracefully

    if args.no_auth:
        os.environ["KB_AUTH_ENABLED"] = "false"

    if args.kb_data_dir:
        os.environ["KB_DATA_DIR"] = args.kb_data_dir

    # --- Resolve host and port: CLI flag > env var > default -------------
    host = args.host or os.environ.get("HOST", "0.0.0.0")
    port = args.port or int(os.environ.get("PORT", "8085"))

    os.environ["HOST"] = host
    os.environ["PORT"] = str(port)

    log_level = args.log_level or "info"

    # Use uvloop only on non-Windows systems for better performance
    loop_type = "uvloop" if platform.system() != "Windows" else "asyncio"

    logging.getLogger("oai_kb_registry").info(
        "Starting Knowledge Base Registry on %s:%s using %s loop", host, port, loop_type
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,
        reload=args.reload,
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_keep_alive=300,
        loop=loop_type,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
