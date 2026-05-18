"""
InfraManager — starts infrastructure Docker Compose services (postgres, valkey)
before the MCP registry backend initialises its database connection.

Usage (called automatically from MCPRegistry.initialize when
registry_config.auto_start_infra is True):

    # Step 1: generate compose file + start infra via the deployer (sync)
    deployer.start_infra_services()

    # Step 2: async TCP safety-net poll (class method, no compose interaction)
    await InfraManager.wait_for_postgres(timeout=60)

The TCP poll is a safety net for Docker Compose versions that pre-date --wait
support (< v2.4) where the command returns before pg_isready passes.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Services we always start for the registry backend.
_DEFAULT_INFRA_SERVICES: List[str] = ["postgres", "valkey"]


class InfraManager:
    """Manages lifecycle of infrastructure Docker services."""

    def __init__(
        self,
        compose_file: Path,
        project_name: str = "mcp-registry",
        services: Optional[List[str]] = None,
        startup_timeout: int = 60,
    ) -> None:
        self.compose_file = Path(compose_file)
        self.project_name = project_name
        self.services = services if services is not None else list(_DEFAULT_INFRA_SERVICES)
        self.startup_timeout = startup_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start infra services and wait until Postgres is reachable.

        Idempotent — safe to call when services are already running.
        Prefer calling deployer.start_infra_services() + InfraManager.wait_for_postgres()
        instead so the compose file is always regenerated first.
        """
        if not self.compose_file.exists():
            raise FileNotFoundError(
                f"Infra compose file not found: {self.compose_file}"
            )

        logger.info(
            "auto_start_infra: starting %s via %s",
            self.services,
            self.compose_file,
        )
        await self._compose_up()

        pg_host = os.environ.get("LOGGING_DB_HOST", "localhost")
        pg_port = int(os.environ.get("LOGGING_DB_PORT", "5432"))
        await self._wait_for_tcp(pg_host, pg_port)
        logger.info("Postgres is ready at %s:%s", pg_host, pg_port)

    @classmethod
    async def wait_for_postgres(
        cls,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: int = 60,
    ) -> None:
        """Poll TCP until Postgres accepts connections — no compose interaction.

        Use this as a safety-net after :py:meth:`DockerComposeManager.start_infra_services`
        returns so that the database logger can connect immediately.  Falls back
        gracefully on Docker Compose versions that do not support ``--wait``.

        Args:
            host:    Postgres host (default: ``LOGGING_DB_HOST`` env var or ``localhost``)
            port:    Postgres port (default: ``LOGGING_DB_PORT`` env var or ``5432``)
            timeout: Maximum seconds to wait before raising ``TimeoutError``.
        """
        pg_host = host or os.environ.get("LOGGING_DB_HOST", "localhost")
        pg_port = port or int(os.environ.get("LOGGING_DB_PORT", "5432"))
        # Reuse _wait_for_tcp from a minimal instance (compose_file not used here).
        instance = cls(compose_file=Path("."), startup_timeout=timeout)
        await instance._wait_for_tcp(pg_host, pg_port)
        logger.info("Postgres is ready at %s:%s", pg_host, pg_port)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _compose_up(self) -> None:
        """Run docker compose up -d --wait <services>.

        --wait (Compose v2.4+) blocks until healthchecks pass.
        Falls back to starting without --wait for older versions.
        """
        cmd_base = [
            "docker", "compose",
            "-f", str(self.compose_file),
            "-p", self.project_name,
            "up", "-d",
        ]

        for use_wait in (True, False):
            cmd = cmd_base + (["--wait"] if use_wait else []) + self.services
            logger.debug("Running: %s", " ".join(cmd))
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            output = stdout.decode("utf-8", errors="replace").strip()

            if proc.returncode == 0:
                if output:
                    logger.debug("docker compose output:\n%s", output)
                return

            if use_wait and (
                "unknown flag" in output.lower()
                or "unknown shorthand flag" in output.lower()
            ):
                logger.debug(
                    "--wait flag not supported by this Docker Compose version; retrying without it"
                )
                continue

            raise RuntimeError(
                f"docker compose up failed (exit {proc.returncode}):\n{output}"
            )

    async def _wait_for_tcp(self, host: str, port: int) -> None:
        """Poll TCP until Postgres accepts connections or startup_timeout elapses."""
        deadline = asyncio.get_event_loop().time() + self.startup_timeout
        attempt = 0

        while True:
            attempt += 1
            try:
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port), timeout=2.0
                )
                writer.close()
                await writer.wait_closed()
                logger.debug(
                    "Postgres TCP reachable at %s:%s after %d attempt(s)",
                    host, port, attempt,
                )
                return
            except (OSError, asyncio.TimeoutError):
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Postgres at {host}:{port} did not become reachable "
                        f"within {self.startup_timeout}s"
                    )
                wait = min(2 ** min(attempt - 1, 3), 10)
                wait = min(wait, remaining)
                logger.debug(
                    "Postgres not ready yet (attempt %d), retrying in %.0fs ...",
                    attempt, wait,
                )
                await asyncio.sleep(wait)
