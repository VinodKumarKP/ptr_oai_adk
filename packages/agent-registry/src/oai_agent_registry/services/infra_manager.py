"""
InfraManager — starts infrastructure Docker Compose services (postgres, valkey)
before the registry backend initialises its database connection.

Usage (called automatically from AgentRegistry.initialize when
registry_config.auto_start_infra is True):

    manager = InfraManager(
        compose_file=Path("/path/to/docker-compose.yaml"),
        startup_timeout=60,
    )
    await manager.start()

The manager:
  1. Runs  docker compose up -d --wait postgres valkey
     --wait blocks until both services' healthchecks pass (pg_isready / redis ping).
  2. Falls back to a TCP poll on the Postgres port for Docker Compose installs
     that pre-date --wait support (< v2.4).
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Services we always start for the registry backend.
# Clients can override via the constructor if needed.
_DEFAULT_INFRA_SERVICES: List[str] = ["postgres", "valkey"]


class InfraManager:
    """Manages lifecycle of infrastructure Docker services."""

    def __init__(
        self,
        compose_file: Path,
        project_name: str = "agent-registry",
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

        --wait (Compose v2.4+) blocks until healthchecks pass, which means
        postgres is accepting connections before the command returns.
        If the flag is unrecognised we retry without it and rely on the
        TCP poll in start() instead.
        """
        cmd_base = [
            "docker", "compose",
            "-f", str(self.compose_file),
            "-p", self.project_name,
            "up", "-d",
        ]

        # Preferred: --wait honours the healthchecks defined in the compose file.
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

            # If --wait is not supported the error mentions "unknown flag"
            if use_wait and (
                "unknown flag" in output.lower()
                or "unknown shorthand flag" in output.lower()
            ):
                logger.debug(
                    "--wait flag not supported by this Docker Compose version; retrying without it"
                )
                continue  # retry without --wait

            raise RuntimeError(
                f"docker compose up failed (exit {proc.returncode}):\n{output}"
            )

    async def _wait_for_tcp(self, host: str, port: int) -> None:
        """Poll until Postgres accepts authenticated connections.

        Prefers an asyncpg round-trip over raw TCP so that Docker Desktop's
        port-proxy race condition is avoided: the Docker proxy can complete a
        TCP handshake before Postgres is ready to serve clients, which would
        cause asyncpg to fail immediately after this check returns.

        Falls back to a TCP-only check when asyncpg is not installed.
        """
        try:
            import asyncpg as _asyncpg  # local import — may not be installed
        except ImportError:
            _asyncpg = None

        pg_user = os.environ.get("LOGGING_DB_USER", "postgres")
        pg_password = os.environ.get("LOGGING_DB_PASSWORD", "postgres")

        deadline = asyncio.get_event_loop().time() + self.startup_timeout
        attempt = 0

        while True:
            attempt += 1
            try:
                if _asyncpg is not None:
                    # Full authentication test — only succeeds when Postgres is
                    # ready to serve clients, not just when the port is open.
                    conn = await asyncio.wait_for(
                        _asyncpg.connect(
                            host=host,
                            port=port,
                            database="postgres",
                            user=pg_user,
                            password=pg_password,
                        ),
                        timeout=3.0,
                    )
                    await conn.close()
                else:
                    # Fallback: plain TCP handshake
                    _, writer = await asyncio.wait_for(
                        asyncio.open_connection(host, port), timeout=2.0
                    )
                    writer.close()
                    await writer.wait_closed()

                logger.debug(
                    "Postgres ready at %s:%s after %d attempt(s)",
                    host, port, attempt,
                )
                return

            except Exception:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Postgres at {host}:{port} did not become ready "
                        f"within {self.startup_timeout}s"
                    )
                wait = min(2 ** min(attempt - 1, 3), 10)
                wait = min(wait, remaining)
                logger.debug(
                    "Postgres not ready yet (attempt %d), retrying in %.0fs ...",
                    attempt, wait,
                )
                await asyncio.sleep(wait)
