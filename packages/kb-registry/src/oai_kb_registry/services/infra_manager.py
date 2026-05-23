"""
InfraManager — starts infrastructure Docker Compose services before the
KB Registry initialises its database connection.

Startup sequence
----------------
Phase 1 — at process startup (called from initialize_registry()):
    InfraManager.start_services()   → docker compose up -d postgres valkey
    await InfraManager.wait_for_postgres()  → TCP/asyncpg poll

Phase 2 — at KB registration time (called from KBRegistry.register_knowledge_base()):
    await InfraManager.ensure_vector_store_service(vector_db_type, compose_file)
    → starts pgvector or chromadb on demand, only when first needed

This on-demand approach means you never pay for a pgvector container when you
only use Chroma KBs, and vice versa.

Services in docker-compose.yaml
--------------------------------
  postgres   — metadata DB for KB registry log / document tracking (port 5436)
  valkey     — token / session store (port 6383)
  pgvector   — builtin Postgres + pgvector extension (port 5435)
  chromadb   — builtin ChromaDB HTTP server (port 8010)

Environment variables read by this module
------------------------------------------
  LOGGING_DB_HOST          — Postgres host          (default: localhost)
  LOGGING_DB_PORT          — Postgres port          (default: 5436)
  INFRA_COMPOSE_FILE       — Override compose file path (optional)
  AUTO_START_INFRA         — "true" to enable       (default: false)
  INFRA_STARTUP_TIMEOUT    — Seconds to wait        (default: 60)
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Services started by default (metadata DB + token store).
# Vector-store services (pgvector, chromadb) can be added via the CLI flag
# --infra-services or the services= parameter in start_services().
_DEFAULT_INFRA_SERVICES: List[str] = ["postgres", "valkey"]

# Bundled compose file shipped with this package.
_BUNDLED_COMPOSE = (
    Path(__file__).parent.parent / "resources" / "docker" / "docker-compose.yaml"
)


class InfraManager:
    """Manages lifecycle of infrastructure Docker services for KB Registry."""

    def __init__(
        self,
        compose_file: Path,
        project_name: str = "kb-registry",
        services: Optional[List[str]] = None,
        startup_timeout: int = 60,
    ) -> None:
        self.compose_file    = Path(compose_file)
        self.project_name    = project_name
        self.services        = services if services is not None else list(_DEFAULT_INFRA_SERVICES)
        self.startup_timeout = startup_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start infra services and wait until the metadata Postgres is reachable.

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
        pg_port = int(os.environ.get("LOGGING_DB_PORT", "5436"))
        await self._wait_for_tcp(pg_host, pg_port)
        logger.info("Postgres is ready at %s:%s", pg_host, pg_port)

    @classmethod
    def start_services(
        cls,
        compose_file: Optional[Path] = None,
        project_name: str = "kb-registry",
        services: Optional[List[str]] = None,
    ) -> None:
        """Start infra services synchronously (no event loop needed).

        Runs ``docker compose up -d --wait <services>`` and falls back to
        starting without ``--wait`` for older Docker Compose versions.

        Args:
            compose_file:  Path to docker-compose file; defaults to the bundled file.
            project_name:  Docker Compose project name.
            services:      Services to start; defaults to ["postgres", "valkey"].
        """
        target_file     = compose_file or _BUNDLED_COMPOSE
        target_services = services    or list(_DEFAULT_INFRA_SERVICES)

        if not target_file.exists():
            raise FileNotFoundError(
                f"Infra compose file not found: {target_file}"
            )

        for use_wait in (True, False):
            cmd = [
                "docker", "compose",
                "-f", str(target_file),
                "-p", project_name,
                "up", "-d",
            ]
            if use_wait:
                cmd.append("--wait")
            cmd.extend(target_services)

            logger.info("Running: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)

            combined = (result.stdout + result.stderr).strip()
            if result.returncode == 0:
                if combined:
                    logger.debug("docker compose output:\n%s", combined)
                logger.info("Infra services started successfully.")
                return

            output_lower = combined.lower()
            if use_wait and (
                "unknown flag"      in output_lower
                or "unknown shorthand" in output_lower
            ):
                logger.debug("--wait not supported; retrying without it")
                continue

            raise RuntimeError(
                f"docker compose up failed for infra services "
                f"(exit {result.returncode}):\n{combined}"
            )

    @classmethod
    def stop_services(
        cls,
        compose_file: Optional[Path] = None,
        project_name: str = "kb-registry",
        services: Optional[List[str]] = None,
    ) -> None:
        """Stop infra services synchronously.

        If *services* is given, only those containers are stopped.
        If *services* is ``None`` the entire Compose project is torn down.

        A non-zero exit code is logged as a warning rather than raising so
        that infra teardown never blocks the rest of the shutdown sequence.

        Args:
            compose_file:  Path to the docker-compose file; defaults to the bundled file.
            project_name:  Docker Compose project name.
            services:      Optional list of service names to stop individually.
        """
        target_file = compose_file or _BUNDLED_COMPOSE

        if services:
            cmd = [
                "docker", "compose",
                "-f", str(target_file),
                "-p", project_name,
                "stop",
            ] + list(services)
            action = "stop"
        else:
            cmd = [
                "docker", "compose",
                "-f", str(target_file),
                "-p", project_name,
                "down",
            ]
            action = "down"

        logger.info("auto_stop_infra: running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        combined = (result.stdout + result.stderr).strip()

        if result.returncode == 0:
            if combined:
                logger.debug("docker compose %s output:\n%s", action, combined)
            logger.info("auto_stop_infra: infra services stopped successfully.")
            return

        logger.warning(
            "auto_stop_infra: docker compose %s returned exit %d:\n%s",
            action, result.returncode, combined,
        )

    @classmethod
    async def wait_for_postgres(
        cls,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: int = 60,
    ) -> None:
        """Poll TCP until the metadata Postgres accepts connections.

        Use this as a safety-net after :meth:`start_services` returns so that
        the database logger can connect immediately.

        Args:
            host:    Postgres host (default: ``LOGGING_DB_HOST`` env var or ``localhost``)
            port:    Postgres port (default: ``LOGGING_DB_PORT`` env var or ``5436``)
            timeout: Maximum seconds to wait before raising ``TimeoutError``.
        """
        pg_host = host or os.environ.get("LOGGING_DB_HOST", "localhost")
        pg_port = port or int(os.environ.get("LOGGING_DB_PORT", "5436"))
        instance = cls(compose_file=Path("."), startup_timeout=timeout)
        await instance._wait_for_tcp(pg_host, pg_port)
        logger.info("Postgres is ready at %s:%s", pg_host, pg_port)

    # ------------------------------------------------------------------
    # On-demand vector store provisioning
    # ------------------------------------------------------------------

    # Maps vector_db_type → (compose service name, readiness check function)
    _VECTOR_SERVICE_MAP: dict = {
        "postgres": "pgvector",
        "chroma":   "chromadb",
        # "s3" intentionally absent — no container needed
    }

    @classmethod
    async def ensure_vector_store_service(
        cls,
        vector_db_type: str,
        compose_file: Optional[Path] = None,
        project_name: str = "kb-registry",
        timeout: int = 60,
    ) -> None:
        """Start the vector store container for *vector_db_type* if not already running.

        Called at KB registration time so the container is provisioned on first
        use rather than speculatively at startup.

        - ``postgres`` → starts the ``pgvector`` service and waits for its TCP port (5435)
        - ``chroma``   → starts the ``chromadb`` service and waits for its HTTP heartbeat
        - ``s3``       → no-op (AWS S3 needs no local container)

        Idempotent: if the service is already healthy the docker compose call
        returns immediately and no unnecessary work is done.

        Args:
            vector_db_type: "chroma", "postgres", or "s3".
            compose_file:   Path to the compose file; defaults to the bundled file.
            project_name:   Docker Compose project name.
            timeout:        Seconds to wait for the service to become ready.
        """
        service = cls._VECTOR_SERVICE_MAP.get(vector_db_type.lower())
        if service is None:
            # s3 or unrecognised type — nothing to start
            logger.debug(
                "ensure_vector_store_service: no container needed for vector_db_type=%r",
                vector_db_type,
            )
            return

        target_file = compose_file or _BUNDLED_COMPOSE
        if not target_file.exists():
            raise FileNotFoundError(
                f"Infra compose file not found: {target_file}"
            )

        logger.info(
            "ensure_vector_store_service: ensuring '%s' container is running "
            "(vector_db_type=%r)",
            service, vector_db_type,
        )

        # Start the service (idempotent — safe if already running)
        cls.start_services(
            compose_file=target_file,
            project_name=project_name,
            services=[service],
        )

        # Wait until the service is actually accepting connections
        instance = cls(
            compose_file=target_file,
            project_name=project_name,
            services=[service],
            startup_timeout=timeout,
        )
        await instance._wait_for_vector_service(vector_db_type)
        logger.info(
            "ensure_vector_store_service: '%s' is ready (vector_db_type=%r)",
            service, vector_db_type,
        )

    async def _wait_for_vector_service(self, vector_db_type: str) -> None:
        """Wait until the vector store service accepts connections."""
        vdb = vector_db_type.lower()

        if vdb == "postgres":
            # pgvector is a Postgres instance — reuse the asyncpg TCP poll
            host = os.environ.get("KB_PGVECTOR_HOST", "localhost")
            port = int(os.environ.get("KB_PGVECTOR_PORT", "5435"))
            await self._wait_for_tcp(host, port)

        elif vdb == "chroma":
            # ChromaDB exposes a /api/v1/heartbeat endpoint
            host = os.environ.get("KB_CHROMA_HOST", "localhost")
            port = int(os.environ.get("KB_CHROMA_PORT", "8010"))
            await self._wait_for_chroma_heartbeat(host, port)

    async def _wait_for_chroma_heartbeat(self, host: str, port: int) -> None:
        """Poll ChromaDB's /api/v1/heartbeat until it responds 200."""
        deadline = asyncio.get_event_loop().time() + self.startup_timeout
        attempt  = 0
        url      = f"http://{host}:{port}/api/v1/heartbeat"

        while True:
            attempt += 1
            try:
                # Use a raw asyncio TCP connect + minimal HTTP request to avoid
                # pulling in httpx/aiohttp as hard dependencies here.
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port), timeout=2.0
                )
                writer.write(f"GET /api/v1/heartbeat HTTP/1.0\r\nHost: {host}\r\n\r\n".encode())
                await writer.drain()
                response_line = await asyncio.wait_for(reader.readline(), timeout=2.0)
                writer.close()
                await writer.wait_closed()

                if b"200" in response_line:
                    logger.debug(
                        "ChromaDB ready at %s:%s after %d attempt(s)", host, port, attempt
                    )
                    return

            except Exception:
                pass

            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"ChromaDB at {host}:{port} did not become ready "
                    f"within {self.startup_timeout}s"
                )
            wait = min(2 ** min(attempt - 1, 3), 10, remaining)
            logger.debug(
                "ChromaDB not ready yet (attempt %d), retrying in %.0fs …",
                attempt, wait,
            )
            await asyncio.sleep(wait)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _compose_up(self) -> None:
        """Run docker compose up -d [--wait] <services> asynchronously."""
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
                "unknown flag"          in output.lower()
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
        """Poll until Postgres accepts authenticated connections.

        Prefers an asyncpg round-trip over a raw TCP check to avoid the
        Docker Desktop port-proxy race condition (the proxy can complete a TCP
        handshake before Postgres is ready to serve clients).

        Falls back to a raw TCP check when asyncpg is not installed.
        """
        try:
            import asyncpg as _asyncpg  # noqa: PLC0415
        except ImportError:
            _asyncpg = None

        pg_user     = os.environ.get("LOGGING_DB_USER",     "postgres")
        pg_password = os.environ.get("LOGGING_DB_PASSWORD", "postgres")

        deadline = asyncio.get_event_loop().time() + self.startup_timeout
        attempt  = 0

        while True:
            attempt += 1
            try:
                if _asyncpg is not None:
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
                    "Postgres not ready yet (attempt %d), retrying in %.0fs …",
                    attempt, wait,
                )
                await asyncio.sleep(wait)
