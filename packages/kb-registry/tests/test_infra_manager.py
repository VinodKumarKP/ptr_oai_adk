"""
Unit tests for InfraManager.
"""

import sys
import os
import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from oai_kb_registry.services.infra_manager import InfraManager


class TestInfraManager:
    """Tests for InfraManager."""

    @pytest.mark.asyncio
    async def test_start_success(self):
        manager = InfraManager(compose_file=Path("dummy_docker-compose.yaml"))
        with patch.object(Path, "exists", return_value=True), \
             patch.object(manager, "_compose_up", new_callable=AsyncMock) as mock_up, \
             patch.object(manager, "_wait_for_tcp", new_callable=AsyncMock) as mock_tcp:
            await manager.start()
            mock_up.assert_called_once()
            mock_tcp.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_file_not_found(self):
        manager = InfraManager(compose_file=Path("missing_docker-compose.yaml"))
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Infra compose file not found"):
                await manager.start()

    def test_start_services_success(self):
        with patch.object(Path, "exists", return_value=True), \
             patch("subprocess.run") as mock_run:
            mock_res = MagicMock()
            mock_res.returncode = 0
            mock_res.stdout = "OK"
            mock_res.stderr = ""
            mock_run.return_value = mock_res

            InfraManager.start_services(compose_file=Path("dummy.yaml"))
            mock_run.assert_called_once()

    def test_start_services_file_not_found(self):
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Infra compose file not found"):
                InfraManager.start_services(compose_file=Path("missing.yaml"))

    def test_start_services_failure_no_wait_fallback(self):
        with patch.object(Path, "exists", return_value=True), \
             patch("subprocess.run") as mock_run:
            # First attempt (with --wait) fails with "unknown flag"
            # Second attempt (without --wait) fails with general error
            mock_res_1 = MagicMock()
            mock_res_1.returncode = 1
            mock_res_1.stdout = ""
            mock_res_1.stderr = "unknown flag: --wait"

            mock_res_2 = MagicMock()
            mock_res_2.returncode = 1
            mock_res_2.stdout = ""
            mock_res_2.stderr = "general compose error"

            mock_run.side_effect = [mock_res_1, mock_res_2]

            with pytest.raises(RuntimeError, match="docker compose up failed"):
                InfraManager.start_services(compose_file=Path("dummy.yaml"))

            assert mock_run.call_count == 2

    def test_stop_services_down(self):
        with patch("subprocess.run") as mock_run:
            mock_res = MagicMock()
            mock_res.returncode = 0
            mock_res.stdout = "stopped"
            mock_res.stderr = ""
            mock_run.return_value = mock_res

            InfraManager.stop_services(compose_file=Path("dummy.yaml"))
            mock_run.assert_called_once()
            args, _ = mock_run.call_args
            assert "down" in args[0]

    def test_stop_services_stop_specific(self):
        with patch("subprocess.run") as mock_run:
            mock_res = MagicMock()
            mock_res.returncode = 0
            mock_res.stdout = "stopped"
            mock_res.stderr = ""
            mock_run.return_value = mock_res

            InfraManager.stop_services(compose_file=Path("dummy.yaml"), services=["postgres"])
            mock_run.assert_called_once()
            args, _ = mock_run.call_args
            assert "stop" in args[0]
            assert "postgres" in args[0]

    def test_stop_services_warning_ignored(self):
        with patch("subprocess.run") as mock_run:
            mock_res = MagicMock()
            mock_res.returncode = 1
            mock_res.stdout = ""
            mock_res.stderr = "compose down warning/error"
            mock_run.return_value = mock_res

            # Should not raise, just logs warning
            InfraManager.stop_services(compose_file=Path("dummy.yaml"))
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_postgres(self):
        with patch.object(InfraManager, "_wait_for_tcp", new_callable=AsyncMock) as mock_tcp:
            await InfraManager.wait_for_postgres()
            mock_tcp.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_vector_store_service_s3(self):
        with patch.object(InfraManager, "start_services") as mock_start:
            await InfraManager.ensure_vector_store_service("s3")
            mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_vector_store_service_postgres(self):
        with patch.object(Path, "exists", return_value=True), \
             patch.object(InfraManager, "start_services") as mock_start, \
             patch.object(InfraManager, "_wait_for_vector_service", new_callable=AsyncMock) as mock_wait:
            await InfraManager.ensure_vector_store_service("postgres", compose_file=Path("dummy.yaml"))
            mock_start.assert_called_once()
            mock_wait.assert_called_once_with("postgres")

    @pytest.mark.asyncio
    async def test_ensure_vector_store_service_file_not_found(self):
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Infra compose file not found"):
                await InfraManager.ensure_vector_store_service("postgres", compose_file=Path("missing.yaml"))

    @pytest.mark.asyncio
    async def test_wait_for_vector_service_postgres(self):
        manager = InfraManager(compose_file=Path("dummy.yaml"))
        with patch.object(manager, "_wait_for_tcp", new_callable=AsyncMock) as mock_tcp:
            await manager._wait_for_vector_service("postgres")
            mock_tcp.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_vector_service_chroma(self):
        manager = InfraManager(compose_file=Path("dummy.yaml"))
        with patch.object(manager, "_wait_for_chroma_heartbeat", new_callable=AsyncMock) as mock_heartbeat:
            await manager._wait_for_vector_service("chroma")
            mock_heartbeat.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_chroma_heartbeat_success(self):
        manager = InfraManager(compose_file=Path("dummy.yaml"), startup_timeout=2)
        mock_reader = AsyncMock()
        mock_reader.readline.return_value = b"HTTP/1.1 200 OK\r\n"
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.wait_closed = AsyncMock()

        with patch("asyncio.open_connection", new_callable=AsyncMock, return_value=(mock_reader, mock_writer)):
            await manager._wait_for_chroma_heartbeat("localhost", 8010)

    @pytest.mark.asyncio
    async def test_wait_for_chroma_heartbeat_timeout(self):
        manager = InfraManager(compose_file=Path("dummy.yaml"), startup_timeout=1)
        mock_reader = AsyncMock()
        mock_reader.readline.side_effect = Exception("Connection closed")
        mock_writer = MagicMock()

        # Mock asyncio.sleep to instantly fast-forward time to avoid hanging the test
        async def mock_sleep(sec):
            pass

        with patch("asyncio.open_connection", new_callable=AsyncMock, side_effect=Exception("TCP connect error")), \
             patch("asyncio.sleep", side_effect=mock_sleep), \
             patch("asyncio.get_event_loop") as mock_loop:
            # We mock time to trigger the deadline check immediately
            mock_loop_instance = MagicMock()
            mock_loop_instance.time.side_effect = [10.0, 15.0]
            mock_loop.return_value = mock_loop_instance

            with pytest.raises(TimeoutError, match="ChromaDB at localhost:8010 did not become ready"):
                await manager._wait_for_chroma_heartbeat("localhost", 8010)

    @pytest.mark.asyncio
    async def test_compose_up_success(self):
        manager = InfraManager(compose_file=Path("dummy.yaml"))
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"Created container", None))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc) as mock_exec:
            await manager._compose_up()
            mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_compose_up_fallback_no_wait(self):
        manager = InfraManager(compose_file=Path("dummy.yaml"))
        
        mock_proc_1 = MagicMock()
        mock_proc_1.returncode = 1
        mock_proc_1.communicate = AsyncMock(return_value=(b"unknown flag: --wait", None))

        mock_proc_2 = MagicMock()
        mock_proc_2.returncode = 0
        mock_proc_2.communicate = AsyncMock(return_value=(b"Created container without wait", None))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, side_effect=[mock_proc_1, mock_proc_2]) as mock_exec:
            await manager._compose_up()
            assert mock_exec.call_count == 2

    @pytest.mark.asyncio
    async def test_compose_up_failure(self):
        manager = InfraManager(compose_file=Path("dummy.yaml"))
        
        mock_proc_1 = MagicMock()
        mock_proc_1.returncode = 1
        mock_proc_1.communicate = AsyncMock(return_value=(b"unknown flag: --wait", None))

        mock_proc_2 = MagicMock()
        mock_proc_2.returncode = 1
        mock_proc_2.communicate = AsyncMock(return_value=(b"general compose error", None))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, side_effect=[mock_proc_1, mock_proc_2]):
            with pytest.raises(RuntimeError, match="docker compose up failed"):
                await manager._compose_up()

    @pytest.mark.asyncio
    async def test_wait_for_tcp_success_asyncpg(self):
        manager = InfraManager(compose_file=Path("dummy.yaml"), startup_timeout=2)
        mock_conn = AsyncMock()
        
        orig_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == "asyncpg":
                mock_asyncpg = MagicMock()
                mock_asyncpg.connect = AsyncMock(return_value=mock_conn)
                return mock_asyncpg
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            await manager._wait_for_tcp("localhost", 5436)

    @pytest.mark.asyncio
    async def test_wait_for_tcp_fallback_no_asyncpg(self):
        manager = InfraManager(compose_file=Path("dummy.yaml"), startup_timeout=2)
        mock_writer = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        orig_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == "asyncpg":
                raise ImportError()
            return orig_import(name, *args, **kwargs)

        # Simulate asyncpg is not installed by letting builtins.__import__ raise ImportError
        with patch("builtins.__import__", side_effect=mock_import), \
             patch("asyncio.open_connection", new_callable=AsyncMock, return_value=(AsyncMock(), mock_writer)) as mock_connect:
            await manager._wait_for_tcp("localhost", 5436)
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_tcp_timeout(self):
        manager = InfraManager(compose_file=Path("dummy.yaml"), startup_timeout=1)
        
        async def mock_sleep(sec):
            pass

        orig_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == "asyncpg":
                raise ImportError()
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import), \
             patch("asyncio.open_connection", new_callable=AsyncMock, side_effect=Exception("TCP Connect Refused")), \
             patch("asyncio.sleep", side_effect=mock_sleep), \
             patch("asyncio.get_event_loop") as mock_loop:
            
            mock_loop_instance = MagicMock()
            mock_loop_instance.time.side_effect = [10.0, 15.0]
            mock_loop.return_value = mock_loop_instance

            with pytest.raises(TimeoutError, match="Postgres at localhost:5436 did not become ready"):
                await manager._wait_for_tcp("localhost", 5436)
