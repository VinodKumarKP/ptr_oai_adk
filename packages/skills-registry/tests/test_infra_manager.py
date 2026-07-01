import asyncio
import os
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from oai_skills_registry.services.infra_manager import InfraManager


def test_infra_manager_init():
    """Test InfraManager initialization."""
    compose_path = Path("docker-compose.yaml")
    manager = InfraManager(compose_file=compose_path, project_name="custom", services=["postgres"], startup_timeout=30)
    assert manager.compose_file == compose_path
    assert manager.project_name == "custom"
    assert manager.services == ["postgres"]
    assert manager.startup_timeout == 30


@pytest.mark.asyncio
async def test_infra_manager_start_missing_file():
    """Test start raises FileNotFoundError if compose file does not exist."""
    manager = InfraManager(compose_file=Path("non_existent_file.yaml"))
    with pytest.raises(FileNotFoundError):
        await manager.start()


@pytest.mark.asyncio
async def test_infra_manager_start_success():
    """Test successful startup sequence."""
    manager = InfraManager(compose_file=Path(__file__))  # use this file so it exists
    
    with patch.object(manager, "_compose_up", new_callable=AsyncMock) as mock_up, \
         patch.object(manager, "_wait_for_tcp", new_callable=AsyncMock) as mock_wait:
        await manager.start()
        mock_up.assert_called_once()
        mock_wait.assert_called_once_with("localhost", 5434)


def test_start_services_success():
    """Test start_services runs subprocess and succeeds."""
    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "Started"
    mock_run.return_value.stderr = ""

    with patch("subprocess.run", mock_run), \
         patch("pathlib.Path.exists", return_value=True):
        InfraManager.start_services()
        # Verify first call was with --wait flag
        assert mock_run.call_count == 1
        args = mock_run.call_args[0][0]
        assert "--wait" in args


def test_start_services_wait_unsupported():
    """Test start_services retries without --wait when unsupported."""
    # First call fails with unknown flag, second call succeeds
    result1 = MagicMock(returncode=1, stdout="", stderr="docker: unknown flag: --wait")
    result2 = MagicMock(returncode=0, stdout="Started", stderr="")
    
    mock_run = MagicMock(side_effect=[result1, result2])

    with patch("subprocess.run", mock_run), \
         patch("pathlib.Path.exists", return_value=True):
        InfraManager.start_services()
        assert mock_run.call_count == 2
        # First call args
        args1 = mock_run.call_args_list[0][0][0]
        assert "--wait" in args1
        # Second call args
        args2 = mock_run.call_args_list[1][0][0]
        assert "--wait" not in args2


def test_start_services_missing_file():
    """Test start_services raises FileNotFoundError if compose file does not exist."""
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            InfraManager.start_services(compose_file=Path("missing.yaml"))


def test_start_services_raise_runtime_error():
    """Test start_services raises RuntimeError on general failures."""
    result = MagicMock(returncode=1, stdout="Failed", stderr="Error log")
    with patch("subprocess.run", return_value=result), \
         patch("pathlib.Path.exists", return_value=True):
        with pytest.raises(RuntimeError, match="docker compose up failed"):
            InfraManager.start_services()


def test_stop_services_with_services():
    """Test stop_services stops specific containers."""
    mock_run = MagicMock(returncode=0, stdout="stopped", stderr="")
    with patch("subprocess.run", return_value=mock_run) as mock_subprocess:
        InfraManager.stop_services(services=["postgres"])
        args = mock_subprocess.call_args[0][0]
        assert "stop" in args
        assert "postgres" in args


def test_stop_services_down():
    """Test stop_services downs compose project when no services are specified."""
    mock_run = MagicMock(returncode=0, stdout="downed", stderr="")
    with patch("subprocess.run", return_value=mock_run) as mock_subprocess:
        InfraManager.stop_services()
        args = mock_subprocess.call_args[0][0]
        assert "down" in args


def test_stop_services_warning_on_failure():
    """Test stop_services only logs warning and does not raise exception on failure."""
    mock_run = MagicMock(returncode=1, stdout="", stderr="shutdown failed")
    with patch("subprocess.run", return_value=mock_run):
        # Should not raise exception
        InfraManager.stop_services()


@pytest.mark.asyncio
async def test_wait_for_postgres_wrapper():
    """Test wait_for_postgres entrypoint."""
    with patch.object(InfraManager, "_wait_for_tcp", new_callable=AsyncMock) as mock_wait:
        await InfraManager.wait_for_postgres(host="127.0.0.1", port=9999, timeout=10)
        mock_wait.assert_called_once_with("127.0.0.1", 9999)


@pytest.mark.asyncio
async def test_compose_up_async_success():
    """Test async _compose_up method success path."""
    manager = InfraManager(compose_file=Path("docker-compose.yaml"))
    
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"Started", b"")
    
    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        await manager._compose_up()
        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert "--wait" in args


@pytest.mark.asyncio
async def test_compose_up_async_wait_unsupported():
    """Test async _compose_up retries without --wait flag when unsupported."""
    manager = InfraManager(compose_file=Path("docker-compose.yaml"))
    
    proc1 = AsyncMock()
    proc1.returncode = 1
    proc1.communicate.return_value = (b"unknown flag: --wait", b"")
    
    proc2 = AsyncMock()
    proc2.returncode = 0
    proc2.communicate.return_value = (b"Started", b"")
    
    with patch("asyncio.create_subprocess_exec", side_effect=[proc1, proc2]) as mock_exec:
        await manager._compose_up()
        assert mock_exec.call_count == 2
        # Verify first call had --wait
        assert "--wait" in mock_exec.call_args_list[0][0]
        # Verify second call did not have --wait
        assert "--wait" not in mock_exec.call_args_list[1][0]


@pytest.mark.asyncio
async def test_compose_up_async_runtime_error():
    """Test async _compose_up raises RuntimeError on execution failure."""
    manager = InfraManager(compose_file=Path("docker-compose.yaml"))
    
    proc = AsyncMock()
    proc.returncode = 1
    proc.communicate.return_value = (b"General compose error", b"")
    
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        with pytest.raises(RuntimeError, match="docker compose up failed"):
            await manager._compose_up()


@pytest.mark.asyncio
async def test_wait_for_tcp_asyncpg():
    """Test _wait_for_tcp using asyncpg success path."""
    manager = InfraManager(compose_file=Path("docker-compose.yaml"), startup_timeout=5)
    
    mock_conn = AsyncMock()
    
    with patch("asyncpg.connect", return_value=mock_conn) as mock_connect:
        await manager._wait_for_tcp("localhost", 5434)
        mock_connect.assert_called_once()
        mock_conn.close.assert_called_once()


@pytest.mark.asyncio
async def test_wait_for_tcp_fallback_no_asyncpg():
    """Test _wait_for_tcp falls back to raw TCP socket handshake when asyncpg is not available."""
    manager = InfraManager(compose_file=Path("docker-compose.yaml"), startup_timeout=5)
    
    mock_writer = AsyncMock()
    
    # Hide asyncpg module to trigger raw TCP socket fallback
    with patch.dict("sys.modules", {"asyncpg": None}), \
         patch("asyncio.open_connection", new_callable=AsyncMock, return_value=(MagicMock(), mock_writer)) as mock_conn:
        await manager._wait_for_tcp("localhost", 5434)
        mock_conn.assert_called_once()
        mock_writer.close.assert_called_once()
        mock_writer.wait_closed.assert_called_once()


@pytest.mark.asyncio
async def test_wait_for_tcp_timeout():
    """Test _wait_for_tcp timeout raises TimeoutError."""
    manager = InfraManager(compose_file=Path("docker-compose.yaml"), startup_timeout=1)
    
    # Force connection attempt to raise exception to trigger retry loop until deadline
    with patch("asyncpg.connect", side_effect=Exception("Connection refused")), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(TimeoutError, match="did not become ready"):
            await manager._wait_for_tcp("localhost", 5434)
        assert mock_sleep.called
