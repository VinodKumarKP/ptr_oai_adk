import asyncio
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from oai_agent_registry.services.infra_manager import InfraManager

@pytest.fixture
def temp_compose_file(tmp_path):
    compose_file = tmp_path / "docker-compose.yaml"
    compose_file.write_text("version: '3'")
    return compose_file

@pytest.mark.asyncio
async def test_infra_manager_init(temp_compose_file):
    manager = InfraManager(compose_file=temp_compose_file)
    assert manager.compose_file == temp_compose_file
    assert manager.project_name == "agent-registry"
    assert "postgres" in manager.services
    assert "valkey" in manager.services
    assert manager.startup_timeout == 60

@pytest.mark.asyncio
@patch("oai_agent_registry.services.infra_manager.asyncio.create_subprocess_exec", new_callable=AsyncMock)
async def test_compose_up_success(mock_create_subprocess, temp_compose_file):
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"started", b""))
    mock_create_subprocess.return_value = mock_proc

    manager = InfraManager(compose_file=temp_compose_file)
    await manager._compose_up()
    
    mock_create_subprocess.assert_called_once()
    args, kwargs = mock_create_subprocess.call_args
    assert "--wait" in args

@pytest.mark.asyncio
@patch("oai_agent_registry.services.infra_manager.asyncio.create_subprocess_exec", new_callable=AsyncMock)
async def test_compose_up_fallback_without_wait(mock_create_subprocess, temp_compose_file):
    # First call with --wait fails with "unknown flag", second call succeeds
    mock_proc1 = MagicMock()
    mock_proc1.returncode = 1
    mock_proc1.communicate = AsyncMock(return_value=(b"unknown flag --wait", b""))
    
    mock_proc2 = MagicMock()
    mock_proc2.returncode = 0
    mock_proc2.communicate = AsyncMock(return_value=(b"started without wait", b""))
    
    mock_create_subprocess.side_effect = [mock_proc1, mock_proc2]

    manager = InfraManager(compose_file=temp_compose_file)
    await manager._compose_up()
    
    assert mock_create_subprocess.call_count == 2
    args1, _ = mock_create_subprocess.call_args_list[0]
    args2, _ = mock_create_subprocess.call_args_list[1]
    
    assert "--wait" in args1
    assert "--wait" not in args2

@pytest.mark.asyncio
@patch("oai_agent_registry.services.infra_manager.asyncio.create_subprocess_exec", new_callable=AsyncMock)
async def test_compose_up_failure(mock_create_subprocess, temp_compose_file):
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(return_value=(b"error starting", b""))
    mock_create_subprocess.return_value = mock_proc

    manager = InfraManager(compose_file=temp_compose_file)
    with pytest.raises(RuntimeError, match="docker compose up failed"):
        await manager._compose_up()

@pytest.mark.asyncio
@patch("oai_agent_registry.services.infra_manager.asyncio.open_connection", new_callable=AsyncMock)
async def test_wait_for_tcp_fallback(mock_open_connection, temp_compose_file):
    # Test fallback to TCP when asyncpg is not imported/fails
    mock_writer = MagicMock()
    mock_writer.wait_closed = AsyncMock()
    mock_open_connection.return_value = (MagicMock(), mock_writer)
    
    manager = InfraManager(compose_file=temp_compose_file, startup_timeout=1)
    with patch.dict("sys.modules", {"asyncpg": None}):
        await manager._wait_for_tcp("localhost", 5433)
        
    mock_open_connection.assert_called_once_with("localhost", 5433)
    mock_writer.close.assert_called_once()

@pytest.mark.asyncio
@patch.object(InfraManager, "_compose_up", new_callable=AsyncMock)
@patch.object(InfraManager, "_wait_for_tcp", new_callable=AsyncMock)
async def test_start(mock_wait_tcp, mock_compose_up, temp_compose_file):
    manager = InfraManager(compose_file=temp_compose_file)
    await manager.start()
    
    mock_compose_up.assert_called_once()
    mock_wait_tcp.assert_called_once_with("localhost", 5433)

@pytest.mark.asyncio
async def test_start_file_not_found(tmp_path):
    non_existent = tmp_path / "not_there.yaml"
    manager = InfraManager(compose_file=non_existent)
    
    with pytest.raises(FileNotFoundError, match="Infra compose file not found"):
        await manager.start()

@pytest.mark.asyncio
@patch.object(InfraManager, "_wait_for_tcp", new_callable=AsyncMock)
async def test_wait_for_postgres_classmethod(mock_wait_tcp):
    await InfraManager.wait_for_postgres(host="test_host", port=9999, timeout=30)
    mock_wait_tcp.assert_called_once_with("test_host", 9999)
