import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import Request, HTTPException

from oai_skills_registry import dependencies
from oai_skills_registry.services.skills_registry import SkillsRegistry
from oai_skills_registry.services.infra_manager import InfraManager


@pytest.fixture(autouse=True)
def cleanup_registry_state():
    """Reset module-level registry state before and after each test."""
    dependencies._skills_registry = None
    dependencies._db_logger = None
    dependencies._infra_started = False
    dependencies._infra_compose_file_used = None
    dependencies._auto_start_infra = None
    dependencies._infra_compose_file = None
    dependencies._infra_startup_timeout = None
    yield
    dependencies._skills_registry = None
    dependencies._db_logger = None
    dependencies._infra_started = False
    dependencies._infra_compose_file_used = None
    dependencies._auto_start_infra = None
    dependencies._infra_compose_file = None
    dependencies._infra_startup_timeout = None


def test_get_registry_raises_uninitialized():
    """Test get_registry raises RuntimeError if not initialized."""
    with pytest.raises(RuntimeError, match="Registry not initialized"):
        dependencies.get_registry()


@pytest.mark.asyncio
async def test_initialize_registry_no_auto_start():
    """Test successful initialization without infra auto-start."""
    mock_db_logger_cls = MagicMock()
    mock_db_instance = AsyncMock()
    mock_db_instance.initialize.return_value = True
    mock_db_logger_cls.return_value = mock_db_instance

    with patch("oai_skills_registry.dependencies.SkillsDatabaseLogger", mock_db_logger_cls):
        registry = await dependencies.initialize_registry()
        assert isinstance(registry, SkillsRegistry)
        assert dependencies.get_registry() == registry
        assert dependencies._infra_started is False


@pytest.mark.asyncio
async def test_initialize_registry_with_auto_start():
    """Test registry initialization with infra auto-start enabled."""
    mock_db_instance = AsyncMock()
    mock_db_instance.initialize.return_value = True

    # Enable auto start via module variable
    dependencies._auto_start_infra = True
    dependencies._infra_compose_file = "/custom/compose.yaml"
    dependencies._infra_startup_timeout = 30

    with patch("oai_skills_registry.dependencies.SkillsDatabaseLogger", return_value=mock_db_instance), \
         patch.object(InfraManager, "start_services") as mock_start, \
         patch.object(InfraManager, "wait_for_postgres", new_callable=AsyncMock) as mock_wait:
        
        await dependencies.initialize_registry()
        
        mock_start.assert_called_once_with(compose_file=Path("/custom/compose.yaml"))
        mock_wait.assert_called_once_with(timeout=30)
        assert dependencies._infra_started is True
        assert dependencies._infra_compose_file_used == Path("/custom/compose.yaml")


@pytest.mark.asyncio
async def test_initialize_registry_auto_start_failures():
    """Test initialize_registry handles docker compose startup failures gracefully."""
    mock_db_instance = AsyncMock()
    mock_db_instance.initialize.return_value = True

    dependencies._auto_start_infra = True

    with patch("oai_skills_registry.dependencies.SkillsDatabaseLogger", return_value=mock_db_instance), \
         patch.object(InfraManager, "start_services", side_effect=Exception("Compose fail")) as mock_start, \
         patch.object(InfraManager, "wait_for_postgres", new_callable=AsyncMock, side_effect=TimeoutError("Postgres timeout")) as mock_wait:
        
        await dependencies.initialize_registry()
        
        assert dependencies._infra_started is False  # infra did not start successfully
        mock_start.assert_called_once()
        mock_wait.assert_called_once()


@pytest.mark.asyncio
async def test_verify_bearer_token():
    """Test verify_bearer_token calls security validator."""
    mock_request = MagicMock(spec=Request)
    
    with patch("oai_skills_registry.dependencies._validate_token") as mock_validate:
        res = await dependencies.verify_bearer_token(mock_request)
        assert res is True
        mock_validate.assert_called_once_with(mock_request)


@pytest.mark.asyncio
async def test_get_auth_user():
    """Test get_auth_user extracts appropriate request state fields."""
    mock_request = MagicMock(spec=Request)
    
    # 1. Case: user_email is present (SAML)
    mock_request.state.user_email = "saml@example.com"
    mock_request.state.user_id = "apikey-user"
    assert await dependencies.get_auth_user(mock_request) == "saml@example.com"

    # 2. Case: user_email is missing, user_id is present (API key)
    mock_request.state.user_email = None
    mock_request.state.user_id = "apikey-user"
    assert await dependencies.get_auth_user(mock_request) == "apikey-user"

    # 3. Case: both missing
    mock_request.state.user_email = None
    mock_request.state.user_id = None
    assert await dependencies.get_auth_user(mock_request) == "authenticated_user"


@pytest.mark.asyncio
async def test_close_registry_tears_down_infra():
    """Test close_registry tears down the initialized registry and stops compose infra."""
    mock_registry = AsyncMock(spec=SkillsRegistry)
    dependencies._skills_registry = mock_registry
    dependencies._infra_started = True
    dependencies._infra_compose_file_used = "/path/to/compose.yaml"

    mock_infra_manager = MagicMock()
    mock_infra_manager.stop_services = MagicMock()

    with patch("oai_skills_registry.services.infra_manager.InfraManager", mock_infra_manager):
        await dependencies.close_registry()
        
        mock_registry.close.assert_called_once()
        mock_infra_manager.stop_services.assert_called_once_with(compose_file="/path/to/compose.yaml")
        assert dependencies._infra_started is False
        assert dependencies._infra_compose_file_used is None


@pytest.mark.asyncio
async def test_close_registry_handles_stop_failures():
    """Test close_registry logs and ignores errors during infra teardown."""
    mock_registry = AsyncMock(spec=SkillsRegistry)
    dependencies._skills_registry = mock_registry
    dependencies._infra_started = True
    dependencies._infra_compose_file_used = "/path/to/compose.yaml"

    mock_infra_manager = MagicMock()
    mock_infra_manager.stop_services = MagicMock(side_effect=Exception("Stop fail"))

    with patch("oai_skills_registry.services.infra_manager.InfraManager", mock_infra_manager):
        await dependencies.close_registry()
        # Should not raise exception
        mock_registry.close.assert_called_once()
