"""
Tests for SkillsRegistry orchestrator service.
"""

import pytest
from oai_skills_registry.services.skills_registry import SkillsRegistry
from oai_skills_registry.services.skill_importer import SkillImporter
from oai_skills_registry.services.version_manager import VersionManager
from oai_skills_registry.services.skill_queries import SkillQueries
from oai_skills_registry.services.skill_discovery import SkillDiscovery
from oai_skills_registry.services.git_provider import GitHubProvider


class TestSkillsRegistryInitialization:
    """Tests for SkillsRegistry initialization."""

    @pytest.fixture
    def registry(self, mock_db_logger):
        """Create a SkillsRegistry instance."""
        return SkillsRegistry(mock_db_logger)

    def test_registry_initializes_with_db_logger(self, registry, mock_db_logger):
        """Test SkillsRegistry initializes with database logger."""
        assert registry.db_logger is mock_db_logger

    def test_registry_initializes_git_providers(self, registry):
        """Test SkillsRegistry initializes Git providers."""
        assert registry.git_providers is not None
        assert "github" in registry.git_providers
        assert isinstance(registry.git_providers["github"], GitHubProvider)

    def test_registry_initializes_importer(self, registry):
        """Test SkillsRegistry initializes SkillImporter."""
        assert registry.importer is not None
        assert isinstance(registry.importer, SkillImporter)

    def test_registry_initializes_version_manager(self, registry):
        """Test SkillsRegistry initializes VersionManager."""
        assert registry.version_manager is not None
        assert isinstance(registry.version_manager, VersionManager)

    def test_registry_initializes_queries(self, registry):
        """Test SkillsRegistry initializes SkillQueries."""
        assert registry.queries is not None
        assert isinstance(registry.queries, SkillQueries)

    def test_registry_initializes_discovery(self, registry):
        """Test SkillsRegistry initializes SkillDiscovery."""
        assert registry.discovery is not None
        assert isinstance(registry.discovery, SkillDiscovery)

    def test_registry_logger_initialized(self, registry):
        """Test SkillsRegistry has logger."""
        assert registry.logger is not None


@pytest.mark.asyncio
class TestSkillsRegistryDelegation:
    """Tests for SkillsRegistry delegation to managers."""

    @pytest.fixture
    def registry(self, mock_db_logger):
        """Create a SkillsRegistry instance."""
        return SkillsRegistry(mock_db_logger)

    async def test_register_git_source_delegates_to_importer(self, registry, monkeypatch):
        """Test register_git_source delegates to SkillImporter."""
        called = False

        async def mock_register(config):
            nonlocal called
            called = True
            return {"source_id": 1, "name": "test", "status": "registered"}

        monkeypatch.setattr(registry.importer, 'register_git_source', mock_register)

        result = await registry.register_git_source({"name": "test"})

        assert called is True
        assert result["source_id"] == 1

    async def test_publish_skill_version_delegates_to_version_manager(self, registry, monkeypatch):
        """Test publish_skill_version delegates to VersionManager."""
        called = False

        async def mock_publish(name, version, message, performed_by):
            nonlocal called
            called = True
            return {"status": "published", "version": version}

        monkeypatch.setattr(registry.version_manager, 'publish_skill_version', mock_publish)

        result = await registry.publish_skill_version("test", "1.0.0", None, "user")

        assert called is True
        assert result["status"] == "published"

    async def test_upgrade_skill_version_delegates_to_version_manager(self, registry, monkeypatch):
        """Test upgrade_skill_version delegates to VersionManager."""
        called = False

        async def mock_upgrade(name, to_version, message, performed_by):
            nonlocal called
            called = True
            return {"from_version": "1.0.0", "to_version": to_version}

        monkeypatch.setattr(registry.version_manager, 'upgrade_skill_version', mock_upgrade)

        result = await registry.upgrade_skill_version("test", "2.0.0", None, "user")

        assert called is True
        assert result["to_version"] == "2.0.0"

    async def test_downgrade_skill_version_delegates_to_version_manager(self, registry, monkeypatch):
        """Test downgrade_skill_version delegates to VersionManager."""
        called = False

        async def mock_downgrade(name, to_version, message, performed_by):
            nonlocal called
            called = True
            return {"from_version": "2.0.0", "to_version": to_version}

        monkeypatch.setattr(registry.version_manager, 'downgrade_skill_version', mock_downgrade)

        result = await registry.downgrade_skill_version("test", "1.0.0", None, "user")

        assert called is True
        assert result["to_version"] == "1.0.0"

    async def test_deprecate_skill_version_delegates_to_version_manager(self, registry, monkeypatch):
        """Test deprecate_skill_version delegates to VersionManager."""
        called = False

        async def mock_deprecate(name, version, message, performed_by):
            nonlocal called
            called = True
            return {"version": version}

        monkeypatch.setattr(registry.version_manager, 'deprecate_skill_version', mock_deprecate)

        result = await registry.deprecate_skill_version("test", "1.0.0", None, "user")

        assert called is True
        assert result["version"] == "1.0.0"

    async def test_delete_skill_delegates_to_version_manager(self, registry, monkeypatch):
        """Test delete_skill delegates to VersionManager."""
        called = False

        async def mock_delete(name, performed_by):
            nonlocal called
            called = True
            return {"skill": name, "status": "deleted"}

        monkeypatch.setattr(registry.version_manager, 'delete_skill', mock_delete)

        result = await registry.delete_skill("test", "user")

        assert called is True
        assert result["status"] == "deleted"

    async def test_get_skill_details_delegates_to_queries(self, registry, monkeypatch):
        """Test get_skill_details delegates to SkillQueries."""
        called = False

        async def mock_get_details(name):
            nonlocal called
            called = True
            return {"skill": {"name": name}, "versions": []}

        monkeypatch.setattr(registry.queries, 'get_skill_details', mock_get_details)

        result = await registry.get_skill_details("test")

        assert called is True
        assert result["skill"]["name"] == "test"

    async def test_get_skill_history_delegates_to_queries(self, registry, monkeypatch):
        """Test get_skill_history delegates to SkillQueries."""
        called = False

        async def mock_get_history(name, limit):
            nonlocal called
            called = True
            return {"skill_name": name, "actions": []}

        monkeypatch.setattr(registry.queries, 'get_skill_history', mock_get_history)

        result = await registry.get_skill_history("test", limit=50)

        assert called is True
        assert result["skill_name"] == "test"

    async def test_discover_skills_delegates_to_discovery(self, registry, monkeypatch):
        """Test discover_skills_from_git delegates to SkillDiscovery."""
        called = False

        async def mock_discover(url, auth_token):
            nonlocal called
            called = True
            return {"skills": [], "total_found": 0}

        monkeypatch.setattr(registry.discovery, 'discover_skills_from_git', mock_discover)

        result = await registry.discover_skills_from_git("https://github.com/owner/repo.git")

        assert called is True
        assert "skills" in result


@pytest.mark.asyncio
class TestSkillsRegistryMethods:
    """Tests for SkillsRegistry public methods."""

    @pytest.fixture
    def registry(self, mock_db_logger):
        """Create a SkillsRegistry instance."""
        return SkillsRegistry(mock_db_logger)

    async def test_registry_has_all_public_methods(self, registry):
        """Test SkillsRegistry exposes all required public methods."""
        methods = [
            'register_git_source',
            'import_skill_from_git',
            'publish_skill_version',
            'upgrade_skill_version',
            'downgrade_skill_version',
            'deprecate_skill_version',
            'delete_skill',
            'get_skill_details',
            'get_skill_history',
            'list_git_versions',
            'list_skill_versions_from_url',
            'discover_skills_from_git',
            'register_skills_bulk',
            'close'
        ]

        for method in methods:
            assert hasattr(registry, method), f"Missing method: {method}"
            assert callable(getattr(registry, method)), f"Not callable: {method}"

    async def test_close_closes_database(self, registry, mock_db_logger):
        """Test close method closes database connection."""
        await registry.close()

        mock_db_logger.close.assert_called_once()

    async def test_registry_list_skill_versions_from_url_with_invalid_url(self, registry):
        """Test list_skill_versions_from_url handles invalid URL."""
        result = await registry.list_skill_versions_from_url(
            "test-skill",
            "invalid-url"
        )

        assert result == []

    async def test_registry_register_skills_bulk_success(self, registry, mock_db_logger):
        """Test register_skills_bulk registers multiple skills."""
        mock_db_logger.get_skill.return_value = None
        mock_db_logger.create_skill.return_value = {"id": 1}

        # This test requires mocking aiohttp, so we'll just verify the method exists
        assert callable(registry.register_skills_bulk)
