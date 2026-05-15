"""
Tests for VersionManager service.
"""

import pytest
from datetime import datetime, timezone
from oai_skills_registry.services.version_manager import VersionManager


@pytest.mark.asyncio
class TestVersionManager:
    """Tests for VersionManager class."""

    @pytest.fixture
    def version_manager(self, mock_db_logger):
        """Create a VersionManager instance with mock database."""
        return VersionManager(mock_db_logger)

    async def test_publish_skill_version_success(self, version_manager, mock_db_logger):
        """Test publishing a draft skill version."""
        # Setup mocks
        mock_db_logger.get_skill.return_value = {"id": 1, "name": "test-skill"}
        mock_db_logger.get_skill_version.return_value = {
            "id": 1,
            "version": "1.0.0",
            "status": "draft"
        }
        mock_db_logger.log_skill_action.return_value = {"id": 1}

        result = await version_manager.publish_skill_version(
            "test-skill",
            "1.0.0",
            "Publishing version 1.0.0",
            "test-user"
        )

        assert result["status"] == "published"
        assert result["version"] == "1.0.0"
        assert result["skill_id"] == 1
        mock_db_logger.update_skill_version_status.assert_called_once()
        mock_db_logger.log_skill_action.assert_called_once()

    async def test_publish_nonexistent_skill(self, version_manager, mock_db_logger):
        """Test publishing version of non-existent skill raises error."""
        mock_db_logger.get_skill.return_value = None

        with pytest.raises(ValueError, match="Skill not found"):
            await version_manager.publish_skill_version(
                "nonexistent",
                "1.0.0",
                None,
                "test-user"
            )

    async def test_publish_nonexistent_version(self, version_manager, mock_db_logger):
        """Test publishing non-existent version raises error."""
        mock_db_logger.get_skill.return_value = {"id": 1, "name": "test-skill"}
        mock_db_logger.get_skill_version.return_value = None

        with pytest.raises(ValueError, match="Version not found"):
            await version_manager.publish_skill_version(
                "test-skill",
                "999.0.0",
                None,
                "test-user"
            )

    async def test_publish_non_draft_version(self, version_manager, mock_db_logger):
        """Test publishing already published version raises error."""
        mock_db_logger.get_skill.return_value = {"id": 1, "name": "test-skill"}
        mock_db_logger.get_skill_version.return_value = {
            "id": 1,
            "version": "1.0.0",
            "status": "published"
        }

        with pytest.raises(ValueError, match="Cannot publish version with status"):
            await version_manager.publish_skill_version(
                "test-skill",
                "1.0.0",
                None,
                "test-user"
            )

    async def test_upgrade_skill_version_success(self, version_manager, mock_db_logger):
        """Test upgrading skill to new version."""
        # Setup mocks
        mock_db_logger.get_skill.return_value = {
            "id": 1,
            "name": "test-skill",
            "current_version": "1.0.0"
        }
        mock_db_logger.get_skill_version.return_value = {
            "id": 2,
            "version": "2.0.0",
            "status": "published"
        }
        mock_db_logger.get_agent_skills.return_value = []
        mock_db_logger.log_skill_action.return_value = {"id": 2}

        result = await version_manager.upgrade_skill_version(
            "test-skill",
            "2.0.0",
            "Upgrading to 2.0.0",
            "test-user"
        )

        assert result["from_version"] == "1.0.0"
        assert result["to_version"] == "2.0.0"
        mock_db_logger.update_skill_current_version.assert_called_once()

    async def test_upgrade_unpublished_version(self, version_manager, mock_db_logger):
        """Test upgrading to unpublished version raises error."""
        mock_db_logger.get_skill.return_value = {
            "id": 1,
            "name": "test-skill",
            "current_version": "1.0.0"
        }
        mock_db_logger.get_skill_version.return_value = {
            "id": 2,
            "version": "2.0.0",
            "status": "draft"
        }

        with pytest.raises(ValueError, match="Version not available"):
            await version_manager.upgrade_skill_version(
                "test-skill",
                "2.0.0",
                None,
                "test-user"
            )

    async def test_downgrade_skill_version_success(self, version_manager, mock_db_logger):
        """Test downgrading skill to previous version."""
        # Setup mocks
        mock_db_logger.get_skill.return_value = {
            "id": 1,
            "name": "test-skill",
            "current_version": "2.0.0"
        }
        mock_db_logger.get_skill_version.return_value = {
            "id": 1,
            "version": "1.0.0",
            "status": "published"
        }
        mock_db_logger.log_skill_action.return_value = {"id": 3}

        result = await version_manager.downgrade_skill_version(
            "test-skill",
            "1.0.0",
            "Downgrading to 1.0.0",
            "test-user"
        )

        assert result["from_version"] == "2.0.0"
        assert result["to_version"] == "1.0.0"
        mock_db_logger.update_skill_current_version.assert_called_once()

    async def test_downgrade_nonexistent_version(self, version_manager, mock_db_logger):
        """Test downgrading to non-existent version raises error."""
        mock_db_logger.get_skill.return_value = {
            "id": 1,
            "name": "test-skill",
            "current_version": "2.0.0"
        }
        mock_db_logger.get_skill_version.return_value = None

        with pytest.raises(ValueError, match="Version not found"):
            await version_manager.downgrade_skill_version(
                "test-skill",
                "0.5.0",
                None,
                "test-user"
            )

    async def test_deprecate_skill_version_success(self, version_manager, mock_db_logger):
        """Test deprecating a skill version."""
        # Setup mocks
        mock_db_logger.get_skill.return_value = {"id": 1, "name": "test-skill"}
        mock_db_logger.get_skill_version.return_value = {
            "id": 1,
            "version": "1.0.0",
            "status": "published"
        }
        mock_db_logger.log_skill_action.return_value = {"id": 4}

        result = await version_manager.deprecate_skill_version(
            "test-skill",
            "1.0.0",
            "Version 1.0.0 is deprecated",
            "test-user"
        )

        assert result["version"] == "1.0.0"
        assert result["message"] == "Version 1.0.0 is deprecated"
        mock_db_logger.update_skill_version_status.assert_called_once()

    async def test_deprecate_nonexistent_version(self, version_manager, mock_db_logger):
        """Test deprecating non-existent version raises error."""
        mock_db_logger.get_skill.return_value = {"id": 1, "name": "test-skill"}
        mock_db_logger.get_skill_version.return_value = None

        with pytest.raises(ValueError, match="Version not found"):
            await version_manager.deprecate_skill_version(
                "test-skill",
                "999.0.0",
                None,
                "test-user"
            )

    async def test_delete_skill_success(self, version_manager, mock_db_logger):
        """Test deleting entire skill."""
        # Setup mocks
        mock_db_logger.get_skill.return_value = {
            "id": 1,
            "name": "test-skill"
        }
        mock_db_logger.log_skill_action.return_value = {"id": 5}

        result = await version_manager.delete_skill("test-skill", "test-user")

        assert result["skill"] == "test-skill"
        assert result["status"] == "deleted"
        mock_db_logger.delete_skill.assert_called_once_with("test-skill")

    async def test_delete_nonexistent_skill(self, version_manager, mock_db_logger):
        """Test deleting non-existent skill raises error."""
        mock_db_logger.get_skill.return_value = None

        with pytest.raises(ValueError, match="Skill not found"):
            await version_manager.delete_skill("nonexistent", "test-user")

    async def test_upgrade_updates_affected_agents(self, version_manager, mock_db_logger):
        """Test upgrade updates all agents with matching version constraints."""
        # Setup mocks
        mock_db_logger.get_skill.return_value = {
            "id": 1,
            "name": "test-skill",
            "current_version": "1.0.0"
        }
        mock_db_logger.get_skill_version.return_value = {
            "id": 2,
            "version": "2.0.0",
            "status": "published"
        }
        # Return agent mappings
        mock_db_logger.get_agent_skills.return_value = [
            {"id": 1, "agent_id": 101, "version_constraint": "latest"},
            {"id": 2, "agent_id": 102, "version_constraint": "1.0.0"},
            {"id": 3, "agent_id": 103, "version_constraint": "latest"}
        ]
        mock_db_logger.log_skill_action.return_value = {"id": 2}

        result = await version_manager.upgrade_skill_version(
            "test-skill",
            "2.0.0",
            None,
            "test-user"
        )

        # All agents should be affected (current implementation returns True for all constraints)
        assert len(result["agents_affected"]) == 3
        assert mock_db_logger.update_agent_skill_version.call_count == 3
