"""
Tests for SkillQueries service.
"""

import pytest
from oai_skills_registry.services.skill_queries import SkillQueries


@pytest.mark.asyncio
class TestSkillQueries:
    """Tests for SkillQueries class."""

    @pytest.fixture
    def skill_queries(self, mock_db_logger, mock_git_provider):
        """Create a SkillQueries instance with mocks."""
        git_providers = {"github": mock_git_provider}
        return SkillQueries(mock_db_logger, git_providers)

    async def test_get_skill_details_success(self, skill_queries, mock_db_logger):
        """Test getting full skill details with versions."""
        # Setup mocks
        skill_record = {
            "id": 1,
            "name": "test-skill",
            "description": "Test skill",
            "version": "1.0.0"
        }
        versions = [
            {"id": 1, "version": "1.0.0", "status": "published"},
            {"id": 2, "version": "0.9.0", "status": "published"}
        ]
        mock_db_logger.get_skill.return_value = skill_record
        mock_db_logger.get_skill_versions.return_value = versions

        result = await skill_queries.get_skill_details("test-skill")

        assert result is not None
        assert result["skill"]["name"] == "test-skill"
        assert result["version_count"] == 2
        assert len(result["versions"]) == 2

    async def test_get_skill_details_nonexistent_skill(self, skill_queries, mock_db_logger):
        """Test getting details of non-existent skill returns None."""
        mock_db_logger.get_skill.return_value = None

        result = await skill_queries.get_skill_details("nonexistent")

        assert result is None

    async def test_get_skill_details_no_versions(self, skill_queries, mock_db_logger):
        """Test getting details of skill with no versions."""
        mock_db_logger.get_skill.return_value = {"id": 1, "name": "test-skill"}
        mock_db_logger.get_skill_versions.return_value = []

        result = await skill_queries.get_skill_details("test-skill")

        assert result is not None
        assert result["version_count"] == 0
        assert result["versions"] == []

    async def test_get_skill_history_success(self, skill_queries, mock_db_logger):
        """Test getting skill action history."""
        # Setup mocks
        skill_record = {"id": 1, "name": "test-skill"}
        actions = [
            {"id": 1, "action": "publish", "performed_by": "user1", "timestamp": "2024-01-01"},
            {"id": 2, "action": "upgrade", "performed_by": "user2", "timestamp": "2024-01-02"}
        ]
        mock_db_logger.get_skill.return_value = skill_record
        mock_db_logger.get_skill_actions.return_value = actions
        mock_db_logger.get_skill_action_count.return_value = 2

        result = await skill_queries.get_skill_history("test-skill", limit=100)

        assert result is not None
        assert result["skill_name"] == "test-skill"
        assert result["total_count"] == 2
        assert len(result["actions"]) == 2
        mock_db_logger.get_skill_actions.assert_called_once_with(1, 100)

    async def test_get_skill_history_nonexistent_skill(self, skill_queries, mock_db_logger):
        """Test getting history of non-existent skill returns None."""
        mock_db_logger.get_skill.return_value = None

        result = await skill_queries.get_skill_history("nonexistent")

        assert result is None

    async def test_get_skill_history_with_custom_limit(self, skill_queries, mock_db_logger):
        """Test getting skill history with custom limit."""
        mock_db_logger.get_skill.return_value = {"id": 1, "name": "test-skill"}
        mock_db_logger.get_skill_actions.return_value = []
        mock_db_logger.get_skill_action_count.return_value = 0

        await skill_queries.get_skill_history("test-skill", limit=50)

        mock_db_logger.get_skill_actions.assert_called_once_with(1, 50)

    async def test_list_git_versions_success(self, skill_queries, mock_db_logger, mock_git_provider):
        """Test listing available versions from Git."""
        # Setup mocks
        git_source = {
            "id": 1,
            "git_provider": "github",
            "repository": "owner/repo",
            "auth_token": "token"
        }
        mock_db_logger.get_git_source.return_value = git_source
        mock_git_provider.get_tags.return_value = [
            {
                "name": "v1.0.0",
                "commit": {"sha": "abc123"},
                "created_at": "2024-01-01"
            },
            {
                "name": "v0.9.0",
                "commit": {"sha": "def456"},
                "created_at": "2023-12-01"
            }
        ]
        mock_git_provider.fetch_skill_files.return_value = {
            "SKILL.md": "---\nname: test-skill\nversion: 1.0.0\n---\nContent"
        }

        result = await skill_queries.list_git_versions("test-skill", source_id=1)

        assert result is not None
        assert isinstance(result, list)

    async def test_list_git_versions_nonexistent_source(self, skill_queries, mock_db_logger):
        """Test listing versions from non-existent source raises error."""
        mock_db_logger.get_git_source.return_value = None

        with pytest.raises(ValueError, match="Git source not found"):
            await skill_queries.list_git_versions("test-skill", source_id=999)

    async def test_list_git_versions_unknown_provider(self, skill_queries, mock_db_logger):
        """Test listing versions with unknown provider raises error."""
        git_source = {
            "id": 1,
            "git_provider": "unknown-provider",
            "repository": "owner/repo"
        }
        mock_db_logger.get_git_source.return_value = git_source

        with pytest.raises(ValueError, match="Unknown Git provider"):
            await skill_queries.list_git_versions("test-skill", source_id=1)

    async def test_list_git_versions_handles_git_errors(self, skill_queries, mock_db_logger, mock_git_provider):
        """Test listing versions handles Git errors gracefully."""
        git_source = {
            "id": 1,
            "git_provider": "github",
            "repository": "owner/repo"
        }
        mock_db_logger.get_git_source.return_value = git_source
        mock_git_provider.get_tags.side_effect = Exception("Git error")

        result = await skill_queries.list_git_versions("test-skill", source_id=1)

        assert result == []

    async def test_list_git_versions_filters_matching_skills(self, skill_queries, mock_db_logger, mock_git_provider):
        """Test listing versions only includes matching skill names."""
        git_source = {
            "id": 1,
            "git_provider": "github",
            "repository": "owner/repo"
        }
        mock_db_logger.get_git_source.return_value = git_source
        mock_git_provider.get_tags.return_value = [
            {"name": "v1.0.0", "created_at": "2024-01-01"}
        ]

        # First call returns matching skill
        # Second call returns non-matching skill
        mock_git_provider.fetch_skill_files.side_effect = [
            {"SKILL.md": "---\nname: test-skill\nversion: 1.0.0\n---\nContent"},
            {"SKILL.md": "---\nname: other-skill\nversion: 2.0.0\n---\nContent"}
        ]

        result = await skill_queries.list_git_versions("test-skill", source_id=1)

        # Only the matching skill should be included
        assert all(v["version"] == "1.0.0" for v in result)
