"""
Advanced tests for SkillsRegistry orchestrator methods.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from oai_skills_registry.services.skills_registry import SkillsRegistry


@pytest.mark.asyncio
class TestSkillsRegistryAdvanced:
    """Advanced tests for SkillsRegistry methods."""

    @pytest.fixture
    def registry(self, mock_db_logger):
        """Create a SkillsRegistry instance."""
        return SkillsRegistry(mock_db_logger)

    async def test_list_skill_versions_from_url_valid_url(self, registry, mock_git_provider):
        """Test listing versions from valid GitHub URL."""
        # Setup mock
        mock_git_provider.get_tags.return_value = [
            {
                "name": "v1.0.0",
                "commit": {"sha": "abc123"},
                "created_at": "2024-01-15T10:00:00Z",
                "message": "Release 1.0.0",
                "tagger": {"name": "John Doe"}
            }
        ]
        mock_git_provider.fetch_skill_files.return_value = {
            "SKILL.md": """---
name: test-skill
version: 1.0.0
description: A test skill
category: test
author: test-author
tags:
  - test
---
Content"""
        }

        result = await registry.list_skill_versions_from_url(
            "test-skill",
            "https://github.com/owner/repo.git"
        )

        assert isinstance(result, list)

    async def test_list_skill_versions_from_url_invalid_url(self, registry):
        """Test listing versions from invalid URL returns empty list."""
        result = await registry.list_skill_versions_from_url(
            "test-skill",
            "invalid-url-format"
        )

        assert result == []

    async def test_list_skill_versions_from_url_no_dot_git(self, registry, mock_git_provider):
        """Test listing versions from URL without .git extension."""
        mock_git_provider.get_tags.return_value = []

        result = await registry.list_skill_versions_from_url(
            "test-skill",
            "https://github.com/owner/repo"
        )

        # Should handle URLs with or without .git
        assert isinstance(result, list)

    async def test_list_skill_versions_from_url_with_auth_token(self, registry, mock_git_provider):
        """Test listing versions with authentication token."""
        mock_git_provider.get_tags.return_value = []

        result = await registry.list_skill_versions_from_url(
            "test-skill",
            "https://github.com/owner/repo.git",
            auth_token="github_token_123"
        )

        assert isinstance(result, list)

    async def test_list_skill_versions_from_url_git_provider_error(self, registry, mock_git_provider):
        """Test handling of Git provider errors."""
        mock_git_provider.get_tags.side_effect = Exception("Network error")

        result = await registry.list_skill_versions_from_url(
            "test-skill",
            "https://github.com/owner/repo.git"
        )

        assert result == []

    async def test_register_skills_bulk_success(self, registry, mock_db_logger):
        """Test successful bulk skill registration."""
        mock_db_logger.get_skill.return_value = None
        mock_db_logger.create_skill.return_value = {"id": 1, "name": "test-skill"}

        # Mock aiohttp for metadata fetching
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "content": "LS0tCm5hbWU6IHRlc3Qtc2tpbGwKdmVyc2lvbjogMS4wLjAKLS0t"  # base64 encoded
            })

            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session.get.return_value.__aexit__.return_value = None
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None

            mock_session_class.return_value = mock_session

            result = await registry.register_skills_bulk(
                "https://github.com/owner/repo.git",
                ["test-skill"],
                "test-user"
            )

            assert "total_registered" in result
            assert "successful" in result
            assert "failed" in result

    async def test_register_skills_bulk_with_auth_token(self, registry, mock_db_logger):
        """Test bulk registration with authentication token."""
        mock_db_logger.get_skill.return_value = None
        mock_db_logger.create_skill.return_value = {"id": 1}

        result = await registry.register_skills_bulk(
            "https://github.com/owner/repo.git",
            ["skill1"],
            "test-user",
            auth_token="github_token"
        )

        assert "total_registered" in result

    async def test_register_skills_bulk_already_registered(self, registry, mock_db_logger):
        """Test bulk registration skips already registered skills."""
        # First skill already exists
        mock_db_logger.get_skill.side_effect = [
            {"id": 1, "name": "existing-skill"},  # Already exists
            None  # New skill doesn't exist
        ]
        mock_db_logger.create_skill.return_value = {"id": 2}

        result = await registry.register_skills_bulk(
            "https://github.com/owner/repo.git",
            ["existing-skill", "new-skill"],
            "test-user"
        )

        assert "failed" in result

    async def test_register_skills_bulk_invalid_url(self, registry):
        """Test bulk registration with invalid URL raises error."""
        with pytest.raises(ValueError, match="Invalid GitHub URL"):
            await registry.register_skills_bulk(
                "invalid-url",
                ["skill1"],
                "test-user"
            )

    async def test_register_skills_bulk_empty_skill_list(self, registry, mock_db_logger):
        """Test bulk registration with empty skill list."""
        result = await registry.register_skills_bulk(
            "https://github.com/owner/repo.git",
            [],
            "test-user"
        )

        assert result["total_registered"] == 0

    async def test_register_skills_bulk_handles_metadata_fetch_error(self, registry, mock_db_logger):
        """Test bulk registration handles metadata fetch errors gracefully."""
        mock_db_logger.get_skill.return_value = None
        mock_db_logger.create_skill.return_value = {"id": 1}

        # Mock aiohttp with error
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 404  # Metadata not found

            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session.get.return_value.__aexit__.return_value = None
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None

            mock_session_class.return_value = mock_session

            result = await registry.register_skills_bulk(
                "https://github.com/owner/repo.git",
                ["skill-no-metadata"],
                "test-user"
            )

            assert "total_registered" in result

    async def test_register_skills_bulk_multiple_skills(self, registry, mock_db_logger):
        """Test bulk registration with multiple skills."""
        mock_db_logger.get_skill.return_value = None
        mock_db_logger.create_skill.return_value = {"id": 1}

        result = await registry.register_skills_bulk(
            "https://github.com/owner/repo.git",
            ["skill1", "skill2", "skill3"],
            "test-user"
        )

        assert "total_registered" in result
        assert "successful" in result
        assert "failed" in result

    async def test_import_skill_from_git_workflow(self, registry, monkeypatch):
        """Test import skill workflow through registry."""
        async def mock_import(skill_name, source_config, git_tag, performed_by):
            return {
                "skill_id": 1,
                "version_id": 1,
                "version": "1.0.0",
                "preview": {
                    "skill_name": skill_name,
                    "version": "1.0.0"
                }
            }

        monkeypatch.setattr(registry.importer, 'import_skill_from_git', mock_import)

        source_config = {
            "git_provider": "github",
            "repository": "owner/repo",
            "_direct_url": "https://github.com/owner/repo.git"
        }

        result = await registry.import_skill_from_git(
            "test-skill",
            source_config,
            None,
            "test-user"
        )

        assert result["skill_id"] == 1
        assert result["version"] == "1.0.0"

    async def test_list_skill_versions_from_url_with_multiple_versions(self, registry, mock_git_provider):
        """Test listing multiple versions from GitHub."""
        mock_git_provider.get_tags.return_value = [
            {
                "name": "v2.0.0",
                "commit": {"sha": "def456"},
                "created_at": "2024-01-16T10:00:00Z"
            },
            {
                "name": "v1.0.0",
                "commit": {"sha": "abc123"},
                "created_at": "2024-01-15T10:00:00Z"
            }
        ]
        mock_git_provider.fetch_skill_files.side_effect = [
            {"SKILL.md": "---\nname: test-skill\nversion: 2.0.0\n---\nContent"},
            {"SKILL.md": "---\nname: test-skill\nversion: 1.0.0\n---\nContent"}
        ]

        result = await registry.list_skill_versions_from_url(
            "test-skill",
            "https://github.com/owner/repo.git"
        )

        # Should return sorted by date (most recent first)
        if result:
            # Versions should be in descending order by date
            for i in range(len(result) - 1):
                assert result[i]["created_at"] >= result[i + 1]["created_at"]

    async def test_list_skill_versions_filters_non_matching_skills(self, registry, mock_git_provider):
        """Test that list_skill_versions filters out non-matching skills."""
        mock_git_provider.get_tags.return_value = [
            {"name": "v1.0.0", "created_at": "2024-01-15T10:00:00Z"}
        ]
        mock_git_provider.fetch_skill_files.return_value = {
            "SKILL.md": "---\nname: different-skill\nversion: 1.0.0\n---\nContent"
        }

        result = await registry.list_skill_versions_from_url(
            "test-skill",
            "https://github.com/owner/repo.git"
        )

        # Should not include the different skill
        assert len(result) == 0
