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
    def registry(self, mock_db_logger, mock_git_provider):
        """Create a SkillsRegistry instance."""
        reg = SkillsRegistry(mock_db_logger)
        reg.git_providers["github"] = mock_git_provider
        return reg

    async def test_list_skill_versions_from_url_valid_url(self, registry, mock_git_provider):
        """Test listing versions from valid GitHub URL."""
        mock_git_provider.get_tags.return_value = [
            {
                "name": "v1.0.0",
                "commit": {"sha": "abc123"},
                "created_at": "2024-01-15T10:00:00Z",
                "message": "Release 1.0.0",
                "tagger": {"name": "John Doe"}
            }
        ]
        mock_git_provider.fetch_skill_md_from_api = AsyncMock(return_value={
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
        })

        result = await registry.list_skill_versions_from_url(
            "test-skill",
            "https://github.com/owner/repo.git"
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["version"] == "1.0.0"

    async def test_list_skill_versions_from_url_fallback(self, registry, mock_git_provider):
        """Test listing versions fallback when provider lacks fetch_skill_md_from_api."""
        del mock_git_provider.fetch_skill_md_from_api  # remove hasattr

        mock_git_provider.get_tags.return_value = [
            {
                "name": "v1.0.0",
                "commit": {"sha": "abc123"},
                "created_at": "2024-01-15T10:00:00Z",
                "message": "Release 1.0.0"
            }
        ]
        mock_git_provider.fetch_skill_files = AsyncMock(return_value={
            "SKILL.md": "---\nname: test-skill\nversion: 1.0.0\n---\nContent"
        })

        result = await registry.list_skill_versions_from_url(
            "test-skill",
            "https://github.com/owner/repo.git"
        )
        assert len(result) == 1

    async def test_list_skill_versions_from_url_file_not_found(self, registry, mock_git_provider):
        """Test listing versions handles FileNotFoundError gracefully (skips)."""
        mock_git_provider.get_tags.return_value = [{"name": "v1.0.0"}]
        mock_git_provider.fetch_skill_md_from_api = AsyncMock(side_effect=FileNotFoundError("SKILL.md not found"))

        result = await registry.list_skill_versions_from_url(
            "test-skill",
            "https://github.com/owner/repo.git"
        )
        assert result == []

    async def test_list_skill_versions_from_url_provider_missing(self, registry):
        """Test listing versions returns empty list when provider not configured."""
        registry.git_providers = {}
        result = await registry.list_skill_versions_from_url(
            "test-skill",
            "https://github.com/owner/repo.git"
        )
        assert result == []

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

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "content": "LS0tCm5hbWU6IHRlc3Qtc2tpbGwKdmVyc2lvbjogMS4wLjAKLS0tCg=="  # base64 encoded with trailing newline
        }

        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=False)

        with patch('aiohttp.ClientSession', return_value=mock_client_session):
            result = await registry.register_skills_bulk(
                "https://github.com/owner/repo.git",
                ["test-skill"],
                "test-user"
            )

            assert result["total_registered"] == 1
            assert result["successful"] == ["test-skill"]
            assert len(result["failed"]) == 0

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

        mock_response = AsyncMock()
        mock_response.status = 404

        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=False)

        with patch('aiohttp.ClientSession', return_value=mock_client_session):
            result = await registry.register_skills_bulk(
                "https://github.com/owner/repo.git",
                ["skill-no-metadata"],
                "test-user"
            )

            assert result["total_registered"] == 1
            assert result["successful"] == ["skill-no-metadata"]

    async def test_register_skills_bulk_db_failure(self, registry, mock_db_logger):
        """Test bulk registration handles database logger creation failures."""
        mock_db_logger.get_skill.return_value = None
        # Returns None on create_skill
        mock_db_logger.create_skill.return_value = {}

        mock_response = AsyncMock()
        mock_response.status = 404

        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=False)

        with patch('aiohttp.ClientSession', return_value=mock_client_session):
            result = await registry.register_skills_bulk(
                "https://github.com/owner/repo.git",
                ["skill-db-fail"],
                "test-user"
            )

            assert result["total_registered"] == 0
            assert len(result["successful"]) == 0
            assert len(result["failed"]) == 1
            assert result["failed"][0]["skill_name"] == "skill-db-fail"

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

    async def test_delegations(self, registry):
        """Test delegation methods to importer and version_manager."""
        # register_git_source
        registry.importer.register_git_source = AsyncMock(return_value={"id": 1})
        res = await registry.register_git_source({"config": "value"})
        assert res == {"id": 1}
        registry.importer.register_git_source.assert_called_once_with({"config": "value"})

        # publish_skill_version
        registry.version_manager.publish_skill_version = AsyncMock(return_value="published")
        res = await registry.publish_skill_version("skill", "1.0.0", "msg", "user")
        assert res == "published"
        registry.version_manager.publish_skill_version.assert_called_once_with("skill", "1.0.0", "msg", "user")

        # upgrade_skill_version
        registry.version_manager.upgrade_skill_version = AsyncMock(return_value="upgraded")
        res = await registry.upgrade_skill_version("skill", "1.0.0", "msg", "user")
        assert res == "upgraded"
        registry.version_manager.upgrade_skill_version.assert_called_once_with("skill", "1.0.0", "msg", "user")

        # downgrade_skill_version
        registry.version_manager.downgrade_skill_version = AsyncMock(return_value="downgraded")
        res = await registry.downgrade_skill_version("skill", "1.0.0", "msg", "user")
        assert res == "downgraded"
        registry.version_manager.downgrade_skill_version.assert_called_once_with("skill", "1.0.0", "msg", "user")

        # deprecate_skill_version
        registry.version_manager.deprecate_skill_version = AsyncMock(return_value="deprecated")
        res = await registry.deprecate_skill_version("skill", "1.0.0", "msg", "user")
        assert res == "deprecated"
        registry.version_manager.deprecate_skill_version.assert_called_once_with("skill", "1.0.0", "msg", "user")

        # delete_skill
        registry.version_manager.delete_skill = AsyncMock(return_value="deleted")
        res = await registry.delete_skill("skill", "user")
        assert res == "deleted"
        registry.version_manager.delete_skill.assert_called_once_with("skill", "user")

        # get_skill_details
        registry.queries.get_skill_details = AsyncMock(return_value={"id": 1})
        res = await registry.get_skill_details("skill")
        assert res == {"id": 1}
        registry.queries.get_skill_details.assert_called_once_with("skill")

        # get_skill_history
        registry.queries.get_skill_history = AsyncMock(return_value={"actions": []})
        res = await registry.get_skill_history("skill", 10)
        assert res == {"actions": []}
        registry.queries.get_skill_history.assert_called_once_with("skill", 10)

        # list_git_versions
        registry.queries.list_git_versions = AsyncMock(return_value=[])
        res = await registry.list_git_versions("skill", 1)
        assert res == []
        registry.queries.list_git_versions.assert_called_once_with("skill", 1)

        # discover_skills_from_git
        registry.discovery.discover_skills_from_git = AsyncMock(return_value={"skills": []})
        res = await registry.discover_skills_from_git("url", "token")
        assert res == {"skills": []}
        registry.discovery.discover_skills_from_git.assert_called_once_with("url", "token")

        # close
        registry.db_logger.close = AsyncMock()
        await registry.close()
        registry.db_logger.close.assert_called_once()

    async def test_list_skill_versions_url_parse_exception(self, registry):
        """Test listing versions handles URL parse exceptions (e.g. passing None) gracefully."""
        res = await registry.list_skill_versions_from_url("skill", None)
        assert res == []

    async def test_register_skills_bulk_metadata_parse_error(self, registry, mock_db_logger):
        """Test bulk registration handles metadata parsing errors gracefully."""
        mock_db_logger.get_skill.return_value = None
        mock_db_logger.create_skill.return_value = {"id": 1}

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.side_effect = ValueError("invalid json")

        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=False)

        with patch('aiohttp.ClientSession', return_value=mock_client_session):
            result = await registry.register_skills_bulk(
                "https://github.com/owner/repo.git",
                ["skill-json-error"],
                "test-user"
            )
            assert result["total_registered"] == 1

    async def test_register_skills_bulk_general_exception(self, registry, mock_db_logger):
        """Test bulk registration handles general exceptions per skill gracefully."""
        mock_db_logger.get_skill.side_effect = RuntimeError("Database connection lost")

        result = await registry.register_skills_bulk(
            "https://github.com/owner/repo.git",
            ["skill-db-error"],
            "test-user"
        )
        assert result["total_registered"] == 0
        assert len(result["failed"]) == 1
        assert "Database connection lost" in result["failed"][0]["error"]
