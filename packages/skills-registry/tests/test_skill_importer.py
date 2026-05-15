"""
Tests for SkillImporter service.
"""

import pytest
from oai_skills_registry.services.skill_importer import SkillImporter


@pytest.mark.asyncio
class TestSkillImporter:
    """Tests for SkillImporter class."""

    @pytest.fixture
    def skill_importer(self, mock_db_logger, mock_git_provider):
        """Create a SkillImporter instance with mocks."""
        git_providers = {"github": mock_git_provider}
        return SkillImporter(mock_db_logger, git_providers)

    async def test_register_git_source_success(self, skill_importer, mock_db_logger):
        """Test registering a new Git source."""
        source_config = {
            "name": "test-source",
            "git_provider": "github",
            "repository": "owner/repo"
        }
        mock_db_logger.register_git_source.return_value = 1

        result = await skill_importer.register_git_source(source_config)

        assert result["source_id"] == 1
        assert result["name"] == "test-source"
        assert result["status"] == "registered"
        mock_db_logger.register_git_source.assert_called_once_with(source_config)

    async def test_import_skill_from_registered_source(self, skill_importer, mock_db_logger, mock_git_provider, sample_skill_md):
        """Test importing skill from a registered Git source."""
        source_config = {
            "id": 1,
            "git_provider": "github",
            "repository": "owner/repo",
            "branch": "main",
            "git_url": "https://github.com/owner/repo.git"
        }
        mock_db_logger.get_skill.return_value = None
        mock_db_logger.create_skill.return_value = {"id": 1, "name": "test-skill"}
        mock_db_logger.create_skill_version.return_value = {"id": 1, "version": "1.0.0"}
        mock_git_provider.fetch_skill_files.return_value = {
            "SKILL.md": sample_skill_md
        }
        mock_git_provider.get_commit_info.return_value = {
            "commit_sha": "abc123",
            "commit_message": "Initial commit"
        }

        result = await skill_importer.import_skill_from_git(
            "test-skill",
            source_config,
            None,
            "test-user"
        )

        assert result["skill_id"] == 1
        assert result["version"] == "1.0.0"
        assert "preview" in result
        mock_db_logger.create_skill.assert_called_once()
        mock_db_logger.create_skill_version.assert_called_once()

    async def test_import_skill_from_direct_url(self, skill_importer, mock_db_logger, mock_git_provider, sample_skill_md):
        """Test importing skill directly from GitHub URL."""
        source_config = {
            "git_provider": "github",
            "_direct_url": "https://github.com/owner/repo.git",
            "branch": "main"
        }
        mock_db_logger.get_skill.return_value = None
        mock_db_logger.create_skill.return_value = {"id": 1, "name": "test-skill"}
        mock_db_logger.create_skill_version.return_value = {"id": 1, "version": "1.0.0"}
        mock_git_provider.fetch_skill_files.return_value = {
            "SKILL.md": sample_skill_md
        }
        mock_git_provider.get_commit_info.return_value = {
            "commit_sha": "abc123",
            "commit_message": "Initial commit"
        }

        result = await skill_importer.import_skill_from_git(
            "test-skill",
            source_config,
            None,
            "test-user"
        )

        assert result["skill_id"] == 1
        assert result["version"] == "1.0.0"

    async def test_import_skill_with_git_tag(self, skill_importer, mock_db_logger, mock_git_provider, sample_skill_md):
        """Test importing skill from specific Git tag."""
        source_config = {
            "git_provider": "github",
            "repository": "owner/repo",
            "branch": "main",
            "git_url": "https://github.com/owner/repo.git"
        }
        mock_db_logger.get_skill.return_value = None
        mock_db_logger.create_skill.return_value = {"id": 1, "name": "test-skill"}
        mock_db_logger.create_skill_version.return_value = {"id": 1, "version": "1.0.0"}
        mock_git_provider.fetch_skill_files.return_value = {
            "SKILL.md": sample_skill_md
        }
        mock_git_provider.get_commit_info.return_value = {
            "commit_sha": "tag123",
            "commit_message": "Release v1.0.0"
        }

        result = await skill_importer.import_skill_from_git(
            "test-skill",
            source_config,
            "v1.0.0",
            "test-user"
        )

        # Verify git tag was passed
        mock_git_provider.fetch_skill_files.assert_called_once()
        call_args = mock_git_provider.fetch_skill_files.call_args
        assert call_args.kwargs["tag"] == "v1.0.0"

    async def test_import_skill_creates_new_skill_if_not_exists(self, skill_importer, mock_db_logger, mock_git_provider, sample_skill_md):
        """Test import creates skill record if it doesn't exist."""
        source_config = {
            "git_provider": "github",
            "repository": "owner/repo",
            "branch": "main",
            "git_url": "https://github.com/owner/repo.git"
        }
        mock_db_logger.get_skill.return_value = None
        mock_db_logger.create_skill.return_value = {"id": 1, "name": "test-skill"}
        mock_db_logger.create_skill_version.return_value = {"id": 1, "version": "1.0.0"}
        mock_git_provider.fetch_skill_files.return_value = {
            "SKILL.md": sample_skill_md
        }
        mock_git_provider.get_commit_info.return_value = {
            "commit_sha": "abc123",
            "commit_message": "Initial commit"
        }

        await skill_importer.import_skill_from_git(
            "test-skill",
            source_config,
            None,
            "test-user"
        )

        # Verify skill was created
        mock_db_logger.create_skill.assert_called_once()
        call_args = mock_db_logger.create_skill.call_args
        assert call_args.kwargs["name"] == "test-skill"

    async def test_import_skill_uses_existing_skill(self, skill_importer, mock_db_logger, mock_git_provider, sample_skill_md):
        """Test import uses existing skill record."""
        source_config = {
            "git_provider": "github",
            "repository": "owner/repo",
            "branch": "main",
            "git_url": "https://github.com/owner/repo.git"
        }
        existing_skill = {"id": 1, "name": "test-skill"}
        mock_db_logger.get_skill.return_value = existing_skill
        mock_db_logger.create_skill_version.return_value = {"id": 2, "version": "2.0.0"}
        mock_git_provider.fetch_skill_files.return_value = {
            "SKILL.md": sample_skill_md
        }
        mock_git_provider.get_commit_info.return_value = {
            "commit_sha": "abc123",
            "commit_message": "New version"
        }

        await skill_importer.import_skill_from_git(
            "test-skill",
            source_config,
            None,
            "test-user"
        )

        # Verify skill was not created
        mock_db_logger.create_skill.assert_not_called()

    async def test_import_skill_name_mismatch_raises_error(self, skill_importer, mock_db_logger, mock_git_provider):
        """Test import fails if skill name doesn't match."""
        source_config = {
            "git_provider": "github",
            "repository": "owner/repo"
        }
        mock_git_provider.fetch_skill_files.return_value = {
            "SKILL.md": "---\nname: different-skill\nversion: 1.0.0\n---\nContent"
        }

        with pytest.raises(ValueError, match="Skill name mismatch"):
            await skill_importer.import_skill_from_git(
                "test-skill",
                source_config,
                None,
                "test-user"
            )

    async def test_import_skill_unknown_provider_raises_error(self, skill_importer, mock_db_logger):
        """Test import fails with unknown provider."""
        source_config = {
            "git_provider": "unknown-provider",
            "repository": "owner/repo"
        }

        with pytest.raises(ValueError, match="Unknown Git provider"):
            await skill_importer.import_skill_from_git(
                "test-skill",
                source_config,
                None,
                "test-user"
            )

    async def test_import_skill_git_fetch_error_raises_error(self, skill_importer, mock_db_logger, mock_git_provider):
        """Test import fails when Git fetch fails."""
        source_config = {
            "git_provider": "github",
            "repository": "owner/repo"
        }
        mock_git_provider.fetch_skill_files.side_effect = Exception("Git fetch failed")

        with pytest.raises(Exception, match="Failed to fetch skill from Git"):
            await skill_importer.import_skill_from_git(
                "test-skill",
                source_config,
                None,
                "test-user"
            )

    async def test_import_skill_invalid_git_url_raises_error(self, skill_importer, mock_db_logger):
        """Test import fails with invalid direct GitHub URL."""
        source_config = {
            "git_provider": "github",
            "_direct_url": "invalid-url-format"
        }

        with pytest.raises(ValueError, match="Invalid GitHub URL format"):
            await skill_importer.import_skill_from_git(
                "test-skill",
                source_config,
                None,
                "test-user"
            )

    async def test_import_skill_parses_config_yaml(self, skill_importer, mock_db_logger, mock_git_provider, sample_skill_md):
        """Test import parses skill_config.yaml if present."""
        source_config = {
            "git_provider": "github",
            "repository": "owner/repo",
            "git_url": "https://github.com/owner/repo.git"
        }
        mock_db_logger.get_skill.return_value = None
        mock_db_logger.create_skill.return_value = {"id": 1, "name": "test-skill"}
        mock_db_logger.create_skill_version.return_value = {"id": 1, "version": "1.0.0"}
        mock_git_provider.fetch_skill_files.return_value = {
            "SKILL.md": sample_skill_md,
            "skill_config.yaml": "timeout: 300\nretries: 3"
        }
        mock_git_provider.get_commit_info.return_value = {
            "commit_sha": "abc123",
            "commit_message": "Initial commit"
        }

        await skill_importer.import_skill_from_git(
            "test-skill",
            source_config,
            None,
            "test-user"
        )

        # Verify version was created with config
        mock_db_logger.create_skill_version.assert_called_once()
        call_args = mock_db_logger.create_skill_version.call_args
        assert call_args.kwargs["config"] is not None
