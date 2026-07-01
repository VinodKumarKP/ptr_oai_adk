"""
Advanced tests for Git Provider async operations with full mocking.
"""

import pytest
import asyncio
from pathlib import Path
import shutil
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp
from oai_skills_registry.services.git_provider import GitHubProvider


@pytest.mark.asyncio
class TestGitHubProviderAsync:
    """Advanced async tests for GitHubProvider with no real network/process calls."""

    @pytest.fixture
    def provider(self):
        return GitHubProvider()

    async def test_fetch_skill_files_success(self, provider):
        """Test fetching skill files successfully."""
        def side_effect(cmd, *args, **kwargs):
            # cmd[-1] is the target temp directory
            temp_dir = Path(cmd[-1])
            skill_dir = temp_dir / "skills" / "test-skill"
            skill_dir.mkdir(parents=True, exist_ok=True)
            with open(skill_dir / "SKILL.md", "w") as f:
                f.write("mock markdown content")
            with open(skill_dir / "skill_config.yaml", "w") as f:
                f.write("mock yaml config")
            res = MagicMock()
            res.returncode = 0
            return res

        with patch("subprocess.run", side_effect=side_effect):
            files = await provider.fetch_skill_files(
                repo="owner/repo",
                branch="main",
                tag=None,
                skill_name="test-skill",
                auth_token="token"
            )
            assert files["SKILL.md"] == "mock markdown content"
            assert files["skill_config.yaml"] == "mock yaml config"

    async def test_fetch_skill_files_clone_fail(self, provider):
        """Test fetch_skill_files handles clone failure."""
        res = MagicMock()
        res.returncode = 1
        res.stderr = b"Permission denied"

        with patch("subprocess.run", return_value=res):
            with pytest.raises(Exception, match="Git clone failed"):
                await provider.fetch_skill_files(
                    repo="owner/repo",
                    branch="main",
                    tag=None,
                    skill_name="test-skill",
                    auth_token=None
                )

    async def test_fetch_skill_files_missing_skill_dir(self, provider):
        """Test fetch_skill_files handles missing skill folder."""
        res = MagicMock()
        res.returncode = 0

        with patch("subprocess.run", return_value=res):
            with pytest.raises(Exception, match="Skill not found: test-skill"):
                await provider.fetch_skill_files(
                    repo="owner/repo",
                    branch="main",
                    tag=None,
                    skill_name="test-skill",
                    auth_token=None
                )

    async def test_fetch_skill_files_missing_skill_md(self, provider):
        """Test fetch_skill_files handles missing SKILL.md."""
        def side_effect(cmd, *args, **kwargs):
            temp_dir = Path(cmd[-1])
            skill_dir = temp_dir / "skills" / "test-skill"
            skill_dir.mkdir(parents=True, exist_ok=True)
            res = MagicMock()
            res.returncode = 0
            return res

        with patch("subprocess.run", side_effect=side_effect):
            with pytest.raises(Exception, match="SKILL.md not found"):
                await provider.fetch_skill_files(
                    repo="owner/repo",
                    branch="main",
                    tag=None,
                    skill_name="test-skill",
                    auth_token=None
                )

    async def test_get_commit_info_success(self, provider):
        """Test retrieving commit info successfully."""
        def side_effect(cmd, *args, **kwargs):
            res = MagicMock()
            res.returncode = 0
            if "rev-parse" in cmd:
                res.stdout = "sha123\n"
            elif "log" in cmd:
                res.stdout = "commit message\n"
            return res

        with patch("subprocess.run", side_effect=side_effect):
            info = await provider.get_commit_info(
                repo="owner/repo",
                tag="v1.0.0",
                auth_token="token"
            )
            assert info["commit_sha"] == "sha123"
            assert info["commit_message"] == "commit message"

    async def test_fetch_skill_md_from_api_success(self, provider):
        """Test fetching skill md via API successfully."""
        mock_resp_md = AsyncMock()
        mock_resp_md.status = 200
        mock_resp_md.text.return_value = "markdown content"

        mock_resp_cfg = AsyncMock()
        mock_resp_cfg.status = 200
        mock_resp_cfg.text.return_value = "config content"

        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=[mock_resp_md, mock_resp_cfg])
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_client_session):
            files = await provider.fetch_skill_md_from_api(
                repo="owner/repo",
                tag="v1.0.0",
                skill_name="test-skill",
                auth_token="token"
            )
            assert files["SKILL.md"] == "markdown content"
            assert files["skill_config.yaml"] == "config content"

    async def test_fetch_skill_md_from_api_not_found(self, provider):
        """Test fetch_skill_md_from_api handles 404 for SKILL.md."""
        mock_resp = AsyncMock()
        mock_resp.status = 404

        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_client_session):
            with pytest.raises(FileNotFoundError, match="SKILL.md not found"):
                await provider.fetch_skill_md_from_api(
                    repo="owner/repo",
                    tag="v1.0.0",
                    skill_name="test-skill",
                    auth_token=None
                )

    async def test_fetch_skill_md_from_api_error(self, provider):
        """Test fetch_skill_md_from_api handles other HTTP errors."""
        mock_resp = AsyncMock()
        mock_resp.status = 500

        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_client_session):
            with pytest.raises(Exception, match="GitHub API error 500"):
                await provider.fetch_skill_md_from_api(
                    repo="owner/repo",
                    tag="v1.0.0",
                    skill_name="test-skill",
                    auth_token=None
                )

    async def test_get_tags_success(self, provider):
        """Test getting and enriching tags successfully."""
        mock_tags_resp = AsyncMock()
        mock_tags_resp.status = 200
        mock_tags_resp.json.return_value = [
            {"name": "v1.0.0", "commit": {"sha": "sha1"}}
        ]

        mock_commit_resp = AsyncMock()
        mock_commit_resp.status = 200
        mock_commit_resp.json.return_value = {
            "commit": {
                "committer": {"date": "2026-07-01T00:00:00Z"},
                "message": "commit message\nline2",
                "author": {"name": "author-name"}
            }
        }

        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=[mock_tags_resp, mock_commit_resp])
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_client_session):
            tags = await provider.get_tags(repo="owner/repo", auth_token="token")
            assert len(tags) == 1
            assert tags[0]["name"] == "v1.0.0"
            assert tags[0]["created_at"] == "2026-07-01T00:00:00Z"
            assert tags[0]["message"] == "commit message"
            assert tags[0]["tagger"]["name"] == "author-name"

    async def test_get_tags_http_error(self, provider):
        """Test get_tags returns empty list on API failure."""
        mock_resp = AsyncMock()
        mock_resp.status = 400

        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_client_session):
            tags = await provider.get_tags(repo="owner/repo", auth_token=None)
            assert tags == []

    async def test_get_tags_enrich_fail(self, provider):
        """Test get_tags continues even if enrich request fails."""
        mock_tags_resp = AsyncMock()
        mock_tags_resp.status = 200
        mock_tags_resp.json.return_value = [
            {"name": "v1.0.0", "commit": {"sha": "sha1"}}
        ]

        mock_commit_resp = AsyncMock()
        mock_commit_resp.status = 500

        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=[mock_tags_resp, mock_commit_resp])
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_client_session):
            tags = await provider.get_tags(repo="owner/repo", auth_token=None)
            assert len(tags) == 1
            assert tags[0]["name"] == "v1.0.0"
            assert "created_at" not in tags[0]
