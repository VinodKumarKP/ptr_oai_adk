"""
Advanced tests for SkillDiscovery service with HTTP operations.
"""

import pytest
import base64
from unittest.mock import AsyncMock, patch, MagicMock
from oai_skills_registry.services.skill_discovery import SkillDiscovery


@pytest.mark.asyncio
class TestSkillDiscoveryAdvanced:
    """Advanced tests for SkillDiscovery with mocked HTTP and DB."""

    @pytest.fixture
    def mock_db_logger(self):
        return MagicMock()

    @pytest.fixture
    def skill_discovery(self, mock_db_logger):
        """Create a SkillDiscovery instance."""
        return SkillDiscovery(mock_db_logger)

    async def test_discover_skills_success(self, skill_discovery, mock_db_logger):
        """Test successful skill discovery from repository."""
        # Mock database response: skill1 is not registered, skill2 is
        async def mock_get_skill(name):
            return {"id": 1} if name == "skill2" else None
        mock_db_logger.get_skill = AsyncMock(side_effect=mock_get_skill)

        # Contents of skills/ directory
        mock_resp_contents = AsyncMock()
        mock_resp_contents.status = 200
        mock_resp_contents.json.return_value = [
            {"name": "skill1", "type": "dir"},
            {"name": "skill2", "type": "dir"},
            {"name": "README.md", "type": "file"}
        ]

        # SKILL.md contents
        skill1_md = base64.b64encode(b"---\nname: skill1\nversion: 1.0.0\ndescription: skill1 desc\ncategory: utility\nauthor: tester\n---\n").decode()
        skill2_md = base64.b64encode(b"---\nname: skill2\nversion: 2.0.0\ndescription: skill2 desc\ncategory: utility\nauthor: tester\n---\n").decode()

        mock_resp_md1 = AsyncMock()
        mock_resp_md1.status = 200
        mock_resp_md1.json.return_value = {"content": skill1_md}

        mock_resp_md2 = AsyncMock()
        mock_resp_md2.status = 200
        mock_resp_md2.json.return_value = {"content": skill2_md}

        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=[mock_resp_contents, mock_resp_md1, mock_resp_md2])
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_client_session):
            result = await skill_discovery.discover_skills_from_git(
                "https://github.com/owner/repo.git",
                auth_token="token"
            )
            assert result["total_found"] == 2
            assert result["available_to_register"] == 1
            assert result["already_registered"] == 1
            assert result["skills"][0]["name"] == "skill1"
            assert result["skills"][0]["status"] == "available"
            assert result["skills"][1]["name"] == "skill2"
            assert result["skills"][1]["status"] == "already_registered"

    async def test_discover_skills_invalid_url(self, skill_discovery):
        """Test discovery with invalid URL raises error."""
        with pytest.raises(ValueError, match="Failed to parse GitHub URL"):
            await skill_discovery.discover_skills_from_git("invalid-url")

    async def test_discover_skills_api_error(self, skill_discovery):
        """Test discovery handles API errors gracefully."""
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
            with pytest.raises(ValueError, match="Failed to access skills directory"):
                await skill_discovery.discover_skills_from_git("https://github.com/owner/repo.git")

    async def test_discover_skills_empty_directory(self, skill_discovery, mock_db_logger):
        """Test discovery with empty skills directory."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = []

        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_client_session):
            result = await skill_discovery.discover_skills_from_git("https://github.com/owner/repo")
            assert result["total_found"] == 0

    async def test_discover_skills_single_item_dict(self, skill_discovery, mock_db_logger):
        """Test discovery handles single file response (dict) from API."""
        mock_db_logger.get_skill = AsyncMock(return_value=None)

        mock_resp_contents = AsyncMock()
        mock_resp_contents.status = 200
        mock_resp_contents.json.return_value = {"name": "skill1", "type": "dir"}

        mock_resp_md = AsyncMock()
        mock_resp_md.status = 200
        mock_resp_md.json.return_value = {"content": base64.b64encode(b"---\nname: skill1\n---\n").decode()}

        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=[mock_resp_contents, mock_resp_md])
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_client_session):
            result = await skill_discovery.discover_skills_from_git("https://github.com/owner/repo")
            assert result["total_found"] == 1
            assert result["skills"][0]["name"] == "skill1"

    async def test_fetch_skill_metadata_error_handling(self, skill_discovery, mock_db_logger):
        """Test that fetching individual skill metadata failure is handled (returns invalid status)."""
        mock_resp_contents = AsyncMock()
        mock_resp_contents.status = 200
        mock_resp_contents.json.return_value = [{"name": "skill1", "type": "dir"}]

        # Metadata fetch returns 500 error
        mock_resp_md = AsyncMock()
        mock_resp_md.status = 500

        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=[mock_resp_contents, mock_resp_md])
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.return_value = mock_cm

        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_client_session):
            result = await skill_discovery.discover_skills_from_git("https://github.com/owner/repo")
            assert result["total_found"] == 1
            assert result["skills"][0]["name"] == "skill1"
            assert result["skills"][0]["status"] == "invalid"
