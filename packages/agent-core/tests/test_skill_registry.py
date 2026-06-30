import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
import httpx

from oai_agent_core.components.skills.skill_registry import SkillRegistry
from oai_agent_core.components.skills.models import SkillProperties

@pytest.fixture
def mock_logger():
    return MagicMock()

def test_init_local(mock_logger):
    registry = SkillRegistry(logger=mock_logger, project_root="/root")
    assert registry.registry_url is None
    assert registry.skills == {}
    assert registry.skills_cache_dir is None

def test_init_hybrid(mock_logger, tmp_path):
    git_prov = MagicMock()
    registry = SkillRegistry(
        logger=mock_logger,
        project_root=str(tmp_path),
        registry_url="http://registry.com/",
        auth_token="token123",
        git_provider=git_prov,
        skills_cache_dir="./test_cache"
    )
    assert registry.registry_url == "http://registry.com"
    assert registry.auth_token == "token123"
    assert registry.git_provider is git_prov
    assert registry.skills_cache_dir == tmp_path / "test_cache"
    assert registry.skills_cache_dir.exists()

def test_get_registry_status(mock_logger):
    registry = SkillRegistry(logger=mock_logger, registry_url="http://registry.com")
    status = registry.get_registry_status()
    assert status["mode"] == "hybrid"
    assert status["registry_url"] == "http://registry.com"

def test_get_auth_headers(mock_logger):
    registry = SkillRegistry(logger=mock_logger, auth_token="tok")
    assert registry._get_auth_headers() == {"Authorization": "Bearer tok"}
    
    registry_no_auth = SkillRegistry(logger=mock_logger)
    assert registry_no_auth._get_auth_headers() == {}

def test_discover_skills(mock_logger, tmp_path):
    registry = SkillRegistry(logger=mock_logger, project_root=str(tmp_path))
    
    # Create mock local skill
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "my_skill"
    skill_dir.mkdir(parents=True)
    
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: my_skill\n"
        "description: This is a test skill\n"
        "---\n"
        "# My Skill\n"
    )
    
    registry.discover_skills(str(skills_dir))
    assert "my_skill" in registry.skills
    skill = registry.get_skill("my_skill")
    assert skill.name == "my_skill"
    assert skill.description == "This is a test skill"

@pytest.mark.asyncio
async def test_load_remote_skill_metadata(mock_logger):
    registry = SkillRegistry(logger=mock_logger, registry_url="http://registry.com")
    
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "skills": [
            {"name": "skillA", "description": "Desc A", "git_repository_url": "gitA"},
            {"name": "skillB", "description": "Desc B", "git_repository_url": "gitB"}
        ]
    }
    
    with patch("httpx.AsyncClient.get", return_value=mock_response):
        await registry._load_remote_skill_metadata()
        
    assert "skillA" in registry.skill_metadata_cache
    assert registry.skill_metadata_cache["skillA"]["description"] == "Desc A"

@pytest.mark.asyncio
async def test_get_skill_metadata(mock_logger):
    registry = SkillRegistry(logger=mock_logger, registry_url="http://registry.com")
    
    # Cache hit
    registry.skill_metadata_cache["skillA"] = {"name": "skillA"}
    meta = await registry.get_skill_metadata("skillA")
    assert meta == {"name": "skillA"}
    
    # Cache miss - fetch success
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"name": "skillB"}
    
    with patch("httpx.AsyncClient.get", return_value=mock_response):
        meta = await registry.get_skill_metadata("skillB")
        assert meta == {"name": "skillB"}

@pytest.mark.asyncio
async def test_pull_skill_flow(mock_logger, tmp_path):
    git_prov = MagicMock()
    git_prov.fetch_skill_files = AsyncMock(return_value={
        "SKILL.md": (
            "---\n"
            "name: remote_skill\n"
            "description: Parsed Remote Description\n"
            "---\n"
            "# Remote Skill\n"
        )
    })
    
    registry = SkillRegistry(
        logger=mock_logger,
        project_root=str(tmp_path),
        registry_url="http://registry.com",
        git_provider=git_prov,
        skills_cache_dir="./cache"
    )
    
    # Mock metadata cache
    registry.skill_metadata_cache["remote_skill"] = {
        "name": "remote_skill",
        "description": "Remote Description",
        "git_repository_url": "https://github.com/test/remote_skill.git"
    }
    
    # Destination directory
    dest_dir = registry.skills_cache_dir / "remote_skill"
    dest_dir.mkdir(parents=True)
    
    skill_md = dest_dir / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: remote_skill\n"
        "description: Parsed Remote Description\n"
        "---\n"
        "# Remote Skill\n"
    )
    
    # Patch HTTP call for reporting usage
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    
    with patch("httpx.AsyncClient.post", return_value=mock_response):
        success = await registry.pull_skill("remote_skill")
        assert success is True
        
    registry.discover_skills(str(registry.skills_cache_dir))
    skill = registry.get_skill("remote_skill")
    assert skill is not None
    assert skill.name == "remote_skill"
    assert skill.description == "Parsed Remote Description"

def test_generate_skills_prompt(mock_logger):
    registry = SkillRegistry(logger=mock_logger)
    
    skill = SkillProperties(
        name="my_skill",
        description="A cool skill",
        path="/some/path",
        skill_dir="/some/skill_dir"
    )
    
    prompt = registry.generate_skills_prompt([skill])
    assert "my_skill" in prompt
    assert "A cool skill" in prompt
