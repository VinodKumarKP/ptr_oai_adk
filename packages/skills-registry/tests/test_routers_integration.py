import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from oai_skills_registry.main import app
from oai_skills_registry.dependencies import get_registry


@pytest.fixture()
def mock_registry():
    """Mock the SkillsRegistry service."""
    registry = MagicMock()
    registry.db_logger = AsyncMock()
    registry.get_skill_details = AsyncMock()
    registry.publish_skill_version = AsyncMock()
    registry.upgrade_skill_version = AsyncMock()
    registry.downgrade_skill_version = AsyncMock()
    registry.deprecate_skill_version = AsyncMock()
    registry.discover_skills_from_git = AsyncMock()
    registry.list_skill_versions_from_url = AsyncMock()
    registry.register_skills_bulk = AsyncMock()
    registry.import_skill_version = AsyncMock()
    registry.refresh_skill_versions = AsyncMock()
    return registry


@pytest.fixture()
def client(mock_registry, monkeypatch):
    """FastAPI TestClient with auth disabled and mocked registry."""
    monkeypatch.setenv("SKILLS_AUTH_ENABLED", "false")
    monkeypatch.setenv("SKILLS_LOCAL_DIR", "/tmp/local_skills")
    
    app.dependency_overrides[get_registry] = lambda: mock_registry
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Skills Router Tests
# ---------------------------------------------------------------------------

def test_root_and_health(client):
    resp = client.get("/api/v1/skills-registry/")
    assert resp.status_code == 200
    assert "message" in resp.json()

    resp = client.get("/api/v1/skills-registry/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "registry": "skills"}


def test_list_skills_success(client, mock_registry):
    mock_registry.db_logger.get_all_skills.return_value = [{"name": "skill-1"}]
    resp = client.get("/api/v1/skills-registry/skills")
    assert resp.status_code == 200
    assert resp.json() == [{"name": "skill-1"}]


def test_list_skills_failure(client, mock_registry):
    mock_registry.db_logger.get_all_skills.side_effect = Exception("DB error")
    resp = client.get("/api/v1/skills-registry/skills")
    assert resp.status_code == 500
    assert "DB error" in resp.json()["detail"]


def test_download_template(client):
    resp = client.get("/api/v1/skills-registry/skills/template/download")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"


def test_get_skill_details_success(client, mock_registry):
    mock_registry.get_skill_details = AsyncMock(return_value={
        "skill": {
            "name": "skill-1",
            "description": "test description",
            "category": "test category",
            "author": "test author"
        },
        "versions": [{"version": "1.0.0", "status": "draft", "git_tag": "v1.0.0"}],
        "version_count": 1
    })
    resp = client.get("/api/v1/skills-registry/skills/skill-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["skill"]["name"] == "skill-1"
    assert data["version_count"] == 1


def test_get_skill_details_not_found(client, mock_registry):
    mock_registry.get_skill_details = AsyncMock(return_value=None)
    resp = client.get("/api/v1/skills-registry/skills/skill-1")
    assert resp.status_code == 404


def test_get_skill_details_failure(client, mock_registry):
    mock_registry.get_skill_details = AsyncMock(side_effect=Exception("Error fetching details"))
    resp = client.get("/api/v1/skills-registry/skills/skill-1")
    assert resp.status_code == 500


def test_get_published_skill_success(client, mock_registry):
    mock_registry.db_logger.get_skill.return_value = {
        "id": 1,
        "git_repository_url": "http://repo.git",
        "description": "test description",
        "category": "test category",
        "author": "test author"
    }
    mock_registry.db_logger.get_skill_versions.return_value = [
        {
            "version": "1.0.0",
            "status": "published",
            "dependencies": '["requests"]',
            "config": '{"capabilities": ["web"]}',
            "breaking_changes": '["none"]'
        }
    ]
    resp = client.get("/api/v1/skills-registry/skills/skill-1/published")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version"] == "1.0.0"
    assert data["dependencies"] == ["requests"]
    assert data["capabilities"] == ["web"]


def test_get_published_skill_not_found(client, mock_registry):
    # 1. Skill not found
    mock_registry.db_logger.get_skill.return_value = None
    resp = client.get("/api/v1/skills-registry/skills/skill-1/published")
    assert resp.status_code == 404

    # 2. No versions
    mock_registry.db_logger.get_skill.return_value = {"id": 1}
    mock_registry.db_logger.get_skill_versions.return_value = []
    resp = client.get("/api/v1/skills-registry/skills/skill-1/published")
    assert resp.status_code == 404

    # 3. No published version
    mock_registry.db_logger.get_skill_versions.return_value = [{"version": "1.0.0", "status": "draft"}]
    resp = client.get("/api/v1/skills-registry/skills/skill-1/published")
    assert resp.status_code == 404


def test_get_published_skill_failure(client, mock_registry):
    mock_registry.db_logger.get_skill.side_effect = Exception("DB error")
    resp = client.get("/api/v1/skills-registry/skills/skill-1/published")
    assert resp.status_code == 500


def test_register_skill_success(client, mock_registry):
    mock_registry.db_logger.create_skill.return_value = {"id": 123}
    payload = {
        "name": "new-skill",
        "description": "test desc",
        "category": "utility",
        "git_repository_url": "http://github.com/new-skill.git",
        "author": "tester"
    }
    resp = client.post("/api/v1/skills-registry/skills", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"status": "created", "skill_id": 123, "name": "new-skill"}


def test_register_skill_failure(client, mock_registry):
    mock_registry.db_logger.create_skill.side_effect = Exception("Save failed")
    payload = {
        "name": "new-skill",
        "description": "test desc",
        "category": "utility",
        "git_repository_url": "http://github.com/new-skill.git",
        "author": "tester"
    }
    resp = client.post("/api/v1/skills-registry/skills", json=payload)
    assert resp.status_code == 400


def test_readme_endpoints_local(client, mock_registry):
    mock_registry.db_logger.get_skill.return_value = {"id": 1, "name": "skill-1"}
    
    with patch("oai_skills_registry.routers.skills.read_local_readme") as mock_local:
        mock_local.return_value = ("Local README content", {"cached": True, "file_path": "/tmp/README.md"})
        resp = client.get("/api/v1/skills-registry/skills/skill-1/readme")
        assert resp.status_code == 200
        assert resp.json()["content"] == "Local README content"
        assert resp.json()["local"] is True


def test_readme_endpoints_github(client, mock_registry):
    mock_registry.db_logger.get_skill.return_value = {"id": 1, "name": "skill-1", "git_repository_url": "http://repo.git"}
    
    with patch("oai_skills_registry.routers.skills.read_local_readme", return_value=(None, {})), \
         patch("oai_skills_registry.routers.skills.fetch_readme") as mock_fetch:
        mock_fetch.return_value = ("GitHub README content", {"cached": False, "branch": "main", "url": "http://raw.github"})
        resp = client.get("/api/v1/skills-registry/skills/skill-1/readme")
        assert resp.status_code == 200
        assert resp.json()["content"] == "GitHub README content"
        assert resp.json().get("local") is None


def test_readme_endpoints_not_found(client, mock_registry):
    # 1. Skill not found
    mock_registry.db_logger.get_skill.return_value = None
    resp = client.get("/api/v1/skills-registry/skills/skill-1/readme")
    assert resp.status_code == 404

    # 2. No repo url
    mock_registry.db_logger.get_skill.return_value = {"id": 1, "name": "skill-1", "git_repository_url": None}
    with patch("oai_skills_registry.routers.skills.read_local_readme", return_value=(None, {})):
        resp = client.get("/api/v1/skills-registry/skills/skill-1/readme")
        assert resp.status_code == 404


def test_readme_cache_management(client):
    with patch("oai_skills_registry.routers.skills.invalidate_readme_cache") as mock_invalidate:
        resp = client.post("/api/v1/skills-registry/skills/skill-1/readme/invalidate-cache")
        assert resp.status_code == 200
        assert mock_invalidate.call_count == 2
    
    with patch("oai_skills_registry.routers.skills.readme_cache_stats") as mock_stats:
        mock_stats.return_value = {
            "hits": 10,
            "misses": 5,
            "entries": [
                {"key": "skill:skill-1", "size": 100},
                {"key": "other:thing", "size": 200}
            ]
        }
        resp = client.get("/api/v1/skills-registry/readme/cache-stats")
        assert resp.status_code == 200
        entries = resp.json()["entries"]
        assert len(entries) == 1
        assert entries[0]["key"] == "skill:skill-1"


# ---------------------------------------------------------------------------
# Lifecycle Router Tests
# ---------------------------------------------------------------------------

def test_publish_skill(client, mock_registry):
    # Success
    mock_registry.publish_skill_version = AsyncMock(return_value={"version": "1.0.0", "action_id": 42})
    resp = client.post(
        "/api/v1/skills-registry/skills/skill-1/publish",
        json={"version": "1.0.0", "message": "Ready"}
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "published", "skill": "skill-1", "version": "1.0.0", "action_id": 42}

    # Failure
    mock_registry.publish_skill_version = AsyncMock(side_effect=Exception("Failed to publish"))
    resp = client.post(
        "/api/v1/skills-registry/skills/skill-1/publish",
        json={"version": "1.0.0", "message": "Ready"}
    )
    assert resp.status_code == 400


def test_upgrade_skill(client, mock_registry):
    # Success
    mock_registry.upgrade_skill_version = AsyncMock(return_value={
        "from_version": "1.0.0",
        "to_version": "1.1.0",
        "agents_affected": 2,
        "action_id": 43
    })
    resp = client.post(
        "/api/v1/skills-registry/skills/skill-1/upgrade",
        json={"to_version": "1.1.0", "message": "New features"}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "upgraded"

    # Failure
    mock_registry.upgrade_skill_version = AsyncMock(side_effect=Exception("Failed to upgrade"))
    resp = client.post(
        "/api/v1/skills-registry/skills/skill-1/upgrade",
        json={"to_version": "1.1.0", "message": "New features"}
    )
    assert resp.status_code == 400


def test_downgrade_skill(client, mock_registry):
    # Success
    mock_registry.downgrade_skill_version = AsyncMock(return_value={
        "from_version": "1.1.0",
        "to_version": "1.0.0",
        "agents_affected": 2,
        "action_id": 44
    })
    resp = client.post(
        "/api/v1/skills-registry/skills/skill-1/downgrade",
        json={"to_version": "1.0.0", "message": "Bug reversion"}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "downgraded"

    # Failure
    mock_registry.downgrade_skill_version = AsyncMock(side_effect=Exception("Failed to downgrade"))
    resp = client.post(
        "/api/v1/skills-registry/skills/skill-1/downgrade",
        json={"to_version": "1.0.0", "message": "Bug reversion"}
    )
    assert resp.status_code == 400


def test_deprecate_skill(client, mock_registry):
    # Success
    mock_registry.deprecate_skill_version = AsyncMock(return_value={
        "version": "1.0.0",
        "action_id": 45
    })
    resp = client.post(
        "/api/v1/skills-registry/skills/skill-1/deprecate",
        json={"version": "1.0.0", "message": "Obsolete"}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "deprecated"

    # Failure
    mock_registry.deprecate_skill_version = AsyncMock(side_effect=Exception("Failed to deprecate"))
    resp = client.post(
        "/api/v1/skills-registry/skills/skill-1/deprecate",
        json={"version": "1.0.0", "message": "Obsolete"}
    )
    assert resp.status_code == 400


def test_skill_history(client, mock_registry):
    # Success
    mock_registry.get_skill_history = AsyncMock(return_value={
        "skill_name": "skill-1",
        "total_count": 1,
        "actions": [{
            "id": 1,
            "skill_name": "skill-1",
            "action": "publish",
            "from_version": None,
            "to_version": "1.0.0",
            "performed_by": "tester",
            "message": "message",
            "created_at": "2026-07-01T00:00:00Z"
        }]
    })
    resp = client.get("/api/v1/skills-registry/skills/skill-1/history")
    assert resp.status_code == 200
    assert resp.json()["total_count"] == 1

    # Failure
    mock_registry.get_skill_history = AsyncMock(side_effect=Exception("DB fail"))
    resp = client.get("/api/v1/skills-registry/skills/skill-1/history")
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Git Router Tests
# ---------------------------------------------------------------------------

def test_discover_skills(client, mock_registry):
    # Success
    mock_registry.discover_skills_from_git = AsyncMock(return_value={
        "git_repository_url": "http://repo.git",
        "total_found": 1,
        "available_to_register": 1,
        "already_registered": 0,
        "invalid": 0,
        "skills": [{
            "name": "test-skill",
            "version": "1.0.0",
            "description": "test description",
            "category": "utility",
            "author": "tester",
            "tags": [],
            "status": "available",
            "path_in_repo": "skills/test-skill"
        }]
    })
    resp = client.post("/api/v1/skills-registry/skills/discover", json={"git_repository_url": "http://repo.git"})
    assert resp.status_code == 200
    assert resp.json()["total_found"] == 1

    # Missing git_repository_url
    resp = client.post("/api/v1/skills-registry/skills/discover", json={})
    assert resp.status_code == 400

    # Failure
    mock_registry.discover_skills_from_git = AsyncMock(side_effect=Exception("Git fail"))
    resp = client.post("/api/v1/skills-registry/skills/discover", json={"git_repository_url": "http://repo.git"})
    assert resp.status_code == 500


def test_preview_skill_versions(client, mock_registry):
    # Success
    mock_registry.list_skill_versions_from_url = AsyncMock(return_value=[
        {"git_tag": "v1.0.0", "commit_sha": "sha"}
    ])
    payload = {"git_repository_url": "http://repo.git", "skill_name": "skill-1"}
    resp = client.post("/api/v1/skills-registry/skills/preview-versions", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert resp.json()["versions_found"] == 1

    # Missing git_repository_url
    resp = client.post("/api/v1/skills-registry/skills/preview-versions", json={"skill_name": "skill-1"})
    assert resp.status_code == 400

    # Missing skill_name
    resp = client.post("/api/v1/skills-registry/skills/preview-versions", json={"git_repository_url": "http://repo.git"})
    assert resp.status_code == 400

    # Failure
    mock_registry.list_skill_versions_from_url = AsyncMock(side_effect=Exception("Git fail"))
    resp = client.post("/api/v1/skills-registry/skills/preview-versions", json=payload)
    assert resp.status_code == 500


def test_register_skills_bulk(client, mock_registry):
    # Success
    mock_registry.register_skills_bulk = AsyncMock(return_value={
        "total_registered": 2,
        "successful": ["s1", "s2"],
        "failed": []
    })
    payload = {
        "git_repository_url": "http://repo.git",
        "skill_names": ["s1", "s2"],
        "auth_token": "token"
    }
    resp = client.post("/api/v1/skills-registry/skills/register-bulk", json=payload)
    assert resp.status_code == 200
    assert resp.json()["total_registered"] == 2

    # Failure
    mock_registry.register_skills_bulk = AsyncMock(side_effect=Exception("Bulk register error"))
    resp = client.post("/api/v1/skills-registry/skills/register-bulk", json=payload)
    assert resp.status_code == 400


def test_import_skill_from_git(client, mock_registry):
    # Success
    mock_registry.import_skill_from_git = AsyncMock(return_value={"version": "1.0.0", "preview": {"skill_name": "skill-1", "version": "1.0.0", "description": "desc", "category": "utility", "tags": [], "git_commit": "sha"}})
    payload = {"git_tag": "v1.0.0", "git_repository_url": "http://repo.git", "auth_token": "token"}
    resp = client.post("/api/v1/skills-registry/skills/skill-1/import-from-git", json=payload)
    assert resp.status_code == 200
    assert resp.json()["version"] == "1.0.0"

    # Git source ID not found
    mock_registry.db_logger.get_git_source = AsyncMock(return_value=None)
    payload_source_not_found = {"git_tag": "v1.0.0", "git_source_id": 999}
    resp = client.post("/api/v1/skills-registry/skills/skill-1/import-from-git", json=payload_source_not_found)
    assert resp.status_code == 404

    # Neither ID nor URL provided
    payload_neither = {"git_tag": "v1.0.0"}
    resp = client.post("/api/v1/skills-registry/skills/skill-1/import-from-git", json=payload_neither)
    assert resp.status_code == 400

    # Failure
    mock_registry.import_skill_from_git = AsyncMock(side_effect=Exception("Import fail"))
    resp = client.post("/api/v1/skills-registry/skills/skill-1/import-from-git", json=payload)
    assert resp.status_code == 400


def test_refresh_skill_versions(client, mock_registry):
    # Success
    mock_registry.db_logger.get_skill.return_value = {"id": 1, "git_repository_url": "http://repo.git"}
    mock_registry.list_skill_versions_from_url.return_value = [
        {"git_tag": "v1.0.0", "commit_sha": "sha", "commit_message": "msg"}
    ]
    resp = client.post("/api/v1/skills-registry/skills/skill-1/refresh-versions", json={"auth_token": "token"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert resp.json()["versions_found"] == 1

    # Skill not found
    mock_registry.db_logger.get_skill.return_value = None
    resp = client.post("/api/v1/skills-registry/skills/skill-1/refresh-versions", json={})
    assert resp.status_code == 404

    # No git repo url
    mock_registry.db_logger.get_skill.return_value = {"id": 1, "git_repository_url": None}
    resp = client.post("/api/v1/skills-registry/skills/skill-1/refresh-versions", json={})
    assert resp.status_code == 400

    # Failure
    mock_registry.db_logger.get_skill.side_effect = Exception("Refresh fail")
    resp = client.post("/api/v1/skills-registry/skills/skill-1/refresh-versions", json={})
    assert resp.status_code == 500
