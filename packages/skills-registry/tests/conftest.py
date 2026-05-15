"""
Shared test fixtures and configuration for Skills Registry tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone


@pytest.fixture
def mock_db_logger():
    """Create a mock database logger."""
    db_logger = AsyncMock()

    # Setup common mock returns
    db_logger.get_skill = AsyncMock(return_value=None)
    db_logger.create_skill = AsyncMock(return_value={"id": 1, "name": "test-skill"})
    db_logger.get_skill_version = AsyncMock(return_value=None)
    db_logger.create_skill_version = AsyncMock(return_value={"id": 1, "version": "1.0.0"})
    db_logger.get_skill_versions = AsyncMock(return_value=[])
    db_logger.get_all_skills = AsyncMock(return_value=[])
    db_logger.update_skill_version_status = AsyncMock()
    db_logger.update_skill_current_version = AsyncMock()
    db_logger.log_skill_action = AsyncMock(return_value={"id": 1})
    db_logger.delete_skill = AsyncMock()
    db_logger.get_agent_skills = AsyncMock(return_value=[])
    db_logger.update_agent_skill_version = AsyncMock()
    db_logger.get_skill_actions = AsyncMock(return_value=[])
    db_logger.get_skill_action_count = AsyncMock(return_value=0)
    db_logger.get_git_source = AsyncMock(return_value=None)
    db_logger.register_git_source = AsyncMock(return_value=1)
    db_logger.close = AsyncMock()

    return db_logger


@pytest.fixture
def mock_git_provider():
    """Create a mock Git provider."""
    provider = AsyncMock()

    # Setup common mock returns
    provider.fetch_skill_files = AsyncMock(return_value={
        "SKILL.md": "---\nname: test-skill\nversion: 1.0.0\n---\nContent",
        "skill_config.yaml": "config: value"
    })
    provider.get_commit_info = AsyncMock(return_value={
        "commit_sha": "abc123",
        "commit_message": "Test commit"
    })
    provider.get_tags = AsyncMock(return_value=[])

    return provider


@pytest.fixture
def sample_skill_md():
    """Sample SKILL.md content."""
    return """---
name: test-skill
version: 1.0.0
description: A test skill
category: test
author: test-author
tags:
  - test
  - example
dependencies:
  - "requests>=2.28.0"
  - "pydantic>=2.0.0"
breaking_changes:
  - "Breaking change 1"
---

# Test Skill

This is a test skill for testing purposes.
"""


@pytest.fixture
def sample_skill_metadata():
    """Sample parsed skill metadata."""
    return {
        "name": "test-skill",
        "version": "1.0.0",
        "description": "A test skill",
        "category": "test",
        "author": "test-author",
        "tags": ["test", "example"],
        "dependencies": ["requests>=2.28.0", "pydantic>=2.0.0"],
        "breaking_changes": ["Breaking change 1"],
        "content": "\n# Test Skill\n\nThis is a test skill for testing purposes.\n"
    }


@pytest.fixture
def github_url():
    """Sample GitHub repository URL."""
    return "https://github.com/owner/skills-repo.git"


@pytest.fixture
def github_repo():
    """Sample parsed GitHub repository."""
    return "owner/skills-repo"


@pytest.fixture
def source_config():
    """Sample Git source configuration."""
    return {
        "name": "test-github-source",
        "git_provider": "github",
        "repository": "owner/skills-repo",
        "git_url": "https://github.com/owner/skills-repo.git",
        "branch": "main",
        "auth_type": "token",
        "auth_token": "test-token-123"
    }


@pytest.fixture
def direct_url_config():
    """Sample direct GitHub URL configuration."""
    return {
        "git_provider": "github",
        "_direct_url": "https://github.com/owner/skills-repo.git",
        "branch": "main"
    }
