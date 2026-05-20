"""
Skills catalog routes — list, get, register, and template download.
"""

import io
import logging
import zipfile
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse

from oai_skills_registry.dependencies import get_registry, verify_bearer_token
from oai_skills_registry.models import (
    BulkRegistrationRequest,
    BulkRegistrationResult,
    SkillDetails,
    SkillPublishedVersion,
    SkillRegistration,
    SkillVersion,
)
from oai_skills_registry.services.skills_registry import SkillsRegistry

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/")
async def root():
    """Returns a summary of all available Skills Registry endpoints."""
    info = {
        "message": "Skills Registry API — GitHub Integration",
        "description": (
            "Skills management system with GitHub repository support for "
            "publishing, importing, and versioning skills."
        ),
        "endpoints": {
            # catalog
            "GET  /health":                            "Health check",
            "GET  /skills":                            "List all registered skills",
            "GET  /skills/{skill_name}":               "Get skill details with all versions",
            "GET  /skills/{skill_name}/published":     "Get published version (for agent installation)",
            "POST /skills":                            "Register a new skill with a GitHub repository",
            "GET  /skills/template/download":          "Download the skill template ZIP",
            # git
            "POST /skills/discover":                   "Discover skills in a GitHub repository",
            "POST /skills/register-bulk":              "Register multiple discovered skills at once",
            "POST /skills/preview-versions":           "Preview available versions from a GitHub repo",
            "POST /skills/{skill_name}/import-from-git":   "Import a skill version from GitHub",
            "POST /skills/{skill_name}/refresh-versions":  "Refresh available versions from GitHub",
            # lifecycle
            "POST /skills/{skill_name}/publish":       "Publish a draft skill version",
            "POST /skills/{skill_name}/upgrade":       "Upgrade to a newer version",
            "POST /skills/{skill_name}/downgrade":     "Downgrade to a previous version",
            "POST /skills/{skill_name}/deprecate":     "Mark a version as deprecated",
            "GET  /skills/{skill_name}/history":       "Get complete skill action history",
        },
    }
    return JSONResponse(info)


@router.get("/health")
async def health_check(registry: SkillsRegistry = Depends(get_registry)):
    """Health check endpoint."""
    return {"status": "ok", "registry": "skills"}


# ---------------------------------------------------------------------------
# Catalog — read
# ---------------------------------------------------------------------------

@router.get("/skills", response_model=List[Dict])
async def list_skills(
    _auth: bool = Depends(verify_bearer_token),
    registry: SkillsRegistry = Depends(get_registry),
):
    """List all registered skills."""
    try:
        return await registry.db_logger.get_all_skills()
    except Exception as exc:
        logger.error("Failed to list skills: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get("/skills/template/download")
async def download_skill_template(
    _auth: bool = Depends(verify_bearer_token),
):
    """Download a template directory structure for creating new agent skills."""
    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("README.md", _TEMPLATE_README)
            zf.writestr("skills/example-skill/SKILL.md", _TEMPLATE_SKILL_MD)
            zf.writestr("skills/example-skill/skill_config.yaml", _TEMPLATE_SKILL_CONFIG)
            zf.writestr("skills/example-skill/example_skill.py", _TEMPLATE_SKILL_PY)
            zf.writestr("skills/example-skill/requirements.txt", _TEMPLATE_REQUIREMENTS)
            zf.writestr("skills/example-skill/README.md", _TEMPLATE_SKILL_README)
            zf.writestr("skills/example-skill/tests/test_example.py", _TEMPLATE_TEST_PY)
            zf.writestr("skills/example-skill/tests/__init__.py", "")

        zip_buffer.seek(0)
        return StreamingResponse(
            iter([zip_buffer.getvalue()]),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=skill-template.zip"},
        )
    except Exception as exc:
        logger.error("Failed to generate template: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate template: {exc}",
        )


@router.get("/skills/{skill_name}", response_model=SkillDetails)
async def get_skill_details(
    skill_name: str,
    _auth: bool = Depends(verify_bearer_token),
    registry: SkillsRegistry = Depends(get_registry),
):
    """Get skill details with all versions."""
    try:
        details = await registry.get_skill_details(skill_name)
        if not details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Skill not found: {skill_name}",
            )

        parsed_versions = []
        for v in details["versions"]:
            if "skill_name" not in v:
                v["skill_name"] = skill_name
            parsed_versions.append(SkillVersion.from_db(v))

        return SkillDetails(
            skill=details["skill"],
            versions=parsed_versions,
            version_count=details["version_count"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get skill details: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get("/skills/{skill_name}/published", response_model=SkillPublishedVersion)
async def get_published_skill(
    skill_name: str,
    _auth: bool = Depends(verify_bearer_token),
    registry: SkillsRegistry = Depends(get_registry),
):
    """Get the currently published version of a skill for agent installation."""
    import json

    try:
        skill = await registry.db_logger.get_skill(skill_name)
        if not skill:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Skill not found: {skill_name}",
            )

        versions = await registry.db_logger.get_skill_versions(skill["id"])
        if not versions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No versions available for skill: {skill_name}",
            )

        published = next((v for v in versions if v.get("status") == "published"), None)
        if not published:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No published version available for skill: {skill_name}",
            )

        for field in ("dependencies", "config", "breaking_changes"):
            if isinstance(published.get(field), str):
                try:
                    published[field] = json.loads(published[field])
                except (json.JSONDecodeError, TypeError):
                    published[field] = [] if field == "dependencies" else None

        capabilities = []
        if isinstance(published.get("config"), dict):
            capabilities = published["config"].get("capabilities", [])

        return SkillPublishedVersion(
            skill_name=skill_name,
            description=skill.get("description"),
            category=skill.get("category"),
            author=skill.get("author"),
            git_repository_url=skill.get("git_repository_url"),
            version=published.get("version"),
            git_tag=published.get("git_tag"),
            git_commit_sha=published.get("git_commit_sha"),
            config=published.get("config"),
            dependencies=published.get("dependencies"),
            capabilities=capabilities or None,
            breaking_changes=published.get("breaking_changes"),
            published_at=published.get("published_at"),
            published_by=published.get("published_by"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get published skill: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@router.post("/skills", response_model=Dict)
async def register_skill(
    skill: SkillRegistration,
    _auth: bool = Depends(verify_bearer_token),
    registry: SkillsRegistry = Depends(get_registry),
):
    """Register a new skill with a GitHub repository."""
    try:
        created = await registry.db_logger.create_skill(
            name=skill.name,
            description=skill.description,
            category=skill.category,
            tags=skill.tags,
            author=skill.author,
            git_repository_url=skill.git_repository_url,
        )
        return {"status": "created", "skill_id": created.get("id"), "name": skill.name}
    except Exception as exc:
        logger.error("Failed to register skill: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


# ---------------------------------------------------------------------------
# Template content (kept here to avoid a separate data file)
# ---------------------------------------------------------------------------

_TEMPLATE_README = """\
# Agent Skills Repository Template

This is a template repository for creating agent skills compatible with the Skills Registry.

## Directory Structure

```
skills/
├── example-skill/
│   ├── SKILL.md                 # Skill metadata and documentation
│   ├── skill_config.yaml        # Skill configuration
│   ├── example_skill.py         # Python implementation
│   ├── requirements.txt         # Python dependencies
│   ├── README.md                # Skill-specific documentation
│   └── tests/
│       └── test_example.py      # Unit tests
```

## Getting Started

1. Copy `example-skill/` and rename it to your skill name.
2. Update SKILL.md with your skill's metadata.
3. Implement your skill in the Python file.
4. Update `requirements.txt` with your dependencies.
5. Register your skill in the Skills Registry.
"""

_TEMPLATE_SKILL_MD = """\
---
name: example-skill
version: 1.0.0
description: This is an example skill template showing the proper structure and format
category: example
author: developer@company.com
tags:
  - example
  - template
capabilities:
  - process_data
  - generate_output
dependencies:
  - pydantic>=2.0.0
---

# Example Skill

An example skill demonstrating the proper structure for Skills Registry compatibility.
"""

_TEMPLATE_SKILL_CONFIG = """\
metadata:
  name: example-skill
  version: 1.0.0
  description: Example skill template
  category: example

settings:
  timeout: 30
  retry_attempts: 3
  log_level: INFO

features:
  enable_caching: true
  enable_logging: true
"""

_TEMPLATE_SKILL_PY = '''\
"""Example Skill Implementation."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ExampleInput(BaseModel):
    input_field: str = Field(..., description="Input field description")
    another_field: Optional[int] = Field(None, description="Optional numeric field")


class ExampleSkill:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = "example-skill"
        self.version = "1.0.0"

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = ExampleInput(**input_data)
        return {"result": f"Processed: {validated.input_field}", "status": "success"}
'''

_TEMPLATE_REQUIREMENTS = """\
pydantic>=2.0.0
requests>=2.28.0
"""

_TEMPLATE_SKILL_README = """\
# Example Skill

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from example_skill import ExampleSkill
skill = ExampleSkill()
result = skill.process({"input_field": "your input"})
print(result)
```
"""

_TEMPLATE_TEST_PY = '''\
"""Tests for example skill."""

import pytest
from example_skill import ExampleSkill


class TestExampleSkill:
    @pytest.fixture
    def skill(self):
        return ExampleSkill()

    def test_process_valid_input(self, skill):
        result = skill.process({"input_field": "test"})
        assert result["status"] == "success"
        assert "Processed" in result["result"]

    def test_process_invalid_input(self, skill):
        with pytest.raises(Exception):
            skill.process({"invalid_field": "x"})
'''
