"""
FastAPI routes for Skills Registry.
Handles skill lifecycle management and Git integration.
"""

import json
import logging
from typing import Optional, List, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from oai_skills_registry.dependencies import (
    get_registry,
    verify_bearer_token,
    get_auth_user
)
from oai_skills_registry.models import (
    SkillRegistration,
    SkillImportConfig,
    SkillPublishConfig,
    SkillUpgradeConfig,
    SkillDowngradeConfig,
    SkillDeprecateConfig,
    SkillDetails,
    SkillVersion,
    SkillActionHistory,
    SkillPublishedVersion,
    SkillDiscoveryResult,
    BulkRegistrationRequest,
    BulkRegistrationResult,
)
from oai_skills_registry.services.skills_registry import SkillsRegistry

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/")
async def root():
    """Returns a list of all available endpoints."""
    info = {
        "message": "Skills Registry API - GitHub Integration",
        "description": "Skills management system with GitHub repository support for publishing, importing, and versioning skills.",
        "endpoints": {
            "GET /health": "Health check endpoint",
            "GET /skills": "List all skills",
            "GET /skills/{skill_name}": "Get skill details with all versions",
            "GET /skills/{skill_name}/published": "Get published skill version for agent installation (includes git repo URL and download info)",
            "POST /skills": "Register new skill with GitHub repository",
            "POST /skills/discover": "Discover all skills in a GitHub repository",
            "POST /skills/register-bulk": "Register multiple discovered skills at once",
            "POST /skills/preview-versions": "Preview available versions from any GitHub repository (for import dialog)",
            "POST /skills/{skill_name}/import-from-git": "Import new skill version from GitHub (supports both registered sources and direct URLs)",
            "POST /skills/{skill_name}/refresh-versions": "Refresh available versions from the skill's GitHub repository",
            "POST /skills/{skill_name}/publish": "Publish draft skill version for agent use",
            "POST /skills/{skill_name}/upgrade": "Upgrade to newer skill version",
            "POST /skills/{skill_name}/downgrade": "Downgrade to previous skill version",
            "POST /skills/{skill_name}/deprecate": "Mark skill version as deprecated",
            "GET /skills/{skill_name}/history": "Get complete skill action history",
        },
    }
    return JSONResponse(info)


@router.get("/health")
async def health_check(registry: SkillsRegistry = Depends(get_registry)):
    """Health check endpoint."""
    return {
        "status": "ok",
        "registry": "skills"
    }


# ==================== SKILL ENDPOINTS ====================

@router.get("/skills", response_model=List[Dict])
async def list_skills(registry: SkillsRegistry = Depends(get_registry)):
    """List all skills."""
    try:
        skills = await registry.db_logger.get_all_skills()
        return skills
    except Exception as e:
        logger.error(f"Failed to list skills: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/skills", response_model=Dict)
async def register_skill(
    skill: SkillRegistration,
    credentials=Depends(verify_bearer_token),
    registry: SkillsRegistry = Depends(get_registry),
):
    """Register a new skill."""
    try:
        created_skill = await registry.db_logger.create_skill(
            name=skill.name,
            description=skill.description,
            category=skill.category,
            tags=skill.tags,
            author=skill.author,
            git_repository_url=skill.git_repository_url
        )
        return {
            "status": "created",
            "skill_id": created_skill.get("id"),
            "name": skill.name
        }
    except Exception as e:
        logger.error(f"Failed to register skill: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/skills/discover", response_model=SkillDiscoveryResult)
async def discover_skills(
    request_data: Dict,
    registry: SkillsRegistry = Depends(get_registry),
):
    """
    Discover all skills available in a GitHub repository.

    Scans the 'skills/' directory and returns metadata for each skill found.
    No authentication required for public repositories.
    """
    try:
        git_repository_url = request_data.get("git_repository_url")
        if not git_repository_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="git_repository_url is required"
            )

        discovery_result = await registry.discover_skills_from_git(
            git_repository_url=git_repository_url,
            auth_token=request_data.get("auth_token")
        )

        return SkillDiscoveryResult(
            git_repository_url=discovery_result["git_repository_url"],
            total_found=discovery_result["total_found"],
            available_to_register=discovery_result["available_to_register"],
            already_registered=discovery_result["already_registered"],
            invalid=discovery_result["invalid"],
            skills=discovery_result["skills"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to discover skills: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/skills/register-bulk", response_model=BulkRegistrationResult)
async def register_skills_bulk(
    request: BulkRegistrationRequest,
    credentials=Depends(verify_bearer_token),
    auth_user: str = Depends(get_auth_user),
    registry: SkillsRegistry = Depends(get_registry),
):
    """
    Register multiple skills from a GitHub repository in one operation.

    Requires authentication. Author defaults to the authenticated user if not specified.
    """
    try:
        author = request.author or auth_user

        result = await registry.register_skills_bulk(
            git_repository_url=request.git_repository_url,
            skill_names=request.skill_names,
            author=author
        )

        return BulkRegistrationResult(
            total_registered=result["total_registered"],
            successful=result["successful"],
            failed=result["failed"]
        )
    except Exception as e:
        logger.error(f"Failed to register skills: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/skills/{skill_name}", response_model=SkillDetails)
async def get_skill_details(
    skill_name: str,
    registry: SkillsRegistry = Depends(get_registry)
):
    """Get skill details with all versions."""
    try:
        details = await registry.get_skill_details(skill_name)
        if not details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Skill not found: {skill_name}"
            )

        # Parse versions from database rows using from_db method
        parsed_versions = []
        for v in details["versions"]:
            # Add skill_name to the version dict if not present
            if "skill_name" not in v:
                v["skill_name"] = skill_name
            # Convert to SkillVersion using from_db to parse JSON strings
            parsed_versions.append(SkillVersion.from_db(v))

        return SkillDetails(
            skill=details["skill"],
            versions=parsed_versions,
            version_count=details["version_count"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get skill details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/skills/{skill_name}/published", response_model=SkillPublishedVersion)
async def get_published_skill(
    skill_name: str,
    registry: SkillsRegistry = Depends(get_registry)
):
    """
    Get currently published version of a skill for agent installation.
    Returns all information agents need to download and configure the skill.
    """
    try:
        # Get skill metadata
        skill = await registry.db_logger.get_skill(skill_name)
        if not skill:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Skill not found: {skill_name}"
            )

        # Get all versions to find the published one
        versions = await registry.db_logger.get_skill_versions(skill["id"])
        if not versions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No versions available for skill: {skill_name}"
            )

        # Find the published version
        published_version = None
        for v in versions:
            if v.get("status") == "published":
                published_version = v
                break

        if not published_version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No published version available for skill: {skill_name}"
            )

        # Parse dependencies and config if they're JSON strings
        if isinstance(published_version.get("dependencies"), str):
            try:
                published_version["dependencies"] = json.loads(published_version["dependencies"])
            except (json.JSONDecodeError, TypeError):
                published_version["dependencies"] = []

        if isinstance(published_version.get("config"), str):
            try:
                published_version["config"] = json.loads(published_version["config"])
            except (json.JSONDecodeError, TypeError):
                published_version["config"] = None

        if isinstance(published_version.get("breaking_changes"), str):
            try:
                published_version["breaking_changes"] = json.loads(published_version["breaking_changes"])
            except (json.JSONDecodeError, TypeError):
                published_version["breaking_changes"] = None

        # Extract capabilities from config if available
        capabilities = []
        if published_version.get("config"):
            config_data = published_version["config"]
            if isinstance(config_data, dict):
                capabilities = config_data.get("capabilities", [])

        return SkillPublishedVersion(
            skill_name=skill_name,
            description=skill.get("description"),
            category=skill.get("category"),
            author=skill.get("author"),
            git_repository_url=skill.get("git_repository_url"),
            version=published_version.get("version"),
            git_tag=published_version.get("git_tag"),
            git_commit_sha=published_version.get("git_commit_sha"),
            config=published_version.get("config"),
            dependencies=published_version.get("dependencies"),
            capabilities=capabilities or None,
            breaking_changes=published_version.get("breaking_changes"),
            published_at=published_version.get("published_at"),
            published_by=published_version.get("published_by")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get published skill: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== SKILL VERSION IMPORT ENDPOINTS ====================

@router.post("/skills/{skill_name}/import-from-git", response_model=Dict)
async def import_skill_from_git(
    skill_name: str,
    config: SkillImportConfig,
    credentials=Depends(verify_bearer_token),
    auth_user: str = Depends(get_auth_user),
    registry: SkillsRegistry = Depends(get_registry),
):
    """
    Import skill version from Git source.
    Supports two methods:
    1. Using a registered git_source_id
    2. Using a direct git_repository_url (GitHub)
    Fetches SKILL.md and skill_config.yaml from Git and creates unpublished version.
    """
    try:
        source_config = None

        # Method 1: Use registered Git source
        if config.git_source_id:
            source_config = await registry.db_logger.get_git_source(config.git_source_id)
            if not source_config:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Git source not found: {config.git_source_id}"
                )

        # Method 2: Use direct GitHub repository URL
        elif config.git_repository_url:
            source_config = {
                "git_provider": "github",
                "repository": "",  # Will be extracted from URL in import method
                "branch": "main",
                "auth_token": None,
                "git_url": config.git_repository_url
            }
            # Store the URL for direct GitHub import
            source_config["_direct_url"] = config.git_repository_url
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either git_source_id or git_repository_url must be provided"
            )

        # Import from Git
        result = await registry.import_skill_from_git(
            skill_name=skill_name,
            source_config=source_config,
            git_tag=config.git_tag,
            performed_by=auth_user
        )

        return {
            "status": "imported",
            "skill": skill_name,
            "version": result["version"],
            "preview": result["preview"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import skill: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/skills/{skill_name}/refresh-versions", response_model=Dict)
async def refresh_skill_versions(
    skill_name: str,
    credentials=Depends(verify_bearer_token),
    registry: SkillsRegistry = Depends(get_registry),
):
    """Fetch and refresh available versions from the skill's Git repository."""
    try:
        skill = await registry.db_logger.get_skill(skill_name)
        if not skill:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Skill not found: {skill_name}"
            )

        git_repository_url = skill.get('git_repository_url')
        if not git_repository_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Skill does not have a Git repository configured"
            )

        # Fetch versions from Git repository
        versions = await registry.list_skill_versions_from_url(
            skill_name=skill_name,
            git_repository_url=git_repository_url
        )

        return {
            "status": "success",
            "skill": skill_name,
            "versions": versions,
            "versions_found": len(versions) if versions else 0
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to refresh skill versions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/skills/preview-versions", response_model=Dict)
async def preview_skill_versions(
    request_data: Dict,
    registry: SkillsRegistry = Depends(get_registry),
):
    """Fetch available versions from a GitHub repository for preview (before registering skill).

    Used by import dialog to show versions before registering the skill.
    No authentication required for public repositories.
    """
    try:
        git_repository_url = request_data.get('git_repository_url')
        skill_name = request_data.get('skill_name')

        if not git_repository_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="git_repository_url is required"
            )

        if not skill_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="skill_name is required"
            )

        # Fetch versions from Git repository
        versions = await registry.list_skill_versions_from_url(
            skill_name=skill_name,
            git_repository_url=git_repository_url
        )

        return {
            "status": "success",
            "skill": skill_name,
            "versions": versions,
            "versions_found": len(versions) if versions else 0
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to preview skill versions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== SKILL LIFECYCLE ENDPOINTS ====================

@router.post("/skills/{skill_name}/publish", response_model=Dict)
async def publish_skill(
    skill_name: str,
    config: SkillPublishConfig,
    credentials=Depends(verify_bearer_token),
    auth_user: str = Depends(get_auth_user),
    registry: SkillsRegistry = Depends(get_registry),
):
    """
    Publish unpublished skill version.
    Makes it available for agents to use.
    """
    try:
        result = await registry.publish_skill_version(
            skill_name=skill_name,
            version=config.version,
            message=config.message,
            performed_by=auth_user
        )
        return {
            "status": "published",
            "skill": skill_name,
            "version": result["version"],
            "action_id": result.get("action_id")
        }
    except Exception as e:
        logger.error(f"Failed to publish skill: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/skills/{skill_name}/upgrade", response_model=Dict)
async def upgrade_skill(
    skill_name: str,
    config: SkillUpgradeConfig,
    credentials=Depends(verify_bearer_token),
    auth_user: str = Depends(get_auth_user),
    registry: SkillsRegistry = Depends(get_registry),
):
    """Upgrade skill to new version."""
    try:
        result = await registry.upgrade_skill_version(
            skill_name=skill_name,
            to_version=config.to_version,
            message=config.message,
            performed_by=auth_user
        )
        return {
            "status": "upgraded",
            "skill": skill_name,
            "from_version": result["from_version"],
            "to_version": result["to_version"],
            "agents_affected": result["agents_affected"],
            "action_id": result.get("action_id")
        }
    except Exception as e:
        logger.error(f"Failed to upgrade skill: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/skills/{skill_name}/downgrade", response_model=Dict)
async def downgrade_skill(
    skill_name: str,
    config: SkillDowngradeConfig,
    credentials=Depends(verify_bearer_token),
    auth_user: str = Depends(get_auth_user),
    registry: SkillsRegistry = Depends(get_registry),
):
    """Downgrade skill to previous version."""
    try:
        result = await registry.downgrade_skill_version(
            skill_name=skill_name,
            to_version=config.to_version,
            message=config.message,
            performed_by=auth_user
        )
        return {
            "status": "downgraded",
            "skill": skill_name,
            "from_version": result["from_version"],
            "to_version": result["to_version"],
            "action_id": result.get("action_id")
        }
    except Exception as e:
        logger.error(f"Failed to downgrade skill: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/skills/{skill_name}/deprecate", response_model=Dict)
async def deprecate_skill(
    skill_name: str,
    config: SkillDeprecateConfig,
    credentials=Depends(verify_bearer_token),
    auth_user: str = Depends(get_auth_user),
    registry: SkillsRegistry = Depends(get_registry),
):
    """Mark skill version as deprecated."""
    try:
        result = await registry.deprecate_skill_version(
            skill_name=skill_name,
            version=config.version,
            message=config.message,
            performed_by=auth_user
        )
        return {
            "status": "deprecated",
            "skill": skill_name,
            "version": result["version"],
            "action_id": result.get("action_id")
        }
    except Exception as e:
        logger.error(f"Failed to deprecate skill: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# ==================== HISTORY ENDPOINTS ====================

@router.get("/skills/{skill_name}/history", response_model=SkillActionHistory)
async def get_skill_history(
    skill_name: str,
    limit: int = 100,
    registry: SkillsRegistry = Depends(get_registry)
):
    """Get skill action history."""
    try:
        history = await registry.get_skill_history(skill_name, limit)
        if not history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Skill not found: {skill_name}"
            )

        # Add skill_name to each action for Pydantic validation
        actions = history["actions"]
        for action in actions:
            if "skill_name" not in action:
                action["skill_name"] = skill_name

        return SkillActionHistory(
            skill_name=history["skill_name"],
            total_count=history["total_count"],
            actions=actions
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get skill history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/skills/template/download")
async def download_skill_template():
    """
    Download a template directory structure for creating new agent skills.

    Returns a ZIP file with:
    - skills/example-skill/ directory with template files
    - README explaining the structure
    - Sample SKILL.md, implementation file, config, etc.

    Developers can extract and customize for their own skills.
    """
    import io
    import zipfile
    from fastapi.responses import StreamingResponse

    try:
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Main README
            readme_content = """# Agent Skills Repository Template

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
└── your-skill/
    ├── SKILL.md
    ├── skill_config.yaml
    ├── your_skill.py
    ├── requirements.txt
    └── README.md
```

## Getting Started

1. Copy the `example-skill/` directory and rename it to your skill name
2. Update SKILL.md with your skill's metadata
3. Implement your skill in the Python file
4. Update requirements.txt with your dependencies
5. Register your skill in the Skills Registry

## SKILL.md Format

Each skill must have a SKILL.md file with YAML frontmatter:

```yaml
---
name: your-skill
version: 1.0.0
description: What does your skill do?
category: category-name
author: your-email@company.com
tags:
  - tag1
  - tag2
capabilities:
  - capability1
  - capability2
dependencies:
  - dependency1>=1.0.0
  - dependency2>=2.0.0
---

# Your Skill

Write your skill documentation here in Markdown.
```

## Registration

Register your skills using the Skills Registry UI:
1. Individual registration: Enter skill name and GitHub repo URL
2. Bulk registration: Upload all skills from a repository
3. Auto-setup: Latest version is automatically published

## For More Information

See the example-skill/ directory for a complete working example.
"""

            zip_file.writestr("README.md", readme_content)

            # Example Skill SKILL.md
            skill_md = """---
name: example-skill
version: 1.0.0
description: This is an example skill template showing the proper structure and format
category: example
author: developer@company.com
tags:
  - example
  - template
  - tutorial
capabilities:
  - process_data
  - generate_output
dependencies:
  - pydantic>=2.0.0
---

# Example Skill

This is an example skill that demonstrates the proper structure for creating agent skills compatible with the Skills Registry.

## Features

- **Feature 1**: Description of what this skill can do
- **Feature 2**: Another capability
- **Feature 3**: More functionality

## Input

The skill accepts input in this format:

```json
{
  "input_field": "value",
  "another_field": 123
}
```

## Output

The skill returns output in this format:

```json
{
  "result": "output value",
  "status": "success"
}
```

## Usage Example

```python
from example_skill import ExampleSkill

skill = ExampleSkill()
result = skill.process({"input_field": "test"})
print(result)
```

## Configuration

See `skill_config.yaml` for skill configuration options.

## Tests

Run tests with:

```bash
pytest tests/
```
"""

            zip_file.writestr("skills/example-skill/SKILL.md", skill_md)

            # Example skill_config.yaml
            skill_config = """# Skill Configuration

# Metadata (also defined in SKILL.md frontmatter)
metadata:
  name: example-skill
  version: 1.0.0
  description: Example skill template
  category: example

# Skill-specific settings
settings:
  timeout: 30
  retry_attempts: 3
  log_level: INFO

# Feature flags
features:
  enable_caching: true
  enable_logging: true

# Integration settings
integrations:
  - type: api
    url: https://api.example.com
    auth: oauth2
"""

            zip_file.writestr("skills/example-skill/skill_config.yaml", skill_config)

            # Example Python implementation
            example_skill_py = '''"""
Example Skill Implementation

Shows the basic structure for implementing an agent skill.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ExampleInput(BaseModel):
    """Input schema for the example skill."""
    input_field: str = Field(..., description="Input field description")
    another_field: Optional[int] = Field(None, description="Optional numeric field")


class ExampleOutput(BaseModel):
    """Output schema for the example skill."""
    result: str = Field(..., description="Result of processing")
    status: str = Field(default="success", description="Processing status")


class ExampleSkill:
    """
    Example skill implementation.

    Replace this with your actual skill implementation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the skill.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.name = "example-skill"
        self.version = "1.0.0"

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and return output.

        Args:
            input_data: Input data to process

        Returns:
            Processed output as dictionary

        Raises:
            ValueError: If input is invalid
        """
        # Validate input
        try:
            validated_input = ExampleInput(**input_data)
        except ValueError as e:
            raise ValueError(f"Invalid input: {e}")

        # Process the input
        result = self._process_internal(validated_input)

        # Return output
        return {
            "result": result,
            "status": "success"
        }

    def _process_internal(self, input_data: ExampleInput) -> str:
        """
        Internal processing logic.

        Replace this with your actual implementation.
        """
        # Example: simple concatenation
        return f"Processed: {input_data.input_field}"


# For testing
if __name__ == "__main__":
    skill = ExampleSkill()

    # Example usage
    test_input = {
        "input_field": "test value",
        "another_field": 42
    }

    result = skill.process(test_input)
    print(f"Result: {result}")
'''

            zip_file.writestr("skills/example-skill/example_skill.py", example_skill_py)

            # Example requirements.txt
            requirements = """# Skill Dependencies

pydantic>=2.0.0
requests>=2.28.0

# Add your skill-specific dependencies here
# Example:
# pandas>=1.5.0
# numpy>=1.23.0
"""

            zip_file.writestr("skills/example-skill/requirements.txt", requirements)

            # Example skill README
            skill_readme = """# Example Skill

This is an example skill showing the template structure.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from example_skill import ExampleSkill

skill = ExampleSkill()
result = skill.process({
    "input_field": "your input",
    "another_field": 123
})
print(result)
```

## Testing

```bash
pytest tests/
```

## Configuration

Edit `skill_config.yaml` to customize the skill behavior.

## Development

1. Update the class in `example_skill.py`
2. Update the input/output schemas (ExampleInput, ExampleOutput)
3. Update SKILL.md with your skill metadata
4. Add tests in `tests/`
5. Update `requirements.txt` with dependencies
6. Register in Skills Registry

## Notes

- Ensure SKILL.md frontmatter is valid YAML
- Update skill name, version, and metadata in SKILL.md
- Version should follow semantic versioning (e.g., 1.0.0)
- Include meaningful description and tags
- List all Python dependencies in requirements.txt
"""

            zip_file.writestr("skills/example-skill/README.md", skill_readme)

            # Example test file
            test_file = '''"""
Tests for example skill.

Run with: pytest tests/test_example.py
"""

import pytest
from example_skill import ExampleSkill, ExampleInput, ExampleOutput


class TestExampleSkill:
    """Test cases for ExampleSkill."""

    @pytest.fixture
    def skill(self):
        """Create skill instance for testing."""
        return ExampleSkill()

    def test_skill_initialization(self, skill):
        """Test that skill initializes correctly."""
        assert skill.name == "example-skill"
        assert skill.version == "1.0.0"

    def test_process_valid_input(self, skill):
        """Test processing with valid input."""
        input_data = {
            "input_field": "test",
            "another_field": 42
        }

        result = skill.process(input_data)

        assert result["status"] == "success"
        assert "Processed" in result["result"]

    def test_process_minimal_input(self, skill):
        """Test processing with minimal required input."""
        input_data = {"input_field": "minimal"}

        result = skill.process(input_data)

        assert result["status"] == "success"

    def test_process_invalid_input(self, skill):
        """Test that invalid input raises error."""
        input_data = {"invalid_field": "test"}

        with pytest.raises(ValueError):
            skill.process(input_data)


def test_input_schema_validation():
    """Test input schema validation."""
    # Valid input
    valid = ExampleInput(input_field="test", another_field=123)
    assert valid.input_field == "test"

    # Missing required field
    with pytest.raises(ValueError):
        ExampleInput(another_field=123)
'''

            zip_file.writestr("skills/example-skill/tests/test_example.py", test_file)

            # Create empty __init__.py for tests
            zip_file.writestr("skills/example-skill/tests/__init__.py", "")

        # Prepare response
        zip_buffer.seek(0)

        return StreamingResponse(
            iter([zip_buffer.getvalue()]),
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=skill-template.zip"
            }
        )

    except Exception as e:
        logger.error(f"Failed to generate template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate template: {str(e)}"
        )
