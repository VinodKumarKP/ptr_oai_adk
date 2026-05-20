"""
Git / discovery routes — discover, bulk-register, preview versions,
import a version, and refresh available versions.
"""

import logging
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status

from oai_skills_registry.dependencies import get_auth_user, get_registry, verify_bearer_token
from oai_skills_registry.models import (
    BulkRegistrationRequest,
    BulkRegistrationResult,
    SkillDiscoveryResult,
    SkillImportConfig,
)
from oai_skills_registry.services.skills_registry import SkillsRegistry

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discovery (no auth — public repos)
# ---------------------------------------------------------------------------

@router.post("/skills/discover", response_model=SkillDiscoveryResult)
async def discover_skills(
    request_data: Dict,
    registry: SkillsRegistry = Depends(get_registry),
):
    """
    Discover all skills available in a GitHub repository.

    Scans the ``skills/`` directory and returns metadata for each skill found.
    No authentication required for public repositories.
    """
    try:
        git_repository_url = request_data.get("git_repository_url")
        if not git_repository_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="git_repository_url is required",
            )

        result = await registry.discover_skills_from_git(
            git_repository_url=git_repository_url,
            auth_token=request_data.get("auth_token"),
        )

        return SkillDiscoveryResult(
            git_repository_url=result["git_repository_url"],
            total_found=result["total_found"],
            available_to_register=result["available_to_register"],
            already_registered=result["already_registered"],
            invalid=result["invalid"],
            skills=result["skills"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to discover skills: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.post("/skills/preview-versions", response_model=Dict)
async def preview_skill_versions(
    request_data: Dict,
    registry: SkillsRegistry = Depends(get_registry),
):
    """
    Preview available versions from a GitHub repository before registering a skill.

    No authentication required for public repositories.
    """
    try:
        git_repository_url = request_data.get("git_repository_url")
        skill_name = request_data.get("skill_name")

        if not git_repository_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="git_repository_url is required",
            )
        if not skill_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="skill_name is required",
            )

        versions = await registry.list_skill_versions_from_url(
            skill_name=skill_name,
            git_repository_url=git_repository_url,
        )

        return {
            "status": "success",
            "skill": skill_name,
            "versions": versions,
            "versions_found": len(versions) if versions else 0,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to preview skill versions: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Bulk registration (auth required)
# ---------------------------------------------------------------------------

@router.post("/skills/register-bulk", response_model=BulkRegistrationResult)
async def register_skills_bulk(
    request: BulkRegistrationRequest,
    _auth: bool = Depends(verify_bearer_token),
    auth_user: str = Depends(get_auth_user),
    registry: SkillsRegistry = Depends(get_registry),
):
    """
    Register multiple skills from a GitHub repository in one operation.

    Author defaults to the authenticated user when not specified.
    """
    try:
        result = await registry.register_skills_bulk(
            git_repository_url=request.git_repository_url,
            skill_names=request.skill_names,
            author=request.author or auth_user,
        )
        return BulkRegistrationResult(
            total_registered=result["total_registered"],
            successful=result["successful"],
            failed=result["failed"],
        )
    except Exception as exc:
        logger.error("Failed to bulk-register skills: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


# ---------------------------------------------------------------------------
# Per-skill Git operations (auth required)
# ---------------------------------------------------------------------------

@router.post("/skills/{skill_name}/import-from-git", response_model=Dict)
async def import_skill_from_git(
    skill_name: str,
    config: SkillImportConfig,
    _auth: bool = Depends(verify_bearer_token),
    auth_user: str = Depends(get_auth_user),
    registry: SkillsRegistry = Depends(get_registry),
):
    """
    Import a skill version from GitHub.

    Supports two methods:

    1. ``git_source_id`` — use a registered Git source.
    2. ``git_repository_url`` — use a direct GitHub URL.

    Fetches ``SKILL.md`` and ``skill_config.yaml`` and creates an unpublished version.
    """
    try:
        if config.git_source_id:
            source_config = await registry.db_logger.get_git_source(config.git_source_id)
            if not source_config:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Git source not found: {config.git_source_id}",
                )
        elif config.git_repository_url:
            source_config = {
                "git_provider": "github",
                "repository": "",
                "branch": "main",
                "auth_token": None,
                "git_url": config.git_repository_url,
                "_direct_url": config.git_repository_url,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either git_source_id or git_repository_url must be provided",
            )

        result = await registry.import_skill_from_git(
            skill_name=skill_name,
            source_config=source_config,
            git_tag=config.git_tag,
            performed_by=auth_user,
        )

        return {
            "status": "imported",
            "skill": skill_name,
            "version": result["version"],
            "preview": result["preview"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to import skill: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/skills/{skill_name}/refresh-versions", response_model=Dict)
async def refresh_skill_versions(
    skill_name: str,
    _auth: bool = Depends(verify_bearer_token),
    registry: SkillsRegistry = Depends(get_registry),
):
    """Fetch and return the available versions from the skill's GitHub repository."""
    try:
        skill = await registry.db_logger.get_skill(skill_name)
        if not skill:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Skill not found: {skill_name}",
            )

        git_repository_url = skill.get("git_repository_url")
        if not git_repository_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Skill does not have a Git repository configured",
            )

        versions = await registry.list_skill_versions_from_url(
            skill_name=skill_name,
            git_repository_url=git_repository_url,
        )

        return {
            "status": "success",
            "skill": skill_name,
            "versions": versions,
            "versions_found": len(versions) if versions else 0,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to refresh skill versions: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
