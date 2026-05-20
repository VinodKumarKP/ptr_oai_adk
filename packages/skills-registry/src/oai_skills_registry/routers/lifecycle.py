"""
Skill lifecycle routes — publish, upgrade, downgrade, deprecate, and history.
"""

import logging
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status

from oai_skills_registry.dependencies import get_auth_user, get_registry, verify_bearer_token
from oai_skills_registry.models import (
    SkillActionHistory,
    SkillDeprecateConfig,
    SkillDowngradeConfig,
    SkillPublishConfig,
    SkillUpgradeConfig,
)
from oai_skills_registry.services.skills_registry import SkillsRegistry

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/skills/{skill_name}/publish", response_model=Dict)
async def publish_skill(
    skill_name: str,
    config: SkillPublishConfig,
    _auth: bool = Depends(verify_bearer_token),
    auth_user: str = Depends(get_auth_user),
    registry: SkillsRegistry = Depends(get_registry),
):
    """Publish an unpublished skill version, making it available for agents."""
    try:
        result = await registry.publish_skill_version(
            skill_name=skill_name,
            version=config.version,
            message=config.message,
            performed_by=auth_user,
        )
        return {
            "status": "published",
            "skill": skill_name,
            "version": result["version"],
            "action_id": result.get("action_id"),
        }
    except Exception as exc:
        logger.error("Failed to publish skill: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/skills/{skill_name}/upgrade", response_model=Dict)
async def upgrade_skill(
    skill_name: str,
    config: SkillUpgradeConfig,
    _auth: bool = Depends(verify_bearer_token),
    auth_user: str = Depends(get_auth_user),
    registry: SkillsRegistry = Depends(get_registry),
):
    """Upgrade a skill to a newer version."""
    try:
        result = await registry.upgrade_skill_version(
            skill_name=skill_name,
            to_version=config.to_version,
            message=config.message,
            performed_by=auth_user,
        )
        return {
            "status": "upgraded",
            "skill": skill_name,
            "from_version": result["from_version"],
            "to_version": result["to_version"],
            "agents_affected": result["agents_affected"],
            "action_id": result.get("action_id"),
        }
    except Exception as exc:
        logger.error("Failed to upgrade skill: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/skills/{skill_name}/downgrade", response_model=Dict)
async def downgrade_skill(
    skill_name: str,
    config: SkillDowngradeConfig,
    _auth: bool = Depends(verify_bearer_token),
    auth_user: str = Depends(get_auth_user),
    registry: SkillsRegistry = Depends(get_registry),
):
    """Downgrade a skill to a previous version."""
    try:
        result = await registry.downgrade_skill_version(
            skill_name=skill_name,
            to_version=config.to_version,
            message=config.message,
            performed_by=auth_user,
        )
        return {
            "status": "downgraded",
            "skill": skill_name,
            "from_version": result["from_version"],
            "to_version": result["to_version"],
            "action_id": result.get("action_id"),
        }
    except Exception as exc:
        logger.error("Failed to downgrade skill: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/skills/{skill_name}/deprecate", response_model=Dict)
async def deprecate_skill(
    skill_name: str,
    config: SkillDeprecateConfig,
    _auth: bool = Depends(verify_bearer_token),
    auth_user: str = Depends(get_auth_user),
    registry: SkillsRegistry = Depends(get_registry),
):
    """Mark a skill version as deprecated."""
    try:
        result = await registry.deprecate_skill_version(
            skill_name=skill_name,
            version=config.version,
            message=config.message,
            performed_by=auth_user,
        )
        return {
            "status": "deprecated",
            "skill": skill_name,
            "version": result["version"],
            "action_id": result.get("action_id"),
        }
    except Exception as exc:
        logger.error("Failed to deprecate skill: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/skills/{skill_name}/history", response_model=SkillActionHistory)
async def get_skill_history(
    skill_name: str,
    limit: int = 100,
    _auth: bool = Depends(verify_bearer_token),
    registry: SkillsRegistry = Depends(get_registry),
):
    """Get the complete action history for a skill."""
    try:
        history = await registry.get_skill_history(skill_name, limit)
        if not history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Skill not found: {skill_name}",
            )

        actions = history["actions"]
        for action in actions:
            if "skill_name" not in action:
                action["skill_name"] = skill_name

        return SkillActionHistory(
            skill_name=history["skill_name"],
            total_count=history["total_count"],
            actions=actions,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get skill history: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
