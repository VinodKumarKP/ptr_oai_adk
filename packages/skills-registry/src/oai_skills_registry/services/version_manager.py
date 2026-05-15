"""
Version Manager
Handles skill version lifecycle: publish, upgrade, downgrade, deprecate, delete.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from oai_skills_registry.services.db.database_logger import SkillsDatabaseLogger
from oai_skills_registry.services.utils import version_matches_constraint


class VersionManager:
    """Manages skill version lifecycle operations."""

    def __init__(self, db_logger: SkillsDatabaseLogger, logger: Optional[logging.Logger] = None):
        self.db_logger = db_logger
        self.logger = logger or logging.getLogger(__name__)

    async def publish_skill_version(self, skill_name: str, version: str,
                                    message: Optional[str],
                                    performed_by: str) -> Dict[str, Any]:
        """Publish unpublished skill version.
        Makes it available for agents to use.

        Args:
            skill_name: Name of the skill
            version: Version to publish
            message: Optional message describing the publish action
            performed_by: User performing the action

        Returns:
            Result with skill_id, version, action_id, and status
        """
        skill = await self.db_logger.get_skill(skill_name)
        if not skill:
            raise ValueError(f"Skill not found: {skill_name}")

        skill_version = await self.db_logger.get_skill_version(skill["id"], version)
        if not skill_version:
            raise ValueError(f"Version not found: {version}")

        if skill_version["status"] != "draft":
            raise ValueError(f"Cannot publish version with status: {skill_version['status']}")

        # Update version status to published
        now = datetime.now(timezone.utc)
        await self.db_logger.update_skill_version_status(
            skill_version["id"],
            status="published",
            published_by=performed_by,
            published_at=now
        )

        # Update skill's current version
        await self.db_logger.update_skill_current_version(skill_name, version)

        # Create action history
        action = await self.db_logger.log_skill_action(
            skill_id=skill["id"],
            action="publish",
            from_version=None,
            to_version=version,
            performed_by=performed_by,
            message=message
        )

        self.logger.info(f"Published skill {skill_name} version {version}")

        return {
            "skill_id": skill["id"],
            "version": version,
            "action_id": action.get("id"),
            "status": "published"
        }

    async def upgrade_skill_version(self, skill_name: str, to_version: str,
                                    message: Optional[str],
                                    performed_by: str) -> Dict[str, Any]:
        """Upgrade skill to new version.
        Updates agents based on their version constraints.

        Args:
            skill_name: Name of the skill
            to_version: Version to upgrade to
            message: Optional message describing the upgrade
            performed_by: User performing the action

        Returns:
            Result with from/to versions, affected agents, and action_id
        """
        skill = await self.db_logger.get_skill(skill_name)
        if not skill:
            raise ValueError(f"Skill not found: {skill_name}")

        from_version = skill.get("current_version")

        skill_version = await self.db_logger.get_skill_version(skill["id"], to_version)
        if not skill_version or skill_version["status"] != "published":
            raise ValueError(f"Version not available: {to_version}")

        # Update skill current version
        await self.db_logger.update_skill_current_version(skill_name, to_version)

        # Get affected agents (would need to implement version constraint matching)
        agent_mappings = await self.db_logger.get_agent_skills(skill["id"])
        affected_agents = []

        for mapping in agent_mappings:
            constraint = mapping.get("version_constraint", "latest")
            if version_matches_constraint(to_version, constraint):
                await self.db_logger.update_agent_skill_version(mapping["id"], to_version)
                affected_agents.append(mapping["agent_id"])

        # Create action history
        action = await self.db_logger.log_skill_action(
            skill_id=skill["id"],
            action="upgrade",
            from_version=from_version,
            to_version=to_version,
            performed_by=performed_by,
            message=message
        )

        self.logger.info(f"Upgraded skill {skill_name} from {from_version} to {to_version}")

        return {
            "from_version": from_version,
            "to_version": to_version,
            "agents_affected": affected_agents,
            "action_id": action.get("id")
        }

    async def downgrade_skill_version(self, skill_name: str, to_version: str,
                                      message: Optional[str],
                                      performed_by: str) -> Dict[str, Any]:
        """Downgrade skill to previous version.

        Args:
            skill_name: Name of the skill
            to_version: Version to downgrade to
            message: Optional message describing the downgrade
            performed_by: User performing the action

        Returns:
            Result with from/to versions and action_id
        """
        skill = await self.db_logger.get_skill(skill_name)
        if not skill:
            raise ValueError(f"Skill not found: {skill_name}")

        from_version = skill.get("current_version")

        skill_version = await self.db_logger.get_skill_version(skill["id"], to_version)
        if not skill_version:
            raise ValueError(f"Version not found: {to_version}")

        # Update skill current version
        await self.db_logger.update_skill_current_version(skill_name, to_version)

        # Create action history
        action = await self.db_logger.log_skill_action(
            skill_id=skill["id"],
            action="downgrade",
            from_version=from_version,
            to_version=to_version,
            performed_by=performed_by,
            message=message
        )

        self.logger.info(f"Downgraded skill {skill_name} from {from_version} to {to_version}")

        return {
            "from_version": from_version,
            "to_version": to_version,
            "action_id": action.get("id")
        }

    async def deprecate_skill_version(self, skill_name: str, version: str,
                                      message: Optional[str],
                                      performed_by: str) -> Dict[str, Any]:
        """Mark skill version as deprecated.

        Args:
            skill_name: Name of the skill
            version: Version to deprecate
            message: Optional deprecation message
            performed_by: User performing the action

        Returns:
            Result with version and action_id
        """
        skill = await self.db_logger.get_skill(skill_name)
        if not skill:
            raise ValueError(f"Skill not found: {skill_name}")

        skill_version = await self.db_logger.get_skill_version(skill["id"], version)
        if not skill_version:
            raise ValueError(f"Version not found: {version}")

        now = datetime.now(timezone.utc)

        # Update version status to deprecated
        await self.db_logger.update_skill_version_status(
            skill_version["id"],
            status="deprecated",
            deprecated_at=now
        )

        # Create action history
        action = await self.db_logger.log_skill_action(
            skill_id=skill["id"],
            action="deprecate",
            from_version=None,
            to_version=version,
            performed_by=performed_by,
            message=message
        )

        self.logger.info(f"Deprecated skill {skill_name} version {version}")

        return {
            "version": version,
            "action_id": action.get("id"),
            "message": message
        }

    async def delete_skill(self, skill_name: str, performed_by: str) -> Dict[str, Any]:
        """Delete entire skill and all versions.

        Args:
            skill_name: Name of the skill to delete
            performed_by: User performing the action

        Returns:
            Result with skill name and action_id
        """
        skill = await self.db_logger.get_skill(skill_name)
        if not skill:
            raise ValueError(f"Skill not found: {skill_name}")

        await self.db_logger.delete_skill(skill_name)

        # Create action history
        action = await self.db_logger.log_skill_action(
            skill_id=skill["id"],
            action="delete",
            from_version=None,
            to_version=None,
            performed_by=performed_by
        )

        self.logger.info(f"Deleted skill {skill_name}")

        return {
            "skill": skill_name,
            "action_id": action.get("id"),
            "status": "deleted"
        }
