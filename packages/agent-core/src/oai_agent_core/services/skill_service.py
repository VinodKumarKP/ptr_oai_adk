"""Skill discovery and management service."""

import logging
from typing import Dict, Any, Optional, List


class SkillService:
    """Service for skill discovery and management.

    Handles:
    - Initializing skill registry
    - Discovering local skills
    - Pulling remote skills from registry
    - Managing skill lifecycle
    """

    def __init__(self, project_root: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the skill service.

        Args:
            project_root: Project root directory
            logger: Optional logger instance
        """
        self.project_root = project_root
        self.logger = logger or logging.getLogger(__name__)
        self._skill_registry = None

    def create_skill_registry(self, skills_config: Dict[str, Any]) -> Any:
        """Create and initialize a skill registry.

        Args:
            skills_config: Skills configuration from agent config

        Returns:
            SkillRegistry instance, or None if not configured
        """
        if not skills_config:
            return None

        try:
            import os
            from oai_agent_core.components.skills.skill_registry import SkillRegistry

            skill_dir = skills_config.get('skill_dir')
            registry_config = skills_config.get('registry', {})

            registry = SkillRegistry(
                logger=self.logger,
                project_root=self.project_root,
                registry_url=registry_config.get('url') or os.environ.get('SKILLS_REGISTRY_URL'),
                auth_token=registry_config.get('auth_token') or os.environ.get('SKILLS_REGISTRY_AUTH_TOKEN'),
                skills_cache_dir=skill_dir
            )

            self._skill_registry = registry
            self.logger.debug("Created skill registry")
            return registry
        except Exception as e:
            self.logger.error(f"Failed to create skill registry: {e}")
            return None

    async def pull_required_skills(self, registry: Any,
                                   required_skills: List[str]) -> int:
        """Pull required skills from remote registry.

        Args:
            registry: Skill registry instance
            required_skills: List of skill names to pull

        Returns:
            Number of skills successfully pulled
        """
        if not registry or not required_skills:
            return 0

        pulled_count = 0
        for skill_name in required_skills:
            try:
                success = await registry.pull_skill(skill_name)
                if success:
                    pulled_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to pull skill '{skill_name}': {e}")

        self.logger.info(f"Pulled {pulled_count} of {len(required_skills)} skills")
        return pulled_count

    def discover_skills(self, registry: Any, skill_dir: str) -> int:
        """Discover local skills in a directory.

        Args:
            registry: Skill registry instance
            skill_dir: Directory to search for skills

        Returns:
            Number of skills discovered
        """
        if not registry or not skill_dir:
            return 0

        try:
            registry.discover_skills(skills_dir=skill_dir)
            skill_count = len(registry.skills)
            self.logger.info(f"Discovered {skill_count} skills")
            return skill_count
        except Exception as e:
            self.logger.error(f"Failed to discover skills: {e}")
            return 0
