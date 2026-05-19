"""
Skill Queries
Handles skill information retrieval: details, history, versions from git.
"""

import logging
from typing import Optional, List, Dict, Any

from oai_skills_registry.services.db.database_logger import SkillsDatabaseLogger
from oai_skills_registry.services.git_provider import GitProvider
from oai_skills_registry.services.utils import parse_skill_md


class SkillQueries:
    """Handles skill queries and information retrieval."""

    def __init__(self, db_logger: SkillsDatabaseLogger, git_providers: Dict[str, GitProvider],
                 logger: Optional[logging.Logger] = None):
        self.db_logger = db_logger
        self.git_providers = git_providers
        self.logger = logger or logging.getLogger(__name__)

    async def get_skill_details(self, skill_name: str) -> Dict[str, Any]:
        """Get full skill details with all versions.

        Args:
            skill_name: Name of the skill

        Returns:
            Dictionary with skill info and all versions
        """
        skill = await self.db_logger.get_skill(skill_name)
        if not skill:
            return None

        versions = await self.db_logger.get_skill_versions(skill["id"])

        return {
            "skill": skill,
            "versions": versions,
            "version_count": len(versions)
        }

    async def get_skill_history(self, skill_name: str, limit: int = 100) -> Dict[str, Any]:
        """Get action history for a skill.

        Args:
            skill_name: Name of the skill
            limit: Maximum number of actions to return

        Returns:
            Dictionary with skill name, total count, and actions
        """
        skill = await self.db_logger.get_skill(skill_name)
        if not skill:
            return None

        actions = await self.db_logger.get_skill_actions(skill["id"], limit)
        count = await self.db_logger.get_skill_action_count(skill["id"])

        return {
            "skill_name": skill_name,
            "total_count": count,
            "actions": actions
        }

    async def list_git_versions(self, skill_name: str, source_id: int) -> List[Dict]:
        """List available versions in Git (tags).

        Args:
            skill_name: Name of the skill
            source_id: ID of the registered git source

        Returns:
            List of available versions with metadata
        """
        git_source = await self.db_logger.get_git_source(source_id)
        if not git_source:
            raise ValueError(f"Git source not found: {source_id}")

        provider = self.git_providers.get(git_source["git_provider"])
        if not provider:
            raise ValueError(f"Unknown Git provider: {git_source['git_provider']}")

        # Get tags from Git
        try:
            tags = await provider.get_tags(
                repo=git_source["repository"],
                auth_token=git_source.get("auth_token")
            )
        except Exception as e:
            self.logger.warning(f"Failed to get tags: {e}")
            return []

        versions = []
        auth_token = git_source.get("auth_token")
        repository = git_source["repository"]

        for tag in tags:
            tag_name = tag.get("name", "")
            try:
                # Use fast GitHub Contents API when available; fall back to git clone.
                if hasattr(provider, "fetch_skill_md_from_api"):
                    files = await provider.fetch_skill_md_from_api(
                        repo=repository,
                        tag=tag_name,
                        skill_name=skill_name,
                        auth_token=auth_token,
                    )
                else:
                    files = await provider.fetch_skill_files(
                        repo=repository,
                        branch=None,
                        tag=tag_name,
                        skill_name=skill_name,
                        auth_token=auth_token,
                    )

                skill_data = parse_skill_md(files["SKILL.md"])
                if skill_data.get("name") == skill_name:
                    versions.append({
                        "version": skill_data.get("version"),
                        "git_tag": tag_name,
                        "commit_sha": tag.get("commit", {}).get("sha", ""),
                        "created_at": tag.get("created_at"),
                        "message": tag.get("message", ""),
                        "author": tag.get("tagger", {}).get("name", "") or skill_data.get("author", ""),
                    })
            except FileNotFoundError:
                self.logger.debug("Skill %s not found at tag %s, skipping", skill_name, tag_name)
                continue
            except Exception as e:
                self.logger.debug("Failed to parse version from tag %s: %s", tag_name, e)
                continue

        return sorted(versions, key=lambda x: x.get("created_at") or "", reverse=True)
