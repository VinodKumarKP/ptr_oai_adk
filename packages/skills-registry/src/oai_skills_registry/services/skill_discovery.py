"""
Skill Discovery
Handles skill discovery from GitHub repositories.
"""

import logging
import re
import base64
from typing import Optional, Dict, Any

import aiohttp

from oai_skills_registry.services.db.database_logger import SkillsDatabaseLogger
from oai_skills_registry.services.utils import parse_skill_md, parse_github_url


class SkillDiscovery:
    """Handles skill discovery from Git repositories."""

    def __init__(self, db_logger: SkillsDatabaseLogger, logger: Optional[logging.Logger] = None):
        self.db_logger = db_logger
        self.logger = logger or logging.getLogger(__name__)

    async def discover_skills_from_git(self, git_repository_url: str,
                                      auth_token: Optional[str] = None) -> Dict[str, Any]:
        """Discover all skills available in a GitHub repository.

        Scans the 'skills/' directory in the repository and returns metadata for each skill found.

        Args:
            git_repository_url: Full GitHub repository URL
            auth_token: Optional GitHub authentication token

        Returns:
            Dictionary with discovered skills and their status
        """
        # Parse GitHub URL
        try:
            repository = parse_github_url(git_repository_url)
        except ValueError as e:
            raise ValueError(f"Failed to parse GitHub URL: {e}")

        discovered_skills = []

        try:
            # Use GitHub API to list contents of skills/ directory
            async with aiohttp.ClientSession() as session:
                headers = {}
                if auth_token:
                    headers["Authorization"] = f"token {auth_token}"

                # Get contents of skills/ directory
                url = f"https://api.github.com/repos/{repository}/contents/skills"

                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        raise ValueError(f"Failed to access skills directory: HTTP {resp.status}")

                    contents = await resp.json()

                    # Filter for directories
                    if isinstance(contents, dict):
                        # Single file/dir response
                        contents = [contents]

                    skill_dirs = [item for item in contents if item.get("type") == "dir"]

                    # For each skill directory, fetch SKILL.md
                    for skill_dir in skill_dirs:
                        skill_name = skill_dir.get("name")
                        discovered_skills.append(
                            await self._fetch_skill_metadata(
                                session, repository, skill_name, headers
                            )
                        )

        except Exception as e:
            self.logger.error(f"Failed to discover skills: {e}")
            raise

        # Count by status
        available = [s for s in discovered_skills if s["status"] == "available"]
        already_registered = [s for s in discovered_skills if s["status"] == "already_registered"]
        invalid = [s for s in discovered_skills if s["status"] == "invalid"]

        return {
            "git_repository_url": git_repository_url,
            "total_found": len(discovered_skills),
            "available_to_register": len(available),
            "already_registered": len(already_registered),
            "invalid": len(invalid),
            "skills": discovered_skills
        }

    async def _fetch_skill_metadata(self, session: aiohttp.ClientSession, repository: str,
                                    skill_name: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """Fetch metadata for a single skill from GitHub.

        Args:
            session: aiohttp session
            repository: Repository in owner/repo format
            skill_name: Name of the skill
            headers: HTTP headers (includes auth if needed)

        Returns:
            Dictionary with skill metadata and status
        """
        try:
            # Fetch SKILL.md from this skill
            skill_md_url = f"https://api.github.com/repos/{repository}/contents/skills/{skill_name}/SKILL.md"

            async with session.get(skill_md_url, headers=headers) as skill_resp:
                if skill_resp.status == 200:
                    skill_content = await skill_resp.json()
                    # Decode base64 content
                    content_str = base64.b64decode(skill_content.get("content", "")).decode("utf-8")

                    # Parse SKILL.md
                    skill_data = parse_skill_md(content_str)

                    # Check if already registered
                    existing = await self.db_logger.get_skill(skill_name)
                    status = "already_registered" if existing else "available"

                    return {
                        "name": skill_data.get("name", skill_name),
                        "version": skill_data.get("version", "0.0.1"),
                        "description": skill_data.get("description", ""),
                        "category": skill_data.get("category", ""),
                        "author": skill_data.get("author", ""),
                        "tags": skill_data.get("tags", []),
                        "status": status,
                        "path_in_repo": f"skills/{skill_name}"
                    }
        except Exception as e:
            self.logger.warning(f"Failed to parse skill {skill_name}: {e}")

        return {
            "name": skill_name,
            "version": "unknown",
            "description": "Failed to parse",
            "category": "unknown",
            "author": "unknown",
            "tags": [],
            "status": "invalid",
            "path_in_repo": f"skills/{skill_name}"
        }
