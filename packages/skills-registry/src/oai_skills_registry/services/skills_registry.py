"""
Skills Registry Service
Main orchestrator for skill lifecycle management and Git integration.

This module acts as a facade that coordinates multiple specialized managers:
- SkillImporter: Handles importing skills from Git repositories
- VersionManager: Manages skill version lifecycle (publish, upgrade, downgrade, etc.)
- SkillQueries: Retrieves skill information and history
- SkillDiscovery: Discovers skills in repositories
- GitHubProvider: Handles Git provider operations
"""

import logging
from typing import Optional, List, Dict, Any

from oai_skills_registry.services.db.database_logger import SkillsDatabaseLogger
from oai_skills_registry.services.git_provider import GitHubProvider
from oai_skills_registry.services.skill_importer import SkillImporter
from oai_skills_registry.services.version_manager import VersionManager
from oai_skills_registry.services.skill_queries import SkillQueries
from oai_skills_registry.services.skill_discovery import SkillDiscovery


class SkillsRegistry:
    """Main skills registry service - orchestrates all skill management operations."""

    def __init__(self, db_logger: SkillsDatabaseLogger, logger: Optional[logging.Logger] = None):
        self.db_logger = db_logger
        self.logger = logger or logging.getLogger(__name__)

        # Initialize Git providers
        self.git_providers = {
            "github": GitHubProvider(),
            # "gitlab": GitLabProvider(),
            # "gitea": GiteaProvider(),
        }

        # Initialize specialized managers
        self.importer = SkillImporter(db_logger, self.git_providers, self.logger)
        self.version_manager = VersionManager(db_logger, self.logger)
        self.queries = SkillQueries(db_logger, self.git_providers, self.logger)
        self.discovery = SkillDiscovery(db_logger, self.logger)

    # ========== Skill Import Operations ==========

    async def register_git_source(self, source_config: Dict) -> Dict:
        """Register a new Git repository as skill source.

        Delegates to SkillImporter.
        """
        return await self.importer.register_git_source(source_config)

    async def import_skill_from_git(self, skill_name: str, source_config: Dict,
                                     git_tag: Optional[str] = None,
                                     performed_by: str = "system") -> Dict[str, Any]:
        """Import skill version from Git source.

        Supports both registered sources and direct GitHub URLs.
        Creates unpublished version record.
        Delegates to SkillImporter.
        """
        return await self.importer.import_skill_from_git(skill_name, source_config, git_tag, performed_by)

    # ========== Skill Version Lifecycle Operations ==========

    async def publish_skill_version(self, skill_name: str, version: str,
                                    message: Optional[str] = None,
                                    performed_by: str = "system") -> Dict[str, Any]:
        """Publish unpublished skill version.

        Makes it available for agents to use.
        Delegates to VersionManager.
        """
        return await self.version_manager.publish_skill_version(skill_name, version, message, performed_by)

    async def upgrade_skill_version(self, skill_name: str, to_version: str,
                                    message: Optional[str] = None,
                                    performed_by: str = "system") -> Dict[str, Any]:
        """Upgrade skill to new version.

        Updates agents based on their version constraints.
        Delegates to VersionManager.
        """
        return await self.version_manager.upgrade_skill_version(skill_name, to_version, message, performed_by)

    async def downgrade_skill_version(self, skill_name: str, to_version: str,
                                      message: Optional[str] = None,
                                      performed_by: str = "system") -> Dict[str, Any]:
        """Downgrade skill to previous version.

        Delegates to VersionManager.
        """
        return await self.version_manager.downgrade_skill_version(skill_name, to_version, message, performed_by)

    async def deprecate_skill_version(self, skill_name: str, version: str,
                                      message: Optional[str] = None,
                                      performed_by: str = "system") -> Dict[str, Any]:
        """Mark skill version as deprecated.

        Delegates to VersionManager.
        """
        return await self.version_manager.deprecate_skill_version(skill_name, version, message, performed_by)

    async def delete_skill(self, skill_name: str, performed_by: str = "system") -> Dict[str, Any]:
        """Delete entire skill and all versions.

        Delegates to VersionManager.
        """
        return await self.version_manager.delete_skill(skill_name, performed_by)

    # ========== Skill Information Queries ==========

    async def get_skill_details(self, skill_name: str) -> Dict[str, Any]:
        """Get full skill details with all versions.

        Delegates to SkillQueries.
        """
        return await self.queries.get_skill_details(skill_name)

    async def get_skill_history(self, skill_name: str, limit: int = 100) -> Dict[str, Any]:
        """Get action history for a skill.

        Delegates to SkillQueries.
        """
        return await self.queries.get_skill_history(skill_name, limit)

    async def list_git_versions(self, skill_name: str, source_id: int) -> List[Dict]:
        """List available versions in Git (tags) from a registered source.

        Delegates to SkillQueries.
        """
        return await self.queries.list_git_versions(skill_name, source_id)

    async def list_skill_versions_from_url(self, skill_name: str, git_repository_url: str,
                                          auth_token: Optional[str] = None) -> List[Dict]:
        """List available versions from a GitHub repository URL.

        This method fetches versions directly from a GitHub repository without requiring
        a registered git source. Useful for refresh operations and direct imports.

        Args:
            skill_name: Name of the skill to match in SKILL.md
            git_repository_url: Full GitHub repository URL (e.g., https://github.com/owner/repo.git)
            auth_token: Optional GitHub authentication token

        Returns:
            List of available versions with metadata
        """
        import re
        from oai_skills_registry.services.utils import parse_skill_md

        # Parse GitHub URL to extract repository
        # Expected format: https://github.com/{owner}/{repo}.git or https://github.com/{owner}/{repo}
        try:
            # Extract owner/repo from URL
            match = re.search(r'github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$', git_repository_url)
            if not match:
                self.logger.error(f"Invalid GitHub URL format: {git_repository_url}")
                return []

            owner, repo = match.groups()
            repository = f"{owner}/{repo}"
        except Exception as e:
            self.logger.error(f"Failed to parse GitHub URL: {e}")
            return []

        # Use GitHub provider
        provider = self.git_providers.get("github")
        if not provider:
            self.logger.error("GitHub provider not available")
            return []

        # Get tags from GitHub
        try:
            tags = await provider.get_tags(
                repo=repository,
                auth_token=auth_token
            )
        except Exception as e:
            self.logger.warning(f"Failed to get tags from {repository}: {e}")
            return []

        versions = []

        for tag in tags:
            try:
                # Try to fetch SKILL.md from this tag
                files = await provider.fetch_skill_files(
                    repo=repository,
                    branch=None,
                    tag=tag.get("name"),
                    skill_name=skill_name,
                    auth_token=auth_token
                )
                skill_data = parse_skill_md(files["SKILL.md"])
                if skill_data.get("name") == skill_name:
                    version_info = {
                        "version": skill_data.get("version"),
                        "git_tag": tag.get("name"),
                        "commit_sha": tag.get("commit", {}).get("sha", ""),
                        "created_at": tag.get("created_at"),
                        "message": tag.get("message", ""),
                        "tagger": tag.get("tagger", {}).get("name", ""),
                        # Include metadata from SKILL.md frontmatter
                        "description": skill_data.get("description", ""),
                        "category": skill_data.get("category", ""),
                        "author": skill_data.get("author", ""),
                        "tags": skill_data.get("tags", [])
                    }
                    self.logger.info(f"Found {skill_name} version {version_info.get('version')}: description={version_info.get('description')}, category={version_info.get('category')}")
                    versions.append(version_info)
            except Exception as e:
                self.logger.error(f"Failed to parse version from tag {tag.get('name')}: {e}")
                continue

        return sorted(versions, key=lambda x: x.get("created_at", ""), reverse=True)

    # ========== Skill Discovery Operations ==========

    async def discover_skills_from_git(self, git_repository_url: str,
                                      auth_token: Optional[str] = None) -> Dict[str, Any]:
        """Discover all skills available in a GitHub repository.

        Scans the 'skills/' directory in the repository and returns metadata for each skill found.
        Delegates to SkillDiscovery.

        Args:
            git_repository_url: Full GitHub repository URL
            auth_token: Optional GitHub authentication token

        Returns:
            Dictionary with discovered skills and their status
        """
        return await self.discovery.discover_skills_from_git(git_repository_url, auth_token)

    async def register_skills_bulk(self, git_repository_url: str, skill_names: List[str],
                                  author: str, auth_token: Optional[str] = None) -> Dict[str, Any]:
        """Register multiple skills from a GitHub repository.

        Fetches metadata from SKILL.md for each skill and uses actual frontmatter data
        instead of placeholder values.

        Note: This method is implemented in this file (not delegated) because it orchestrates
        multiple operations and needs to be close to the bulk workflow logic.

        Args:
            git_repository_url: Full GitHub repository URL
            skill_names: List of skill names to register
            author: Author/owner registering the skills
            auth_token: Optional GitHub authentication token

        Returns:
            Result with successful and failed registrations
        """
        import re
        import aiohttp
        import base64

        successful = []
        failed = []

        # Parse GitHub URL to get owner/repo
        try:
            match = re.search(r'github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$', git_repository_url)
            if not match:
                raise ValueError(f"Invalid GitHub URL format: {git_repository_url}")
            owner, repo = match.groups()
            repository = f"{owner}/{repo}"
        except Exception as e:
            raise ValueError(f"Failed to parse GitHub URL: {e}")

        for skill_name in skill_names:
            try:
                # Check if already registered
                existing = await self.db_logger.get_skill(skill_name)
                if existing:
                    failed.append({"skill_name": skill_name, "error": "Already registered"})
                    continue

                # Fetch SKILL.md from GitHub to get actual metadata
                skill_metadata = None
                try:
                    async with aiohttp.ClientSession() as session:
                        headers = {}
                        if auth_token:
                            headers["Authorization"] = f"token {auth_token}"

                        skill_md_url = f"https://api.github.com/repos/{repository}/contents/skills/{skill_name}/SKILL.md"
                        async with session.get(skill_md_url, headers=headers) as skill_resp:
                            if skill_resp.status == 200:
                                skill_content = await skill_resp.json()
                                # Decode base64 content
                                content_str = base64.b64decode(skill_content.get("content", "")).decode("utf-8")
                                # Parse SKILL.md frontmatter
                                from oai_skills_registry.services.utils import parse_skill_md
                                skill_metadata = parse_skill_md(content_str)
                except Exception as metadata_err:
                    self.logger.warning(f"Could not fetch metadata for {skill_name}: {metadata_err}")

                # Create skill in database with actual metadata from SKILL.md
                skill_description = (skill_metadata.get("description", "") if skill_metadata
                                     else f"Discovered from {git_repository_url}")
                skill_category = (skill_metadata.get("category", "") if skill_metadata
                                 else "")
                skill_tags = (skill_metadata.get("tags", []) if skill_metadata
                             else [])
                skill_author = (skill_metadata.get("author", author) if skill_metadata
                               else author)

                result = await self.db_logger.create_skill(
                    name=skill_name,
                    description=skill_description,
                    category=skill_category,
                    tags=skill_tags,
                    author=skill_author,
                    git_repository_url=git_repository_url
                )

                if result.get("id"):
                    successful.append(skill_name)
                    self.logger.info(f"Registered skill: {skill_name} with metadata from SKILL.md")
                else:
                    failed.append({"skill_name": skill_name, "error": "Failed to create in database"})
            except Exception as e:
                failed.append({"skill_name": skill_name, "error": str(e)})
                self.logger.error(f"Failed to register skill {skill_name}: {e}")

        return {
            "total_registered": len(successful),
            "successful": successful,
            "failed": failed
        }

    # ========== Connection Management ==========

    async def close(self) -> None:
        """Close database connection."""
        await self.db_logger.close()
