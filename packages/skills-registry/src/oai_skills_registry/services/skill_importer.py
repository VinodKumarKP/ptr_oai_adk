"""
Skill Importer
Handles importing skills from Git repositories.
"""

import logging
import yaml
import re
from typing import Optional, Dict, Any

from oai_skills_registry.services.db.database_logger import SkillsDatabaseLogger
from oai_skills_registry.services.git_provider import GitProvider
from oai_skills_registry.services.utils import parse_skill_md, convert_datetime_to_str
from oai_skills_registry.models import SkillImportPreview


class SkillImporter:
    """Handles skill import operations from Git repositories."""

    def __init__(self, db_logger: SkillsDatabaseLogger, git_providers: Dict[str, GitProvider],
                 logger: Optional[logging.Logger] = None):
        self.db_logger = db_logger
        self.git_providers = git_providers
        self.logger = logger or logging.getLogger(__name__)

    async def register_git_source(self, source_config: Dict) -> Dict:
        """Register a new Git repository as skill source.

        Args:
            source_config: Configuration for the Git source

        Returns:
            Dictionary with source_id, name, and status
        """
        source_id = await self.db_logger.register_git_source(source_config)
        self.logger.info(f"Registered Git source: {source_config['name']}")
        return {
            "source_id": source_id,
            "name": source_config["name"],
            "status": "registered"
        }

    async def import_skill_from_git(self, skill_name: str, source_config: Dict,
                                     git_tag: Optional[str],
                                     performed_by: str) -> Dict[str, Any]:
        """Import skill version from Git source.

        Supports both registered sources and direct GitHub URLs.
        Creates unpublished version record.
        Returns import preview with metadata.

        Args:
            skill_name: Name of the skill to import
            source_config: Git source configuration
            git_tag: Optional Git tag to checkout
            performed_by: User performing the import

        Returns:
            Dictionary with skill_id, version_id, version, and preview
        """
        # Get provider
        provider = self.git_providers.get(source_config["git_provider"])
        if not provider:
            raise ValueError(f"Unknown Git provider: {source_config['git_provider']}")

        # Handle direct GitHub URL import (parse URL to extract repository)
        repository = source_config.get("repository")
        if not repository and "_direct_url" in source_config:
            # Parse GitHub URL to extract owner/repo
            try:
                match = re.search(r'github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$', source_config["_direct_url"])
                if match:
                    owner, repo = match.groups()
                    repository = f"{owner}/{repo}"
                else:
                    raise ValueError(f"Invalid GitHub URL format: {source_config['_direct_url']}")
            except Exception as e:
                raise ValueError(f"Failed to parse GitHub URL: {e}")

        if not repository:
            raise ValueError("Repository not specified in source config")

        # Fetch skill files from Git
        try:
            files = await provider.fetch_skill_files(
                repo=repository,
                branch=source_config.get("branch", "main"),
                tag=git_tag,
                skill_name=skill_name,
                auth_token=source_config.get("auth_token")
            )
        except Exception as e:
            raise Exception(f"Failed to fetch skill from Git: {e}")

        # Parse SKILL.md
        skill_data = parse_skill_md(files["SKILL.md"])
        if skill_data["name"] != skill_name:
            raise ValueError(f"Skill name mismatch: {skill_data['name']} != {skill_name}")

        # Parse skill_config.yaml
        config_data = None
        if "skill_config.yaml" in files:
            config_data = yaml.safe_load(files["skill_config.yaml"])
            # Convert any datetime objects to strings for JSON serialization
            if config_data:
                config_data = convert_datetime_to_str(config_data)

        # Get Git commit info
        try:
            git_info = await provider.get_commit_info(
                repo=repository,
                tag=git_tag,
                auth_token=source_config.get("auth_token")
            )
        except Exception as e:
            self.logger.warning(f"Failed to get commit info: {e}")
            git_info = {"commit_sha": None, "commit_message": ""}

        version = skill_data.get("version", "0.0.1")

        # Get git repository URL for agent downloads
        git_repository_url = None
        if "_direct_url" in source_config:
            git_repository_url = source_config["_direct_url"]
        elif "git_url" in source_config:
            git_repository_url = source_config["git_url"]

        # Create skill if doesn't exist
        existing_skill = await self.db_logger.get_skill(skill_name)
        if not existing_skill:
            existing_skill = await self.db_logger.create_skill(
                name=skill_name,
                description=skill_data.get("description", ""),
                category=skill_data.get("category", ""),
                tags=skill_data.get("tags", []),
                author=performed_by,
                git_repository_url=git_repository_url
            )

        # Create unpublished version
        skill_version = await self.db_logger.create_skill_version(
            skill_id=existing_skill["id"],
            version=version,
            git_source_id=source_config.get("id"),
            git_branch=source_config.get("branch"),
            git_commit_sha=git_info.get("commit_sha"),
            git_tag=git_tag,
            content=files["SKILL.md"],
            config=config_data,
            dependencies=skill_data.get("dependencies", {}),
            breaking_changes=skill_data.get("breaking_changes"),
            status="draft",
            published_by=performed_by
        )

        # Create import preview
        preview = SkillImportPreview(
            skill_name=skill_name,
            version=version,
            description=skill_data.get("description", ""),
            category=skill_data.get("category", ""),
            tags=skill_data.get("tags", []),
            changes=skill_data.get("changes"),
            git_commit=git_info.get("commit_sha", ""),
            git_tag=git_tag,
            breaking_changes=skill_data.get("breaking_changes"),
            dependencies=skill_data.get("dependencies", {})
        )

        return {
            "skill_id": existing_skill["id"],
            "version_id": skill_version.get("id"),
            "version": version,
            "preview": preview.model_dump()
        }
