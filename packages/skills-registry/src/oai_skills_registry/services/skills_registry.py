"""
Skills Registry Service
Handles skill lifecycle management and Git integration.
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from abc import ABC, abstractmethod
import tempfile
import shutil
from pathlib import Path
import yaml
import re
import json
import subprocess

from oai_skills_registry.services.db.database_logger import SkillsDatabaseLogger
from oai_skills_registry.models import (
    SkillImportPreview,
    SkillMetadata,
)


class GitProvider(ABC):
    """Base class for Git providers."""

    @abstractmethod
    async def fetch_skill_files(self, repo: str, branch: Optional[str],
                                tag: Optional[str], skill_name: str,
                                auth_token: Optional[str]) -> Dict[str, str]:
        """Fetch SKILL.md and skill_config.yaml from Git.
        Returns: {"SKILL.md": content, "skill_config.yaml": content}
        """
        pass

    @abstractmethod
    async def get_commit_info(self, repo: str, tag: Optional[str],
                              auth_token: Optional[str]) -> Dict[str, str]:
        """Get commit SHA and message for a tag or latest commit."""
        pass

    @abstractmethod
    async def get_tags(self, repo: str, auth_token: Optional[str]) -> List[Dict]:
        """Get all tags from repository."""
        pass


class GitHubProvider(GitProvider):
    """GitHub provider implementation."""

    async def fetch_skill_files(self, repo: str, branch: Optional[str],
                                tag: Optional[str], skill_name: str,
                                auth_token: Optional[str]) -> Dict[str, str]:
        """Fetch files from GitHub using Git CLI."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Determine reference (tag or branch)
            ref = tag if tag else (branch or "main")

            # Use git-archive to get files without full clone
            import subprocess

            # Clone shallow
            clone_cmd = [
                "git", "clone",
                "-b", ref,
                "--depth", "1",
                "--single-branch",
                f"https://{f'oauth2:{auth_token}@' if auth_token else ''}github.com/{repo}.git",
                str(temp_dir)
            ]

            result = subprocess.run(clone_cmd, capture_output=True, timeout=60)
            if result.returncode != 0:
                raise Exception(f"Git clone failed: {result.stderr.decode()}")

            skill_path = temp_dir / "skills" / skill_name
            if not skill_path.exists():
                raise Exception(f"Skill not found: {skill_name}")

            files = {}

            # Read SKILL.md
            skill_md = skill_path / "SKILL.md"
            if skill_md.exists():
                with open(skill_md) as f:
                    files["SKILL.md"] = f.read()
            else:
                raise Exception(f"SKILL.md not found in {skill_path}")

            # Read skill_config.yaml if exists
            config_file = skill_path / "skill_config.yaml"
            if config_file.exists():
                with open(config_file) as f:
                    files["skill_config.yaml"] = f.read()

            return files

        except Exception as e:
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def get_commit_info(self, repo: str, tag: Optional[str],
                              auth_token: Optional[str]) -> Dict[str, str]:
        """Get commit info for a tag or current HEAD."""
        import subprocess

        temp_dir = Path(tempfile.mkdtemp())
        try:
            ref = tag or "HEAD"

            clone_cmd = [
                "git", "clone",
                "--depth", "1",
                f"https://{f'oauth2:{auth_token}@' if auth_token else ''}github.com/{repo}.git",
                str(temp_dir)
            ]
            subprocess.run(clone_cmd, capture_output=True, timeout=60, check=True)

            # Get commit SHA
            sha_cmd = ["git", "rev-parse", ref]
            result = subprocess.run(
                sha_cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            commit_sha = result.stdout.strip()

            # Get commit message
            msg_cmd = ["git", "log", "-1", "--format=%B", ref]
            result = subprocess.run(
                msg_cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            commit_message = result.stdout.strip()

            return {
                "commit_sha": commit_sha,
                "commit_message": commit_message
            }

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def get_tags(self, repo: str, auth_token: Optional[str]) -> List[Dict]:
        """Get tags from GitHub using REST API."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            headers = {}
            if auth_token:
                headers["Authorization"] = f"token {auth_token}"

            url = f"https://api.github.com/repos/{repo}/tags?per_page=100"

            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return []


class SkillsRegistry:
    """Main skills registry service."""

    def __init__(self, db_logger: SkillsDatabaseLogger, logger: Optional[logging.Logger] = None):
        self.db_logger = db_logger
        self.logger = logger or logging.getLogger(__name__)
        self.git_providers = {
            "github": GitHubProvider(),
            # "gitlab": GitLabProvider(),
            # "gitea": GiteaProvider(),
        }

    async def register_git_source(self, source_config: Dict) -> Dict:
        """Register a new Git repository as skill source."""
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
        """
        Import skill version from Git source.
        Supports both registered sources and direct GitHub URLs.
        Creates unpublished version record.
        Returns import preview with metadata.
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
                import re
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
        skill_data = self._parse_skill_md(files["SKILL.md"])
        if skill_data["name"] != skill_name:
            raise ValueError(f"Skill name mismatch: {skill_data['name']} != {skill_name}")

        # Parse skill_config.yaml
        config_data = None
        if "skill_config.yaml" in files:
            config_data = yaml.safe_load(files["skill_config.yaml"])
            # Convert any datetime objects to strings for JSON serialization
            if config_data:
                config_data = self._convert_datetime_to_str(config_data)

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
            "preview": preview.dict()
        }

    def _parse_skill_md(self, content: str) -> Dict[str, Any]:
        """Parse SKILL.md frontmatter and content."""
        # Extract YAML frontmatter
        match = re.match(r'^---\n(.*?)\n---\n(.*)', content, re.DOTALL)
        if not match:
            raise ValueError("Invalid SKILL.md format - missing frontmatter")

        metadata = yaml.safe_load(match.group(1))
        body = match.group(2)

        metadata["content"] = body
        return metadata

    def _convert_datetime_to_str(self, obj: Any) -> Any:
        """Recursively convert datetime objects to ISO format strings."""
        if isinstance(obj, dict):
            return {k: self._convert_datetime_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_datetime_to_str(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

    async def publish_skill_version(self, skill_name: str, version: str,
                                    message: Optional[str],
                                    performed_by: str) -> Dict[str, Any]:
        """
        Publish unpublished skill version.
        Makes it available for agents to use.
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
        """
        Upgrade skill to new version.
        Updates agents based on their version constraints.
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
            if self._version_matches_constraint(to_version, constraint):
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
        """Downgrade skill to previous version."""
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
        """Mark skill version as deprecated."""
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
        """Delete entire skill and all versions."""
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

    def _version_matches_constraint(self, version: str, constraint: str) -> bool:
        """Check if version matches constraint."""
        if constraint == "latest":
            return True
        if constraint == version:
            return True
        # Could add more sophisticated version constraint matching
        return True

    async def get_skill_details(self, skill_name: str) -> Dict[str, Any]:
        """Get full skill details with all versions."""
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
        """Get action history for a skill."""
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
        """List available versions in Git (tags)."""
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

        for tag in tags:
            try:
                # Try to fetch SKILL.md from this tag
                files = await provider.fetch_skill_files(
                    repo=git_source["repository"],
                    branch=None,
                    tag=tag.get("name"),
                    skill_name=skill_name,
                    auth_token=git_source.get("auth_token")
                )
                skill_data = self._parse_skill_md(files["SKILL.md"])
                if skill_data.get("name") == skill_name:
                    versions.append({
                        "version": skill_data.get("version"),
                        "git_tag": tag.get("name"),
                        "commit_sha": tag.get("commit", {}).get("sha", ""),
                        "created_at": tag.get("created_at")
                    })
            except Exception as e:
                self.logger.debug(f"Failed to parse version from tag {tag.get('name')}: {e}")
                continue

        return sorted(versions, key=lambda x: x.get("created_at", ""), reverse=True)

    async def list_skill_versions_from_url(self, skill_name: str, git_repository_url: str,
                                          auth_token: Optional[str] = None) -> List[Dict]:
        """List available versions from a GitHub repository URL.

        This method fetches versions directly from a GitHub repository without requiring
        a registered git source. Useful for refresh operations.

        Args:
            skill_name: Name of the skill to match in SKILL.md
            git_repository_url: Full GitHub repository URL (e.g., https://github.com/owner/repo.git)
            auth_token: Optional GitHub authentication token

        Returns:
            List of available versions with metadata
        """
        # Parse GitHub URL to extract repository
        # Expected format: https://github.com/{owner}/{repo}.git or https://github.com/{owner}/{repo}
        try:
            # Extract owner/repo from URL
            import re
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
                skill_data = self._parse_skill_md(files["SKILL.md"])
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

    async def discover_skills_from_git(self, git_repository_url: str,
                                      auth_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Discover all skills available in a GitHub repository.

        Scans the 'skills/' directory in the repository and returns metadata for each skill found.

        Args:
            git_repository_url: Full GitHub repository URL
            auth_token: Optional GitHub authentication token

        Returns:
            Dictionary with discovered skills and their status
        """
        import re
        import aiohttp

        # Parse GitHub URL
        try:
            match = re.search(r'github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$', git_repository_url)
            if not match:
                raise ValueError(f"Invalid GitHub URL format: {git_repository_url}")
            owner, repo = match.groups()
            repository = f"{owner}/{repo}"
        except Exception as e:
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

                        try:
                            # Fetch SKILL.md from this skill
                            skill_md_url = f"https://api.github.com/repos/{repository}/contents/skills/{skill_name}/SKILL.md"

                            async with session.get(skill_md_url, headers=headers) as skill_resp:
                                if skill_resp.status == 200:
                                    skill_content = await skill_resp.json()
                                    # Decode base64 content
                                    import base64
                                    content_str = base64.b64decode(skill_content.get("content", "")).decode("utf-8")

                                    # Parse SKILL.md
                                    skill_data = self._parse_skill_md(content_str)

                                    # Check if already registered
                                    existing = await self.db_logger.get_skill(skill_name)
                                    status = "already_registered" if existing else "available"

                                    discovered_skills.append({
                                        "name": skill_data.get("name", skill_name),
                                        "version": skill_data.get("version", "0.0.1"),
                                        "description": skill_data.get("description", ""),
                                        "category": skill_data.get("category", ""),
                                        "author": skill_data.get("author", ""),
                                        "tags": skill_data.get("tags", []),
                                        "status": status,
                                        "path_in_repo": f"skills/{skill_name}"
                                    })
                        except Exception as e:
                            self.logger.warning(f"Failed to parse skill {skill_name}: {e}")
                            discovered_skills.append({
                                "name": skill_name,
                                "version": "unknown",
                                "description": "Failed to parse",
                                "category": "unknown",
                                "author": "unknown",
                                "tags": [],
                                "status": "invalid",
                                "path_in_repo": f"skills/{skill_name}"
                            })
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

    async def register_skills_bulk(self, git_repository_url: str, skill_names: List[str],
                                  author: str, auth_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Register multiple skills from a GitHub repository.

        Fetches metadata from SKILL.md for each skill and uses actual frontmatter data
        instead of placeholder values.

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
                                import base64
                                content_str = base64.b64decode(skill_content.get("content", "")).decode("utf-8")
                                # Parse SKILL.md frontmatter
                                skill_metadata = self._parse_skill_md(content_str)
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

    async def close(self) -> None:
        """Close database connection."""
        await self.db_logger.close()
