"""
Hybrid skill registry combining local file-based discovery with remote API fallback.

This module provides a SkillRegistry that:
1. Discovers skills from local filesystem (like base SkillRegistry)
2. Falls back to skills-registry API if skill not found locally
3. Can pull remote skills and cache them locally
4. Maintains backward compatibility with existing local skill setup
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import httpx

from oai_agent_core.components.skills.skill_registry import SkillRegistry
from oai_agent_core.components.skills.models import SkillProperties
from oai_agent_core.utils.path_utils import resolve_path


class HybridSkillRegistry(SkillRegistry):
    """
    Hybrid registry combining local filesystem discovery with remote API lookup.

    Load Order:
    1. Local skills directory (fast, no network)
    2. Remote registry API cache (if skill not found locally)
    3. Fallback: Attempt to pull from remote (if configured)

    Attributes:
        registry_url: Base URL for skills-registry API
        auth_token: Optional authentication token for API calls
        remote_skills_cache: In-memory cache of remote skills metadata
        cache_timeout: Cache expiration time (seconds), 0 = never
    """

    def __init__(
        self,
        registry_url: str = "http://localhost:8083/api/v1/skills-registry",
        auth_token: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        project_root: Optional[str] = None,
        cache_timeout: int = 3600,  # 1 hour
        enable_auto_pull: bool = False,
    ):
        """
        Initialize hybrid skill registry.

        Args:
            registry_url: Base URL of skills-registry API
            auth_token: Bearer token for API authentication
            logger: Optional logger instance
            project_root: Project root for relative paths
            cache_timeout: Cache expiration in seconds (0 = never expires)
            enable_auto_pull: Automatically pull missing skills from registry
        """
        super().__init__(logger=logger, project_root=project_root)
        self.registry_url = registry_url.rstrip('/')
        self.auth_token = auth_token
        self.remote_skills_cache: Dict[str, SkillProperties] = {}
        self.cache_timestamp: Dict[str, float] = {}
        self.cache_timeout = cache_timeout
        self.enable_auto_pull = enable_auto_pull
        self._http_client: Optional[httpx.AsyncClient] = None

    async def initialize(self):
        """
        Initialize registry by loading remote skills.

        This should be called after local discovery.
        Must be called in an async context.
        """
        try:
            await self._load_remote_skills_metadata()
            self.logger.info("Hybrid skill registry initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize remote skills: {e}")
            self.logger.info("Continuing with local-only skills discovery")

    async def _load_remote_skills_metadata(self):
        """Fetch available skills metadata from remote registry."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                headers = self._get_auth_headers()

                response = await client.get(
                    f"{self.registry_url}/skills",
                    headers=headers
                )
                response.raise_for_status()

                remote_skills_data = response.json()

                # Parse response - could be list or dict
                if isinstance(remote_skills_data, dict):
                    skills_list = remote_skills_data.get('skills', [])
                else:
                    skills_list = remote_skills_data

                for skill_data in skills_list:
                    skill = SkillProperties(
                        name=skill_data.get('name', ''),
                        description=skill_data.get('description', ''),
                        dependencies=skill_data.get('dependencies', []),
                        author=skill_data.get('author', ''),
                        category=skill_data.get('category', ''),
                        tags=skill_data.get('tags', []),
                        version=skill_data.get('current_version', ''),
                        status=skill_data.get('status', 'active'),
                    )
                    self.remote_skills_cache[skill.name] = skill

                self.logger.debug(
                    f"Loaded {len(self.remote_skills_cache)} skills from registry"
                )

        except httpx.HTTPError as e:
            self.logger.warning(f"HTTP error loading remote skills: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading remote skills: {e}")

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        headers = {}
        if self.auth_token:
            # Support both "token" and "Bearer token" formats
            token = self.auth_token
            if not token.startswith('Bearer '):
                token = f'Bearer {token}'
            headers['Authorization'] = token
        return headers

    def get_skill(self, name: str) -> Optional[SkillProperties]:
        """
        Get skill from local first, then remote cache.

        Order:
        1. Local filesystem
        2. Remote cache
        3. None if not found

        Args:
            name: Skill name to retrieve

        Returns:
            SkillProperties if found, None otherwise
        """
        # Try local first
        skill = self.skills.get(name)
        if skill:
            self.logger.debug(f"Found skill '{name}' in local registry")
            return skill

        # Try remote cache
        if name in self.remote_skills_cache:
            self.logger.debug(f"Found skill '{name}' in remote cache")
            return self.remote_skills_cache[name]

        self.logger.debug(f"Skill '{name}' not found in local or remote")
        return None

    def get_all_skills(self) -> List[SkillProperties]:
        """
        Get all skills (local + remote, deduplicated).

        Local skills take precedence over remote with same name.

        Returns:
            Sorted list of all available skills
        """
        all_skills = {}

        # Add remote skills first
        all_skills.update({s.name: s for s in self.remote_skills_cache.values()})

        # Override with local skills (precedence)
        all_skills.update({s.name: s for s in self.skills.values()})

        return sorted(all_skills.values(), key=lambda s: s.name)

    async def pull_skill(
        self,
        skill_name: str,
        version: str = "latest",
        target_dir: Optional[Path] = None,
    ) -> bool:
        """
        Download a skill from remote registry and save locally.

        Creates local directory structure:
            {target_dir}/{skill_name}/
            ├── SKILL.md
            └── skill_config.yaml (if available)

        Args:
            skill_name: Name of skill to pull
            version: Version to pull (default: "latest")
            target_dir: Where to save (default: {project_root}/skills)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Fetch skill content from API
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = self._get_auth_headers()

                endpoint = f"{self.registry_url}/skills/{skill_name}/versions/{version}"
                self.logger.debug(f"Pulling skill from {endpoint}")

                response = await client.get(endpoint, headers=headers)
                response.raise_for_status()

                skill_data = response.json()

            # Determine target directory
            if target_dir is None:
                base_path = self.project_root or '.'
                target_dir = Path(base_path) / 'skills' / skill_name
            else:
                target_dir = Path(target_dir) / skill_name

            # Create directory
            target_dir.mkdir(parents=True, exist_ok=True)

            # Save SKILL.md
            skill_md_path = target_dir / 'SKILL.md'
            skill_md_path.write_text(skill_data.get('content', ''), encoding='utf-8')
            self.logger.debug(f"Saved {skill_md_path}")

            # Save skill_config.yaml if available
            if 'config' in skill_data:
                config_path = target_dir / 'skill_config.yaml'
                config_path.write_text(skill_data['config'], encoding='utf-8')
                self.logger.debug(f"Saved {config_path}")

            # Save dependencies if available
            if 'dependencies' in skill_data and skill_data['dependencies']:
                deps_path = target_dir / 'requirements.txt'
                deps_text = '\n'.join(skill_data['dependencies'])
                deps_path.write_text(deps_text, encoding='utf-8')
                self.logger.debug(f"Saved {deps_path}")

            self.logger.info(
                f"Successfully pulled skill '{skill_name}' "
                f"version '{version}' to {target_dir}"
            )
            return True

        except httpx.HTTPError as e:
            self.logger.error(f"HTTP error pulling skill '{skill_name}': {e}")
            return False
        except IOError as e:
            self.logger.error(f"IO error saving skill '{skill_name}': {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error pulling skill '{skill_name}': {e}")
            return False

    async def get_remote_versions(self, skill_name: str) -> List[Dict[str, Any]]:
        """
        Get list of available versions for a skill from remote registry.

        Args:
            skill_name: Name of skill

        Returns:
            List of version info dicts with 'version', 'status', 'created_at', etc.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                headers = self._get_auth_headers()

                response = await client.get(
                    f"{self.registry_url}/skills/{skill_name}/versions",
                    headers=headers
                )
                response.raise_for_status()

                data = response.json()
                return data.get('versions', []) if isinstance(data, dict) else data

        except Exception as e:
            self.logger.error(f"Failed to get versions for '{skill_name}': {e}")
            return []

    async def publish_skill_usage(
        self,
        agent_id: str,
        skill_name: str,
        version: str,
        success: bool,
        duration_ms: int = 0,
        error: Optional[str] = None,
    ) -> bool:
        """
        Report skill usage to registry for analytics.

        Args:
            agent_id: ID of agent using skill
            skill_name: Name of skill
            version: Version of skill
            success: Whether execution was successful
            duration_ms: Execution duration in milliseconds
            error: Error message if failed

        Returns:
            True if reported successfully
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                headers = self._get_auth_headers()
                headers['Content-Type'] = 'application/json'

                payload = {
                    'agent_id': agent_id,
                    'skill_name': skill_name,
                    'version': version,
                    'success': success,
                    'duration_ms': duration_ms,
                    'error': error,
                }

                response = await client.post(
                    f"{self.registry_url}/skills/{skill_name}/usage",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                return True

        except Exception as e:
            # Non-critical - don't fail if usage reporting fails
            self.logger.debug(f"Failed to report skill usage: {e}")
            return False

    def get_skill_source(self, name: str) -> str:
        """
        Identify where a skill is loaded from.

        Returns:
            "local" if from filesystem, "remote" if from API cache, "unknown" if not found
        """
        if name in self.skills:
            return "local"
        if name in self.remote_skills_cache:
            return "remote"
        return "unknown"

    def get_registry_status(self) -> Dict[str, Any]:
        """
        Get status of the hybrid registry.

        Returns:
            Status dict with counts and configuration info
        """
        return {
            'local_skills': len(self.skills),
            'remote_skills_cached': len(self.remote_skills_cache),
            'total_skills': len(self.get_all_skills()),
            'registry_url': self.registry_url,
            'authenticated': bool(self.auth_token),
            'auto_pull_enabled': self.enable_auto_pull,
        }


# Async context manager for initialization
class HybridSkillRegistryContext:
    """Context manager for async initialization of hybrid registry."""

    def __init__(self, registry: HybridSkillRegistry):
        self.registry = registry

    async def __aenter__(self):
        await self.registry.initialize()
        return self.registry

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass
