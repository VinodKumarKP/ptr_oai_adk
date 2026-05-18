"""
Corrected Hybrid Skill Registry Implementation

This registry properly integrates with the Skills Registry Backend which stores
METADATA (name, description, git_repository_url, versions) but NOT the actual skill code.

The actual skill code is stored in GitHub repositories, referenced by git_repository_url.

Flow:
1. Query Skills Registry API for skill metadata (includes git_repository_url)
2. Clone/fetch actual skill code from GitHub
3. Cache skill code locally
4. Load and use locally cached skill
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import httpx

from oai_agent_core.components.skills.skill_registry import SkillRegistry
from oai_agent_core.components.skills.models import SkillProperties


class HybridSkillRegistry(SkillRegistry):
    """
    Hybrid registry that integrates with:

    1. Skills Registry Backend (HTTP API) - Stores metadata & git URLs
    2. GitHub (or other Git hosts) - Stores actual skill code
    3. Local Filesystem - Caches skill code for fast access

    Does NOT assume skills registry stores code - it only stores metadata.

    Attributes:
        registry_url: Skills Registry API base URL
        auth_token: API authentication token
        git_provider: GitProvider instance for fetching code from GitHub
        skill_metadata_cache: In-memory cache of skill metadata
    """

    def __init__(
        self,
        registry_url: str = "http://localhost:8083/api/v1/skills-registry",
        auth_token: Optional[str] = None,
        git_provider: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        project_root: Optional[str] = None,
        skills_cache_dir: Optional[str] = None,
    ):
        """
        Initialize hybrid skill registry.

        Args:
            registry_url: Skills Registry API base URL
            auth_token: Bearer token for API authentication
            git_provider: GitProvider instance (GitHubProvider, etc.)
                         If None, will try to import and instantiate
            logger: Optional logger instance
            project_root: Project root for relative paths
            skills_cache_dir: Where to cache fetched skills (default: {project_root}/skills)
        """
        super().__init__(logger=logger, project_root=project_root)

        self.registry_url = registry_url.rstrip('/')
        self.auth_token = auth_token
        self.git_provider = git_provider

        # Initialize git provider if not provided
        if self.git_provider is None:
            self._initialize_git_provider()

        self.skill_metadata_cache: Dict[str, Dict[str, Any]] = {}
        self.skills_cache_dir = Path(skills_cache_dir or (project_root or '.') / 'skills')
        self.skills_cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"HybridSkillRegistry initialized")
        self.logger.info(f"  Registry API: {self.registry_url}")
        self.logger.info(f"  Skills cache: {self.skills_cache_dir}")

    def _initialize_git_provider(self):
        """Initialize default git provider (GitHub)."""
        try:
            from oai_skills_registry.services.git_provider import GitHubProvider
            self.git_provider = GitHubProvider()
            self.logger.debug("Initialized GitHubProvider for skill code fetching")
        except ImportError:
            self.logger.warning(
                "Could not import GitHubProvider. "
                "Skills requiring git operations will fail. "
                "Install skills-registry package or provide git_provider."
            )

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        headers = {}
        if self.auth_token:
            if not self.auth_token.startswith('Bearer '):
                headers['Authorization'] = f'Bearer {self.auth_token}'
            else:
                headers['Authorization'] = self.auth_token
        return headers

    async def initialize(self):
        """
        Initialize registry by loading remote skill metadata.

        This loads metadata (not code) from Skills Registry API.
        Actual skill code is fetched on-demand from GitHub.
        """
        try:
            await self._load_remote_skill_metadata()
            self.logger.info(
                f"Loaded metadata for {len(self.skill_metadata_cache)} skills from registry"
            )
        except Exception as e:
            self.logger.warning(f"Failed to load remote skill metadata: {e}")
            self.logger.info("Continuing with local-only skills discovery")

    async def _load_remote_skill_metadata(self):
        """
        Fetch skill metadata from Skills Registry API.

        Metadata includes:
        - name, description, category, author, tags
        - git_repository_url (points to GitHub)
        - current_version, status (published/draft/deprecated)
        - dependencies, breaking_changes

        Does NOT fetch actual skill code - just metadata.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = self._get_auth_headers()

                response = await client.get(
                    f"{self.registry_url}/skills",
                    headers=headers
                )
                response.raise_for_status()

                data = response.json()

                # Parse response - could be list or dict
                if isinstance(data, dict):
                    skills_list = data.get('skills', [])
                else:
                    skills_list = data

                # Cache metadata (NOT the code)
                for skill_data in skills_list:
                    skill_name = skill_data.get('name')
                    if skill_name:
                        self.skill_metadata_cache[skill_name] = skill_data
                        self.logger.debug(
                            f"Loaded metadata: {skill_name} "
                            f"(git: {skill_data.get('git_repository_url', 'none')})"
                        )

                self.logger.info(
                    f"Loaded metadata for {len(self.skill_metadata_cache)} skills"
                )

        except httpx.HTTPError as e:
            self.logger.error(f"HTTP error loading skill metadata: {e}")
        except Exception as e:
            self.logger.error(f"Error loading skill metadata: {e}")

    async def get_skill_metadata(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """
        Get skill metadata from registry.

        Returns metadata dict with:
        - name, description, category, etc.
        - git_repository_url (where actual code lives)
        - current_version
        - status (published/draft/deprecated)

        Does NOT fetch the actual skill code.
        """
        # Try cache first
        if skill_name in self.skill_metadata_cache:
            return self.skill_metadata_cache[skill_name]

        # Try fetching from API
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                headers = self._get_auth_headers()

                response = await client.get(
                    f"{self.registry_url}/skills/{skill_name}",
                    headers=headers
                )
                response.raise_for_status()

                metadata = response.json()
                self.skill_metadata_cache[skill_name] = metadata

                self.logger.debug(f"Fetched metadata for skill: {skill_name}")
                return metadata

        except Exception as e:
            self.logger.debug(f"Failed to fetch metadata for '{skill_name}': {e}")
            return None

    async def _fetch_skill_code_from_github(
        self,
        git_repo_url: str,
        skill_name: str,
        branch: str = "main",
        target_dir: Optional[Path] = None,
    ) -> bool:
        """
        Fetch actual skill code from GitHub repository.

        This is where the actual code clone happens using the git provider.
        Never fetches code from registry API - always clones from GitHub.

        Args:
            git_repo_url: GitHub repository URL (from skill metadata)
            skill_name: Name of the skill
            branch: Git branch/tag to fetch (default: main)
            target_dir: Where to save (default: skills_cache_dir/skill_name)

        Returns:
            True if successful, False otherwise
        """
        if not self.git_provider:
            self.logger.error(
                "Git provider not available - cannot fetch skill code from GitHub. "
                "Install GitHubProvider or pass git_provider to __init__"
            )
            return False

        try:
            if target_dir is None:
                target_dir = self.skills_cache_dir / skill_name

            target_dir = Path(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(
                f"Cloning skill '{skill_name}' from GitHub repository: {git_repo_url} "
                f"(branch/tag: {branch}) → {target_dir}"
            )

            # Use git provider to fetch/clone from GitHub
            # This could be: clone, fetch specific files, etc.
            # The exact method depends on GitProvider implementation

            if hasattr(self.git_provider, 'fetch_skill_files'):
                self.logger.debug(
                    f"Using GitProvider.fetch_skill_files() to fetch from {git_repo_url}"
                )
                await self.git_provider.fetch_skill_files(
                    repo_url=git_repo_url,
                    skill_name=skill_name,
                    branch=branch,
                    target_dir=target_dir,
                )
            elif hasattr(self.git_provider, 'clone_repository'):
                self.logger.debug(
                    f"Using GitProvider.clone_repository() to clone from {git_repo_url}"
                )
                await self.git_provider.clone_repository(
                    repo_url=git_repo_url,
                    target_dir=target_dir,
                    branch=branch,
                )
            else:
                self.logger.error(
                    f"Git provider {type(self.git_provider).__name__} does not support "
                    "fetch_skill_files() or clone_repository() methods"
                )
                return False

            self.logger.info(
                f"✓ Successfully cloned skill code from GitHub to {target_dir}"
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to clone skill '{skill_name}' from GitHub: {e}",
                exc_info=True
            )
            return False

    async def get_skill(self, name: str) -> Optional[SkillProperties]:
        """
        Get skill, fetching code from GitHub if needed.

        Process:
        1. Check local filesystem (fast)
        2. If not found, get metadata from registry (includes git_repository_url)
        3. If still not found locally, fetch code from GitHub
        4. Parse and return skill

        Args:
            name: Skill name

        Returns:
            SkillProperties if found, None otherwise
        """
        # Check local filesystem first
        skill = self.skills.get(name)
        if skill:
            self.logger.debug(f"Found skill '{name}' in local registry")
            return skill

        # Get metadata from registry (includes git URL)
        metadata = await self.get_skill_metadata(name)
        if not metadata:
            self.logger.debug(f"Skill '{name}' not found in registry")
            return None

        # Check if code cached locally
        skill_path = self.skills_cache_dir / name
        if not skill_path.exists():
            # Fetch from GitHub
            git_url = metadata.get('git_repository_url')
            if not git_url:
                self.logger.warning(
                    f"No git_repository_url in metadata for '{name}'"
                )
                return None

            success = await self._fetch_skill_code_from_github(
                git_repo_url=git_url,
                skill_name=name,
                branch=metadata.get('git_branch', 'main'),
                target_dir=skill_path,
            )

            if not success:
                return None

        # Discover locally cached skill
        try:
            from oai_agent_core.components.skills.parser import load_metadata
            skill_props = load_metadata(skill_path)
            self.skills[name] = skill_props
            return skill_props
        except Exception as e:
            self.logger.error(f"Failed to load skill '{name}' from cache: {e}")
            return None

    async def list_available_versions(
        self,
        skill_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get list of available versions for a skill from registry.

        Returns version info from registry API including:
        - version number
        - git_tag
        - status (published/draft/deprecated)
        - created_at, author, etc.
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
                versions = data.get('versions', []) if isinstance(data, dict) else data

                self.logger.debug(
                    f"Found {len(versions)} versions for skill '{skill_name}'"
                )
                return versions

        except Exception as e:
            self.logger.error(f"Failed to list versions for '{skill_name}': {e}")
            return []

    async def pull_skill(
        self,
        skill_name: str,
        version: str = "latest",
        target_dir: Optional[Path] = None,
    ) -> bool:
        """
        Pull a skill from its GitHub repository (NOT from registry API).

        This method:
        1. Gets skill metadata from registry (metadata ONLY, not code)
        2. Extracts git_repository_url from metadata
        3. Clones/fetches the skill code from GitHub
        4. Caches locally for reuse

        CRITICAL: Always fetches code from git_repository_url in metadata.
        Never tries to fetch skill code directly from registry API.

        Args:
            skill_name: Skill name
            version: Version to pull (default: latest)
            target_dir: Where to save (default: skills_cache_dir/skill_name)

        Returns:
            True if successfully cloned from GitHub, False otherwise
        """
        try:
            # STEP 1: Get metadata from registry (metadata ONLY, not code)
            self.logger.debug(
                f"Getting metadata for skill '{skill_name}' from registry..."
            )
            metadata = await self.get_skill_metadata(skill_name)
            if not metadata:
                self.logger.error(f"Skill '{skill_name}' not found in registry")
                return False

            # STEP 2: Extract git repository URL from metadata
            git_repository_url = metadata.get('git_repository_url')
            if not git_repository_url:
                self.logger.error(
                    f"No git_repository_url found in metadata for '{skill_name}'. "
                    f"Cannot clone skill code from GitHub."
                )
                return False

            self.logger.debug(
                f"Found git_repository_url: {git_repository_url}"
            )

            # STEP 3: Determine target directory
            if target_dir is None:
                target_dir = self.skills_cache_dir / skill_name

            # STEP 4: Determine git reference (branch, tag, or commit)
            git_branch = metadata.get('git_branch', 'main')

            # If specific version requested, try to get the git tag/commit
            if version != "latest":
                self.logger.debug(
                    f"Looking up version '{version}' to find git tag/commit..."
                )
                versions = await self.list_available_versions(skill_name)
                if versions:
                    version_info = next(
                        (v for v in versions if v.get('version') == version),
                        None
                    )
                    if version_info:
                        git_branch = version_info.get('git_tag', version)
                        self.logger.info(
                            f"Using git tag/branch: {git_branch} for version {version}"
                        )
                    else:
                        self.logger.warning(
                            f"Version '{version}' not found in registry. "
                            f"Will use default branch: {git_branch}"
                        )

            # STEP 5: Clone from GitHub using git provider (NOT from registry)
            self.logger.info(
                f"Pulling skill '{skill_name}' from GitHub repository: {git_repository_url} "
                f"(branch/tag: {git_branch})"
            )

            success = await self._fetch_skill_code_from_github(
                git_repo_url=git_repository_url,
                skill_name=skill_name,
                branch=git_branch,
                target_dir=target_dir,
            )

            if success:
                self.logger.info(f"✓ Successfully pulled skill '{skill_name}' from GitHub")
                return True
            else:
                self.logger.error(
                    f"✗ Failed to pull skill '{skill_name}' from GitHub"
                )
                return False

        except Exception as e:
            self.logger.error(
                f"Error pulling skill '{skill_name}' from GitHub: {e}",
                exc_info=True
            )
            return False

    async def pull_skill_version(
        self,
        skill_name: str,
        version: str = "latest",
        target_dir: Optional[Path] = None,
    ) -> bool:
        """
        Deprecated: Use pull_skill() instead.

        This method is kept for backwards compatibility.
        It delegates to pull_skill().
        """
        self.logger.warning(
            "pull_skill_version() is deprecated. Use pull_skill() instead."
        )
        return await self.pull_skill(skill_name, version, target_dir)

    async def report_skill_usage(
        self,
        skill_name: str,
        agent_id: str,
        success: bool,
        duration_ms: int = 0,
        error: Optional[str] = None,
    ) -> bool:
        """
        Report skill usage to registry for analytics.

        Args:
            skill_name: Skill that was used
            agent_id: Agent that used it
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
                    'success': success,
                    'duration_ms': duration_ms,
                    'error': error,
                    'timestamp': datetime.utcnow().isoformat(),
                }

                response = await client.post(
                    f"{self.registry_url}/skills/{skill_name}/usage",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()

                self.logger.debug(
                    f"Reported usage: {skill_name} "
                    f"(agent: {agent_id}, success: {success})"
                )
                return True

        except Exception as e:
            # Non-critical - don't fail if usage reporting fails
            self.logger.debug(f"Failed to report usage: {e}")
            return False

    def get_registry_status(self) -> Dict[str, Any]:
        """
        Get status of the hybrid registry.

        Shows:
        - Number of skills discovered locally
        - Number of skills metadata cached from registry
        - Total available skills
        - Registry connection status
        """
        return {
            'local_skills_count': len(self.skills),
            'remote_metadata_cached': len(self.skill_metadata_cache),
            'total_skills': len(self.get_all_skills()),
            'registry_url': self.registry_url,
            'registry_authenticated': bool(self.auth_token),
            'skills_cache_dir': str(self.skills_cache_dir),
            'git_provider_available': self.git_provider is not None,
            'status': 'healthy' if self.git_provider else 'degraded',
        }

    def get_all_skills(self) -> List[SkillProperties]:
        """
        Get all available skills (local + remote metadata).

        Returns local skills first, then any remote-only skills
        (skills in registry but not cached locally yet).
        """
        all_skills = {}

        # Add locally discovered skills
        for skill in self.skills.values():
            all_skills[skill.name] = skill

        # Add skills from remote metadata (that aren't cached locally yet)
        for skill_name, metadata in self.skill_metadata_cache.items():
            if skill_name not in all_skills:
                # Create lightweight SkillProperties from metadata
                skill = SkillProperties(
                    name=metadata.get('name', skill_name),
                    description=metadata.get('description', ''),
                    author=metadata.get('author', ''),
                    category=metadata.get('category', ''),
                    dependencies=metadata.get('dependencies', []),
                    tags=metadata.get('tags', []),
                    version=metadata.get('current_version', ''),
                )
                all_skills[skill_name] = skill

        return sorted(all_skills.values(), key=lambda s: s.name)
