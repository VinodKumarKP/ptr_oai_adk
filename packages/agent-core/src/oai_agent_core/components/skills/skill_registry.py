"""
Skill Registry - Unified Local and Remote (Hybrid) Skill Management

This registry supports:
1. Local skill discovery from filesystem
2. Remote skill registry integration (metadata + GitHub)
3. Git provider integration for pulling skills from GitHub
4. Caching of remotely fetched skills locally

Architecture:
- Registry API: Provides skill metadata (name, description, git_repository_url, versions)
- GitHub: Stores actual skill code (SKILL.md, skill_config.yaml, src/, etc.)
- Local Cache: Fast access to previously fetched skills
"""

import asyncio
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import httpx

from oai_agent_core.components.skills.errors import ParseError, ValidationError
from oai_agent_core.components.skills.models import SkillProperties
from oai_agent_core.components.skills.parser import load_metadata, find_skill_md
from oai_agent_core.utils.path_utils import resolve_path, is_safe_path


class SkillRegistry:
    """
    Unified registry for discovering, loading, and managing agent skills.

    Supports both local skills and remote skills from a Skills Registry backend.

    Local Discovery:
    - Scans designated directory for skills
    - Parses metadata from SKILL.md files
    - Provides access to locally available skills

    Hybrid/Remote Discovery:
    - Queries Skills Registry API for skill metadata (includes git_repository_url)
    - Uses git provider to clone skill code from GitHub
    - Caches cloned skills locally for fast access
    - Supports version management and usage tracking

    Three-Tier Architecture:
    - Tier 1: Skills Registry API (metadata only)
    - Tier 2: GitHub Repositories (actual code)
    - Tier 3: Local Cache (fast access)
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        project_root: Optional[str] = None,
        registry_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        git_provider: Optional[Any] = None,
        skills_cache_dir: Optional[str] = None,
    ):
        """
        Initialize the skill registry.

        Args:
            logger: Logger instance for debugging and info messages
            project_root: Root directory for relative path resolution
            registry_url: Skills Registry API base URL (optional, for hybrid mode)
            auth_token: Bearer token for registry API authentication
            git_provider: GitProvider instance for fetching code from GitHub
            skills_cache_dir: Directory for caching fetched skills
        """
        self.logger = logger or logging.getLogger(__name__)
        self.project_root = project_root
        self.skills: Dict[str, SkillProperties] = {}

        # Hybrid/Remote configuration
        self.registry_url = registry_url.rstrip('/') if registry_url else None
        self.auth_token = auth_token
        self.git_provider = git_provider
        self.skill_metadata_cache: Dict[str, Dict[str, Any]] = {}

        # Initialize skills cache directory
        # IMPORTANT: resolve_path is used here (same function as discover_skills uses)
        # so that relative paths like "./skills" are resolved against project_root,
        # ensuring pull_skill() and discover_skills() point to the same directory.
        if self.registry_url:
            cache_dir_str = skills_cache_dir or 'skills_cache'
            self.skills_cache_dir = resolve_path(cache_dir_str, project_root)
            self.skills_cache_dir.mkdir(parents=True, exist_ok=True)

            # Initialize git provider if not provided
            if self.git_provider is None:
                self._initialize_git_provider()

            self.logger.info(f"SkillRegistry initialized in hybrid mode")
            self.logger.info(f"  Registry API: {self.registry_url}")
            self.logger.info(f"  Skills cache: {self.skills_cache_dir}")
        else:
            self.skills_cache_dir = None
            self.logger.info("SkillRegistry initialized in local-only mode")

    def _initialize_git_provider(self):
        """Initialize default git provider (GitHub)."""
        try:
            from oai_skills_registry.services.git_provider import GitHubProvider

            self.git_provider = GitHubProvider()
            self.logger.debug("Initialized GitHubProvider for skill code fetching")
        except ImportError:
            self.logger.warning(
                "Could not import GitHubProvider. "
                "Remote skill operations will fail. "
                "Install oai_skills_registry package or provide git_provider."
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
        if not self.registry_url:
            self.logger.debug("Not in hybrid mode. Skipping remote initialization.")
            return

        try:
            await self._load_remote_skill_metadata()
            self.logger.info(
                f"Loaded metadata for {len(self.skill_metadata_cache)} skills from registry"
            )
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            self.logger.warning(f"Failed to load remote skill metadata (network error): {e}")
            self.logger.info("Continuing with local-only skills discovery")
        except Exception as e:
            self.logger.warning(f"Failed to load remote skill metadata: {e}", exc_info=True)
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
        except asyncio.TimeoutError as e:
            self.logger.error(f"Timeout loading skill metadata: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading skill metadata: {e}", exc_info=True)

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

        # Try fetching from API if in hybrid mode
        if not self.registry_url:
            return None

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

        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            self.logger.debug(f"Failed to fetch metadata for '{skill_name}' (network error): {e}")
            return None
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
            if hasattr(self.git_provider, 'fetch_skill_files'):
                self.logger.debug(
                    f"Using GitProvider.fetch_skill_files() to fetch from {git_repo_url}"
                )
                # Parse repository from git_repo_url (format: owner/repo or https://github.com/owner/repo)
                repo_match = re.search(r'github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$', git_repo_url)
                if repo_match:
                    repo = f"{repo_match.group(1)}/{repo_match.group(2)}"
                else:
                    # Assume it's already in owner/repo format
                    repo = git_repo_url

                self.logger.debug(f"Parsed repository: {repo} from URL: {git_repo_url}")

                # Fetch skill files from GitHub
                skill_files = await self.git_provider.fetch_skill_files(
                    repo=repo,
                    branch=None,
                    tag=branch,  # Use branch/tag as git reference
                    skill_name=skill_name,
                    auth_token=self.auth_token,
                )

                # Save files to target directory
                target_dir.mkdir(parents=True, exist_ok=True)
                for file_name, content in skill_files.items():
                    file_path = target_dir / file_name
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content)
                    self.logger.debug(f"Saved {file_name} to {file_path}")

                self.logger.debug(f"Successfully fetched skill files to {target_dir}")
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

        except (OSError, IOError) as e:
            self.logger.error(
                f"Failed to clone skill '{skill_name}' from GitHub (file/command error): {e}"
            )
            return False
        except Exception as e:
            self.logger.error(
                f"Failed to clone skill '{skill_name}' from GitHub: {e}",
                exc_info=True
            )
            return False

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
        if not self.registry_url:
            self.logger.error("Cannot pull remote skill - not in hybrid mode")
            return False

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

            self.logger.debug(f"Found git_repository_url: {git_repository_url}")

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

        except (OSError, IOError) as e:
            self.logger.error(
                f"Error pulling skill '{skill_name}' (file/network error): {e}"
            )
            return False
        except Exception as e:
            self.logger.error(
                f"Unexpected error pulling skill '{skill_name}': {e}",
                exc_info=True
            )
            return False

    async def list_available_versions(
        self, skill_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get list of available versions for a skill from registry.

        Returns version info from registry API including:
        - version number
        - git_tag
        - status (published/draft/deprecated)
        - created_at, author, etc.
        """
        if not self.registry_url:
            return []

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

        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            self.logger.error(f"Failed to list versions for '{skill_name}' (network error): {e}")
            return []
        except Exception as e:
            self.logger.error(f"Failed to list versions for '{skill_name}': {e}")
            return []

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
        if not self.registry_url:
            return False

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

        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            # Non-critical - don't fail if usage reporting fails
            self.logger.debug(f"Failed to report usage (network error): {e}")
            return False
        except Exception as e:
            # Non-critical - don't fail if usage reporting fails
            self.logger.debug(f"Failed to report usage: {e}")
            return False

    def get_registry_status(self) -> Dict[str, Any]:
        """
        Get status of the skill registry.

        Shows:
        - Number of skills discovered locally
        - Number of skills metadata cached from registry
        - Total available skills
        - Registry connection status
        """
        return {
            'mode': 'hybrid' if self.registry_url else 'local-only',
            'local_skills_count': len(self.skills),
            'remote_metadata_cached': len(self.skill_metadata_cache),
            'total_skills': len(self.get_all_skills()),
            'registry_url': self.registry_url,
            'registry_authenticated': bool(self.auth_token),
            'skills_cache_dir': str(self.skills_cache_dir) if self.skills_cache_dir else None,
            'git_provider_available': self.git_provider is not None,
            'status': 'healthy' if (not self.registry_url or self.git_provider) else 'degraded',
        }

    def discover_skills(self, skills_dir: str) -> None:
        """
        Discovers all skills in a directory and populates the registry.

        Scans the specified directory for subdirectories, each representing a skill,
        and loads metadata from each skill's SKILL.md file.

        Args:
            skills_dir: The path to the main skills directory.
        """
        if not skills_dir:
            self.logger.info("No skills directory provided. Skipping skill discovery.")
            return

        resolved_skills_dir = resolve_path(skills_dir, self.project_root)

        if not resolved_skills_dir.exists() or not resolved_skills_dir.is_dir():
            self.logger.warning(f"Skills directory not found or not a directory: {resolved_skills_dir}")
            return

        self.logger.info(f"Discovering skills in: {resolved_skills_dir}")

        for skill_dir in resolved_skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            if not is_safe_path(skill_dir, resolved_skills_dir):
                self.logger.warning(f"Skipping potentially unsafe skill path: {skill_dir}")
                continue

            try:
                skill_md_path = find_skill_md(skill_dir)
                if not skill_md_path:
                    self.logger.debug(f"No SKILL.md found in {skill_dir}, skipping.")
                    continue

                skill_properties = load_metadata(skill_dir)
                self.skills[skill_properties.name] = skill_properties
                self.logger.debug(f"Successfully discovered skill: {skill_properties.name}")

            except (ParseError, ValidationError) as e:
                self.logger.warning(f"Skipping invalid skill in {skill_dir}: {e}")
            except (OSError, IOError) as e:
                self.logger.error(f"Failed to read skill in {skill_dir} (file error): {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error parsing skill in {skill_dir}: {e}", exc_info=True)

        self.logger.info(f"Discovery complete. Found {len(self.skills)} skills.")

    def get_skill(self, name: str) -> Optional[SkillProperties]:
        """
        Retrieves a skill by its name.

        Returns only locally loaded skills. For remote skills, use get_skill_metadata()
        instead, then pull_skill() to fetch the code.

        Args:
            name: The name of the skill to retrieve.

        Returns:
            A SkillProperties object if the skill is found locally, otherwise None.
        """
        # Only return locally loaded skills
        return self.skills.get(name)

    def get_skills(self, skill_names: List[str]) -> List[SkillProperties]:
        """
        Retrieves a list of skills from a list of names.

        Args:
            skill_names: A list of skill names to retrieve.

        Returns:
            A list of SkillProperties objects for the found skills.
        """
        return [skill for skill in (self.get_skill(name) for name in skill_names) if skill]

    def get_all_skills(self) -> List[SkillProperties]:
        """
        Retrieves all discovered skills (local + remote metadata).

        Returns local skills first, then any remote-only skills
        (skills in registry but not cached locally yet).

        Returns:
            A list of all SkillProperties objects in the registry, sorted by name.
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

    def generate_skills_prompt(self, skills: List[SkillProperties]) -> str:
        """
        Generates the <available_skills> XML block for the main system prompt.
        This creates a concise list of available skills (Phase 1 of Progressive Disclosure) so the agent knows what it can do.
        Params:
        skills – A list of discovered SkillProperties objects.
        Returns:
        A string containing the formatted <available_skills> block or an empty string if no skills are provided.
        """
        from oai_agent_core.components.skills.prompt import generate_skills_prompt
        return generate_skills_prompt(skills)
