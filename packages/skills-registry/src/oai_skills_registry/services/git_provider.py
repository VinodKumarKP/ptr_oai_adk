"""
Git Provider Interface and Implementations
Handles Git integration for skill repository operations.
"""

import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import subprocess
import aiohttp


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

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def fetch_skill_files(self, repo: str, branch: Optional[str],
                                tag: Optional[str], skill_name: str,
                                auth_token: Optional[str]) -> Dict[str, str]:
        """Fetch files from GitHub using Git CLI."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Determine reference (tag or branch)
            ref = tag if tag else (branch or "main")

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

    async def fetch_skill_md_from_api(
        self,
        repo: str,
        tag: str,
        skill_name: str,
        auth_token: Optional[str] = None,
    ) -> Dict[str, str]:
        """Fetch SKILL.md (and optional skill_config.yaml) via the GitHub Contents API.

        Much faster than ``fetch_skill_files`` for listing/previewing versions because
        it avoids a full git clone.  Use this for version discovery; use
        ``fetch_skill_files`` for actual imports where the full tree is needed.

        Args:
            repo:       ``owner/repo`` string
            tag:        Git tag or branch ref
            skill_name: Skill directory name inside ``skills/``
            auth_token: Optional GitHub personal-access token

        Returns:
            Dict with keys ``"SKILL.md"`` (always present) and optionally
            ``"skill_config.yaml"``.

        Raises:
            FileNotFoundError: when ``skills/{skill_name}/SKILL.md`` is absent at *tag*
            Exception: on any other HTTP error
        """
        headers = {"Accept": "application/vnd.github.v3.raw"}
        if auth_token:
            headers["Authorization"] = f"token {auth_token}"

        params = {"ref": tag}
        files: Dict[str, str] = {}

        async with aiohttp.ClientSession() as session:
            # Fetch SKILL.md (required)
            skill_md_url = (
                f"https://api.github.com/repos/{repo}/contents"
                f"/skills/{skill_name}/SKILL.md"
            )
            async with session.get(skill_md_url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    files["SKILL.md"] = await resp.text()
                elif resp.status == 404:
                    raise FileNotFoundError(
                        f"skills/{skill_name}/SKILL.md not found at ref '{tag}'"
                    )
                else:
                    raise Exception(
                        f"GitHub API error {resp.status} fetching SKILL.md "
                        f"for {skill_name}@{tag}"
                    )

            # Fetch skill_config.yaml (optional — 404 is fine)
            config_url = (
                f"https://api.github.com/repos/{repo}/contents"
                f"/skills/{skill_name}/skill_config.yaml"
            )
            async with session.get(config_url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    files["skill_config.yaml"] = await resp.text()

        return files

    async def get_tags(self, repo: str, auth_token: Optional[str]) -> List[Dict]:
        """Get all tags from GitHub REST API, enriched with commit dates.

        Fetches ``/repos/{repo}/tags`` then concurrently retrieves the
        committer/author date for each tag so that callers can sort and display
        version timelines without extra round-trips.

        Each returned dict has at minimum:
            ``name``       – tag name (e.g. "v1.2.0")
            ``commit.sha`` – commit SHA the tag points to
            ``created_at`` – ISO-8601 date string (from GitHub commit API)
            ``message``    – commit message (first line)
            ``tagger``     – ``{"name": "<author name>"}``
        """
        headers: Dict[str, str] = {}
        if auth_token:
            headers["Authorization"] = f"token {auth_token}"

        async with aiohttp.ClientSession() as session:
            url = f"https://api.github.com/repos/{repo}/tags?per_page=100"
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    self.logger.warning(
                        "GitHub tags API returned %s for %s", resp.status, repo
                    )
                    return []
                tags = await resp.json()

            # Enrich each tag with commit date / message / author concurrently
            async def _enrich(tag: Dict) -> Dict:
                sha = tag.get("commit", {}).get("sha", "")
                if not sha:
                    return tag
                try:
                    commit_url = (
                        f"https://api.github.com/repos/{repo}/commits/{sha}"
                    )
                    async with session.get(commit_url, headers=headers) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            commit = data.get("commit", {})
                            # Prefer committer date; fall back to author date
                            tag["created_at"] = (
                                commit.get("committer", {}).get("date")
                                or commit.get("author", {}).get("date")
                            )
                            tag["message"] = commit.get("message", "").split("\n")[0]
                            tag["tagger"] = {
                                "name": commit.get("author", {}).get("name", "")
                            }
                except Exception as exc:
                    self.logger.debug(
                        "Could not fetch commit info for tag %s (%s): %s",
                        tag.get("name"), sha, exc,
                    )
                return tag

            enriched = await asyncio.gather(*[_enrich(t) for t in tags])

        return list(enriched)
