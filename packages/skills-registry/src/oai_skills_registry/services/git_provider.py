"""
Git Provider Interface and Implementations
Handles Git integration for skill repository operations.
"""

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

    async def get_tags(self, repo: str, auth_token: Optional[str]) -> List[Dict]:
        """Get tags from GitHub using REST API."""
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
