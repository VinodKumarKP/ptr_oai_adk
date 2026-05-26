"""
readme_fetcher.py — fetch README.md from local filesystem or a GitHub repository.

Priority
--------
1. Local filesystem: ``{local_base_dir}/{item_name}/README.md``
   (or any filename in the candidates list)
2. GitHub raw content API: fetched from ``{repo_url}`` at ``{subpath}/{filename}``
   with in-process TTL caching.

Usage
-----
from oai_platform_core.readme_fetcher import read_local_readme, fetch_readme

# Local first, GitHub fallback
content, meta = read_local_readme("/opt/agents", "git_pr_insight_agent")
if content is None:
    content, meta = await fetch_readme(
        repo_url="https://github.com/owner/repo.git",
        cache_key="agent:git_pr_insight_agent",
        github_token=os.getenv("GITHUB_TOKEN"),
        subpath="agentic_registry_agents/agents/git_pr_insight_agent",
    )
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

# { cache_key: {"content": str, "fetched_at": float, "branch": str, "url": str, "local": bool} }
_readme_cache: Dict[str, Dict[str, Any]] = {}

# Default TTL: 1 hour.  Override per-call or via env var README_CACHE_TTL_SECONDS.
_DEFAULT_TTL = int(os.getenv("README_CACHE_TTL_SECONDS", "3600"))

# Branches tried in order when no explicit branch is given
_BRANCH_CANDIDATES = ("main", "master")

# README filenames tried in order (case-sensitive on GitHub raw)
_README_CANDIDATES = ("README.md", "readme.md", "README.MD", "Readme.md")

# Skill-specific candidates (SKILL.md preferred for skills)
_SKILL_README_CANDIDATES = ("README.md", "SKILL.md", "readme.md")


def invalidate_readme_cache(cache_key: str) -> bool:
    """Remove a single entry from the cache.  Returns True if it existed."""
    return _readme_cache.pop(cache_key, None) is not None


def clear_readme_cache() -> int:
    """Clear the entire cache.  Returns the number of entries removed."""
    count = len(_readme_cache)
    _readme_cache.clear()
    return count


def readme_cache_stats() -> Dict[str, Any]:
    """Return lightweight cache statistics (no content)."""
    now = time.time()
    entries = []
    for key, entry in _readme_cache.items():
        age = now - entry["fetched_at"]
        entries.append({
            "key":        key,
            "branch":     entry.get("branch"),
            "local":      entry.get("local", False),
            "age_seconds": int(age),
            "expires_in":  max(0, int(_DEFAULT_TTL - age)),
        })
    return {"ttl_seconds": _DEFAULT_TTL, "entries": entries}


# ---------------------------------------------------------------------------
# Local filesystem reader
# ---------------------------------------------------------------------------

def read_local_readme(
    local_base_dir: str,
    item_name: str,
    filenames: Optional[List[str]] = None,
    ttl: int = _DEFAULT_TTL,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Read a README from the local filesystem.

    Looks for ``{local_base_dir}/{item_name}/{filename}`` where *filename* is
    tried in the order given by *filenames* (default: README.md, SKILL.md,
    readme.md).

    Returns
    -------
    (content, meta)
        ``content`` is the file text, or ``None`` if nothing was found.
        ``meta`` keys: ``local``, ``file_path``, ``cached``.
    """
    if filenames is None:
        filenames = list(_README_CANDIDATES) + ["SKILL.md"]

    cache_key = f"local:{local_base_dir}:{item_name}"

    # ── Cache hit ────────────────────────────────────────────────────────────
    cached = _readme_cache.get(cache_key)
    if cached and (time.time() - cached["fetched_at"]) < ttl:
        logger.debug("README local cache hit for '%s'", item_name)
        return cached["content"], {
            "local":     True,
            "cached":    True,
            "file_path": cached.get("file_path"),
        }

    # ── Filesystem read ──────────────────────────────────────────────────────
    base = Path(local_base_dir) / item_name
    for fname in filenames:
        candidate = base / fname
        if candidate.is_file():
            try:
                content = candidate.read_text(encoding="utf-8")
                _readme_cache[cache_key] = {
                    "content":    content,
                    "fetched_at": time.time(),
                    "local":      True,
                    "file_path":  str(candidate),
                }
                logger.info(
                    "README read from local file '%s' (%d bytes)",
                    candidate, len(content)
                )
                return content, {
                    "local":     True,
                    "cached":    False,
                    "file_path": str(candidate),
                }
            except OSError as exc:
                logger.warning("Could not read local README '%s': %s", candidate, exc)

    logger.debug("No local README found for '%s' in %s", item_name, local_base_dir)
    return None, {"local": True, "cached": False, "error": f"No README found in {base}"}


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def _parse_github_repo(url: str) -> Optional[str]:
    """Return 'owner/repo' from a GitHub URL, or None if not a GitHub URL."""
    url = url.strip().rstrip("/")
    # SSH:  git@github.com:owner/repo.git
    ssh = re.match(r"git@github\.com:([^/]+/[^/]+?)(?:\.git)?$", url)
    if ssh:
        return ssh.group(1)
    # HTTPS: https://github.com/owner/repo[.git]
    https = re.match(r"https?://github\.com/([^/]+/[^/]+?)(?:\.git)?$", url)
    if https:
        return https.group(1)
    return None


def _raw_url(owner_repo: str, branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{path}"


# ---------------------------------------------------------------------------
# GitHub fetcher
# ---------------------------------------------------------------------------

async def fetch_readme(
    repo_url: str,
    cache_key: str,
    github_token: Optional[str] = None,
    ttl: int = _DEFAULT_TTL,
    subpath: Optional[str] = None,
    filenames: Optional[List[str]] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Fetch README.md from a GitHub repository with in-process TTL caching.

    Parameters
    ----------
    repo_url:
        The git repository URL, e.g. ``https://github.com/owner/repo.git``
    cache_key:
        Unique identifier for the cache entry.
    github_token:
        Optional GitHub PAT for private repositories.
    ttl:
        Cache lifetime in seconds (default: README_CACHE_TTL_SECONDS env var or 3600).
    subpath:
        Optional subdirectory path within the repository, e.g.
        ``"mcp_registry_servers/servers/jira_server"``.  When given, the README
        is fetched from ``{subpath}/{filename}`` rather than the repo root.
    filenames:
        Ordered list of filenames to try.  Defaults to standard README candidates.

    Returns
    -------
    (content, meta)
        ``content`` is the README text, or ``None`` if not found.
        ``meta`` is a dict with keys: ``cached``, ``branch``, ``url``, ``error``.
    """
    if filenames is None:
        filenames = list(_README_CANDIDATES)

    # ── Cache hit ────────────────────────────────────────────────────────────
    cached = _readme_cache.get(cache_key)
    if cached and (time.time() - cached["fetched_at"]) < ttl:
        logger.debug("README cache hit for '%s'", cache_key)
        return cached["content"], {
            "cached":  True,
            "branch":  cached["branch"],
            "url":     cached["url"],
        }

    # ── Parse URL ────────────────────────────────────────────────────────────
    owner_repo = _parse_github_repo(repo_url)
    if not owner_repo:
        logger.warning("README fetch: unrecognised git URL for '%s': %s", cache_key, repo_url)
        return None, {"cached": False, "error": f"Not a recognised GitHub URL: {repo_url}"}

    # ── Build path prefix ─────────────────────────────────────────────────────
    path_prefix = f"{subpath.strip('/')}/" if subpath else ""

    # ── Fetch from GitHub ─────────────────────────────────────────────────────
    headers: Dict[str, str] = {"Accept": "text/plain"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        for branch in _BRANCH_CANDIDATES:
            for filename in filenames:
                url = _raw_url(owner_repo, branch, f"{path_prefix}{filename}")
                try:
                    resp = await client.get(url, headers=headers)
                    if resp.status_code == 200:
                        content = resp.text
                        _readme_cache[cache_key] = {
                            "content":    content,
                            "fetched_at": time.time(),
                            "branch":     branch,
                            "url":        url,
                            "local":      False,
                        }
                        logger.info(
                            "README fetched for '%s' from %s (%d bytes)",
                            cache_key, url, len(content)
                        )
                        return content, {
                            "cached":      False,
                            "branch":      branch,
                            "url":         url,
                            "source_url":  url,
                        }
                    if resp.status_code == 401:
                        return None, {"cached": False, "error": "GitHub authentication failed (check GITHUB_TOKEN)"}
                    if resp.status_code == 403:
                        return None, {"cached": False, "error": "GitHub rate limit or access denied"}
                except httpx.RequestError as exc:
                    logger.warning("README fetch request error for '%s': %s", cache_key, exc)

    logger.warning("README not found for '%s' in repo %s (subpath: %s)", cache_key, repo_url, subpath)
    return None, {"cached": False, "error": "README not found in repository"}
