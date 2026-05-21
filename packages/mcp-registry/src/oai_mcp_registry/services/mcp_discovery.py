"""
MCPDiscovery — fetches and parses MCP server config YAMLs from a GitHub repository.

Expected repository structure (checked in order):
  {repo}/mcp_registry_servers/servers_config/*.yaml   ← primary
  {repo}/servers_config/*.yaml                        ← fallback 1
  {repo}/mcp_servers_config/*.yaml                    ← fallback 2
  {repo}/config/servers/*.yaml                        ← fallback 3
  {repo}/servers/*.yaml                               ← fallback 4

A custom ``config_path`` can be supplied at call time to bypass auto-detection.
Authentication uses explicit ``auth_token`` first, then falls back to the
``GITHUB_TOKEN`` environment variable so private repositories work without
passing tokens from the UI.
"""
from __future__ import annotations

import logging
import os
import re
from io import StringIO
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Candidate directories (tried in order, first non-empty wins)
_CONFIG_DIRS = [
    "mcp_registry_servers/servers_config",
    "servers_config",
    "mcp_servers_config",
    "config/servers",
    "servers",
]

_GITHUB_CONTENTS_API = "https://api.github.com/repos/{repo}/contents/{path}"

# Heuristic patterns that mark a variable name as sensitive
_SENSITIVE_PATTERNS = frozenset([
    "password", "passwd", "pwd", "secret", "token", "key", "apikey",
    "api_key", "auth", "credential", "private", "cert", "certificate",
])


def _is_sensitive_by_name(name: str) -> bool:
    """Return True if the variable name looks like it holds a secret."""
    lower = name.lower()
    return any(pat in lower for pat in _SENSITIVE_PATTERNS)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _parse_github_repo(git_url: str) -> Optional[str]:
    """Return ``'owner/repo'`` from a GitHub HTTPS or .git URL, else ``None``."""
    m = re.match(
        r"https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$",
        git_url.strip(),
    )
    return m.group(1) if m else None


def _normalize_server_name(raw: str) -> str:
    """Lower-case, strip non-identifier chars, collapse to underscores."""
    return re.sub(r"[^a-z0-9]+", "_", raw.lower().strip()).strip("_")


def _load_yaml(text: str) -> Optional[Dict]:
    """Parse a YAML string. Tries ruamel.yaml first, falls back to PyYAML."""
    try:
        from ruamel.yaml import YAML as _RuamelYAML
        _yaml = _RuamelYAML()
        data = _yaml.load(StringIO(text))
        return dict(data) if data else None
    except Exception:
        pass
    try:
        import yaml as _pyyaml
        return _pyyaml.safe_load(text)
    except Exception as exc:
        logger.debug("YAML parse failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Main service
# --------------------------------------------------------------------------- #

class MCPDiscovery:
    """Discovers MCP server config YAMLs from a GitHub repository via the Contents API."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    async def discover(
        self,
        git_repository_url: str,
        existing_server_names: Optional[List[str]] = None,
        auth_token: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Scan *git_repository_url* for MCP server config YAMLs and return metadata.

        Args:
            git_repository_url:   Full GitHub repository URL.
            existing_server_names: Server names already in the registry
                (used to set ``status = "already_registered"``).
            auth_token:           Optional GitHub PAT for private repos.
            config_path:          Optional explicit path inside the repo that
                contains server YAML files. When supplied, auto-detection is
                skipped and only this path is tried.

        Returns:
            Dict with keys: git_repository_url, total_found,
            available_to_register, already_registered, invalid, servers.
        """
        repo = _parse_github_repo(git_repository_url)
        if not repo:
            raise ValueError(f"Cannot parse GitHub URL: {git_repository_url!r}")

        existing = set(existing_server_names or [])

        # Resolve auth: explicit token > GITHUB_TOKEN env var > unauthenticated
        resolved_token = auth_token or os.environ.get("GITHUB_TOKEN")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if resolved_token:
            headers["Authorization"] = f"token {resolved_token}"
            self.logger.debug(
                "GitHub auth: using %s",
                "explicit token" if auth_token else "GITHUB_TOKEN env var",
            )
        else:
            self.logger.debug("GitHub auth: unauthenticated (public repo only)")

        async with httpx.AsyncClient(timeout=30.0) as client:
            yaml_entries = await self._find_yaml_entries(
                client, repo, headers, config_path=config_path
            )

            servers: List[Dict[str, Any]] = []
            for entry in yaml_entries:
                info = await self._parse_yaml_entry(
                    client, entry, git_repository_url, headers
                )
                if info is not None:
                    if info.get("error"):
                        info["status"] = "invalid"
                    elif info["name"] in existing:
                        info["status"] = "already_registered"
                    else:
                        info["status"] = "available"
                    servers.append(info)

        total = len(servers)
        available = sum(1 for s in servers if s["status"] == "available")
        registered = sum(1 for s in servers if s["status"] == "already_registered")
        invalid = sum(1 for s in servers if s["status"] == "invalid")

        return {
            "git_repository_url": git_repository_url,
            "total_found": total,
            "available_to_register": available,
            "already_registered": registered,
            "invalid": invalid,
            "servers": servers,
        }

    async def _find_yaml_entries(
        self,
        client: httpx.AsyncClient,
        repo: str,
        headers: Dict[str, str],
        config_path: Optional[str] = None,
    ) -> List[Dict]:
        """Return `.yaml` file entries from the first matching directory."""
        candidates = [config_path.strip("/").strip()] if config_path else _CONFIG_DIRS

        for candidate in candidates:
            url = _GITHUB_CONTENTS_API.format(repo=repo, path=candidate)
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                self.logger.debug(
                    "Contents API returned %s for %s/%s", resp.status_code, repo, candidate
                )
                if config_path:
                    self.logger.warning(
                        "Explicit config_path '%s' not found in %s (HTTP %s)",
                        config_path, repo, resp.status_code,
                    )
                continue

            entries = resp.json()
            if not isinstance(entries, list):
                continue

            yaml_files = [
                e for e in entries
                if isinstance(e, dict)
                and e.get("type") == "file"
                and e.get("name", "").endswith(".yaml")
                and not e.get("name", "").startswith("template")
            ]
            if yaml_files:
                self.logger.info(
                    "Found %d MCP server YAML(s) in %s/%s", len(yaml_files), repo, candidate
                )
                return yaml_files

        self.logger.info("No MCP server config YAML files found in %s", repo)
        return []

    async def _parse_yaml_entry(
        self,
        client: httpx.AsyncClient,
        entry: Dict,
        repo_url: str,
        headers: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """Fetch and parse a single YAML entry, returning a dict of server metadata."""
        config_file = entry.get("name", "unknown.yaml")
        fetch_url = entry.get("download_url") or entry.get("url", "")

        try:
            fetch_headers = {**headers, "Accept": "application/vnd.github.v3.raw"}
            resp = await client.get(fetch_url, headers=fetch_headers, follow_redirects=True)

            if resp.status_code != 200:
                return {
                    "name": _normalize_server_name(config_file.replace(".yaml", "")),
                    "config_file": config_file,
                    "error": f"HTTP {resp.status_code} fetching YAML",
                }

            data = _load_yaml(resp.text)
            if not isinstance(data, dict):
                return {
                    "name": _normalize_server_name(config_file.replace(".yaml", "")),
                    "config_file": config_file,
                    "error": "YAML did not parse to a mapping",
                }

            # Always use filename as the unique identity key
            name_key = _normalize_server_name(config_file.replace(".yaml", ""))

            tags = data.get("tags") or []
            port_raw = data.get("port")
            source = str(data.get("source") or data.get("source_url") or repo_url)

            # Surface YAML name in description if it differs from filename
            yaml_name = str(data.get("name") or "").strip()
            description = str(data.get("description") or "")
            if yaml_name and yaml_name != name_key:
                description = f"{yaml_name} — {description}" if description else yaml_name

            # ── Env vars ──────────────────────────────────────────────────
            # Accept both dict ({VAR: value}) and list ([{name: VAR, default: x}])
            raw_env = data.get("env") or data.get("env_vars") or {}
            env_vars: Dict[str, str] = {}
            if isinstance(raw_env, dict):
                env_vars = {str(k).upper(): str(v) for k, v in raw_env.items()}
            elif isinstance(raw_env, list):
                for item in raw_env:
                    if isinstance(item, dict):
                        k = str(item.get("name") or "").upper()
                        v = str(item.get("default") or item.get("value") or "")
                        if k:
                            env_vars[k] = v

            # Explicit sensitive list from YAML, supplemented by heuristic
            explicit_sensitive = [
                str(n).upper()
                for n in (data.get("env_sensitive") or data.get("sensitive_vars") or [])
            ]
            sensitive_vars: List[str] = list({
                name
                for name in env_vars
                if name in explicit_sensitive or _is_sensitive_by_name(name)
            })

            return {
                "name": name_key,
                "description": description,
                "tags": list(tags) if isinstance(tags, (list, tuple)) else [],
                "port": int(port_raw) if port_raw else None,
                "source": source,
                "config_file": config_file,
                "env_vars": env_vars,
                "sensitive_vars": sensitive_vars,
                "error": None,
            }

        except Exception as exc:
            self.logger.warning("Failed to parse YAML %s: %s", config_file, exc)
            return {
                "name": _normalize_server_name(config_file.replace(".yaml", "")),
                "config_file": config_file,
                "error": str(exc),
            }
