"""
AgentDiscovery — fetches and parses agent config YAMLs from a GitHub repository.

Expected repository structure (checked in order):
  {repo}/agentic_registry_agents/agents_config/*.yaml   ← primary
  {repo}/agents_config/*.yaml                           ← fallback 1
  {repo}/agents/*.yaml                                  ← fallback 2
  {repo}/config/agents/*.yaml                           ← fallback 3
  {repo}/configs/*.yaml                                 ← fallback 4
  {repo}/configs/agents/*.yaml                          ← fallback 5

A custom ``config_path`` can be supplied at call time to bypass auto-detection.

Each YAML file is expected to contain at minimum a ``name`` (or its filename is
used as the agent name) and optional ``description``, ``type``, ``tags``,
``prompts``, ``port``, and ``source`` fields.

YAML ``type`` → registry ``framework`` mapping:
  langgraph   → langgraph
  openai      → openai
  crewai      → crewai
  aws-strands → strands
  strands     → strands
  mcp         → openai
  (anything else) → None (user selects at import time)
"""
from __future__ import annotations

import logging
import os
import re
from io import StringIO
from typing import Any, Dict, List, Optional

import httpx

# Matches ${VAR} and ${VAR:-default}
_PLACEHOLDER_RE = re.compile(r'^\$\{([^}:]+)(?::-([^}]*))?\}$')

# Heuristic: variable names containing these substrings are treated as sensitive
# when the YAML does not provide an explicit `env_sensitive:` list.
_SENSITIVE_PATTERNS = frozenset([
    "password", "passwd", "pwd", "secret", "token", "apikey", "api_key",
    "auth", "credential", "private", "cert", "key", "signature", "access_key",
    "client_secret",
])

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

AGENT_TYPE_TO_FRAMEWORK: Dict[str, str] = {
    "langgraph": "langgraph",
    "openai": "openai",
    "crewai": "crewai",
    "aws-strands": "strands",
    "strands": "strands",
    "mcp": "openai",
}

# Candidate directories (tried in order, first non-empty wins).
# A caller-supplied ``config_path`` overrides this list entirely.
_CONFIG_DIRS = [
    "agentic_registry_agents/agents_config",
    "agents_config",
    "agents",
    "config/agents",
    "configs",
    "configs/agents",
]

_GITHUB_CONTENTS_API = "https://api.github.com/repos/{repo}/contents/{path}"


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


def _normalize_agent_name(raw: str) -> str:
    """Lower-case, strip non-identifier chars, collapse to underscores."""
    return re.sub(r"[^a-z0-9]+", "_", raw.lower().strip()).strip("_")


def _is_sensitive_by_name(name: str) -> bool:
    """Return True if the variable name suggests it holds a secret value."""
    lower = name.lower()
    return any(pat in lower for pat in _SENSITIVE_PATTERNS)


def _parse_env_requirements(
    env_section: Any,
    explicit_sensitive: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract env var requirements from the ``env:`` block of an agent YAML.

    Each entry in *env_section* is expected to be one of:
      - ``KEY: ${KEY}``            → name=KEY, default=None, required=True
      - ``KEY: ${KEY:-fallback}``  → name=KEY, default="fallback", required=False
      - ``KEY: literal``           → name=KEY, default="literal", required=False
      - ``KEY: null / ""``         → name=KEY, default=None, required=True

    Sensitivity is determined by (in priority order):
      1. ``explicit_sensitive`` list from the YAML ``env_sensitive:`` key
      2. Heuristic pattern matching on the variable name

    Returns a list of dicts suitable for constructing ``EnvVarRequirement`` models.
    """
    if not isinstance(env_section, dict):
        return []

    explicit_set = {str(v).strip().upper() for v in (explicit_sensitive or [])}

    requirements = []
    for key, value in env_section.items():
        name = str(key).strip()
        str_val = str(value).strip() if value is not None else ""

        # Determine sensitivity
        sensitive = (name.upper() in explicit_set) or _is_sensitive_by_name(name)

        m = _PLACEHOLDER_RE.match(str_val)
        if m:
            # group(1) = var name inside ${}, group(2) = default after :-
            default = m.group(2)  # None when no :- was present
            requirements.append({
                "name": name,
                "default": default,
                "required": default is None,
                "sensitive": sensitive,
            })
        elif str_val:
            # Literal value — treat as optional with that value as default
            requirements.append({
                "name": name,
                "default": str_val,
                "required": False,
                "sensitive": sensitive,
            })
        else:
            # Empty / null value — required, no default
            requirements.append({
                "name": name,
                "default": None,
                "required": True,
                "sensitive": sensitive,
            })

    return requirements


def _load_yaml(text: str) -> Optional[Dict]:
    """Parse a YAML string.  Tries ruamel.yaml first, falls back to PyYAML."""
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

class AgentDiscovery:
    """Discovers agent config YAMLs from a GitHub repository via the Contents API."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def discover(
        self,
        git_repository_url: str,
        existing_agent_names: Optional[List[str]] = None,
        auth_token: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Scan *git_repository_url* for agent config YAMLs and return metadata.

        Args:
            git_repository_url:  Full GitHub repository URL.
            existing_agent_names: Agent names already in the registry
                (used to set ``status = "already_registered"``).
            auth_token:          Optional GitHub PAT for private repos.
            config_path:         Optional explicit path inside the repo that
                contains agent YAML files (e.g. ``"my_project/agent_configs"``).
                When supplied, auto-detection is skipped and only this path is
                tried.

        Returns:
            Dict with keys: git_repository_url, total_found,
            available_to_register, already_registered, invalid, agents.
        """
        repo = _parse_github_repo(git_repository_url)
        if not repo:
            raise ValueError(f"Cannot parse GitHub URL: {git_repository_url!r}")

        existing = set(existing_agent_names or [])

        # Resolve authentication: explicit token > GITHUB_TOKEN env var > none
        resolved_token = auth_token or os.environ.get("GITHUB_TOKEN")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if resolved_token:
            headers["Authorization"] = f"token {resolved_token}"
            self.logger.info(
                "GitHub auth: using %s",
                "explicit token" if auth_token else "GITHUB_TOKEN env var",
            )
        else:
            self.logger.info("GitHub auth: unauthenticated (public repo only)")

        async with httpx.AsyncClient(timeout=30.0) as client:
            yaml_entries = await self._find_yaml_entries(
                client, repo, headers, config_path=config_path
            )

            agents: List[Dict[str, Any]] = []
            for entry in yaml_entries:
                info = await self._parse_yaml_entry(
                    client, entry, git_repository_url, headers
                )
                if info is not None:
                    # Assign status
                    if info.get("error"):
                        info["status"] = "invalid"
                    elif info["name"] in existing:
                        info["status"] = "already_registered"
                    else:
                        info["status"] = "available"
                    agents.append(info)

        total = len(agents)
        available = sum(1 for a in agents if a["status"] == "available")
        registered = sum(1 for a in agents if a["status"] == "already_registered")
        invalid = sum(1 for a in agents if a["status"] == "invalid")

        return {
            "git_repository_url": git_repository_url,
            "total_found": total,
            "available_to_register": available,
            "already_registered": registered,
            "invalid": invalid,
            "agents": agents,
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    async def _find_yaml_entries(
        self,
        client: httpx.AsyncClient,
        repo: str,
        headers: Dict[str, str],
        config_path: Optional[str] = None,
    ) -> List[Dict]:
        """Return the list of ``.yaml`` file entries from the first matching dir.

        If *config_path* is supplied it is tried exclusively (no auto-detection).
        Otherwise all paths in ``_CONFIG_DIRS`` are tried in order.
        """
        candidates = [config_path.strip("/").strip()] if config_path else _CONFIG_DIRS

        for candidate in candidates:
            url = _GITHUB_CONTENTS_API.format(repo=repo, path=candidate)
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                self.logger.debug(
                    "Contents API returned %s for %s/%s", resp.status_code, repo, candidate
                )
                if config_path:
                    # User supplied an explicit path that doesn't exist — surface it clearly
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
                and not e.get("name", "").startswith("template")  # skip template.yaml
            ]
            if yaml_files:
                self.logger.info(
                    "Found %d agent YAML(s) in %s/%s", len(yaml_files), repo, candidate
                )
                return yaml_files

        self.logger.info("No agent config YAML files found in %s", repo)
        return []

    async def _parse_yaml_entry(
        self,
        client: httpx.AsyncClient,
        entry: Dict,
        repo_url: str,
        headers: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """Fetch and parse a single YAML entry, returning a dict of agent metadata."""
        config_file = entry.get("name", "unknown.yaml")
        # Prefer the raw download URL; fall back to the API URL with raw accept header
        fetch_url = entry.get("download_url") or entry.get("url", "")

        try:
            fetch_headers = {**headers, "Accept": "application/vnd.github.v3.raw"}
            resp = await client.get(fetch_url, headers=fetch_headers, follow_redirects=True)

            if resp.status_code != 200:
                return {
                    "name": _normalize_agent_name(config_file.replace(".yaml", "")),
                    "config_file": config_file,
                    "error": f"HTTP {resp.status_code} fetching YAML",
                }

            data = _load_yaml(resp.text)
            if not isinstance(data, dict):
                return {
                    "name": _normalize_agent_name(config_file.replace(".yaml", "")),
                    "config_file": config_file,
                    "error": "YAML did not parse to a mapping",
                }

            # --- Extract fields ---
            # Always derive the agent identity from the YAML *filename* so that
            # each file produces a unique key even when multiple YAMLs share the
            # same 'name:' field value.
            name_key = _normalize_agent_name(config_file.replace(".yaml", ""))

            agent_type = str(data.get("type", "")).strip()
            framework = AGENT_TYPE_TO_FRAMEWORK.get(agent_type)

            tags = data.get("tags") or []
            prompts = data.get("prompts") or []
            port_raw = data.get("port")
            source = str(data.get("source") or repo_url)
            # Use the YAML 'name' field as display label in description if different
            yaml_name = str(data.get("name") or "").strip()
            description = str(data.get("description") or "")
            if yaml_name and yaml_name != name_key:
                description = f"{yaml_name} — {description}" if description else yaml_name

            # Parse required environment variables from the `env:` section.
            # `env_sensitive:` is an optional list of var names that are explicitly
            # flagged as sensitive (e.g. passwords, API keys). Names not in this
            # list fall back to the heuristic pattern matcher.
            env_section = data.get("env") or {}
            env_sensitive_list = data.get("env_sensitive") or []
            required_env = _parse_env_requirements(env_section, explicit_sensitive=env_sensitive_list)

            return {
                "name": name_key,
                "description": description,
                "framework": framework,
                "agent_type": agent_type or None,
                "tags": list(tags) if isinstance(tags, (list, tuple)) else [],
                "prompts": list(prompts) if isinstance(prompts, (list, tuple)) else [],
                "port": int(port_raw) if port_raw else None,
                "source": source,
                "config_file": config_file,
                "error": None,
                "required_env": required_env,
            }

        except Exception as exc:
            self.logger.warning("Failed to parse YAML %s: %s", config_file, exc)
            return {
                "name": _normalize_agent_name(config_file.replace(".yaml", "")),
                "config_file": config_file,
                "error": str(exc),
            }
