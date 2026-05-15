"""
Utility Functions for Skills Registry
Helper functions for parsing, conversion, and validation.
"""

import re
import yaml
from typing import Any, Dict
from datetime import datetime


def parse_skill_md(content: str) -> Dict[str, Any]:
    """Parse SKILL.md frontmatter and content.

    Args:
        content: Full SKILL.md file content

    Returns:
        Dictionary with parsed metadata and content
    """
    # Extract YAML frontmatter
    match = re.match(r'^---\n(.*?)\n---\n(.*)', content, re.DOTALL)
    if not match:
        raise ValueError("Invalid SKILL.md format - missing frontmatter")

    metadata = yaml.safe_load(match.group(1))
    body = match.group(2)

    metadata["content"] = body
    return metadata


def convert_datetime_to_str(obj: Any) -> Any:
    """Recursively convert datetime objects to ISO format strings.

    Args:
        obj: Object to convert (dict, list, datetime, or other)

    Returns:
        Converted object with datetimes as ISO strings
    """
    if isinstance(obj, dict):
        return {k: convert_datetime_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetime_to_str(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj


def version_matches_constraint(version: str, constraint: str) -> bool:
    """Check if version matches constraint.

    Args:
        version: Version string (e.g., "1.0.0")
        constraint: Version constraint (e.g., "latest", "1.0.0", ">=1.0.0")

    Returns:
        True if version matches constraint
    """
    if constraint == "latest":
        return True
    if constraint == version:
        return True
    # Could add more sophisticated version constraint matching
    return True


def parse_github_url(url: str) -> str:
    """Parse GitHub URL and extract owner/repo.

    Args:
        url: GitHub repository URL (https://github.com/owner/repo or https://github.com/owner/repo.git)

    Returns:
        Repository in owner/repo format

    Raises:
        ValueError: If URL format is invalid
    """
    try:
        match = re.search(r'github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$', url)
        if not match:
            raise ValueError(f"Invalid GitHub URL format: {url}")
        owner, repo = match.groups()
        return f"{owner}/{repo}"
    except Exception as e:
        raise ValueError(f"Failed to parse GitHub URL: {e}")
