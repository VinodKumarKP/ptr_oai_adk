"""Utility functions for the Agent Skills component."""

from pathlib import Path


def resolve_path(path_str: str, project_root: str = None) -> Path:
    """
    Resolves a path, handling paths relative to a project root.

    Args:
        path_str: The path string from the configuration.
        project_root: The absolute path to the project's root directory.

    Returns:
        A resolved, absolute Path object.
    """
    if project_root and path_str and path_str.startswith('.'):
        # Interpret '.' as the project root
        base_path = Path(project_root)
        relative_part = path_str.lstrip('./')
        return (base_path / relative_part).resolve()
    else:
        return Path(path_str).expanduser().resolve()


def is_safe_path(path_to_check: Path, base_dir: Path) -> bool:
    """
    Validates that a path is safely contained within a base directory.

    This is a security measure to prevent directory traversal attacks where a
    maliciously crafted path (e.g., "../../../etc/passwd") could access
    sensitive files outside the intended directory.

    Args:
        path_to_check: The path to validate.
        base_dir: The directory that should contain the path.

    Returns:
        True if the path is safe, False otherwise.
    """
    try:
        # Ensure both paths are absolute and resolved.
        resolved_path = path_to_check.resolve()
        resolved_base = base_dir.resolve()
        # Check if the resolved path is within the base directory.
        resolved_path.relative_to(resolved_base)
        return True
    except (ValueError, OSError, RuntimeError):
        # ValueError is raised if the path is not within the base.
        # Other errors can occur for invalid paths.
        return False
