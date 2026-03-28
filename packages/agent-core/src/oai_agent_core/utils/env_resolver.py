"""Utility for resolving environment variable placeholders in configuration structures.

Supports the following placeholder syntaxes anywhere inside a string value:

    ${VAR}              resolved from os.environ['VAR']
    ${VAR:-default}     resolved from os.environ.get('VAR', 'default')
    $VAR                resolved from os.environ['VAR']

Works recursively on arbitrarily nested dicts, lists, and scalar strings,
so it can be dropped onto any parsed YAML / JSON config structure.

Example usage::

    from env_resolver import EnvResolver

    raw = yaml.safe_load(open("config.yaml"))
    resolved = EnvResolver().resolve(raw)

    # Or resolve a single string:
    api_key = EnvResolver().resolve_value("${CONFLUENCE_API_KEY}")

    # Strict mode (default): raises if a required var is missing.
    # Lenient mode: leaves unresolvable placeholders as-is.
    resolved = EnvResolver(strict=False).resolve(raw)
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Union


# Matches ${VAR}, ${VAR:-default}, or bare $VAR (letter/underscore start)
_ENV_VAR_PATTERN = re.compile(
    r'\$\{([^}]+)\}'           # ${VAR} or ${VAR:-default}
    r'|\$([A-Za-z_][A-Za-z0-9_]*)'  # $VAR
)


class EnvResolutionError(Exception):
    """Raised when a required environment variable is not set and has no default."""
    pass


class EnvResolver:
    """Resolve environment variable placeholders in configuration structures.

    Walks any combination of dicts, lists, and strings, expanding every
    ``${VAR}``, ``${VAR:-default}``, or ``$VAR`` placeholder it finds.
    Non-string values (int, bool, None, etc.) are returned unchanged.

    Args:
        strict: When ``True`` (default), raises :exc:`EnvResolutionError` if a
            placeholder references an env var that is not set and carries no
            default.  When ``False``, unresolvable placeholders are left in
            the string as-is and a warning is logged.
        logger: Optional logger.  A module-level logger is used when omitted.

    Example::

        resolver = EnvResolver()

        # Resolve an entire parsed YAML config in one call
        config = resolver.resolve(yaml.safe_load(open("config.yaml")))

        # Resolve a single value
        token = resolver.resolve_value("${GITHUB_TOKEN}")

        # Lenient — never raises, useful for optional configs
        resolver = EnvResolver(strict=False)
        config = resolver.resolve(raw_config)
    """

    def __init__(
        self,
        strict: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.strict = strict
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, config: Any) -> Any:
        """Recursively resolve all env var placeholders in *config*.

        Args:
            config: Any value — typically the dict returned by
                ``yaml.safe_load`` or ``json.load``, but also accepts lists
                and plain strings.

        Returns:
            A new object of the same shape with all placeholders expanded.
            The original object is never mutated.

        Raises:
            EnvResolutionError: (strict mode only) If a placeholder references
                an env var that is not set and has no default value.
        """
        return self._resolve_recursive(config)

    def resolve_value(self, value: str) -> str:
        """Resolve all env var placeholders in a single string.

        Args:
            value: A string that may contain ``${VAR}``, ``${VAR:-default}``,
                or ``$VAR`` placeholders.

        Returns:
            The string with all placeholders expanded.

        Raises:
            EnvResolutionError: (strict mode only) If a required env var is
                missing.
        """
        if '$' not in value:
            return value
        return self._expand(value)

    def resolve_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience wrapper — resolve a flat or nested dictionary.

        Args:
            d: Dictionary to resolve.

        Returns:
            New dictionary with all string values resolved.
        """
        return self._resolve_recursive(d)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_recursive(self, value: Any) -> Any:
        """Recursively walk *value* and expand placeholders in every string."""
        if isinstance(value, dict):
            return {k: self._resolve_recursive(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_recursive(item) for item in value]
        if isinstance(value, str) and '$' in value:
            return self._expand(value)
        return value

    def _expand(self, text: str) -> str:
        """Expand all ``${VAR}`` / ``${VAR:-default}`` / ``$VAR`` in *text*."""

        def replacer(match: re.Match) -> str:
            # Group 1 → ${...} syntax; Group 2 → $VAR bare syntax
            expr = match.group(1) or match.group(2)

            if ':-' in expr:
                # ${VAR:-default}
                var_name, default = expr.split(':-', 1)
                var_name = var_name.strip()
                resolved = os.environ.get(var_name, default)
                self.logger.debug(
                    "Resolved '${%s:-...}' → %s (from %s)",
                    var_name,
                    resolved,
                    "env" if var_name in os.environ else "default",
                )
                return resolved

            # ${VAR} or $VAR — required, no default
            var_name = expr.strip()
            resolved = os.environ.get(var_name)

            if resolved is not None:
                self.logger.debug("Resolved '$%s' → [redacted]", var_name)
                return resolved

            # Variable not set
            msg = (
                f"Environment variable '{var_name}' is not set and has no "
                f"default value.  Either export it or use the "
                f"'${{{var_name}:-your_default}}' syntax in your config."
            )
            if self.strict:
                raise EnvResolutionError(msg)

            self.logger.warning("%s  Leaving placeholder as-is.", msg)
            return match.group(0)   # return original placeholder unchanged

        return _ENV_VAR_PATTERN.sub(replacer, text)