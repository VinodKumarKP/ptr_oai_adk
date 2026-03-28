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


# ---------------------------------------------------------------------------
# Callable resolution
# ---------------------------------------------------------------------------

class CallableResolutionError(Exception):
    """Raised when a callable placeholder cannot be resolved."""
    pass


# Built-in filter factories — each returns a Callable[[str], bool]
_BUILTIN_FILTERS: Dict[str, Any] = {
    'endswith': lambda exts: (
        lambda path: any(path.endswith(e.strip()) for e in exts.split(','))
    ),
    'startswith': lambda prefixes: (
        lambda path: any(path.startswith(p.strip()) for p in prefixes.split(','))
    ),
    'regex': lambda pattern: (
        lambda path: bool(re.fullmatch(pattern.strip(), path))
    ),
    'contains': lambda substr: (
        lambda path: substr.strip() in path
    ),
}


class ConfigResolver(EnvResolver):
    """Extend :class:`EnvResolver` with callable placeholder resolution.

    Handles three additional placeholder prefixes on top of the standard
    ``${VAR}`` env-var syntax:

    ``filter:<type>:<arg>``
        Expands to a built-in lambda.  Supported types:

        ========================  ==========================================
        ``filter:endswith:.md``   ``lambda p: p.endswith('.md')``
        ``filter:endswith:.md,.py`` matches either extension
        ``filter:startswith:docs/`` path starts with prefix
        ``filter:regex:^docs/.*``  full regex match
        ``filter:contains:test``  substring match
        ========================  ==========================================

    ``callable:<dotted.path.to.function>``
        Imports and returns a named callable from your codebase::

            file_filter: "callable:myproject.filters.is_markdown"

    ``eval:<python_expression>``
        Evaluates a Python expression and returns its result.  **Disabled
        by default** — pass ``allow_eval=True`` to opt in.  Only use in
        environments where the config file is fully trusted::

            file_filter: "eval:lambda p: p.endswith('.md') and 'test' not in p"

    All three prefixes are recognised during :meth:`resolve` /
    :meth:`resolve_dict` so the caller does not need to do anything special.

    Args:
        strict: Inherited from :class:`EnvResolver` — raises on missing env vars.
        allow_eval: Enable the ``eval:`` prefix.  Defaults to ``False``.
        logger: Optional logger.

    Example::

        resolver = ConfigResolver()
        settings = resolver.resolve_dict({
            "repo": "langchain-ai/langchain",
            "branch": "master",
            "access_token": "${GITHUB_ACCESS_TOKEN}",
            "file_filter": "filter:endswith:.md",
        })
        # settings["file_filter"] is now a real callable lambda
        loader = GithubFileLoader(**settings)

        # Use a project-defined function instead:
        settings = resolver.resolve_dict({
            "file_filter": "callable:myproject.filters.is_markdown",
        })

        # Opt in to eval for ad-hoc expressions (trusted configs only):
        resolver = ConfigResolver(allow_eval=True)
        settings = resolver.resolve_dict({
            "file_filter": "eval:lambda p: p.endswith('.md') and 'test' not in p",
        })
    """

    _CALLABLE_PREFIXES = ('filter:', 'callable:', 'eval:')

    def __init__(
        self,
        strict: bool = True,
        allow_eval: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(strict=strict, logger=logger)
        self.allow_eval = allow_eval

    # ------------------------------------------------------------------
    # Override recursive resolver to also handle callable placeholders
    # ------------------------------------------------------------------

    def _resolve_recursive(self, value: Any) -> Any:
        """Extend parent resolution with callable placeholder handling."""
        if isinstance(value, str) and self._is_callable_placeholder(value):
            return self._resolve_callable(value)
        return super()._resolve_recursive(value)

    # ------------------------------------------------------------------
    # Callable resolution helpers
    # ------------------------------------------------------------------

    def _is_callable_placeholder(self, value: str) -> bool:
        """Return True if *value* starts with a known callable prefix."""
        return any(value.startswith(prefix) for prefix in self._CALLABLE_PREFIXES)

    def _resolve_callable(self, placeholder: str) -> Any:
        """Dispatch a callable placeholder to the appropriate resolver."""
        if placeholder.startswith('filter:'):
            return self._resolve_filter(placeholder[len('filter:'):])
        if placeholder.startswith('callable:'):
            return self._resolve_dotted_callable(placeholder[len('callable:'):])
        if placeholder.startswith('eval:'):
            return self._resolve_eval(placeholder[len('eval:'):])
        raise CallableResolutionError(f"Unknown callable prefix in: '{placeholder}'")

    def _resolve_filter(self, spec: str) -> Any:
        """Resolve a ``filter:<type>:<arg>`` placeholder to a lambda.

        Args:
            spec: Everything after the ``filter:`` prefix,
                e.g. ``"endswith:.md"`` or ``"regex:^docs/.*\\.md$"``.

        Returns:
            A ``Callable[[str], bool]``.

        Raises:
            CallableResolutionError: If the filter type is unknown.
        """
        # Split on the first colon only — the arg may itself contain colons
        # (e.g. regex patterns like "^https?://")
        parts = spec.split(':', 1)
        if len(parts) != 2:
            raise CallableResolutionError(
                f"filter: placeholder must be 'filter:<type>:<arg>', got: 'filter:{spec}'"
            )
        filter_type, arg = parts[0].strip(), parts[1].strip()

        factory = _BUILTIN_FILTERS.get(filter_type)
        if factory is None:
            supported = ', '.join(_BUILTIN_FILTERS.keys())
            raise CallableResolutionError(
                f"Unknown filter type '{filter_type}'. "
                f"Supported types: {supported}.  "
                f"For custom logic use 'callable:' or 'eval:'."
            )

        result = factory(arg)
        self.logger.debug("Resolved 'filter:%s:%s' → lambda", filter_type, arg)
        return result

    def _resolve_dotted_callable(self, dotted_path: str) -> Any:
        """Import and return a callable from a dotted module path.

        Args:
            dotted_path: Fully-qualified path, e.g.
                ``"myproject.filters.is_markdown"``.

        Returns:
            The imported callable.

        Raises:
            CallableResolutionError: If the module or attribute cannot be found.
        """
        dotted_path = dotted_path.strip()
        try:
            module_path, attr_name = dotted_path.rsplit('.', 1)
        except ValueError:
            raise CallableResolutionError(
                f"'callable:' value must be a dotted path like "
                f"'mymodule.submodule.function', got: '{dotted_path}'"
            )

        try:
            import importlib
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise CallableResolutionError(
                f"Could not import module '{module_path}': {e}"
            )

        if not hasattr(module, attr_name):
            raise CallableResolutionError(
                f"Module '{module_path}' has no attribute '{attr_name}'."
            )

        result = getattr(module, attr_name)
        if not callable(result):
            raise CallableResolutionError(
                f"'{dotted_path}' is not callable (got {type(result).__name__})."
            )

        self.logger.debug("Resolved 'callable:%s' → %r", dotted_path, result)
        return result

    def _resolve_eval(self, expression: str) -> Any:
        """Evaluate a Python expression string and return the result.

        .. warning::
            This method calls :func:`eval` on arbitrary user-supplied code.
            Only enable via ``allow_eval=True`` when the config source is
            fully trusted (e.g. a file you control in a private repo, not
            user-submitted input).

        Args:
            expression: A Python expression, typically a lambda.

        Returns:
            The result of evaluating *expression*.

        Raises:
            CallableResolutionError: If ``allow_eval`` is False or evaluation
                fails.
        """
        if not self.allow_eval:
            raise CallableResolutionError(
                "The 'eval:' placeholder is disabled by default.  "
                "Pass allow_eval=True to ConfigResolver to enable it.  "
                "Only do this when the config file is fully trusted."
            )

        expression = expression.strip()
        self.logger.warning(
            "Evaluating 'eval:' placeholder — ensure config source is trusted. "
            "Expression: %s", expression
        )
        try:
            result = eval(expression)  # noqa: S307
        except Exception as e:
            raise CallableResolutionError(
                f"Failed to evaluate expression '{expression}': {e}"
            )
        return result