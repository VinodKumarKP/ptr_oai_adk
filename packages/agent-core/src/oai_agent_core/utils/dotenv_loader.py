"""Minimal, dependency-free ``.env`` loader.

Populates ``os.environ`` from a ``{config_root}/.env`` file when present, so that
``${VAR}`` placeholders in YAML configs, env-based settings (tracing/metrics
endpoints, credentials, etc.), and ``os.environ`` lookups all see the values.

Behaviour mirrors the common ``python-dotenv`` defaults:

- Lines are ``KEY=VALUE`` (an optional leading ``export`` is accepted).
- Blank lines and lines starting with ``#`` are ignored.
- Surrounding single/double quotes around the value are stripped.
- For *unquoted* values, an inline ``# comment`` (preceded by whitespace) is
  dropped.
- By default existing ``os.environ`` keys are **not** overwritten, so the real
  process environment wins over the file. Pass ``override=True`` to flip this.
"""

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

ENV_FILENAME = ".env"


def _parse_env_line(line: str) -> Optional[Tuple[str, str]]:
    """Parse a single ``.env`` line into ``(key, value)`` or ``None`` to skip."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export ") or line.startswith("export\t"):
        line = line[len("export"):].lstrip()
    if "=" not in line:
        return None

    key, _, value = line.partition("=")
    key = key.strip()
    if not key:
        return None

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        # Quoted value: take content verbatim (no comment stripping).
        value = value[1:-1]
    else:
        # Unquoted: drop an inline comment introduced by whitespace + '#'.
        for marker in (" #", "\t#"):
            idx = value.find(marker)
            if idx != -1:
                value = value[:idx]
        value = value.strip()

    return key, value


def load_dotenv(config_root: Optional[str], override: bool = False) -> int:
    """Load ``{config_root}/.env`` into ``os.environ`` if the file exists.

    Args:
        config_root: Directory expected to contain a ``.env`` file. If ``None``
            or the file is absent, this is a no-op.
        override: When ``False`` (default), keys already present in
            ``os.environ`` are left untouched (real env wins). When ``True``,
            ``.env`` values overwrite existing ones.

    Returns:
        The number of variables written to ``os.environ``.
    """
    if not config_root:
        return 0

    env_path = os.path.join(config_root, ENV_FILENAME)
    if not os.path.isfile(env_path):
        return 0

    applied = 0
    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                parsed = _parse_env_line(raw_line)
                if parsed is None:
                    continue
                key, value = parsed
                if not override and key in os.environ:
                    continue
                os.environ[key] = value
                applied += 1
        logger.info("Loaded %d environment variable(s) from %s", applied, env_path)
    except Exception as exc:  # never let env loading break startup
        logger.warning("Failed to load .env from %s: %s", env_path, exc)

    return applied
