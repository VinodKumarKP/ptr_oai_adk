"""Central configuration for the Agent HTTP Server.

Every environment variable the server module itself reads is declared here
once — name, default and parsing rule in one place — so the same setting can't
end up with different defaults in different modules. See
``docs/ENVIRONMENT_VARIABLES.md`` for the operator-facing reference.

**Values are read from ``os.environ`` on each access**, not snapshotted at
construction. That is deliberate: the server consults some settings while the
app is being built and others during ``startup()``, and tests set environment
variables between those two points. Reading live keeps the historical timing
semantics exactly; an env lookup is cheap enough that caching would buy
nothing but surprises.

**Not covered here:** the authentication settings read in
``security/dependencies.py``. Those intentionally go through
``get_original_environ()`` — the pre-request-isolation snapshot — so that
per-request environment overrides can never influence an auth decision. Do not
migrate them to this module.
"""

import os
from typing import List, Optional, Set

TRUE_VALUES = ("1", "true", "yes", "on")


def env_flag(name: str, default: bool = False) -> bool:
    """Reads a boolean environment variable.

    Accepts true/1/yes/on case-insensitively; anything else present is False.
    Needed because ``os.environ.get(name, False)`` returns the *string*
    ``"false"`` when the variable is set to that, which is truthy — the
    opposite of what the operator asked for.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in TRUE_VALUES


def env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None else value


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_csv(name: str) -> Optional[List[str]]:
    """Comma-separated list, or None when the variable is unset.

    An empty string is a meaningful value (an explicitly empty list), which is
    why "unset" is signalled with None rather than an empty list.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


class ServerSettings:
    """Environment-backed settings for :class:`AgentHTTPServer`."""

    # -- profile ------------------------------------------------------------

    #: Surfaces enabled unless overridden by ALWAYS_ACTIVE_MODES.
    DEFAULT_MODES: Set[str] = {
        "health", "agent", "chat", "logs", "a2a", "monitoring", "token", "readme",
    }
    #: Surfaces enabled by AGENT_SERVER_PROFILE=lite — protocol plus health,
    #: with none of the operational extras.
    LITE_MODES: Set[str] = {"health", "agent", "chat", "a2a", "token"}

    @property
    def profile(self) -> str:
        return env_str("AGENT_SERVER_PROFILE").strip().lower()

    @property
    def is_lite(self) -> bool:
        return self.profile == "lite"

    def resolve_modes(self) -> Set[str]:
        """Surfaces to enable, honouring ALWAYS_ACTIVE_MODES then the profile."""
        explicit = env_csv("ALWAYS_ACTIVE_MODES")
        if explicit is not None:
            return set(explicit)
        return set(self.LITE_MODES if self.is_lite else self.DEFAULT_MODES)

    @property
    def slim_startup(self) -> bool:
        """Skip warm-up that only pays off on long-lived servers."""
        return env_flag("SLIM_STARTUP") or self.is_lite

    # -- logging ------------------------------------------------------------

    @property
    def log_level(self) -> str:
        return env_str("LOG_LEVEL", "INFO").upper()

    @property
    def log_format(self) -> str:
        return env_str("LOG_FORMAT", "text").lower()

    # -- networking ---------------------------------------------------------

    @property
    def port(self) -> int:
        """Listen port from PORT; 0 means "not set, use the CLI argument"."""
        return env_int("PORT", 0)

    @property
    def cors_allow_any_origin(self) -> bool:
        return env_str("ALLOWED_ORIGINS").strip() == "*"

    def cors_origins(self, fallback: str = "http://localhost:3000") -> List[str]:
        if self.cors_allow_any_origin:
            return ["*"]
        return env_csv("ALLOWED_ORIGINS") or [fallback]

    # -- LLM judge ----------------------------------------------------------

    def llm_judge_enabled(self, monitoring_active: bool) -> bool:
        """Quality evaluation costs an extra LLM call per interaction.

        Defaults to whether the ``monitoring`` surface is active. The flag can
        only *disable*: call sites additionally require the monitoring mode.
        """
        return env_flag("LLM_JUDGE_ENABLED", default=monitoring_active)

    @property
    def llm_judge_model_id(self) -> str:
        return env_str("LLM_JUDGE_MODEL_ID", "bedrock/us.amazon.nova-micro-v1:0")

    # -- protocols ----------------------------------------------------------

    @property
    def a2a_mount_root(self) -> bool:
        """Also mount A2A routes at "/" (the AgentCore data-plane contract)."""
        return env_flag("A2A_MOUNT_ROOT")

    @property
    def scheduler_enabled(self) -> bool:
        return env_str("ENABLE_SCHEDULER", "true").lower() != "false"

    # -- registry -----------------------------------------------------------

    @property
    def agent_registry_url(self) -> str:
        return env_str("AGENT_REGISTRY_URL")

    @property
    def agent_base_url(self) -> str:
        return env_str("AGENT_BASE_URL")

    @property
    def agent_base_url_port(self) -> str:
        return env_str("AGENT_BASE_URL_PORT")

    @property
    def registry_url(self) -> str:
        """Registry endpoint, preferring AGENT_REGISTRY_URL over AGENT_BASE_URL."""
        return self.agent_registry_url or self.agent_base_url


#: Shared instance. Safe to import and hold: every attribute reads live env.
settings = ServerSettings()


class Config:
    """Authentication flags.

    Retained for backwards compatibility. The authoritative auth decisions are
    made in ``security/dependencies.py`` against ``get_original_environ()``;
    the defaults here mirror that module (notably ``FORCE_AUTH`` defaults to
    *true* — auth is required unless an operator explicitly opts out).
    """

    def __init__(self):
        self.agent_auth_enabled = env_flag("AGENT_AUTH_ENABLED", True)
        self.force_auth = env_str("FORCE_AUTH", "true").lower() != "false"
