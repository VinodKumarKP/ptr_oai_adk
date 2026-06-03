"""Model configuration manager for Anthropic agents via claude-agent-sdk + LiteLLM proxy.

Architecture
------------
- **claude-agent-sdk** provides the agentic loop, subagents, hooks, MCP first-class
  support, and extended thinking without manual tool-call loop management.
- **LiteLLM proxy** handles provider routing — point ``litellm_proxy_url`` at a
  running LiteLLM proxy and it transparently routes to Bedrock, Vertex AI, or the
  direct Anthropic API without any code changes.

YAML model config examples
--------------------------
  # Direct Anthropic API (no proxy)
  model:
    model_id: claude-opus-4-5-20251101
    api_key: "${ANTHROPIC_API_KEY}"

  # Via LiteLLM proxy (Bedrock / Vertex / any provider)
  model:
    model_id: anthropic/claude-opus-4-5-20251101   # LiteLLM model-ID format
    litellm_proxy_url: "${LITELLM_PROXY_URL}"       # e.g. http://localhost:4000
    api_key: "${LITELLM_PROXY_API_KEY}"
    max_tokens: 8096
    temperature: 0.7
"""

import os
from typing import Any, Dict, Optional

from oai_agent_core.core.base_model_configuration_manager import BaseModelConfigurationManager


class AnthropicModelConfigurationManager(BaseModelConfigurationManager):
    """Resolves model and SDK options for ``claude_agent_sdk.query()``.

    ``create_model`` returns a dict of kwargs forwarded to ``ClaudeAgentOptions``.

    Because ``BaseModelConfigurationManager._build_default_config`` only preserves
    ``model_id`` and ``params``, authentication fields (``api_key``,
    ``litellm_proxy_url``) are stored separately in ``_raw_config`` so they
    survive the base-class normalisation.
    """

    def __init__(self, default_config: Optional[Dict[str, Any]] = None, **kwargs):
        # Stash the full raw config BEFORE the base class strips it
        self._raw_config: Dict[str, Any] = dict(default_config or {})
        super().__init__(default_config=default_config, **kwargs)

    def create_model(self, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return resolved claude-agent-sdk model kwargs.

        ``ClaudeAgentOptions`` accepts:
            model  – short model alias, e.g. "claude-opus-4-5" or "claude-sonnet-4-5"
            env    – subprocess env vars; used to pass ANTHROPIC_API_KEY and
                     ANTHROPIC_BASE_URL (for LiteLLM proxy routing)

        There are NO ``api_key``, ``api_base_url``, or ``max_tokens`` fields on
        ``ClaudeAgentOptions`` — authentication and routing are done via the
        subprocess environment that the Claude CLI reads.
        """
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError:
            raise ImportError(
                "claude-agent-sdk is required: pip install claude-agent-sdk"
            )

        config = self.default_config if not model_config else self._merge_with_defaults(model_config)
        self._validate_config(config)

        # Strip LiteLLM provider prefix so ClaudeAgentOptions gets a bare model name.
        # "anthropic/claude-opus-4-5-20251101" → "claude-opus-4-5-20251101"
        model_id: str = config["model_id"]
        bare_model = model_id.split("/")[-1] if "/" in model_id else model_id

        kwargs: Dict[str, Any] = {"model": bare_model}

        # Build subprocess env dict for auth + LiteLLM proxy routing.
        env: Dict[str, str] = {}

        # Read auth fields from _raw_config — BaseModelConfigurationManager strips them
        raw = getattr(self, "_raw_config", {})
        api_key = (
            raw.get("api_key")
            or os.getenv("LITELLM_PROXY_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
        )
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key

        # LiteLLM proxy: set ANTHROPIC_BASE_URL so the Claude CLI routes through it
        proxy_url = raw.get("litellm_proxy_url") or os.getenv("LITELLM_PROXY_URL")
        if proxy_url:
            env["ANTHROPIC_BASE_URL"] = proxy_url
            self.logger.debug("Routing through LiteLLM proxy: %s", proxy_url)

        if env:
            kwargs["env"] = env

        self.logger.debug(
            "AnthropicModelConfig: model=%s proxy=%s",
            bare_model,
            proxy_url or "direct",
        )
        return kwargs

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for observability."""
        model_id: str = self.default_config.get("model_id", "")
        raw = getattr(self, "_raw_config", {})
        proxy = raw.get("litellm_proxy_url") or os.getenv("LITELLM_PROXY_URL")
        if model_id.startswith("bedrock/"):
            provider = "bedrock"
        elif model_id.startswith("vertex_ai/"):
            provider = "vertex_ai"
        else:
            provider = "anthropic"
        return {
            "model_id": model_id,
            "provider": provider,
            "via_proxy": bool(proxy),
        }
