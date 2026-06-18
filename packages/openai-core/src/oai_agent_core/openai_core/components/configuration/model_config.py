from typing import Optional, Dict, Any

from oai_agent_core.core.base_model_configuration_manager import BaseModelConfigurationManager


class OpenAIModelConfigurationManager(BaseModelConfigurationManager):
    """Model configuration manager for OpenAI agents using ChatLiteLLM.

    This implementation uses the 'agents' library's LitellmModel wrapper around LiteLLM
    to provide a compatible chat model interface.

    Note:
        ``LitellmModel`` only accepts ``model``/``base_url``/``api_key``. Sampling
        parameters (temperature, top_p, presence_penalty, ...) are applied at the
        agent level via ``ModelSettings`` and are extracted with
        :meth:`build_model_settings`.
    """

    # Keys recognized by agents.ModelSettings that we pass through from config.
    MODEL_SETTINGS_KEYS = frozenset({
        'temperature',
        'top_p',
        'frequency_penalty',
        'presence_penalty',
        'tool_choice',
        'parallel_tool_calls',
        'truncation',
        'max_tokens',
        'reasoning',
        'verbosity',
        'metadata',
        'store',
        'top_logprobs',
    })

    def build_model_settings(self, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract ModelSettings-compatible parameters from a model config.

        Sampling/runtime parameters cannot be passed to ``LitellmModel`` (whose
        constructor only accepts ``model``/``base_url``/``api_key``); they belong
        on ``agents.ModelSettings`` at the agent level. This method resolves the
        config against defaults and returns only the recognized settings keys.

        Both top-level keys (e.g. ``temperature``) and a nested ``params`` dict
        are honored. Note that :meth:`_merge_with_defaults` normalizes the legacy
        top-level ``temperature``/``max_tokens`` into ``params`` first, so by the
        time settings are extracted everything lives under the merged ``params``.

        Args:
            model_config: Optional model configuration (uses defaults if None).

        Returns:
            Dict of recognized ModelSettings keyword arguments (possibly empty).
        """
        if not model_config:
            config = self.default_config or {}
        else:
            config = self._merge_with_defaults(model_config)

        settings: Dict[str, Any] = {}

        for key in self.MODEL_SETTINGS_KEYS:
            value = config.get(key)
            if value is not None:
                settings[key] = value

        # A nested 'params' block overrides top-level keys.
        params = config.get('params')
        if isinstance(params, dict):
            for key, value in params.items():
                if key in self.MODEL_SETTINGS_KEYS and value is not None:
                    settings[key] = value

        return settings

    def create_model(self, model_config: Optional[Dict[str, Any]] = None):
        """Create LitellmModel instance from configuration.

        Only ``model_id`` (and optionally ``base_url``/``api_key``) are used here,
        because ``LitellmModel`` accepts nothing else. Sampling params such as
        ``temperature``/``top_p`` are NOT applied to the model object; they are
        applied at the agent level via :meth:`build_model_settings`.

        Args:
            model_config: Optional model configuration (uses defaults if None).

        Returns:
            Configured LitellmModel instance

        Example:
            >>> manager = OpenAIModelConfigurationManager()
            >>> model = manager.create_model({'model_id': 'gpt-4'})
            >>> settings = manager.build_model_settings({
            ...     'temperature': 0.9,
            ...     'params': {'top_p': 0.95, 'presence_penalty': 0.5}
            ... })
        """
        try:
            from agents.extensions.models.litellm_model import LitellmModel
        except ImportError:
            raise ImportError(
                "Please install openai-agents[litellm]"
            )

        # Use default config if none provided
        if not model_config:
            config = self.default_config
        else:
            config = self._merge_with_defaults(model_config)

        # Validate configuration
        self._validate_config(config)

        try:
            # Extract standard parameters
            model_params = {
                'model': config['model_id'],
            }

            # ChatLiteLLM expects model and standard LLM parameters
            model = LitellmModel(**model_params)

            self.logger.debug(
                f"Created LitellmModel: {config['model_id']} "
                f"with params: {model_params}"
            )

            return model

        except Exception as e:
            self.logger.error(f"Failed to create LitellmModel: {e}")
            raise
