from typing import Optional, Dict, Any

from oai_agent_core.core.base_model_configuration_manager import BaseModelConfigurationManager


class OpenAIModelConfigurationManager(BaseModelConfigurationManager):
    """Model configuration manager for OpenAI agents using ChatLiteLLM.

    This implementation uses the 'agents' library's LitellmModel wrapper around LiteLLM
    to provide a compatible chat model interface.

    Supports additional LLM parameters like top_p, top_k, presence_penalty, etc.
    """

    def create_model(self, model_config: Optional[Dict[str, Any]] = None):
        """Create LitellmModel instance from configuration.

        Args:
            model_config: Optional model configuration (uses defaults if None)
                Additional params can include: top_p, top_k, presence_penalty,
                frequency_penalty, stop, streaming, timeout, etc.

        Returns:
            Configured LitellmModel instance

        Example:
            >>> manager = OpenAIModelConfigurationManager()
            >>> model = manager.create_model({
            ...     'model_id': 'gpt-4',
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
            self.logger.error(f"Failed to create ChatLiteLLM: {e}")
            raise
