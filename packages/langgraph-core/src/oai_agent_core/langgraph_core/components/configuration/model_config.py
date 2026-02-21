from typing import Optional, Dict, Any

from oai_agent_core.core.base_model_configuration_manager import BaseModelConfigurationManager


class LangChainModelConfigurationManager(BaseModelConfigurationManager):
    """Model configuration manager for LangChain using ChatLiteLLM.

    This implementation uses LangChain's ChatLiteLLM wrapper around LiteLLM
    to provide a LangChain-compatible chat model interface.

    Supports additional LLM parameters like top_p, top_k, presence_penalty, etc.
    """

    def create_model(self, model_config: Optional[Dict[str, Any]] = None):
        """Create ChatLiteLLM instance from configuration.

        Args:
            model_config: Optional model configuration (uses defaults if None)
                Additional params can include: top_p, top_k, presence_penalty,
                frequency_penalty, stop, streaming, timeout, etc.

        Returns:
            Configured ChatLiteLLM instance

        Example:
            >>> manager = LangChainModelConfigurationManager()
            >>> model = manager.create_model({
            ...     'temperature': 0.9,
            ...     'params': {'top_p': 0.95, 'presence_penalty': 0.5}
            ... })
        """
        try:
            from langchain_litellm import ChatLiteLLM
        except ImportError:
            raise ImportError(
                "Please install langchain-litellm: "
                "pip install langchain-litellm"
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
                'temperature': config['params']['temperature'],
                'max_tokens': config['params']['max_tokens']
            }

            # Add any additional LLM parameters from config
            additional_params = {
                k: v for k, v in config['params'].items()
                if k not in ['temperature', 'max_tokens']
            }
            model_params.update(additional_params)

            # ChatLiteLLM expects model and standard LLM parameters
            model = ChatLiteLLM(**model_params)

            self.logger.debug(
                f"Created ChatLiteLLM: {config['model_id']} "
                f"with params: {model_params}"
            )

            return model

        except Exception as e:
            self.logger.error(f"Failed to create ChatLiteLLM: {e}")
            raise