from typing import Optional, Dict, Any

from oai_agent_core.core.base_model_configuration_manager import BaseModelConfigurationManager


class StrandsModelConfigurationManager(BaseModelConfigurationManager):
    """Model configuration manager for AWS Strands using LiteLLM.
    
    This implementation uses the strands.models.litellm.LiteLLMModel
    for Strands agent framework integration.
    """

    def create_model(self, model_config: Optional[Dict[str, Any]] = None):
        """Create LiteLLMModel instance from configuration.
        
        Args:
            model_config: Optional model configuration (uses defaults if None)
            
        Returns:
            Configured LiteLLMModel instance
            
        Example:
            >>> manager = StrandsModelConfigurationManager()
            >>> model = manager.create_model({'temperature': 0.9})
        """
        try:
            from strands.models.litellm import LiteLLMModel
        except ImportError:
            raise ImportError(
                "Please install strands-agents with LiteLLM support"
            )

        # Use default config if none provided
        if not model_config:
            config = self.default_config
        else:
            config = self._merge_with_defaults(model_config)

        # Validate configuration
        self._validate_config(config)

        try:
            model = LiteLLMModel(
                model_id=config['model_id'],
                params=config.get('params', {})
            )

            self.logger.debug(
                f"Created LiteLLMModel: {config['model_id']} "
                f"(temp={config.get('params', {}).get('temperature')}, "
                f"max_tokens={config.get('params', {}).get('max_tokens')})"
            )

            return model

        except Exception as e:
            self.logger.error(f"Failed to create LiteLLMModel: {e}")
            raise
