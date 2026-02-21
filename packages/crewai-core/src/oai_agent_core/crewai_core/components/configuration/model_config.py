from typing import Optional, Dict, Any

from oai_agent_core.core.base_model_configuration_manager import BaseModelConfigurationManager


class CrewAIModelConfigurationManager(BaseModelConfigurationManager):
    """Model configuration manager for CrewAI using LiteLLM.
    
    This implementation uses CrewAI's LLM class which supports LiteLLM
    model strings directly.
    """

    def create_model(self, model_config: Optional[Dict[str, Any]] = None):
        """Create CrewAI LLM instance from configuration.
        
        Args:
            model_config: Optional model configuration (uses defaults if None)
            
        Returns:
            Configured CrewAI LLM instance
            
        Example:
            >>> manager = CrewAIModelConfigurationManager()
            >>> model = manager.create_model({'temperature': 0.9})
        """
        try:
            from crewai import LLM
        except ImportError:
            raise ImportError(
                "Please install crewai: pip install crewai"
            )

        # Use default config if none provided
        if not model_config:
            config = self.default_config
        else:
            config = self._merge_with_defaults(model_config)

        # Validate configuration
        self._validate_config(config)

        try:
            # CrewAI's LLM class accepts model string and params
            model = LLM(
                model=config['model_id'],
                temperature=config['params']['temperature'],
                max_tokens=config['params']['max_tokens']
            )

            self.logger.debug(
                f"Created CrewAI LLM: {config['model_id']} "
                f"(temp={config['params']['temperature']}, "
                f"max_tokens={config['params']['max_tokens']})"
            )

            return model

        except Exception as e:
            self.logger.error(f"Failed to create CrewAI LLM: {e}")
            raise
