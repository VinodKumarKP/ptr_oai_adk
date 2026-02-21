"""Model configuration framework using LiteLLM for multiple AI frameworks."""

import logging
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod


class BaseModelConfigurationManager(ABC):
    """Base class for model configuration management across different frameworks.

    This abstract base class provides common configuration management functionality
    and defines the interface that all framework-specific implementations must follow.

    Attributes:
        default_config: Default model configuration
        logger: Logger instance
    """

    DEFAULT_MODEL_ID = 'bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0'
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 4096

    def __init__(
            self,
            default_config: Optional[Dict[str, Any]] = None,
            logger: Optional[logging.Logger] = None
    ):
        """Initialize the base model configuration manager.

        Args:
            default_config: Optional default configuration to override defaults
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.default_config = self._build_default_config(default_config)

        self.logger.debug(
            f"{self.__class__.__name__} initialized with model "
            f"'{self.default_config['model_id']}'"
        )

    def _build_default_config(
            self,
            config_override: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build default configuration with optional overrides.

        Args:
            config_override: Optional configuration to merge with defaults

        Returns:
            Complete default configuration
        """
        base_config = {
            'model_id': self.DEFAULT_MODEL_ID,
            'params': {
                'temperature': self.DEFAULT_TEMPERATURE,
                'max_tokens': self.DEFAULT_MAX_TOKENS
            }
        }

        if config_override:
            # Update model_id if provided
            if 'model_id' in config_override and config_override['model_id']:
                base_config['model_id'] = config_override['model_id']

            # Merge params
            if 'params' in config_override and config_override['params']:
                base_config['params'].update(config_override['params'])

            # Handle legacy root-level params
            for key in ['temperature', 'max_tokens']:
                if key in config_override and config_override[key] is not None:
                    base_config['params'][key] = config_override[key]

        return base_config

    def get_default_config(self) -> Dict[str, Any]:
        """Get the default model configuration.

        Returns:
            Dictionary with default model configuration
        """
        return self.default_config.copy()

    @abstractmethod
    def create_model(self, model_config: Optional[Dict[str, Any]] = None):
        """Create a model instance from configuration.

        Args:
            model_config: Optional model configuration (uses defaults if None)

        Returns:
            Configured model instance for the specific framework
        """
        pass

    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge provided config with defaults.

        Args:
            config: Partial configuration

        Returns:
            Complete configuration with defaults filled in
        """
        merged = self.default_config.copy()

        # Update model_id if provided
        if 'model_id' in config and config['model_id'] is not None:
            merged['model_id'] = config['model_id']

        # Update params if provided
        if 'params' in config and config['params'] is not None:
            merged['params'].update(config['params'])

        # Handle legacy temperature/max_tokens at root level
        if 'temperature' in config and config['temperature'] is not None:
            merged['params']['temperature'] = config['temperature']
        if 'max_tokens' in config and config['max_tokens'] is not None:
            merged['params']['max_tokens'] = config['max_tokens']

        return merged

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate model configuration parameters.

        Args:
            config: Configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate required keys
        required_keys = ['temperature', 'max_tokens']
        missing_keys = [k for k in required_keys if k not in config['params']]

        if missing_keys:
            raise ValueError(
                f"Model configuration missing required keys: {missing_keys}"
            )

        # Validate temperature range
        temp = config['params']['temperature']
        if not isinstance(temp, (int, float)) or not 0 <= temp <= 1:
            raise ValueError(
                f"Temperature must be between 0 and 1, got: {temp}"
            )

        # Validate max_tokens
        max_tokens = config['params']['max_tokens']
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError(
                f"max_tokens must be a positive integer, got: {max_tokens}"
            )

        # Validate model_id format (basic check)
        model_id = config['model_id']
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(
                f"model_id must be a non-empty string, got: {model_id}"
            )

    def update_default_config(self, **kwargs) -> None:
        """Update default configuration parameters.

        Args:
            **kwargs: Configuration parameters to update

        Example:
            >>> manager.update_default_config(temperature=0.9, max_tokens=8192)
        """
        for key, value in kwargs.items():
            if value is not None:
                if key == 'model_id':
                    self.default_config['model_id'] = value
                elif key in ['temperature', 'max_tokens']:
                    self.default_config['params'][key] = value
                self.logger.debug(f"Updated default config: {key}={value}")

    def get_model_info(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get human-readable model information.

        Args:
            config: Optional config to inspect (uses default if None)

        Returns:
            Dictionary with model information
        """
        cfg = config or self.default_config
        params = cfg.get('params', {})

        return {
            'model_id': cfg.get('model_id'),
            'provider': self._extract_provider(cfg.get('model_id', '')),
            'temperature': params.get('temperature'),
            'max_tokens': params.get('max_tokens'),
            'model_family': self._extract_model_family(cfg.get('model_id', ''))
        }

    def _extract_provider(self, model_id: str) -> str:
        """Extract provider from model ID.

        Args:
            model_id: Model identifier

        Returns:
            Provider name
        """
        if model_id.startswith('gpt-') or model_id.startswith('o1-'):
            return 'openai'
        elif 'claude' in model_id:
            return 'anthropic'
        elif 'bedrock/' in model_id:
            return 'aws-bedrock'
        elif 'azure/' in model_id:
            return 'azure'
        elif 'vertex_ai/' in model_id:
            return 'gcp'
        elif 'gemini' in model_id:
            return 'google'
        else:
            return 'unknown'

    def _extract_model_family(self, model_id: str) -> str:
        """Extract model family from model ID.

        Args:
            model_id: Full model identifier

        Returns:
            Model family name
        """
        if 'gpt' in model_id or 'o1' in model_id:
            return 'gpt'
        elif 'claude' in model_id:
            return 'claude'
        elif 'gemini' in model_id:
            return 'gemini'
        elif 'llama' in model_id:
            return 'llama'
        elif 'titan' in model_id:
            return 'titan'
        else:
            return 'unknown'

    def list_supported_parameters(self) -> list:
        """Get list of supported configuration parameters.

        Returns:
            List of parameter names
        """
        return ['model_id', 'temperature', 'max_tokens']

    def __repr__(self) -> str:
        """String representation of the manager."""
        return (
            f"{self.__class__.__name__}("
            f"default_model='{self.default_config['model_id']}')"
        )