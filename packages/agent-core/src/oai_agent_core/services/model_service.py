"""LLM provider initialization and management service."""

import logging
import hashlib
import json
from typing import Dict, Any, Optional


class ModelService:
    """Service for LLM provider initialization and management.

    Handles:
    - Creating LLM instances from configuration
    - Managing model providers (OpenAI, Anthropic, etc.)
    - Validating model configuration
    - Caching LLM instances by config hash (Phase 3.4)

    **Phase 3.4**: Caches model instances to avoid recreating the same LLM
    with identical configurations. Useful when multiple agents share the same model.
    """

    def __init__(self, logger: Optional[logging.Logger] = None,
                 enable_cache: bool = True):
        """Initialize the model service.

        Args:
            logger: Optional logger instance
            enable_cache: Whether to enable model caching (Phase 3.4). Default True.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.enable_cache = enable_cache
        self._model_manager = None
        self._model_cache: Dict[str, Any] = {}  # config_hash -> model instance

    def set_model_manager(self, model_manager: Any) -> None:
        """Set the model manager instance.

        Args:
            model_manager: Model manager for LLM creation
        """
        self._model_manager = model_manager

    def _hash_model_config(self, model_config: Dict[str, Any]) -> str:
        """Generate a hash of the model configuration for caching.

        **Phase 3.4**: Creates a deterministic hash of config settings.

        Args:
            model_config: Model configuration

        Returns:
            Hex string hash of the configuration
        """
        # Create a copy and sort keys for deterministic hashing
        config_copy = dict(model_config)
        config_json = json.dumps(config_copy, sort_keys=True, default=str)
        return hashlib.sha256(config_json.encode()).hexdigest()

    def create_model(self, model_config: Dict[str, Any]) -> Any:
        """Create an LLM instance from configuration (with caching).

        **Phase 3.4**: Checks cache first before creating new model instance.

        Args:
            model_config: Model configuration

        Returns:
            LLM instance, or None if creation failed
        """
        if not self._model_manager:
            self.logger.warning("Model manager not set")
            return None

        # Phase 3.4: Check cache first
        if self.enable_cache:
            config_hash = self._hash_model_config(model_config)
            if config_hash in self._model_cache:
                self.logger.debug(f"Reusing cached model for config hash {config_hash[:8]}")
                return self._model_cache[config_hash]

        try:
            model = self._model_manager.create_model(model_config=model_config)

            # Phase 3.4: Store in cache
            if self.enable_cache and model:
                config_hash = self._hash_model_config(model_config)
                self._model_cache[config_hash] = model
                self.logger.debug(f"Created and cached model: {model_config.get('name', 'unknown')} (hash: {config_hash[:8]})")
            else:
                self.logger.debug(f"Created model: {model_config.get('name', 'unknown')}")

            return model
        except Exception as e:
            self.logger.error(f"Failed to create model: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear the model cache.

        **Phase 3.4**: Useful for testing or when LLM instances need to be recreated.
        """
        self._model_cache.clear()
        self.logger.debug("Cleared model cache")

    def validate_model_config(self, model_config: Dict[str, Any]) -> bool:
        """Validate model configuration.

        Args:
            model_config: Model configuration

        Returns:
            True if valid, False otherwise
        """
        # Required fields
        if 'name' not in model_config:
            self.logger.error("Model configuration missing required 'name' field")
            return False

        if 'provider' not in model_config:
            self.logger.error("Model configuration missing required 'provider' field")
            return False

        self.logger.debug("Model configuration validation passed")
        return True
