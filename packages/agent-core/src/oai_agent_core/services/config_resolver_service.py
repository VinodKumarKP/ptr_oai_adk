"""Configuration loading and resolution service."""

import logging
import os
from typing import Dict, Any, Optional

from oai_agent_core.utils.exceptions import ConfigurationError


class ConfigResolverService:
    """Service for configuration loading and resolution.

    Handles:
    - Loading agent configs from YAML files (with caching)
    - Resolving environment variables and macros
    - Validating configuration structure
    - Merging configuration defaults

    **Phase 3.4**: Includes in-memory caching of loaded configs to reduce disk I/O
    and improve initialization performance for frequently-used agents.
    """

    def __init__(self, config_root: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 enable_cache: bool = True):
        """Initialize the config resolver service.

        Args:
            config_root: Root directory for config files
            logger: Optional logger instance
            enable_cache: Whether to enable config caching (Phase 3.4). Default True.
        """
        self.config_root = config_root or os.getcwd()
        self.logger = logger or logging.getLogger(__name__)
        self.enable_cache = enable_cache
        self._config_cache: Dict[str, Dict[str, Any]] = {}  # agent_name -> config

    def load_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Load agent configuration from YAML file (with caching).

        **Phase 3.4**: Checks in-memory cache first before loading from disk.

        Args:
            agent_name: Name of the agent (used to find YAML file)

        Returns:
            Agent configuration dictionary

        Raises:
            ConfigurationError: If config file not found or invalid
        """
        # Phase 3.4: Check cache first
        if self.enable_cache and agent_name in self._config_cache:
            self.logger.debug(f"Loaded configuration for agent '{agent_name}' from cache")
            return self._config_cache[agent_name]

        try:
            from oai_agent_core.components.configuration.model_config import ConfigManager

            config_manager = ConfigManager(config_root=self.config_root)
            config = config_manager.load_agent_config(agent_name)

            # Phase 3.4: Store in cache
            if self.enable_cache:
                self._config_cache[agent_name] = config
                self.logger.debug(f"Loaded configuration for agent '{agent_name}' from disk (cached)")
            else:
                self.logger.debug(f"Loaded configuration for agent '{agent_name}' from disk")

            return config
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration for agent '{agent_name}': {e}"
            )

    def clear_cache(self) -> None:
        """Clear the configuration cache.

        **Phase 3.4**: Useful for testing or when configs are updated on disk.
        """
        self._config_cache.clear()
        self.logger.debug("Cleared configuration cache")

    def resolve_macros(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve macros in configuration values.

        Replaces ${VAR_NAME} and other macro expressions with resolved values.

        Args:
            config: Configuration with macros

        Returns:
            Configuration with macros resolved
        """
        try:
            from oai_agent_core.macros import MacroProcessor

            processor = MacroProcessor(
                project_root=self.config_root,
                logger=self.logger
            )

            # Process system prompt if present
            if 'system_prompt' in config:
                config['system_prompt'] = processor.process(config['system_prompt'])

            self.logger.debug("Resolved macros in configuration")
            return config
        except Exception as e:
            self.logger.warning(f"Failed to resolve macros: {e}")
            return config

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration structure.

        Args:
            config: Configuration to validate

        Returns:
            True if valid, False otherwise
        """
        # Required fields
        if 'type' not in config:
            self.logger.error("Configuration missing required 'type' field")
            return False

        # Validate nested structures
        if 'model' in config and not isinstance(config['model'], dict):
            self.logger.error("Configuration 'model' must be a dict")
            return False

        if 'tools' in config and not isinstance(config['tools'], dict):
            self.logger.error("Configuration 'tools' must be a dict")
            return False

        self.logger.debug("Configuration validation passed")
        return True
