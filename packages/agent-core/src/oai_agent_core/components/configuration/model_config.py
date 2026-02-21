import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from ruamel.yaml import YAML, comments

logger = logging.getLogger(__name__)


class ConfigManager:
    """Centralized configuration manager for all agent-related configurations.

    This class handles loading, saving, merging, and validating configurations
    for agents and global settings. It supports both YAML and JSON formats.
    """

    def __init__(self, config_root: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_root: Optional root directory for configurations.
                         If None, attempts to auto-detect based on project structure.
        """
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.config_root = Path(config_root) if config_root else None
        
        if config_root is None:
            # Default structure: oai_agent_core/agents_config
            self.agent_config_dir = self.project_root / 'oai_agent_core' / 'agents_config'
            self.global_config_dir = self.project_root / 'oai_agent_core' / 'global_config'
        else:
            self.agent_config_dir = self.config_root / 'agents_config'
            self.global_config_dir = self.config_root / 'oai_agent_core' / 'global_config'
            if not self.global_config_dir.exists():
                self.global_config_dir = self.project_root / 'oai_agent_core' / 'global_config'

        # Cache for loaded configurations
        self._agent_config_cache: Dict[str, Dict[str, Any]] = {}
        self._global_config_cache: Dict[str, Dict[str, Any]] = {}

        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.width = 4096
        self.yaml.indent(mapping=2, sequence=4, offset=2)
        self.yaml.default_flow_style = False
        self.yaml.allow_unicode = True

        logger.info(f"ConfigManager initialized with project root: {self.project_root}")

    def ruamel_to_native(self, obj: Any) -> Any:
        """Convert ruamel.yaml objects to native Python objects (dicts/lists).

        Args:
            obj: The object to convert (CommentedMap, CommentedSeq, or other).

        Returns:
            Native Python object (dict, list, or original value).
        """
        if isinstance(obj, comments.CommentedMap):
            return {k: self.ruamel_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, comments.CommentedSeq):
            return [self.ruamel_to_native(i) for i in obj]
        else:
            return obj

    def _detect_project_root(self) -> Path:
        """Auto-detect the project root directory.

        Searches upwards from the current file location for common project markers.

        Returns:
            Path object representing the project root.
        """
        current_file = Path(__file__).resolve()

        # Look for common project indicators
        indicators = ['agents_config', 'global_config', 'setup.py', 'pyproject.toml', '.git']

        # Start from current directory and go up
        search_path = current_file.parent

        while search_path != search_path.parent:  # Not at filesystem root
            for indicator in indicators:
                if (search_path / indicator).exists():
                    logger.debug(f"Found project root at {search_path} (indicator: {indicator})")
                    return search_path
            search_path = search_path.parent

        # Fallback: assume standard package structure
        fallback_root = current_file.parent.parent.parent.parent
        logger.warning(f"Could not auto-detect project root, using fallback: {fallback_root}")
        return fallback_root

    def load_agent_config(self, agent_name: str, use_cache: bool = True, abort_if_not_found: bool = True) -> Dict[
        str, Any]:
        """
        Load agent configuration from YAML file.

        Args:
            agent_name: Name of the agent (filename without extension).
            use_cache: Whether to use cached configuration if available.
            abort_if_not_found: Whether to raise FileNotFoundError if config is missing.

        Returns:
            Agent configuration dictionary.

        Raises:
            FileNotFoundError: If configuration file doesn't exist and abort_if_not_found is True.
            ValueError: If configuration is invalid/empty.
            RuntimeError: If loading fails.
        """
        if use_cache and agent_name in self._agent_config_cache:
            logger.debug(f"Using cached config for agent '{agent_name}'")
            return self._agent_config_cache[agent_name].copy()

        config_path = self.agent_config_dir / f"{agent_name}.yaml"

        if not config_path.exists():
            if abort_if_not_found:
                raise FileNotFoundError(f"Agent configuration file not found: {config_path}")
            else:
                return {}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = self.yaml.load(f)

            if not config:
                raise ValueError(f"Empty configuration file: {config_path}")

            # Validate required fields
            self._validate_agent_config(config, agent_name)

            # Cache the configuration
            if use_cache:
                self._agent_config_cache[agent_name] = config.copy()

            logger.info(f"Loaded configuration for agent '{agent_name}' from {config_path}")
            return config

        except Exception as e:
            raise RuntimeError(f"Failed to load configuration for agent '{agent_name}': {e}")

    def load_global_config(self, provider: str = 'aws', use_cache: bool = True) -> Dict[str, Any]:
        """
        Load global configuration for cloud provider.

        Args:
            provider: Cloud provider name (e.g., 'aws', 'azure').
            use_cache: Whether to use cached configuration if available.

        Returns:
            Global configuration dictionary.
        """
        if use_cache and provider in self._global_config_cache:
            logger.debug(f"Using cached global config for provider '{provider}'")
            return self._global_config_cache[provider].copy()

        config_path = self.global_config_dir / f"{provider}.yaml"

        if not config_path.exists():
            logger.warning(f"Global configuration file not found: {config_path}, using defaults")
            return self._get_default_global_config()

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = self.yaml.load(f) or {}

            # Apply defaults for missing values
            config = self._apply_global_config_defaults(config)

            # Cache the configuration
            if use_cache:
                self._global_config_cache[provider] = config.copy()

            logger.info(f"Loaded global configuration for provider '{provider}' from {config_path}")
            return config

        except Exception as yaml_error:
            if 'yaml' in str(type(yaml_error)).lower():
                logger.error(f"Invalid YAML in global config file {config_path}: {yaml_error}")
                return self._get_default_global_config()
        except Exception as e:
            logger.error(f"Failed to load global configuration for provider '{provider}': {e}")
            return self._get_default_global_config()

    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries with deep merge for nested dicts.

        Args:
            base_config: Base configuration dictionary.
            override_config: Configuration dictionary to override with.

        Returns:
            Merged configuration dictionary.
        """
        result = base_config.copy()

        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def validate_config_value(self, key: str, value: Any, config_type: str = "agent") -> Any:
        """
        Validate a configuration value.

        Args:
            key: Configuration key.
            value: Value to validate.
            config_type: Type of configuration ('agent' or 'global').

        Returns:
            Validated value.

        Raises:
            ValueError: If value is invalid.
        """
        if key == 'temperature':
            if not isinstance(value, (int, float)) or not 0 <= value <= 2:
                raise ValueError(f"Temperature must be a number between 0 and 2, got: {value}")

        elif key == 'max_tokens':
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"Max tokens must be a positive integer, got: {value}")

        elif key == 'type' and config_type == 'agent':
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Agent type must be a non-empty string, got: {value}")

        return value

    def get_config_value(self, config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """
        Get a nested configuration value using dot notation.

        Args:
            config: Configuration dictionary.
            key_path: Dot-separated path to the key (e.g., 'model.temperature').
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        keys = key_path.split('.')
        value = config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set_config_value(self, config: Dict[str, Any], key_path: str, value: Any) -> None:
        """
        Set a nested configuration value using dot notation.

        Args:
            config: Configuration dictionary to modify.
            key_path: Dot-separated path to the key (e.g., 'model.temperature').
            value: Value to set.
        """
        keys = key_path.split('.')
        current = config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

    def _validate_agent_config(self, config: Dict[str, Any], agent_name: str) -> None:
        """Validate agent configuration structure."""
        required_fields = ['type']

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in configuration for agent '{agent_name}'")

        # Validate agent type
        agent_type = config.get('type')
        if not isinstance(agent_type, str) or not agent_type.strip():
            raise ValueError(f"Invalid agent type '{agent_type}' for agent '{agent_name}'")

    def _apply_global_config_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values to global configuration."""
        defaults = self._get_default_global_config()
        return self.merge_configs(defaults, config)

    def _get_default_global_config(self) -> Dict[str, Any]:
        """Get default global configuration."""
        return {
            'model': {
                'temperature': 0.0,
                'max_tokens': 1000
            }
        }

    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        Clear configuration cache.

        Args:
            cache_type: Type of cache to clear ('agent', 'global', or None for both).
        """
        if cache_type is None or cache_type == 'agent':
            self._agent_config_cache.clear()
            logger.info("Cleared agent configuration cache")

        if cache_type is None or cache_type == 'global':
            self._global_config_cache.clear()
            logger.info("Cleared global configuration cache")

    def list_available_agents(self) -> List[str]:
        """List all available agent configuration files.

        Returns:
            List of agent names (filenames without extension).
        """
        if not self.agent_config_dir.exists():
            return []

        return [f.stem for f in self.agent_config_dir.glob('*.yaml')]

    def list_available_providers(self) -> List[str]:
        """List all available global provider configuration files.

        Returns:
            List of provider names (filenames without extension).
        """
        if not self.global_config_dir.exists():
            return []

        return [f.stem for f in self.global_config_dir.glob('*.yaml')]

    def save_agent_config(self, agent_name: str, config: Dict[str, Any]) -> None:
        """
        Save agent configuration to YAML file.

        Args:
            agent_name: Name of the agent.
            config: Configuration dictionary to save.

        Raises:
            RuntimeError: If saving fails.
        """
        # Validate before saving
        self._validate_agent_config(config, agent_name)

        # Ensure directory exists
        self.agent_config_dir.mkdir(parents=True, exist_ok=True)

        config_path = self.agent_config_dir / f"{agent_name}.yaml"

        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                self.yaml.dump(config, f)

            # Update cache
            self._agent_config_cache[agent_name] = config.copy()

            logger.info(f"Saved configuration for agent '{agent_name}' to {config_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to save configuration for agent '{agent_name}': {e}")

    def load_server_config(self) -> Dict[str, Any]:
        """
        Load server configuration from JSON and YAML files in the agent config directory.

        Returns:
            Dict[str, Any]: Dictionary of server configurations, keyed by filename stem.
        """
        server_config = {}
        # Support .json, .yaml, and .yml files
        config_patterns = [
            os.path.join(self.agent_config_dir, "*.json"),
            os.path.join(self.agent_config_dir, "*.yaml"),
            os.path.join(self.agent_config_dir, "*.yml"),
        ]

        config_files = []
        for pattern in config_patterns:
            config_files.extend(glob.glob(pattern))

        for config_file in config_files:
            server_name = os.path.basename(config_file).split('.')[0]
            ext = os.path.splitext(config_file)[1].lower()
            if ext == ".json":
                server_config[server_name] = self._load_individual_json_config(config_file)
            elif ext in [".yaml", ".yml"]:
                server_config[server_name] = self._load_individual_yaml_config(config_file)

        return server_config

    def _load_individual_yaml_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load individual server configuration from YAML file.

        Args:
            config_file (str): Path to configuration file.

        Returns:
            Dict[str, Any]: Configuration data.
        """
        try:
            with open(config_file, "r", encoding='utf-8') as file:
                return self.yaml.load(file)
        except FileNotFoundError:
            raise Exception(f"Configuration file not found: {config_file}")
        except Exception as e:
            raise Exception(f"Error loading server configuration: {str(e)} for file {config_file}")

    def _load_individual_json_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load individual server configuration from JSON file.

        Args:
            config_file (str): Path to configuration file.

        Returns:
            Dict[str, Any]: Configuration data.
        """
        try:
            with open(config_file, "r", encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            raise Exception(f"Configuration file not found: {config_file}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON in configuration file: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading server configuration: {str(e)}")

    def load_config_from_path(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file.

        Args:
            file_path: Path to the configuration file.

        Returns:
            Configuration dictionary.
        """
        if not os.path.exists(file_path):
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith(('.yaml', '.yml')):
                    return self.yaml.load(f) or {}
                elif file_path.endswith('.json'):
                    return json.load(f)
                else:
                    # Try to detect format by content
                    content = f.read()
                    f.seek(0)
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        try:
                            return self.yaml.load(f) or {}
                        except Exception as ex:
                            raise Exception(f"Error loading config file {file_path}: {ex}")
        except Exception as e:
            raise Exception(f"Error loading config file {file_path}: {e}")


# Create a global instance for easy access
config_manager = ConfigManager()
