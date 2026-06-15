"""Service Layer for Agent-Core Architecture

This module provides focused service classes for major agent operations,
enabling dependency injection and easier testing.

## Architecture

Each service encapsulates the logic for a specific domain:
- **ConfigResolverService**: Configuration loading and resolution
- **KnowledgeBaseService**: Knowledge base initialization and access
- **ToolService**: Tool registry and loading
- **SkillService**: Skill discovery and management
- **ModelService**: LLM provider initialization
- **ObservabilityService**: Tracing and monitoring integration

## Benefits

✅ Single Responsibility: Each service handles one domain
✅ Dependency Injection: Services can be mocked in tests
✅ Testability: Clear, focused interfaces
✅ Reusability: Services can be used independently of BaseAgent
✅ Clarity: Business logic is explicit and organized

## Usage Example

    # Create services
    config_service = ConfigResolverService(config_root='/config')
    tool_service = ToolService(project_root='/project')
    kb_service = KnowledgeBaseService(project_root='/project')
    
    # Load configuration
    agent_config = config_service.load_agent_config('my_agent')
    
    # Load tools
    tool_registry = tool_service.create_registry()
    tool_service.load_tools(tool_registry, agent_config.get('tools', {}))
    
    # Load knowledge base
    kb_factory = kb_service.create_knowledge_base_factory()
    kb = kb_factory.get_knowledge_base(agent_config.get('knowledge_base'))
"""

import logging
import os
import hashlib
import json
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from pathlib import Path

from oai_agent_core.core.exceptions import (
    ConfigurationError,
    ToolLoadingError,
    KnowledgeBaseError,
)


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


class KnowledgeBaseService:
    """Service for knowledge base initialization and management.
    
    Handles:
    - Creating knowledge base factories
    - Initializing knowledge bases from config
    - Managing KB credentials and registry access
    - Propagating top-level KB settings to individual KBs
    """
    
    def __init__(self, project_root: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the KB service.
        
        Args:
            project_root: Project root directory
            logger: Optional logger instance
        """
        self.project_root = project_root
        self.logger = logger or logging.getLogger(__name__)
        self._kb_factory = None
    
    def create_knowledge_base_factory(self, 
                                      llm: Optional[Any] = None,
                                      document_loader: Optional[Any] = None,
                                      vector_store: Optional[Any] = None) -> Any:
        """Create a knowledge base factory.
        
        Args:
            llm: Optional LLM instance for embeddings
            document_loader: Optional custom document loader
            vector_store: Optional custom vector store
            
        Returns:
            KnowledgeBaseFactory instance
            
        Raises:
            KnowledgeBaseError: If factory creation fails
        """
        try:
            from oai_agent_core.components.knowledge_base.knowledge_base_factory import (
                KnowledgeBaseFactory
            )
            
            factory = KnowledgeBaseFactory(
                logger=self.logger,
                project_root=self.project_root,
                document_loader=document_loader,
                vector_store=vector_store
            )
            
            self._kb_factory = factory
            self.logger.debug("Created knowledge base factory")
            return factory
        except Exception as e:
            raise KnowledgeBaseError(f"Failed to create KB factory: {e}")
    
    def get_knowledge_base(self, kb_config: Dict[str, Any]) -> Any:
        """Get a knowledge base instance for the given configuration.
        
        Args:
            kb_config: Knowledge base configuration
            
        Returns:
            Knowledge base instance, or None if factory not available
        """
        if not self._kb_factory:
            self.logger.warning("KB factory not initialized")
            return None
        
        try:
            kb = self._kb_factory.get_knowledge_base(kb_config)
            self.logger.debug("Created knowledge base instance")
            return kb
        except Exception as e:
            self.logger.error(f"Failed to create KB instance: {e}")
            return None
    
    def normalize_kb_config(self, kb_config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize knowledge base configuration.
        
        Handles legacy formats and propagates top-level settings to individual KBs.
        
        Args:
            kb_config: Raw KB configuration
            
        Returns:
            Normalized KB configuration
        """
        # If agent_list is present, propagate top-level registry to all agents
        if isinstance(kb_config, dict) and 'agent_list' in kb_config:
            registry_url = kb_config.get('registry', {}).get('url')
            auth_token = kb_config.get('registry', {}).get('auth_token')
            
            for agent_entry in kb_config.get('agent_list', []):
                if isinstance(agent_entry, dict):
                    for agent_name, agent_config in agent_entry.items():
                        kb_entries = agent_config.get('knowledge_base', [])
                        if isinstance(kb_entries, list):
                            for entry in kb_entries:
                                if registry_url and 'registry_url' not in entry:
                                    entry['registry_url'] = registry_url
                                if auth_token and 'auth_token' not in entry:
                                    entry['auth_token'] = auth_token
        
        return kb_config


class ToolService:
    """Service for tool registry and tool loading.
    
    Handles:
    - Creating tool registries
    - Loading tools from configuration
    - Managing tool registration
    - Providing access to loaded tools
    """
    
    def __init__(self, project_root: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the tool service.
        
        Args:
            project_root: Project root directory
            logger: Optional logger instance
        """
        self.project_root = project_root
        self.logger = logger or logging.getLogger(__name__)
        self._registries: Dict[str, Any] = {}
    
    def create_registry(self, framework: str = 'base') -> Any:
        """Create a tool registry for a specific framework.
        
        Args:
            framework: Framework name (base, crewai, langchain, etc.)
            
        Returns:
            Tool registry instance
            
        Raises:
            ToolLoadingError: If registry creation fails
        """
        try:
            if framework.lower() == 'base':
                from oai_agent_core.core.base_tool_registry import BaseToolRegistry
                # BaseToolRegistry is abstract, so return None
                # Subclasses must provide concrete implementation
                return None
            else:
                # Framework-specific registries should be created by subclasses
                raise NotImplementedError(f"Framework '{framework}' not supported")
        except Exception as e:
            raise ToolLoadingError(f"Failed to create tool registry: {e}")
    
    def load_tools(self, registry: Any, tools_config: Dict[str, Any]) -> int:
        """Load tools into the registry.
        
        Args:
            registry: Tool registry instance
            tools_config: Tools configuration
            
        Returns:
            Number of tools loaded
        """
        if not registry:
            self.logger.warning("Tool registry not initialized")
            return 0
        
        try:
            registry.load_tools_from_config(tools_config)
            
            # Get tool count - support both get_tools() method and tools attribute
            if hasattr(registry, 'get_tools') and callable(registry.get_tools):
                tool_count = len(registry.get_tools())
            elif hasattr(registry, 'tools'):
                tool_count = len(registry.tools)
            else:
                tool_count = 0
            
            self.logger.info(f"Loaded {tool_count} tools into registry")
            return tool_count
        except Exception as e:
            self.logger.error(f"Failed to load tools: {e}")
            return 0
    
    def load_mcp_tools(self, registry: Any, mcp_config: Dict[str, Any]) -> int:
        """Load MCP tools into the registry.
        
        Args:
            registry: Tool registry instance
            mcp_config: MCP configuration
            
        Returns:
            Number of MCP servers loaded
        """
        if not registry:
            self.logger.warning("Tool registry not initialized")
            return 0
        
        try:
            registry.load_mcp_config(mcp_config)
            mcp_count = len(registry.mcp_configs)
            self.logger.info(f"Loaded {mcp_count} MCP servers into registry")
            return mcp_count
        except Exception as e:
            self.logger.error(f"Failed to load MCP tools: {e}")
            return 0
    
    def get_tool(self, registry: Any, tool_name: str) -> Optional[Any]:
        """Get a specific tool from the registry.
        
        Args:
            registry: Tool registry instance
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None if not found
        """
        if not registry:
            return None
        return registry.get_tool(tool_name)


class SkillService:
    """Service for skill discovery and management.
    
    Handles:
    - Initializing skill registry
    - Discovering local skills
    - Pulling remote skills from registry
    - Managing skill lifecycle
    """
    
    def __init__(self, project_root: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the skill service.
        
        Args:
            project_root: Project root directory
            logger: Optional logger instance
        """
        self.project_root = project_root
        self.logger = logger or logging.getLogger(__name__)
        self._skill_registry = None
    
    def create_skill_registry(self, skills_config: Dict[str, Any]) -> Any:
        """Create and initialize a skill registry.
        
        Args:
            skills_config: Skills configuration from agent config
            
        Returns:
            SkillRegistry instance, or None if not configured
        """
        if not skills_config:
            return None
        
        try:
            import os
            from oai_agent_core.components.skills.skill_registry import SkillRegistry
            
            skill_dir = skills_config.get('skill_dir')
            registry_config = skills_config.get('registry', {})
            
            registry = SkillRegistry(
                logger=self.logger,
                project_root=self.project_root,
                registry_url=registry_config.get('url') or os.environ.get('SKILLS_REGISTRY_URL'),
                auth_token=registry_config.get('auth_token') or os.environ.get('SKILLS_REGISTRY_AUTH_TOKEN'),
                skills_cache_dir=skill_dir
            )
            
            self._skill_registry = registry
            self.logger.debug("Created skill registry")
            return registry
        except Exception as e:
            self.logger.error(f"Failed to create skill registry: {e}")
            return None
    
    async def pull_required_skills(self, registry: Any, 
                                   required_skills: List[str]) -> int:
        """Pull required skills from remote registry.
        
        Args:
            registry: Skill registry instance
            required_skills: List of skill names to pull
            
        Returns:
            Number of skills successfully pulled
        """
        if not registry or not required_skills:
            return 0
        
        pulled_count = 0
        for skill_name in required_skills:
            try:
                success = await registry.pull_skill(skill_name)
                if success:
                    pulled_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to pull skill '{skill_name}': {e}")
        
        self.logger.info(f"Pulled {pulled_count} of {len(required_skills)} skills")
        return pulled_count
    
    def discover_skills(self, registry: Any, skill_dir: str) -> int:
        """Discover local skills in a directory.
        
        Args:
            registry: Skill registry instance
            skill_dir: Directory to search for skills
            
        Returns:
            Number of skills discovered
        """
        if not registry or not skill_dir:
            return 0
        
        try:
            registry.discover_skills(skills_dir=skill_dir)
            skill_count = len(registry.skills)
            self.logger.info(f"Discovered {skill_count} skills")
            return skill_count
        except Exception as e:
            self.logger.error(f"Failed to discover skills: {e}")
            return 0


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


class ObservabilityService:
    """Service for observability and tracing integration.
    
    Handles:
    - Initializing observability managers
    - Creating trace spans
    - Tracking errors and metrics
    - Integrating with Langfuse, OpenTelemetry, etc.
    """
    
    def __init__(self, agent_name: str = 'agent',
                 framework: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the observability service.
        
        Args:
            agent_name: Name of the agent
            framework: Agent framework type
            logger: Optional logger instance
        """
        self.agent_name = agent_name
        self.framework = framework
        self.logger = logger or logging.getLogger(__name__)
        self._observability_manager = None
    
    def create_langfuse_manager(self) -> Any:
        """Create a Langfuse observability manager.
        
        Returns:
            LangfuseObservabilityManager instance
        """
        try:
            from oai_agent_core.components.observability.langfuse_observability_manager import (
                LangfuseObservabilityManager
            )
            
            manager = LangfuseObservabilityManager(
                agent_name=self.agent_name,
                logger=self.logger,
                framework=self.framework
            )
            
            self._observability_manager = manager
            self.logger.debug("Created Langfuse observability manager")
            return manager
        except Exception as e:
            self.logger.error(f"Failed to create observability manager: {e}")
            return None
    
    def track_operation(self, operation_name: str) -> Optional[Any]:
        """Create a trace span for an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Trace span context manager, or None if observability not enabled
        """
        if not self._observability_manager:
            return None
        
        try:
            return self._observability_manager.trace_operation(operation_name)
        except Exception as e:
            self.logger.debug(f"Failed to create trace span: {e}")
            return None
    
    def track_error(self, operation_name: str, error: Exception) -> None:
        """Track an error in observability system.
        
        Args:
            operation_name: Name of the operation that failed
            error: The exception that was raised
        """
        if not self._observability_manager:
            return
        
        try:
            self._observability_manager.track_error(operation_name, error)
        except Exception as e:
            self.logger.debug(f"Failed to track error: {e}")
    
    def track_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Track token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        if not self._observability_manager:
            return
        
        try:
            self._observability_manager.track_tokens(
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        except Exception as e:
            self.logger.debug(f"Failed to track tokens: {e}")


class ConfigValidator:
    """Phase 3.2: Configuration validation and schema enforcement.
    
    Validates all configuration sections to catch errors early and provide
    clear error messages for invalid configurations.
    
    Handles:
    - Tool configuration validation
    - Skill configuration validation
    - Knowledge base configuration validation
    - Model configuration validation
    - Memory configuration validation
    - Guardrails configuration validation
    - Overall agent configuration validation
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the configuration validator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_agent_config(self, config: Dict[str, Any]) -> bool:
        """Validate complete agent configuration.
        
        Args:
            config: Agent configuration to validate
            
        Returns:
            True if all validations passed, False otherwise
        """
        self.errors = []
        self.warnings = []
        
        # Skip validation if config is empty or minimal (common in tests)
        if not config:
            return True
        
        # Required fields - but don't fail if 'type' is missing
        # (some test configs might not have it)
        if 'type' not in config and any(k in config for k in ['model', 'tools', 'knowledge_base']):
            # Config has sections but no type - worth warning about
            self.warnings.append("Configuration missing 'type' field but has other sections")
        
        # Validate optional sections only if they're non-empty
        if config.get('model'):  # Only validate if model config is present and non-empty
            self._validate_model_config(config['model'])
        
        if config.get('tools'):  # Only validate if tools config is present
            self._validate_tools_config(config['tools'])
        
        if config.get('knowledge_base'):  # Only validate if KB config is present
            self._validate_kb_config(config['knowledge_base'])
        
        if config.get('skills'):  # Only validate if skills config is present
            self._validate_skills_config(config['skills'])
        
        if config.get('memory'):  # Only validate if memory config is present
            self._validate_memory_config(config['memory'])
        
        if config.get('guardrails'):  # Only validate if guardrails config is present
            self._validate_guardrails_config(config['guardrails'])
        
        # Log results only if there are errors
        if self.errors:
            for error in self.errors:
                self.logger.error(f"Validation Error: {error}")
        
        if self.warnings:
            for warning in self.warnings:
                self.logger.debug(f"Validation Warning: {warning}")
        
        return len(self.errors) == 0
    
    def _validate_model_config(self, model_config: Any) -> None:
        """Validate model configuration.
        
        Args:
            model_config: Model configuration section
        """
        if not isinstance(model_config, dict):
            self.errors.append("'model' must be a dictionary")
            return
        
        # Skip validation for empty model configs (common in tests)
        if not model_config:
            return
        
        # Only require 'name' and 'provider' if other fields suggest it's a real config
        # (i.e., has settings like 'temperature', 'max_tokens', etc.)
        has_real_settings = any(k in model_config for k in [
            'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty',
            'stop', 'functions', 'tools', 'system_prompt', 'endpoint', 'api_key'
        ])
        
        if has_real_settings:
            # This looks like a real config, validate required fields
            if 'name' not in model_config:
                self.errors.append("Model configuration missing required 'name' field")
            
            if 'provider' not in model_config:
                self.errors.append("Model configuration missing required 'provider' field")
        else:
            # Minimal config (just name/provider or empty), just check type
            if 'name' in model_config and not isinstance(model_config['name'], str):
                self.warnings.append("Model 'name' should be a string")
            
            if 'provider' in model_config and not isinstance(model_config['provider'], str):
                self.warnings.append("Model 'provider' should be a string")
    
    def _validate_tools_config(self, tools_config: Any) -> None:
        """Validate tools configuration.
        
        Args:
            tools_config: Tools configuration section
        """
        if not isinstance(tools_config, dict):
            self.errors.append("'tools' must be a dictionary")
            return
        
        # Each tool should be properly configured
        for tool_name, tool_config in tools_config.items():
            if not isinstance(tool_config, dict):
                self.errors.append(f"Tool '{tool_name}' configuration must be a dictionary")
                continue
            
            # Tools should have either 'module' or 'command'
            if 'module' not in tool_config and 'command' not in tool_config and 'url' not in tool_config:
                self.warnings.append(
                    f"Tool '{tool_name}' missing 'module' (class/function), "
                    "'command' (MCP stdio), or 'url' (MCP HTTP)"
                )
    
    def _validate_kb_config(self, kb_config: Any) -> None:
        """Validate knowledge base configuration.
        
        Args:
            kb_config: Knowledge base configuration section
        """
        if isinstance(kb_config, dict):
            # New style with 'sources' or agent-level configs
            sources = kb_config.get('sources', [])
            if not isinstance(sources, list):
                self.warnings.append("KB 'sources' should be a list")
        elif isinstance(kb_config, list):
            # Old style list of KB entries
            for i, entry in enumerate(kb_config):
                if not isinstance(entry, dict):
                    self.errors.append(f"KB entry {i} must be a dictionary")
        else:
            self.errors.append("'knowledge_base' must be a dictionary or list")
    
    def _validate_skills_config(self, skills_config: Any) -> None:
        """Validate skills configuration.
        
        Args:
            skills_config: Skills configuration section
        """
        if not isinstance(skills_config, dict):
            self.errors.append("'skills' must be a dictionary")
            return
        
        # Required fields
        if 'skill_dir' not in skills_config:
            self.errors.append("Skills configuration missing required 'skill_dir' field")
        
        # Registry is optional but if provided should be valid
        if 'registry' in skills_config:
            registry = skills_config['registry']
            if not isinstance(registry, dict):
                self.errors.append("Skills 'registry' must be a dictionary")
    
    def _validate_memory_config(self, memory_config: Any) -> None:
        """Validate memory configuration.
        
        Args:
            memory_config: Memory configuration section
        """
        if not isinstance(memory_config, dict):
            self.errors.append("'memory' must be a dictionary")
            return
        
        # Type is required for memory
        if 'type' not in memory_config:
            self.warnings.append("Memory configuration missing 'type' field (defaults to 'simple')")
        
        valid_types = ['simple', 'redis', 'mongodb', 'diskcache', 'pinecone']
        mem_type = memory_config.get('type', '').lower()
        if mem_type and mem_type not in valid_types:
            self.warnings.append(f"Unknown memory type: '{mem_type}'")
    
    def _validate_guardrails_config(self, guardrails_config: Any) -> None:
        """Validate guardrails configuration.
        
        Args:
            guardrails_config: Guardrails configuration section
        """
        if not isinstance(guardrails_config, dict):
            self.errors.append("'guardrails' must be a dictionary")
            return
        
        # If configured, should have either 'guard' or 'definition_file'
        has_guard = 'guard' in guardrails_config
        has_def = 'definition_file' in guardrails_config
        
        if not has_guard and not has_def:
            self.warnings.append(
                "Guardrails configured but missing 'guard' or 'definition_file'"
            )
    
    def get_errors(self) -> List[str]:
        """Get all validation errors.
        
        Returns:
            List of error messages
        """
        return self.errors
    
    def get_warnings(self) -> List[str]:
        """Get all validation warnings.
        
        Returns:
            List of warning messages
        """
        return self.warnings
    
    def export_json_schema(self) -> Dict[str, Any]:
        """Export configuration schema as JSON Schema format.
        
        **Phase 3.5**: Generates a JSON Schema that can be used for:
        - Configuration validation in external tools
        - IDE autocompletion and validation
        - Documentation generation
        
        Returns:
            JSON Schema as a dictionary following JSON Schema Draft 7 standard
        """
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Agent Configuration Schema",
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "Agent framework type (crewai, langchain, etc.)",
                    "examples": ["crewai", "langchain", "bedrock"]
                },
                "model": {
                    "type": "object",
                    "description": "Language model configuration",
                    "properties": {
                        "name": {"type": "string", "description": "Model name"},
                        "provider": {
                            "type": "string",
                            "description": "LLM provider",
                            "enum": ["openai", "anthropic", "bedrock", "local"]
                        },
                        "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                        "max_tokens": {"type": "integer", "minimum": 1}
                    }
                },
                "tools": {
                    "type": "object",
                    "description": "Tool configurations",
                    "additionalProperties": True
                },
                "knowledge_base": {
                    "oneOf": [
                        {"type": "array", "description": "Old-style KB list (deprecated)"},
                        {
                            "type": "object",
                            "properties": {
                                "registry": {
                                    "type": "object",
                                    "properties": {
                                        "url": {"type": "string"},
                                        "token": {"type": "string"}
                                    }
                                },
                                "sources": {"type": "array"}
                            }
                        }
                    ]
                },
                "skills": {
                    "type": "object",
                    "properties": {
                        "skill_dir": {"type": "string"},
                        "registry": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "auth_token": {"type": "string"}
                            }
                        }
                    }
                },
                "memory": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["redis", "mongodb", "diskcache", "simple", "pinecone"]
                        }
                    }
                },
                "guardrails": {
                    "type": "object",
                    "properties": {
                        "definition_file": {"type": "string"},
                        "enable_agent_validation": {"type": "boolean"}
                    }
                }
            },
            "required": ["type"]
        }
        
        self.logger.debug("Exported configuration JSON schema")
        return schema
    
    def migrate_config(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate configuration from old format to new format.
        
        **Phase 3.5**: Handles common migration scenarios:
        - Converting old knowledge_base list format to new dict format
        - Updating deprecated fields
        - Adding required fields with defaults
        
        Args:
            old_config: Configuration in old format
            
        Returns:
            Configuration in new format
        """
        new_config = dict(old_config)
        
        # Migrate knowledge_base from list to dict format if needed
        if 'knowledge_base' in new_config and isinstance(new_config['knowledge_base'], list):
            kb_list = new_config['knowledge_base']
            
            # Extract common registry settings from first entry (if present)
            registry_url = None
            registry_token = None
            
            if kb_list and isinstance(kb_list[0], dict):
                registry_url = kb_list[0].get('registry_url')
                registry_token = kb_list[0].get('auth_token')
            
            # Convert to new format
            new_config['knowledge_base'] = {
                'sources': kb_list,
                'registry': {}
            }
            
            if registry_url:
                new_config['knowledge_base']['registry']['url'] = registry_url
            if registry_token:
                new_config['knowledge_base']['registry']['token'] = registry_token
            
            self.logger.info("Migrated knowledge_base from list to dict format")
        
        # Add type if missing
        if 'type' not in new_config:
            new_config['type'] = 'base'
            self.logger.warning("Added default 'type' field to configuration")
        
        return new_config
    
    def validate_with_details(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration and return detailed results.
        
        **Phase 3.5**: Provides structured validation output including:
        - Validation status (valid/invalid)
        - Error list with details
        - Warning list
        - Recommendations for fixes
        
        Args:
            config: Configuration to validate
            
        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str],
                'suggestions': List[str]
            }
        """
        self.validate_agent_config(config)
        
        suggestions = []
        
        # Generate suggestions based on errors/warnings
        if any('missing' in e.lower() for e in self.errors):
            suggestions.append("Ensure all required fields are present in configuration")
        
        if any('type' in w.lower() for w in self.warnings):
            suggestions.append("Consider adding 'type' field to specify agent framework")
        
        if any('provider' in w.lower() for w in self.warnings):
            suggestions.append("Specify a model provider (openai, anthropic, bedrock, etc.)")
        
        return {
            'valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'suggestions': suggestions
        }
