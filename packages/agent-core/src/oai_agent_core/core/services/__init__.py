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
    - Loading agent configs from YAML files
    - Resolving environment variables and macros
    - Validating configuration structure
    - Merging configuration defaults
    """
    
    def __init__(self, config_root: Optional[str] = None, 
                 logger: Optional[logging.Logger] = None):
        """Initialize the config resolver service.
        
        Args:
            config_root: Root directory for config files
            logger: Optional logger instance
        """
        self.config_root = config_root or os.getcwd()
        self.logger = logger or logging.getLogger(__name__)
    
    def load_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Load agent configuration from YAML file.
        
        Args:
            agent_name: Name of the agent (used to find YAML file)
            
        Returns:
            Agent configuration dictionary
            
        Raises:
            ConfigurationError: If config file not found or invalid
        """
        try:
            from oai_agent_core.components.configuration.model_config import ConfigManager
            
            config_manager = ConfigManager(config_root=self.config_root)
            config = config_manager.load_agent_config(agent_name)
            
            self.logger.debug(f"Loaded configuration for agent '{agent_name}'")
            return config
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration for agent '{agent_name}': {e}"
            )
    
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
            tool_count = len(registry.get_tools())
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
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the model service.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self._model_manager = None
    
    def set_model_manager(self, model_manager: Any) -> None:
        """Set the model manager instance.
        
        Args:
            model_manager: Model manager for LLM creation
        """
        self._model_manager = model_manager
    
    def create_model(self, model_config: Dict[str, Any]) -> Any:
        """Create an LLM instance from configuration.
        
        Args:
            model_config: Model configuration
            
        Returns:
            LLM instance, or None if creation failed
        """
        if not self._model_manager:
            self.logger.warning("Model manager not set")
            return None
        
        try:
            model = self._model_manager.create_model(model_config=model_config)
            self.logger.debug(f"Created model: {model_config.get('name', 'unknown')}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to create model: {e}")
            return None
    
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
