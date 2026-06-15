"""Agent Mixins for Phase 2 Architectural Refactoring

This module provides mixin classes that decompose BaseAgent's responsibilities
into focused, testable concerns:

- SkillsMixin: Skill discovery, pulling, and registration
- KnowledgeBaseMixin: Knowledge base configuration and initialization
- ToolsMixin: Tool registry coordination and loading
- MemoryMixin: Memory store initialization and context management
- MessageProcessingMixin: Message augmentation and guardrails
- ObservabilityMixin: Observability and tracing integration

## Design Pattern

Mixins are designed to be combined using Python's MRO (Method Resolution Order).
Each mixin is self-contained with:
- Required attributes (documented at top of class)
- Methods for its concern
- Optional dependencies (imported at method level to avoid circular deps)

## Usage Example

    class MyAgent(SkillsMixin, KnowledgeBaseMixin, ToolsMixin,
                  MemoryMixin, MessageProcessingMixin,
                  ObservabilityMixin, ABC):
        
        def __init__(self, ...):
            # Init all mixin attributes
            self.skill_registry = None
            self.global_kb_factory = None
            # ... etc
        
        async def initialize(self):
            # Mixins provide methods for their concerns
            await self._load_skills_if_configured()
            await self._load_knowledge_base()
            await self._load_tools()
            # ... etc

## Benefits

✅ Single Responsibility: Each mixin has one clear purpose
✅ Testability: Mock individual mixin methods in tests
✅ Reusability: Combine mixins with different base classes
✅ Readability: Clear separation of concerns
✅ Backward Compatible: Existing code unchanged
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Type, List
from abc import ABC


class SkillsMixin:
    """Mixin for skill discovery, pulling, and registration.
    
    Required Attributes (set by BaseAgent.__init__):
    - skill_registry: Optional[SkillRegistry]
    - agent_config: Dict[str, Any]
    - config_root: Optional[str]
    - logger: logging.Logger
    
    Provides:
    - Skill initialization from configuration
    - Remote skill pulling from registry
    - Skill registry management
    """
    
    async def _load_skills_if_configured(self) -> None:
        """Load and register skills if configured in agent_config.
        
        Initializes skill registry from configuration and pulls remote skills
        if registry URL is provided. Handles errors gracefully (logs but doesn't fail).
        """
        if not self.skill_registry:
            self.logger.debug("Skill registry not initialized; skipping skill loading")
            return
        
        skills_config = self.agent_config.get('skills', {})
        if not skills_config:
            self.logger.debug("No skills configured for agent")
            return
        
        try:
            # Get remote skills list if registry configured
            remote_skills = skills_config.get('remote_skills', [])
            if remote_skills:
                self.logger.info(f"Pulling {len(remote_skills)} remote skills from registry")
                await self.skill_registry.pull_skills(remote_skills)
            
            self.logger.debug(f"Skills loaded successfully")
        except Exception as e:
            # Non-critical failure - log but don't block agent initialization
            self.logger.warning(f"Failed to load skills: {e}")
    
    def get_skill_registry(self):
        """Get the skill registry instance.
        
        Returns:
            SkillRegistry or None if not initialized.
        """
        return self.skill_registry


class KnowledgeBaseMixin:
    """Mixin for knowledge base configuration and initialization.
    
    Required Attributes (set by BaseAgent.__init__):
    - global_kb_factory: Optional[KnowledgeBaseFactory]
    - agent_config: Dict[str, Any]
    - config_root: Optional[str]
    - document_loader: Optional[Any]
    - vector_store: Optional[Any]
    - logger: logging.Logger
    
    Provides:
    - KB configuration loading and validation
    - KB factory initialization
    - Multi-KB support (agent list with per-agent KBs)
    - Registry credential propagation
    """
    
    def _enrich_agent_list_kb_configs(self) -> None:
        """Propagate top-level KB registry credentials into every agent's KB entries.

        Walks through ``agent_list`` in ``self.agent_config`` and, for any agent
        whose ``knowledge_base`` is a flat list without per-entry credentials,
        injects the top-level ``knowledge_base.registry`` URL and token.

        This ensures that by the time the KB factory receives an agent config,
        each KB entry already has ``registry_url`` and ``auth_token`` set.
        Per-entry values always win over the top-level block.

        Only runs when ``knowledge_base`` is a dict (new-style config).
        """
        kb_config = self.agent_config.get('knowledge_base')
        if not isinstance(kb_config, dict):
            return  # Old-style list or missing KB config
        
        agent_list = kb_config.get('agent_list', [])
        if not agent_list:
            return  # No agent list, nothing to enrich
        
        # Top-level registry config
        registry_url = kb_config.get('registry', {}).get('url')
        auth_token = kb_config.get('registry', {}).get('auth_token')
        
        if not registry_url:
            return  # No top-level registry to propagate
        
        # Enrich each agent's KB entries with registry credentials
        for agent_entry in agent_list:
            kb_entries = agent_entry.get('knowledge_base', [])
            if isinstance(kb_entries, list):
                for entry in kb_entries:
                    # Per-entry values take precedence
                    if 'registry_url' not in entry and registry_url:
                        entry['registry_url'] = registry_url
                    if 'auth_token' not in entry and auth_token:
                        entry['auth_token'] = auth_token
    
    async def _load_knowledge_base(self) -> None:
        """Initialize knowledge base factory from configuration.
        
        Creates KB factory and validates KB config. Handles errors gracefully
        (logs but doesn't block agent initialization).
        
        If using agent_list (multi-KB), enriches each agent's KB entries
        with top-level registry credentials.
        """
        kb_config = self.agent_config.get('knowledge_base')
        if not kb_config:
            self.logger.debug("No knowledge base configured for agent")
            return
        
        try:
            # Enrich agent list KB configs with top-level registry credentials
            if isinstance(kb_config, dict) and kb_config.get('agent_list'):
                self._enrich_agent_list_kb_configs()
            
            # Import here to avoid circular dependencies
            from oai_agent_core.components.knowledge_base.knowledge_base_factory import (
                KnowledgeBaseFactory
            )
            
            self.global_kb_factory = KnowledgeBaseFactory(
                logger=self.logger,
                project_root=self.config_root,
                document_loader=self.document_loader,
                vector_store=self.vector_store
            )
            self.logger.debug("Knowledge base factory initialized")
        except Exception as e:
            # Non-critical failure - log but don't block agent initialization
            self.logger.warning(f"Failed to initialize knowledge base: {e}")
    
    def get_knowledge_base(self, kb_config: Dict[str, Any]):
        """Get a knowledge base instance for the given configuration.
        
        Args:
            kb_config: Knowledge base configuration dict
            
        Returns:
            Knowledge base instance or None if factory not available
        """
        if not self.global_kb_factory:
            return None
        return self.global_kb_factory.get_knowledge_base(kb_config)


class ToolsMixin:
    """Mixin for tool registry coordination and loading.
    
    Required Attributes (set by BaseAgent.__init__):
    - tool_registry: Optional[BaseToolRegistry]
    - agent_config: Dict[str, Any]
    - logger: logging.Logger
    
    Provides:
    - Tool registry initialization
    - Tool loading from configuration
    - Tool registry management
    """
    
    async def _load_tools(self) -> None:
        """Initialize tool registry and load tools from configuration.
        
        Creates tool registry instance and loads all configured tools
        (framework tools, custom tools, MCP tools). Handles errors gracefully
        (logs but doesn't block agent initialization).
        """
        tools_config = self.agent_config.get('tools')
        if not tools_config:
            self.logger.debug("No tools configured for agent")
            return
        
        try:
            # Import here to avoid circular dependencies
            from oai_agent_core.core.base_tool_registry import BaseToolRegistry
            
            # Create tool registry - subclasses override this
            if not hasattr(self, 'get_tool_registry'):
                self.logger.warning("Tool registry creation not implemented in subclass")
                return
            
            self.tool_registry = self.get_tool_registry()
            
            if self.tool_registry:
                # Load framework tools
                if isinstance(tools_config, dict):
                    await self.tool_registry.load_tools_from_config(tools_config)
                    
                    # Load MCP tools separately if configured
                    mcp_config = tools_config.get('mcp')
                    if mcp_config:
                        await self.tool_registry.load_mcp_config(mcp_config)
                
                self.logger.debug(
                    f"Tool registry initialized with "
                    f"{self.tool_registry.get_tool_count()} tools"
                )
        except Exception as e:
            # Non-critical failure - log but don't block agent initialization
            self.logger.warning(f"Failed to load tools: {e}")
    
    def get_tools(self) -> Dict[str, Any]:
        """Get all loaded tools from the tool registry.
        
        Returns:
            Dictionary of tool name -> tool instance, or empty dict if no registry
        """
        if not self.tool_registry:
            return {}
        return self.tool_registry.get_tools()
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get a specific tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Tool instance or None if not found
        """
        if not self.tool_registry:
            return None
        return self.tool_registry.get_tool(tool_name)


class MemoryMixin:
    """Mixin for memory store initialization and context management.
    
    Required Attributes (set by BaseAgent.__init__):
    - memory_store: Optional[Any]
    - agent_config: Dict[str, Any]
    - session_id: str
    - user_id: str
    - logger: logging.Logger
    
    Provides:
    - Memory store initialization
    - Conversation context retrieval
    - Session and user management
    """
    
    async def _load_memory(self) -> None:
        """Initialize memory store from configuration.
        
        Creates memory store instance for conversation history and context.
        Handles errors gracefully (logs but doesn't block agent initialization).
        """
        memory_config = self.agent_config.get('memory')
        if not memory_config:
            self.logger.debug("No memory store configured for agent")
            return
        
        try:
            # Import here to avoid circular dependencies
            from oai_agent_core.components.memory.memory_manager import MemoryManager
            
            memory_type = memory_config.get('type', 'in_memory')
            
            self.memory_store = MemoryManager.create_memory_store(
                memory_type=memory_type,
                config=memory_config,
                user_id=self.user_id,
                session_id=self.session_id,
                logger=self.logger
            )
            self.logger.debug(f"Memory store initialized (type: {memory_type})")
        except Exception as e:
            # Non-critical failure - log but don't block agent initialization
            self.logger.warning(f"Failed to initialize memory store: {e}")
    
    def _get_conversation_context(self, current_message: str) -> str:
        """Retrieve conversation context from memory store.
        
        Fetches previous messages relevant to the current message from the
        memory store to provide context for the LLM.
        
        Args:
            current_message: The current user message
            
        Returns:
            Conversation context string, or empty string if no memory store
        """
        if not self.memory_store:
            return ""
        
        try:
            context = self.memory_store.get_context(
                session_id=self.session_id,
                user_id=self.user_id,
                current_message=current_message
            )
            return context if context else ""
        except Exception as e:
            self.logger.debug(f"Failed to retrieve conversation context: {e}")
            return ""
    
    def store_conversation(self, user_message: str, assistant_response: str) -> None:
        """Store a message exchange in memory.
        
        Args:
            user_message: The user's message
            assistant_response: The agent's response
        """
        if not self.memory_store:
            return
        
        try:
            self.memory_store.store_message(
                session_id=self.session_id,
                user_id=self.user_id,
                role='user',
                content=user_message
            )
            self.memory_store.store_message(
                session_id=self.session_id,
                user_id=self.user_id,
                role='assistant',
                content=assistant_response
            )
        except Exception as e:
            self.logger.debug(f"Failed to store conversation: {e}")


class MessageProcessingMixin:
    """Mixin for message augmentation and guardrails.
    
    Required Attributes (set by BaseAgent.__init__):
    - guardrails_manager: Optional[Any]
    - agent_config: Dict[str, Any]
    - logger: logging.Logger
    - And attributes from MemoryMixin (for context)
    - And attributes from KnowledgeBaseMixin (for KB search)
    
    Provides:
    - System prompt augmentation with KB and memory context
    - Input message validation and guardrails
    - Output message validation and guardrails
    - Message augmentation with macros
    """
    
    def _augment_system_prompt(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Augment system prompt with KB and memory context.
        
        Adds knowledge base search results and conversation memory context
        to the system prompt to provide the LLM with relevant context.
        
        Args:
            config: Agent execution configuration
            
        Returns:
            Updated configuration with augmented system prompt
        """
        try:
            config = deepcopy(config) if config else {}
            system_prompt = config.get('system_prompt', '')
            
            # Import here to avoid circular dependencies
            from oai_agent_core.macros import MacroProcessor
            
            # Add KB search results if available
            kb_context = ""
            if hasattr(self, 'global_kb_factory') and self.global_kb_factory:
                kb_config = self.agent_config.get('knowledge_base')
                if kb_config:
                    kb = self.get_knowledge_base(kb_config)
                    if kb:
                        # Search KB for context (query would come from user message)
                        kb_context = "<!-- Knowledge Base Context will be inserted here -->"
            
            # Add conversation context if available
            conversation_context = ""
            if hasattr(self, '_get_conversation_context'):
                conversation_context = self._get_conversation_context(
                    config.get('current_message', '')
                )
                if conversation_context:
                    conversation_context = f"\n\nPrevious Conversation:\n{conversation_context}"
            
            # Combine contexts
            if kb_context or conversation_context:
                system_prompt = (
                    f"{system_prompt}"
                    f"{kb_context}"
                    f"{conversation_context}"
                ).strip()
            
            config['system_prompt'] = system_prompt
            return config
        except Exception as e:
            self.logger.debug(f"Failed to augment system prompt: {e}")
            return config
    
    def _guardrail_input_message(self, message: str) -> str:
        """Apply guardrails to validate input message.
        
        Validates user input against configured guardrail policies.
        Returns message unchanged if passes or if guardrails not configured.
        
        Args:
            message: The user input message
            
        Returns:
            Validated message (may be modified by guardrails)
            
        Raises:
            GuardrailViolationError: If message violates critical policies
        """
        if not self.guardrails_manager:
            return message
        
        try:
            validated = self.guardrails_manager.validate_input(message)
            return validated
        except Exception as e:
            self.logger.warning(f"Input guardrail validation failed: {e}")
            return message
    
    def _guardrail_output_message(self, message: str) -> str:
        """Apply guardrails to validate output message.
        
        Validates agent output against configured guardrail policies.
        Returns message unchanged if passes or if guardrails not configured.
        
        Args:
            message: The agent output message
            
        Returns:
            Validated message (may be modified by guardrails)
        """
        if not self.guardrails_manager:
            return message
        
        try:
            validated = self.guardrails_manager.validate_output(message)
            return validated
        except Exception as e:
            self.logger.warning(f"Output guardrail validation failed: {e}")
            return message
    
    def _augment_message(self, message: str, original_query: str = None) -> str:
        """Augment message with macro processing.
        
        Processes macros in message templates (e.g., {session_id}, {user_id}).
        
        Args:
            message: The message to augment
            original_query: Optional original user query for context
            
        Returns:
            Message with macros replaced
        """
        try:
            from oai_agent_core.macros import MacroProcessor
            
            processor = MacroProcessor(
                agent_name=getattr(self, 'agent_name', 'agent'),
                session_id=getattr(self, 'session_id', 'default'),
                user_id=getattr(self, 'user_id', 'default')
            )
            return processor.process(message)
        except Exception as e:
            self.logger.debug(f"Failed to augment message with macros: {e}")
            return message


class ObservabilityMixin:
    """Mixin for observability and tracing integration.
    
    Required Attributes (set by BaseAgent.__init__):
    - langfuse_manager: LangfuseObservabilityManager
    - logger: logging.Logger
    - agent_name: str
    
    Provides:
    - Tracing spans for agent operations
    - Error tracking and logging
    - Performance metrics
    """
    
    async def _trace_operation(self, operation_name: str, handler_coro):
        """Trace an agent operation with Langfuse.
        
        Wraps an async operation with observability tracing.
        
        Args:
            operation_name: Name of the operation being traced
            handler_coro: Async coroutine to trace
            
        Returns:
            Result from the handler coroutine
        """
        if not self.langfuse_manager:
            return await handler_coro
        
        try:
            with self.langfuse_manager.trace_operation(operation_name) as trace:
                result = await handler_coro
                trace.end(status='success')
                return result
        except Exception as e:
            self.logger.error(f"Operation {operation_name} failed: {e}", exc_info=True)
            if self.langfuse_manager:
                self.langfuse_manager.track_error(operation_name, e)
            raise
    
    def track_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Track token usage for observability.
        
        Args:
            input_tokens: Number of input tokens consumed
            output_tokens: Number of output tokens generated
        """
        if self.langfuse_manager:
            self.langfuse_manager.track_tokens(
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
    
    def get_trace_context(self) -> Dict[str, Any]:
        """Get current trace context for propagation.
        
        Returns:
            Dictionary with trace context (trace_id, span_id, etc.)
        """
        if self.langfuse_manager:
            return self.langfuse_manager.get_context()
        return {}


# Import deepcopy for message augmentation
from copy import deepcopy
