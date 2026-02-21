import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type

from oai_agent_core.components.configuration.model_config import ConfigManager
from oai_agent_core.components.configuration.model_config import config_manager
from oai_agent_core.components.observability.langfuse_observability_manager import LangfuseObservabilityManager
from oai_agent_core.core.base_model_configuration_manager import BaseModelConfigurationManager


class BaseAgent(ABC):
    """Abstract base class for all agents.

    This class defines the common interface and shared functionality for all agent implementations.
    It handles configuration loading, model initialization, observability setup, and defines
    the contract for agent execution methods.
    """

    def __init__(
            self,
            agent_name: str,
            agent_config: Optional[Dict[str, Any]] = None,
            llm: Any = None,
            session_id: str = "default",
            user_id: str = "default",
            config_root: str = None,
            agent_type: str = None,
            document_loader: Optional[Any] = None,
            vector_store: Optional[Any] = None,
            model_manager: Optional[BaseModelConfigurationManager] = None,
            **kwargs
    ):
        """Initialize the BaseAgent.

        Args:
            agent_name: Unique identifier for the agent.
            agent_config: Dictionary containing agent configuration. If None, loaded from file.
            llm: Language model instance. If None, created using model_manager.
            session_id: Session identifier for tracking execution context.
            user_id: User identifier for tracking and personalization.
            config_root: Root directory for configuration files.
            agent_type: Type of the agent (e.g., 'crewai', 'langchain').
            document_loader: Optional document loader instance for knowledge base.
            vector_store: Optional vector store instance for knowledge base.
            model_manager: Optional manager for creating LLM instances.
            **kwargs: Additional keyword arguments to be set as attributes.
        """
        self.agent_name = agent_name
        self.config_manager = ConfigManager(config_root=config_root)
        self.config_root = config_root

        # Load config from file if not provided
        if agent_config is None:
            agent_config = self.config_manager.load_agent_config(agent_name)

        self.agent_config = agent_config

        # Assign LLM if provided, otherwise create from manager
        self.llm = llm
        if self.llm is None:
            self.model_manager = model_manager
            model_config = agent_config.get('model')
            if self.model_manager:
                self.llm = self.model_manager.create_model(model_config=model_config)
            else:
                self.llm = None  # Handle case where model_manager is not provided

        self.session_id = session_id
        self.user_id = user_id
        self.agent_type = agent_config.get('type') if 'type' in agent_config else agent_type
        self._initialized = False

        # Initialize any additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.logger = logging.getLogger(__name__)

        self.langfuse_manager = LangfuseObservabilityManager(
            agent_name=self.agent_name,
            logger=self.logger,
            framework=self.agent_type
        )
        self.document_loader = document_loader
        self.vector_store = vector_store
        self.memory_store = None
        self.global_kb_factory = None
        self.tool_registry = None
        self.guardrails_manager = None

    def assign_llm(self, model_config: Dict):
        """
        Assign model to the agent.
        :param model_config: Model configuration
        :return:
        """
        self.llm = self.model_manager.create_model(model_config=model_config)

    async def _load_tools_and_kb_and_memory(self, kb_factory_class: Optional[Type] = None) -> None:
        """Load tools and initialize global knowledge base concurrently.

        This method handles:
        1. Loading tools from configuration (using tool_registry).
        2. Loading MCP configuration (using tool_registry).
        3. Initializing global knowledge base (if configured and factory provided).

        Args:
            kb_factory_class: The KnowledgeBaseFactory class to use for initialization.
        """
        if not hasattr(self, 'tool_registry'):
            self.logger.warning("Tool registry not initialized. Skipping tool loading.")
            return

        async def _load_tools_and_mcp():
            tools_config = self.agent_config.get('tools', {})
            if tools_config:
                await asyncio.to_thread(self.tool_registry.load_tools_from_config, tools_config)
                self.logger.info(
                    f"Loaded {len(self.tool_registry.tools)} global tools into registry"
                )

            mcp_config = self.agent_config.get('mcps', self.agent_config.get('servers', {}))
            if mcp_config:
                self.tool_registry.load_mcp_config(mcp_config)

        async def _init_global_kb():
            global_kb_config = self.agent_config.get('knowledge_base', [])
            if global_kb_config and kb_factory_class:
                try:
                    self.global_kb_factory = await asyncio.to_thread(
                        kb_factory_class,
                        knowledge_base_config=global_kb_config,
                        logger=self.logger,
                        project_root=self.config_root,
                        llm=self.llm,
                        document_loader=self.document_loader,
                        vector_store=self.vector_store
                    )
                    self.logger.info("Initialized global knowledge base")
                except ImportError as e:
                    self.logger.warning(f"Could not initialize global knowledge base: missing dependencies {e}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize global knowledge base: {e}")

        async def _init_memory_store():
            memory_config = self.agent_config.get('memory', {})
            if memory_config:
                # Import here to avoid circular dependencies or early import issues
                # We use a generic MemoryStore implementation that uses BaseMemoryStore logic
                # but allows for dynamic vector store creation
                from oai_agent_core.core.base_memory_store import BaseMemoryStore
                
                # Create a concrete implementation of BaseMemoryStore
                class ConfigurableMemoryStore(BaseMemoryStore):
                    pass
                
                try:
                    self.memory_store = ConfigurableMemoryStore(
                        memory_config=memory_config,
                        logger=self.logger,
                        project_root=self.config_root,
                        llm=self.llm
                    )
                    self.logger.info("Initialized memory store")
                except Exception as e:
                    self.logger.error(f"Failed to initialize memory store: {e}")

        async def _init_guardrails():
            guardrails_config = self.agent_config.get('guardrails', {})
            if guardrails_config:
                from oai_agent_core.components.guardrails.guardrails_manager import GuardrailManager
                self.guardrails_manager = GuardrailManager(self.config_root,
                                                           guardrails_config,
                                                           logger=self.logger)
                if guardrails_config.get('enable_agent_validation', True):
                    if len(self.agent_config.get('agent_list', [])) > 1:
                        self.agent_config[
                            'system_prompt'] = f"{self.agent_config.get('system_prompt', '')}\n{self.guardrails_manager.get_guardrails_prompt()}"
                    elif len(self.agent_config.get('agent_list', [])) == 1:
                        agent_name = list((self.agent_config['agent_list'][0]).keys())[0]
                        self.agent_config['agent_list'][0][agent_name][
                            'system_prompt'] = f"{self.agent_config['agent_list'][0][agent_name].get('system_prompt', '')}\n{self.guardrails_manager.get_guardrails_prompt()}"

        await asyncio.gather(_load_tools_and_mcp(), _init_global_kb(), _init_memory_store(), _init_guardrails())

    def _get_conversation_context(self, current_message: str) -> str:
        """Retrieve and format conversation context for the current message.

        Args:
            current_message: The current user message

        Returns:
            Formatted conversation context string
        """
        if not self.memory_store:
            return current_message

        try:
            recent_turns, relevant_turns = self.memory_store.get_relevant_context(
                current_message=current_message,
                session_id=self.session_id,
                user_id=self.user_id,
                include_recent=True
            )

            context = self.memory_store.format_context_for_prompt(
                recent_turns=recent_turns,
                relevant_turns=relevant_turns
            )

            return context

        except Exception as e:
            self.logger.warning(f"Could not retrieve conversation context: {e}")
            return ""

    def _guardrail_input_message(self, message: str):
        """
        Apply guardrails to the input message, if enabled
        :param message: The message to guardrail
        :return: The guarded message, or the original message if guardrails are not enabled
        """
        if self.guardrails_manager:
            return self.guardrails_manager.validate_input(message)
        return message

    def _guardrail_output_message(self, message: str):
        """
        Apply guardrails to the output message, if enabled
        :param message: The message to guardrail
        :return: The guarded message, or the original message if guardrails are not enabled
        """
        if self.guardrails_manager:
            return self.guardrails_manager.validate_output(message)
        return message

    def _augment_message(self, message: str, original_query: str = None) -> str:
        """Augment the message with knowledge base and memory context.

        Args:
            message: The message to augment (usually the formatted prompt).
            original_query: The original user query (used for semantic search).
                            If None, 'message' is used.

        Returns:
            The augmented message string.
        """
        query = original_query if original_query else message
        augmented_message = message

        # Augment with global knowledge base if available
        if self.global_kb_factory:
            try:
                kb_result = self.global_kb_factory.search_custom_knowledge_base(query)
                augmented_message = f"{augmented_message}\n\nRelevant Context from Knowledge Base:\n{kb_result}"
            except Exception as e:
                self.logger.warning(f"Failed to search knowledge base: {e}")

        # Add conversation context
        if self.memory_store:
            conversation_context = self._get_conversation_context(current_message=query)
            if conversation_context:
                augmented_message = f"{conversation_context}\n\n{augmented_message}"

        return augmented_message

    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'model.temperature').

        Args:
            key_path: Dot-separated path to the configuration key.
            default: Default value to return if key is not found.

        Returns:
            The configuration value or default.
        """
        return self.config_manager.get_config_value(self.agent_config, key_path, default)

    def set_config_value(self, key_path: str, value: Any) -> None:
        """Set a configuration value using dot notation.

        Args:
            key_path: Dot-separated path to the configuration key.
            value: Value to set.
        """
        self.config_manager.set_config_value(self.agent_config, key_path, value)
        # Update agent_type if it was changed
        if key_path == 'type':
            self.agent_type = value

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update agent configuration with merge support.

        Args:
            updates: Dictionary of configuration updates to merge.
        """
        self.agent_config = self.config_manager.merge_configs(self.agent_config, updates)
        # Update agent_type if it was changed
        if 'type' in updates:
            self.agent_type = updates['type']

    def validate_config(self) -> None:
        """Validate the current agent configuration.

        Raises:
            ValueError: If required fields are missing.
        """
        required_fields = ['type']
        for field in required_fields:
            if field not in self.agent_config:
                raise ValueError(f"Missing required field '{field}' in configuration for agent '{self.agent_name}'")

    @staticmethod
    def load_agent_config(agent_name: str) -> Dict[str, Any]:
        """Load agent configuration from YAML file (static method for backward compatibility).

        Args:
            agent_name: Name of the agent to load configuration for.

        Returns:
            Dictionary containing agent configuration.
        """
        return config_manager.load_agent_config(agent_name)

    @property
    def is_initialized(self) -> bool:
        """Check if agent is initialized.

        Returns:
            True if initialized, False otherwise.
        """
        return self._initialized

    @abstractmethod
    async def initialize(self):
        """Initialize the agent resources and connections.

        This method must be implemented by subclasses to perform any necessary
        setup before the agent can be used.
        """
        raise NotImplementedError("Subclasses must implement initialize method")

    @abstractmethod
    async def astream(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Asynchronously stream the agent's response.

        Args:
            user_message: The input message from the user.
            config: Optional configuration overrides for this request.

        Yields:
            Chunks of the response.
        """
        raise NotImplementedError("Subclasses must implement astream method")

    @abstractmethod
    async def ainvoke(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Asynchronously invoke the agent and get the full response.

        Args:
            user_message: The input message from the user.
            config: Optional configuration overrides for this request.

        Returns:
            The agent's response.
        """
        raise NotImplementedError("Subclasses must implement ainvoke method")

    @abstractmethod
    def invoke(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Synchronously invoke the agent and get the full response.

        Args:
            user_message: The input message from the user.
            config: Optional configuration overrides for this request.

        Returns:
            The agent's response.
        """
        raise NotImplementedError("Subclasses must implement invoke method")

    @abstractmethod
    async def stream(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Synchronously stream the agent's response (if supported).

        Args:
            user_message: The input message from the user.
            config: Optional configuration overrides for this request.

        Yields:
            Chunks of the response.
        """
        raise NotImplementedError("Subclasses must implement stream method")

    async def _ensure_initialized(self) -> None:
        """Ensure the agent is initialized before use."""
        if not self._initialized:
            await self.initialize()

    def __del__(self):
        """Cleanup resources when the agent is destroyed."""
        if hasattr(self, 'langfuse_manager') and self.langfuse_manager.is_enabled:
            self.langfuse_manager.flush()
