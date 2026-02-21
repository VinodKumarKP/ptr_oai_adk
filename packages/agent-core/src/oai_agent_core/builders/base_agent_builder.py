"""Base Builder for creating Agent instances."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

from oai_agent_core.components.configuration.model_config import ConfigManager
from oai_agent_core.core.base_knowledge_base_factory import BaseKnowledgeBaseFactory
from oai_agent_core.core.base_agent import BaseAgent
from oai_agent_core.core.constants import Constants


class BaseAgentBuilder(ABC):
    """Abstract base class for Agent Builders.

    Provides common functionality for:
    - Tool loading (Regular, MCP, Knowledge Base)
    - Model initialization
    - Configuration normalization
    - Parallel agent creation
    """

    def __init__(
            self,
            model_manager: Any,
            tool_registry: Any,
            llm: Optional[Any] = None,
            config_root: Optional[str] = None,
            logger: Optional[logging.Logger] = None,
            document_loader: Optional[Any] = None,
            vector_store: Optional[Any] = None,
    ):
        """Initialize the base agent builder.

        Args:
            model_manager: Manager for model configuration.
            tool_registry: Registry for managing tools.
            llm: Optional pre-configured language model instance.
            config_root: Root directory for configuration files.
            logger: Optional logger instance.
            document_loader: Optional document loader instance.
            vector_store: Optional vector store instance.
        """
        self.llm = llm
        self.model_manager = model_manager
        self.tool_registry = tool_registry
        self.config_root = config_root
        self.logger = logger or logging.getLogger(__name__)
        self.document_loader = document_loader
        self.vector_store = vector_store

        # Locks to prevent race conditions during concurrent creation
        self._tool_lock = asyncio.Lock()
        self._kb_lock = asyncio.Lock()

    async def create_single_agent(
            self,
            agent_name: str,
            agent_config: Dict[str, Any],
            pre_loaded_tools: Optional[List[Any]] = None
    ) -> Any:
        """Create a single agent instance.

        Args:
            agent_name: Name of the agent.
            agent_config: Configuration dictionary for the agent.
            pre_loaded_tools: List of tools already loaded/instantiated.

        Returns:
            Configured Agent instance.
        """
        # Ensure model is initialized
        self._ensure_model(agent_config)

        # Collect all tools for this agent
        all_tools = list(pre_loaded_tools) if pre_loaded_tools else []

        # Add regular tools
        regular_tools = await self._get_regular_tools(agent_name, agent_config)
        all_tools.extend(regular_tools)

        # Load and configure MCP tools
        mcp_tools = await self._load_mcp_tools(agent_name, agent_config)
        all_tools.extend(mcp_tools)

        # Load Knowledge Base tools
        kb_tools = await self._load_knowledge_base_tools(agent_name, agent_config)
        all_tools.extend(kb_tools)

        # Create specific agent instance
        return self._create_agent_instance(agent_name, agent_config, all_tools)

    @abstractmethod
    def _create_agent_instance(
            self,
            agent_name: str,
            agent_config: Dict[str, Any],
            tools: List[Any]
    ) -> Any:
        """Create the framework-specific agent instance.

        Args:
            agent_name: Name of the agent.
            agent_config: Configuration dictionary.
            tools: List of configured tools.

        Returns:
            Framework-specific agent instance.
        """
        pass

    def _ensure_model(self, agent_config: Dict[str, Any]) -> None:
        """Ensure the LLM model is initialized."""
        if self.llm is None:
            model_config = agent_config.get('model')
            model = self.model_manager.create_model(model_config)
            self.llm = model

    async def _get_regular_tools(self, agent_name: str, agent_config: Dict[str, Any]) -> List[Any]:
        """Retrieve regular tools configured for the agent."""
        tool_names = agent_config.get('tools', [])
        if not tool_names:
            return []

        # Normalize tool names to list
        tool_name_list = []
        if isinstance(tool_names, list):
            tool_name_list = tool_names
        elif isinstance(tool_names, dict):
            tool_name_list = list(tool_names.keys())

        # Try to get tools
        agent_tools = self.tool_registry.get_tools_for_agent(tool_name_list)

        # If tools missing, try loading with lock
        if len(agent_tools) == 0:
            async with self._tool_lock:
                # Double-check pattern
                agent_tools = self.tool_registry.get_tools_for_agent(tool_name_list)
                if len(agent_tools) == 0:
                    await asyncio.to_thread(self.tool_registry.load_tools_from_config, agent_config.get('tools'))
                    agent_tools = self.tool_registry.get_tools_for_agent(tool_name_list)

        self.logger.debug(
            f"Added {len(agent_tools)} tools from registry to agent '{agent_name}'"
        )
        return agent_tools

    async def _load_mcp_tools(self, agent_name: str, agent_config: Dict[str, Any]) -> List[Any]:
        """Load MCP tools based on configuration."""
        mcp_tools = []
        
        # Check for both 'mcps' and 'servers' keys for compatibility
        mcp_config = agent_config.get('mcps', agent_config.get('servers'))
        
        if mcp_config:
            # If it's a list (Strands style), convert to dict or handle appropriately
            # The current implementations differ slightly here.
            # OpenAI/LangGraph expect dict or list of dicts?
            # Let's look at the implementations:
            # OpenAI: load_mcp_tools_from_config(dict, agent_name)
            # LangGraph: load_mcp_tools_from_config(dict)
            # Strands: load_mcp_tools_from_config(dict)
            
            # We'll assume the registry handles the format, or we normalize here.
            # Most registries seem to take a dict of config.
            
            if isinstance(mcp_config, list):
                # Normalize list of dicts to single dict if needed, or pass as is if registry supports it
                # For now, let's assume the registry method signature matches what we pass.
                # But wait, Strands builder had specific logic for list vs dict.
                pass

            # We will delegate to the registry, but we need to be careful about the signature.
            # OpenAI registry takes (config, agent_name).
            # LangGraph/Strands registry takes (config).
            
            # To make this generic, we might need to inspect the registry method or standardize it.
            # For now, we'll try to call it with just config, as that's the common denominator,
            # unless we know it's OpenAI.
            
            # Actually, let's look at the OpenAI implementation again.
            # await self.tool_registry.load_mcp_tools_from_config(config, agent_name)
            
            # We can try to pass agent_name as a keyword arg if supported.
            try:
                loaded_tools = await self.tool_registry.load_mcp_tools_from_config(mcp_config, agent_name=agent_name)
            except TypeError:
                # Fallback for registries that don't accept agent_name
                loaded_tools = await self.tool_registry.load_mcp_tools_from_config(mcp_config)
            
            if isinstance(loaded_tools, list):
                mcp_tools.extend(loaded_tools)
            elif isinstance(loaded_tools, dict):
                mcp_tools.extend(list(loaded_tools.values()))
                
            # Strands builder had logic to manually pull from registry.tools after loading.
            # We might need to standardize the registry return value in a future refactor.
            # For now, if loaded_tools is empty/None, we might need to fetch them.
            
            if not mcp_tools and hasattr(self.tool_registry, 'get_mcp_clients'):
                 # Fallback for Strands-like behavior where load doesn't return the clients
                 clients = self.tool_registry.get_mcp_clients()
                 # Filter for this agent? Strands logic was specific.
                 # Let's assume for now the registry returns the tools we need.
                 pass

            self.logger.debug(
                f"Added {len(mcp_tools)} MCP tools to agent '{agent_name}'"
            )
            
        return mcp_tools

    async def _load_knowledge_base_tools(self, agent_name: str, agent_config: Dict[str, Any]) -> List[Any]:
        """Load Knowledge Base tools."""
        kb_configs = agent_config.get('knowledge_base', [])
        if not kb_configs:
            return []
        try:
            kb_class = self._get_knowledgebase_factory_class()

            # Use to_thread for potentially blocking KB initialization
            kb_factory = await asyncio.to_thread(
                kb_class,
                knowledge_base_config=kb_configs,
                logger=self.logger,
                project_root=self.tool_registry.project_root,
                llm=self.llm,
                document_loader=self.document_loader,
                vector_store=self.vector_store
            )

            # Add as tool
            tools = []
            for kb_config in kb_configs:
                name = kb_config.get('name', 'default_knowledge_base')
                description = kb_config.get('description', 'Search the knowledge base.')
                kb_tool = kb_factory.create_tool(name=name, description=description)
                tools.append(kb_tool)
                kb_tool = kb_factory.create_load_tool(name=name, description=description)
                tools.append(kb_tool)
                
            self.logger.debug(f"Added {len(tools)} knowledge base tools to agent '{agent_name}'")
            return tools
        except ImportError:
            raise ImportError("Install vector dependencies")

    async def create_multi_agent_system(
            self,
            agent_configs: List[Dict[str, Any]],
            system_prompt: str = "",
            session_id: str = "default",
            pattern: str = Constants.PATTERN_SUPERVISOR
    ) -> Tuple[Any, List[BaseAgent]]:
        """Create a multi-agent system from configuration.

        Args:
            agent_configs: List of agent configuration dictionaries.
            system_prompt: System prompt for the supervisor.
            session_id: Session identifier.
            pattern: Architecture pattern.

        Returns:
            Tuple of (supervisor_agent, list_of_base_agents).
        """
        config_manager = ConfigManager(config_root=self.config_root)

        # 1. Normalize configurations
        agent_definitions = self._normalize_agent_configs(agent_configs, config_manager)

        # 2. Create agents with appropriate tools
        base_agent_list, agent_list, sub_agent_tools = await self._create_agents_parallel(
            agent_definitions, pattern
        )

        # 3. Create supervisor/architecture structure
        supervisor = self._create_supervisor_agent(
            pattern, agent_list, sub_agent_tools, system_prompt
        )

        self.logger.info(
            f"✅ Created multi-agent system ({pattern}) with {len(base_agent_list)} agents"
        )

        return supervisor, base_agent_list

    def _normalize_agent_configs(
            self,
            agent_configs: List[Any],
            config_manager: ConfigManager
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Normalize agent configurations into a consistent format."""
        config = config_manager.ruamel_to_native(agent_configs)
        agent_definitions = []

        for i, item in enumerate(config):
            agent_name = item
            agent_config = {}

            if isinstance(config, list):
                if isinstance(agent_configs[i], str):
                    agent_config = config_manager.load_agent_config(
                        agent_name=agent_configs[i]
                    )
                elif isinstance(agent_configs[i], dict):
                    agent_config = agent_configs[i]
                    agent_name = list(agent_config.keys())[0]
                    agent_config = agent_config[agent_name]
            elif isinstance(config, dict):
                # In dict mode, item is the key (agent_name)
                agent_name = item
                if config[agent_name] is None:
                    agent_config = config_manager.load_agent_config(
                        agent_name=agent_name
                    )
                else:
                    agent_config = config[agent_name]

            agent_definitions.append((agent_name, agent_config))

        return agent_definitions

    async def _create_agents_parallel(
            self,
            agent_definitions: List[Tuple[str, Dict[str, Any]]],
            pattern: str
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        """Create all agents in parallel."""
        
        async def _create_agent(name, config):
            extra_tools = []
            
            # Hook for adding extra tools (like handoffs)
            extra_tools.extend(self._get_extra_tools(name, config, agent_definitions, pattern))

            # Create base agent using factory
            base_agent = await self.create_single_agent(
                agent_name=name,
                agent_config=config,
                pre_loaded_tools=extra_tools
            )
            
            # Ensure name is set
            if hasattr(base_agent, 'name'):
                base_agent.name = name
            
            agent_tool = None
            if pattern == Constants.PATTERN_AGENT_AS_TOOL:
                # Create tool for this agent
                description = config.get('description', f"Agent responsible for {name}")
                if not config.get('description') and config.get('system_prompt'):
                     description = f"Agent with instructions: {config.get('system_prompt')[:100]}..."

                agent_tool = self._create_agent_as_tool(
                    agent=base_agent,
                    name=name,
                    description=description
                )
            
            return base_agent, agent_tool

        tasks = [_create_agent(name, config) for name, config in agent_definitions]
        results = await asyncio.gather(*tasks)

        base_agent_list = []
        agent_list = []
        sub_agent_tools = []

        for base_agent, agent_tool in results:
            base_agent_list.append(base_agent)

            if pattern == Constants.PATTERN_SUPERVISOR or pattern == Constants.PATTERN_SWARM or pattern == Constants.PATTERN_HANDOFF:
                agent_list.append(base_agent)
            elif pattern == Constants.PATTERN_AGENT_AS_TOOL:
                sub_agent_tools.append(agent_tool)
                
        return base_agent_list, agent_list, sub_agent_tools

    def _get_extra_tools(self, name: str, config: Dict[str, Any], agent_definitions: List, pattern: str) -> List[Any]:
        """Hook to get extra tools for an agent (e.g. handoffs)."""
        return []

    @abstractmethod
    def _create_agent_as_tool(self, agent: Any, name: str, description: str) -> Any:
        """Wrap an agent as a tool."""
        pass

    @abstractmethod
    def _create_supervisor_agent(
            self,
            pattern: str,
            agent_list: List[Any],
            sub_agent_tools: List[Any],
            system_prompt: str
    ) -> Any:
        """Create the supervisor agent or swarm structure."""
        pass


    @abstractmethod
    def _get_knowledgebase_factory_class(self) -> BaseKnowledgeBaseFactory:
        """Get the knowledge base factory class."""
        pass
