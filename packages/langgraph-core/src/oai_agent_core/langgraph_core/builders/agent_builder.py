"""Builder for creating and configuring LangChain Agent instances."""

import logging
from typing import Dict, Any, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph_supervisor import create_supervisor
from oai_agent_core.builders.base_agent_builder import BaseAgentBuilder
from oai_agent_core.core.constants import Constants

from oai_agent_core.langgraph_core.components.configuration.model_config import \
    LangChainModelConfigurationManager as ModelConfigurationManager
from oai_agent_core.langgraph_core.components.registry.tool_registry import LangChainToolRegistry


class AgentBuilder(BaseAgentBuilder):
    """Builder for creating and configuring LangChain Agent instances.

    Handles:
    - Single agent creation with tools
    - Multi-agent system creation with supervisor
    - Tool loading and assignment from registry
    - Agent wrapper creation for different agent types
    - Context dependency extraction

    Attributes:
        llm: Language model instance
        tool_registry: Tool registry instance
        config_root: Root directory for configurations
        logger: Logger instance
    """

    def __init__(
            self,
            model_manager: ModelConfigurationManager,
            tool_registry: LangChainToolRegistry,
            llm: Optional[Any] = None,
            config_root: Optional[str] = None,
            logger: Optional[logging.Logger] = None,
            document_loader: Optional[Any] = None,
            vector_store: Optional[Any] = None,
    ):
        """Initialize the agent builder.

        Args:
            llm: Language model instance
            tool_registry: Tool registry instance
            config_root: Root directory for configurations
            logger: Optional logger instance
            document_loader: Optional document loader instance
            vector_store: Optional vector store instance
        """
        super().__init__(
            model_manager=model_manager,
            tool_registry=tool_registry,
            llm=llm,
            config_root=config_root,
            logger=logger,
            document_loader=document_loader,
            vector_store=vector_store
        )

    def _create_agent_instance(
            self,
            agent_name: str,
            agent_config: Dict[str, Any],
            tools: List[Any]
    ) -> Any:
        """Create the LangChain agent instance.

        Args:
            agent_name: Name of the agent.
            agent_config: Configuration dictionary.
            tools: List of configured tools.

        Returns:
            Configured Agent instance.
        """
        # Get system prompt
        system_prompt = agent_config.get('system_prompt', '')

        # Create the agent
        agent = create_agent(
            model=self.llm,
            tools=tools if tools else [],
            system_prompt=system_prompt,
            name=agent_name
        )

        return agent

    def _create_agent_as_tool(self, agent: Any, name: str, description: str) -> Any:
        """Wrap an agent as a tool.

        Args:
            agent: The agent instance to wrap
            name: Name of the tool
            description: Description of what the agent does

        Returns:
            A tool that invokes the agent
        """

        @tool
        def agent_tool(query: str) -> str:
            """Delegate work to the sub-agent."""
            import asyncio
            response = asyncio.run(agent.ainvoke({"messages": [HumanMessage(content=query)]}))
            return response["messages"][-1].content

        agent_tool.name = name
        agent_tool.description = description
        return agent_tool

    def _create_supervisor_agent(
            self,
            pattern: str,
            agent_list: List[Any],
            sub_agent_tools: List[Any],
            system_prompt: str
    ) -> Any:
        """Create the supervisor agent or swarm structure."""
        memory = MemorySaver()
        supervisor = None

        if pattern == Constants.PATTERN_SUPERVISOR:
            supervisor = create_supervisor(
                model=self.llm,
                agents=agent_list,
                prompt=system_prompt,
                add_handoff_back_messages=True,
                output_mode="full_history",
                parallel_tool_calls=True
            ).compile(checkpointer=memory)

        elif pattern == Constants.PATTERN_AGENT_AS_TOOL:
            supervisor = create_agent(
                model=self.llm,
                tools=sub_agent_tools,
                system_prompt=system_prompt,
                name="supervisor"
            )
            if hasattr(supervisor, 'compile'):
                supervisor = supervisor.compile(checkpointer=memory)

        elif pattern == Constants.PATTERN_SWARM:
            try:
                from langgraph_swarm import create_swarm
                supervisor = create_swarm(
                    agents=agent_list,
                    default_active_agent=agent_list[0].name
                ).compile(checkpointer=memory)
            except ImportError:
                self.logger.error("langgraph_swarm not found. Cannot create swarm.")
                raise

        else:
            raise ValueError(f"Unknown architecture: {pattern}")

        return supervisor

    def validate_agent_config(
            self,
            agent_name: str,
            agent_config: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Validate agent configuration.

        Args:
            agent_name: Agent identifier
            agent_config: Agent configuration to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check for system prompt
        if not agent_config.get('system_prompt'):
            errors.append(
                f"Agent '{agent_name}' missing system_prompt"
            )

        # Validate tools exist in registry
        tool_names = agent_config.get('tools', [])
        if tool_names:
            # Handle both list and dict formats
            if isinstance(tool_names, list):
                tool_list = tool_names
            elif isinstance(tool_names, dict):
                tool_list = list(tool_names.keys())
            else:
                errors.append(
                    f"Agent '{agent_name}' has invalid 'tools' configuration "
                    "(must be a list or dictionary)"
                )
                tool_list = []

            # Check if tools exist in registry
            for tool_name in tool_list:
                if not self.tool_registry.has_tool(tool_name):
                    errors.append(
                        f"Agent '{agent_name}' references unknown tool '{tool_name}'. "
                        f"Tool must be defined in the global 'tools' section first."
                    )

        # Validate MCP configuration
        mcps = agent_config.get('mcps', agent_config.get('servers', {}))
        if mcps and not isinstance(mcps, dict):
            errors.append(
                f"Agent '{agent_name}' has invalid 'mcps' configuration "
                "(must be a dictionary)"
            )

        is_valid = len(errors) == 0
        return is_valid, errors

    def get_agent_summary(
            self,
            agent_name: str,
            agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get summary information about an agent configuration.

        Args:
            agent_name: Agent identifier
            agent_config: Agent configuration

        Returns:
            Dictionary with agent information
        """
        tools = agent_config.get('tools', [])
        if isinstance(tools, list):
            tool_count = len(tools)
        elif isinstance(tools, dict):
            tool_count = len(tools)
        else:
            tool_count = 0

        mcp_count = len(agent_config.get('mcps', {}))

        return {
            'agent_name': agent_name,
            'has_system_prompt': bool(agent_config.get('system_prompt')),
            'tool_count': tool_count,
            'mcp_count': mcp_count,
            'total_tools': tool_count + mcp_count,
            'agent_type': agent_config.get('type', 'langchain')
        }

    def extract_context_map(
            self,
            agent_configs: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Extract context dependencies from agent configurations.

        Args:
            agent_configs: List of agent configuration dictionaries

        Returns:
            Dictionary mapping agent keys to their context dependencies

        Example:
            >>> # Agent config: {'analyst': {'context': ['researcher']}}
            >>> context_map = builder.extract_context_map(configs)
            >>> # Returns: {'analyst': ['researcher']}
        """
        context_map = {}

        for agent_config in agent_configs:
            if isinstance(agent_config, dict):
                agent_key = list(agent_config.keys())[0]
                agent_data = agent_config[agent_key]

                if isinstance(agent_data, dict):
                    context = agent_data.get('context', [])
                    if context:
                        context_map[agent_key] = context

        return context_map

    def __repr__(self) -> str:
        """String representation of the builder."""
        tool_count = len(self.tool_registry.tools) if hasattr(self.tool_registry, 'tools') else 0
        return (
            f"LangChainAgentBuilder("
            f"llm={type(self.llm).__name__}, "
            f"tools={tool_count}, "
            f"config_root={self.config_root})"
        )

    def _get_knowledgebase_factory_class(self):
        """Get the knowledge base factory class."""
        from oai_agent_core.langgraph_core.components.knowledge.knowledge_base_factory import KnowledgeBaseFactory
        return KnowledgeBaseFactory
