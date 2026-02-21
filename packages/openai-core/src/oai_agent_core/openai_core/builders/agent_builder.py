"""Builder for creating OpenAI Agents."""
import logging
from typing import Dict, Any, Optional, List

from agents import Agent, ModelSettings, function_tool
from oai_agent_core.builders.base_agent_builder import BaseAgentBuilder
from oai_agent_core.core.base_knowledge_base_factory import BaseKnowledgeBaseFactory
from oai_agent_core.core.constants import Constants

from oai_agent_core.openai_core.components.configuration.model_config import \
    OpenAIModelConfigurationManager as ModelConfigurationManager
from oai_agent_core.openai_core.components.knowledge.knowledge_base_factory import KnowledgeBaseFactory
from oai_agent_core.openai_core.components.tools.tool_registry import OpenAIToolRegistry


class AgentBuilder(BaseAgentBuilder):
    """Factory class to build OpenAI agents from configuration.

    Handles the creation of single agents and multi-agent systems, including
    tool configuration, model initialization, and MCP server integration.
    """

    def __init__(
            self,
            model_manager: ModelConfigurationManager,
            tool_registry: OpenAIToolRegistry,
            llm: Optional[Any] = None,
            config_root: Optional[str] = None,
            logger: Optional[logging.Logger] = None,
            document_loader: Optional[Any] = None,
            vector_store: Optional[Any] = None,
    ):
        """Initialize the agent builder.

        Args:
            model_manager: Manager for model configuration.
            tool_registry: Registry for managing tools.
            llm: Optional pre-configured language model instance.
            config_root: Root directory for configuration files.
            logger: Optional logger instance.
            document_loader: Optional document loader instance.
            vector_store: Optional vector store instance.
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
        """Create the OpenAI agent instance.

        Args:
            agent_name: Name of the agent.
            agent_config: Configuration dictionary.
            tools: List of configured tools.

        Returns:
            Configured Agent instance.
        """
        # Store agent-specific MCP config for later activation
        # Note: This is a bit of a hack to attach it to the agent instance
        # Ideally, the Agent class would support this natively or we wrap it.
        # But for now, we follow the existing pattern.
        
        oai_agent = Agent(
            model=self.llm,
            name=agent_name,
            instructions=agent_config.get('system_prompt', "Use MCP tools when they help."),
            mcp_servers=[], # MCPs are already in 'tools' list as clients
            tools=tools,
            model_settings=ModelSettings(include_usage=True),
        )

        # Store agent-specific MCP config for later activation
        oai_agent.mcp_config = agent_config.get('mcps', agent_config.get('servers', {}))
        
        return oai_agent

    def _create_agent_as_tool(self, agent: Any, name: str, description: str) -> Any:
        """Wrap an agent as a tool.

        Args:
            agent: The agent instance to wrap
            name: Name of the tool
            description: Description of what the agent does

        Returns:
            A tool that invokes the agent
        """
        from agents import Runner

        @function_tool
        async def agent_tool(query: str) -> str:
            """Delegate work to the sub-agent.

            Args:
                query: The query or task to delegate to the sub-agent.
            """
            # Run the agent with the query
            result = await Runner.run(agent, query)
            return str(result.final_output)

        # Set the tool name and description
        if hasattr(agent_tool, 'name'):
            agent_tool.name = name
        if hasattr(agent_tool, 'description'):
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
        supervisor = None

        if pattern in [Constants.PATTERN_SUPERVISOR, Constants.PATTERN_HANDOFF]:
            # Default instruction for supervisor to ensure it knows how to route
            default_instruction = (
                "You are a supervisor. Transfer to appropriate agents when required. "
                "Use MCP tools when they help."
            )
            
            instructions = system_prompt or default_instruction
            
            # Ensure the transfer instruction is present if architecture is supervisor
            if pattern == Constants.PATTERN_SUPERVISOR and "Transfer to appropriate agents when required" not in instructions:
                 instructions += " Transfer to appropriate agents when required."

            supervisor = Agent(
                model=self.llm,
                name="Supervisor",
                handoffs=agent_list,
                instructions=instructions
            )

            # For handoff architecture, allow agents to hand off to each other
            if pattern == Constants.PATTERN_HANDOFF:
                for agent in agent_list:
                    agent.handoffs = [a for a in agent_list if a != agent]

            # For supervisor architecture, allow agents to hand off back to supervisor
            elif pattern == Constants.PATTERN_SUPERVISOR:
                for agent in agent_list:
                    agent.handoffs = [supervisor]

        elif pattern == Constants.PATTERN_AGENT_AS_TOOL:
            supervisor = Agent(
                model=self.llm,
                name="Supervisor",
                tools=sub_agent_tools,
                instructions=system_prompt or "Use the available agent tools to answer the user's request."
            )

        return supervisor

    def _get_knowledgebase_factory_class(self) -> type[KnowledgeBaseFactory]:
        """Get the knowledge base factory class."""
        from oai_agent_core.openai_core.components.knowledge.knowledge_base_factory import KnowledgeBaseFactory
        return KnowledgeBaseFactory
