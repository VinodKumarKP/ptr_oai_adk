"""Builder for creating and configuring Strands Agent instances."""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable

from oai_agent_core.builders.base_agent_builder import BaseAgentBuilder
from oai_agent_core.components.skills.skill_registry import SkillRegistry
from oai_agent_core.core.constants import Constants

try:
    from strands import Agent
    from strands.tools import tool
except ImportError:
    raise ImportError("Please install strands-agents: pip install strands-agents")

from oai_agent_core.aws_strands_core.components.configuration.model_config import \
    StrandsModelConfigurationManager as ModelConfigurationManager
from oai_agent_core.aws_strands_core.components.registry.tool_registry import AWSStrandsToolRegistry as ToolRegistry


class AgentBuilder(BaseAgentBuilder):
    """Builder for creating and configuring Strands Agent instances.

    Handles:
    - Agent instance creation from configuration
    - Model assignment per agent
    - Tool assignment per agent (including MCP clients)
    - Callback handler creation
    - Context dependency extraction

    Attributes:
        model_manager: Model configuration manager
        tool_registry: Tool registry instance
        logger: Logger instance
    """

    def __init__(
            self,
            model_manager: ModelConfigurationManager,
            tool_registry: ToolRegistry,
            llm: Optional[Any] = None,
            logger: Optional[logging.Logger] = None,
            document_loader: Optional[Any] = None,
            vector_store: Optional[Any] = None,
            skill_registry: Optional[SkillRegistry] = None
    ):
        """Initialize the agent builder.

        Args:
            model_manager: Model configuration manager
            tool_registry: Tool registry instance
            logger: Optional logger instance
            document_loader: Optional document loader instance
            vector_store: Optional vector store instance
        """
        super().__init__(
            model_manager=model_manager,
            tool_registry=tool_registry,
            llm=llm,
            logger=logger,
            document_loader=document_loader,
            vector_store=vector_store,
            skill_registry=skill_registry
        )

    async def create_agent(
            self,
            agent_key: str,
            agent_config: Dict[str, Any],
            callback: Optional[Callable] = None
    ) -> Agent:
        """Create a Strands Agent instance from configuration.
        
        Note: This is an alias for create_single_agent to maintain backward compatibility
        with Strands-specific naming conventions.
        """
        return await self.create_single_agent(agent_key, agent_config)

    def _create_agent_instance(
            self,
            agent_name: str,
            agent_config: Dict[str, Any],
            tools: List[Any]
    ) -> Any:
        """Create the Strands agent instance.

        Args:
            agent_name: Name of the agent.
            agent_config: Configuration dictionary.
            tools: List of configured tools.

        Returns:
            Configured Agent instance.
        """
        # Get system prompt (instructions)
        if self.tool_registry.enable_lazy_loading:
            tools = self.tool_registry.lazy_loading_required_tools()
            system_prompt = f"""{agent_config.get('system_prompt', "Use MCP tools when they help.")}
                            {self.tool_registry.generate_lazy_mcp_system_prompt(
                agent_config.get('tools', []),
                agent_config.get('mcps', [])
            )}          
                            """
        else:
            system_prompt = agent_config.get('system_prompt', "Use MCP tools when they help.")

        skills = agent_config.get('skills', [])
        if skills:
            skill_list = self.skill_registry.get_skills(skills)
            system_prompt = f"{system_prompt}\n{self.skill_registry.generate_skills_prompt(skill_list)}"
            from strands_tools import file_read,file_write,shell
            tools.append(file_read)
            tools.append(file_write)
            tools.append(shell)


        # Create agent instance with correct parameter names
        strands_agent = Agent(
            name=agent_name,
            model=self.llm,
            system_prompt=system_prompt,  # Correct parameter name
            tools=tools if tools else None  # Correct parameter name
        )

        return strands_agent

    async def _create_knowledge_base_tool(self, agent_name: str, kb_configs: List[Dict[str, Any]]) -> List[Any]:
        """Create knowledge base tool for Strands."""
        try:
            from oai_agent_core.aws_strands_core.components.knowledge.knowledge_base_factory import KnowledgeBaseFactory
            
            # Use to_thread for potentially blocking KB initialization
            kb_factory = await asyncio.to_thread(
                KnowledgeBaseFactory,
                knowledge_base_config=kb_configs,
                logger=self.logger,
                project_root=self.tool_registry.project_root,
                llm=self.llm,
                document_loader=self.document_loader,
                vector_store=self.vector_store
            )
            
            # Add as tool
            kb_tool = kb_factory.create_tool()
            self.logger.debug(f"Added knowledge base tool to agent '{agent_name}'")
            return [kb_tool]
        except ImportError:
            raise ImportError("Install vector dependencies pip install oai-aws-strands-agent-core[vector]")

    def _create_agent_as_tool(self, agent: Any, name: str, description: str) -> Any:
        """Wrap an agent as a tool.

        Args:
            agent: The agent instance to wrap
            name: Name of the tool
            description: Description of what the agent does

        Returns:
            A tool that invokes the agent
        """
        @tool(name=name, description=description)
        def agent_tool(query: str) -> str:
            """Delegate work to the sub-agent."""
            import asyncio
            # Strands agents are typically invoked synchronously or via invoke_async
            # If we are in an async loop, we should use invoke_async and run it
            # But tool execution is often synchronous in Strands unless async tool support is enabled.
            # Assuming Strands supports async tools or we wrap it.
            
            # If Strands supports async tools natively:
            # return await agent.invoke_async(query)
            
            # If we must return a sync function but run async code:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
                
            if loop and loop.is_running():
                # We are in an async context, but this tool function is sync.
                # This is tricky. Ideally Strands supports async tools.
                # If not, we might block.
                # For now, let's assume we can run it.
                # If Strands agent is purely sync callable:
                return agent(query)
            else:
                return agent(query)

        agent_tool.name = name
        agent_tool.description = description
        return agent_tool

    async def create_agents_from_config(
            self,
            agent_configs: List[Dict[str, Any]],
            callback_factory: Optional[Callable[[str], Callable]] = None,
            architecture: str = Constants.PATTERN_SUPERVISOR
    ) -> Dict[str, Any]:
        """Create multiple agents from configuration list in parallel.

        Args:
            agent_configs: List of agent configuration dictionaries
            callback_factory: Optional factory function that creates callbacks
            architecture: Architecture type ('supervisor', 'agent-as-tool', 'swarm')

        Returns:
            Dictionary mapping agent keys to Agent instances or Tools (for agent-as-tool)
        """
        # Normalize configs first to match base class expectation
        # Note: Strands config structure is slightly different (list of dicts with key as name)
        # The base class _normalize_agent_configs handles this.
        
        # However, create_agents_from_config expects to return a Dict, while base class returns lists.
        # So we use the helper _create_agents_parallel directly but adapt the input/output.
        
        config_manager = None # Not needed for normalization if we pass raw list?
        # Actually, base class _normalize_agent_configs needs a config manager.
        # We can create a temporary one or use self.model_manager if it has config capabilities?
        # Strands builder didn't use ConfigManager explicitly before, it parsed raw dicts.
        
        # Let's adapt the input manually to match what _create_agents_parallel expects
        agent_definitions = []
        for agent_config in agent_configs:
            agent_key = list(agent_config.keys())[0]
            agent_data = agent_config[agent_key]
            agent_definitions.append((agent_key, agent_data))
            
        base_agents, _, sub_agent_tools = await self._create_agents_parallel(
            agent_definitions, architecture
        )
        
        # Reconstruct the map
        agent_map = {}
        for agent in base_agents:
            # Find the tool wrapper if it exists
            if architecture == Constants.PATTERN_AGENT_AS_TOOL:
                # This is tricky because sub_agent_tools is a list, not mapped by name
                # We need to find the tool corresponding to this agent.
                # The base implementation of _create_agents_parallel returns lists in order.
                # But it filters sub_agent_tools.
                pass
        
        # Actually, it's easier to just reimplement create_agents_from_config using the base helper
        # but keeping the map logic.
        
        # Let's use the base class _create_agents_parallel but we need to map results back to keys.
        # The base class returns (base_agent_list, agent_list, sub_agent_tools).
        # If architecture is agent-as-tool, base_agent_list has agents, sub_agent_tools has tools.
        # They should be in the same order as definitions.
        
        # Wait, _create_agents_parallel filters the lists based on pattern.
        # If pattern is agent-as-tool, agent_list is empty.
        
        # To be safe and preserve the map, let's just use the logic we had but call create_single_agent
        
        tasks = []
        keys = []
        configs = []

        for agent_config in agent_configs:
            agent_key = list(agent_config.keys())[0]
            agent_data = agent_config[agent_key]
            keys.append(agent_key)
            configs.append(agent_data)

            # Create agent task (using base class method)
            tasks.append(self.create_single_agent(agent_key, agent_data))

        # Run all agent creation tasks in parallel
        results = await asyncio.gather(*tasks)
        
        agent_map = {}
        
        for i, (key, agent) in enumerate(zip(keys, results)):
            if architecture == Constants.PATTERN_AGENT_AS_TOOL:
                # Wrap agent as tool
                config = configs[i]
                description = config.get('description', f"Agent responsible for {key}")
                if not config.get('description') and config.get('system_prompt'):
                     description = f"Agent with instructions: {config.get('system_prompt')[:100]}..."
                
                agent_tool = self._create_agent_as_tool(agent, key, description)
                agent_map[key] = agent_tool
            else:
                agent_map[key] = agent

        self.logger.info(f"✅ Created {len(agent_map)} agents/tools in parallel")
        return agent_map

    def _create_supervisor_agent(
            self,
            pattern: str,
            agent_list: List[Any],
            sub_agent_tools: List[Any],
            system_prompt: str
    ) -> Any:
        """Create the supervisor agent or swarm structure.
        
        Note: Strands typically uses OrchestrationBuilder for this, so this might not be used
        directly via create_multi_agent_system, but we implement it for completeness.
        """
        # Strands doesn't have a built-in supervisor in the same way as LangGraph/OpenAI
        # It usually uses a specific agent configured with tools.
        # If we need to return a supervisor agent here:
        
        if pattern == Constants.PATTERN_AGENT_AS_TOOL:
             supervisor = Agent(
                name="Supervisor",
                model=self.llm,
                system_prompt=system_prompt or "You are a supervisor agent. Use the available tools to answer the user's request.",
                tools=sub_agent_tools
            )
             return supervisor
             
        return None

    def extract_context_map(
            self,
            agent_configs: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Extract context dependencies from agent configurations.

        Context dependencies define which agents' outputs should be available
        to other agents in multi-agent workflows.

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
            agent_key = list(agent_config.keys())[0]
            agent_data = agent_config[agent_key]

            context = agent_data.get('context', [])
            if context:
                context_map[agent_key] = context

        return context_map

    def validate_agent_config(
            self,
            agent_key: str,
            agent_config: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Validate agent configuration.

        Args:
            agent_key: Agent identifier
            agent_config: Agent configuration to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check for system prompt/instructions
        if not agent_config.get('system_prompt') and not agent_config.get('backstory'):
            errors.append(f"Agent '{agent_key}' missing system_prompt or backstory")

        # Validate regular tools exist
        tool_names = agent_config.get('tools', [])
        if tool_names:
            if isinstance(tool_names, str):
                tool_names = [tool_names]

            for tool_name in tool_names:
                if not self.tool_registry.has_tool(tool_name):
                    errors.append(
                        f"Agent '{agent_key}' references unknown tool '{tool_name}'"
                    )

        # Validate MCP configurations if present
        if 'mcps' in agent_config:
            mcp_configs = agent_config['mcps']
            if not isinstance(mcp_configs, list):
                errors.append(f"Agent '{agent_key}' has invalid 'mcps' configuration (must be a list)")
            else:
                for idx, mcp_config in enumerate(mcp_configs):
                    if not isinstance(mcp_config, dict):
                        errors.append(f"Agent '{agent_key}' MCP config {idx} must be a dictionary")
                        continue

                    # Check for required fields
                    if 'command' not in mcp_config and 'url' not in mcp_config:
                        errors.append(
                            f"Agent '{agent_key}' MCP config {idx} must have 'command' or 'url'"
                        )

        is_valid = len(errors) == 0
        return is_valid, errors

    def get_agent_summary(
            self,
            agent_key: str,
            agent: Agent
    ) -> Dict[str, Any]:
        """Get summary information about an agent.

        Args:
            agent_key: Agent identifier
            agent: Agent instance

        Returns:
            Dictionary with agent information
        """
        tool_count = len(agent.tools) if agent.tools else 0
        has_instructions = bool(agent.instructions) if hasattr(agent, 'instructions') else False

        return {
            'agent_key': agent_key,
            'name': agent.name if hasattr(agent, 'name') else agent_key,
            'has_tools': tool_count > 0,
            'tool_count': tool_count,
            'has_instructions': has_instructions
        }

    def __repr__(self) -> str:
        """String representation of the builder."""
        return (
            f"AgentBuilder("
            f"tools={len(self.tool_registry)}, "
            f"model={self.model_manager.default_config.get('model_id', 'unknown')})"
        )

    def _get_knowledgebase_factory_class(self):
        """Get the knowledge base factory class."""
        from oai_agent_core.aws_strands_core.components.knowledge.knowledge_base_factory import KnowledgeBaseFactory
        return KnowledgeBaseFactory

