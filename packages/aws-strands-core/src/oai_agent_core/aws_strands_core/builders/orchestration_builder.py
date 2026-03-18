"""Builder for creating multi-agent orchestration systems (Graph, Swarm)."""

import logging
from typing import Dict, Any, List, Optional, Union

from oai_agent_core.components.output_parser.output_model_registry import OutputModelRegistry
from oai_agent_core.core.constants import Constants

try:
    from strands import Agent
    from strands.multiagent import GraphBuilder, Swarm
except ImportError:
    raise ImportError("Please install strands-agents: pip install strands-agents")


class OrchestrationBuilder:
    """Builder for creating multi-agent orchestration patterns.

    Supports:
    - Graph pattern (deterministic workflows)
    - Swarm pattern (autonomous coordination)
    - Single agent (no orchestration)
    - Agent as Tool (hierarchical tool usage)

    Attributes:
        logger: Logger instance
    """

    PATTERN_GRAPH = Constants.PATTERN_GRAPH
    PATTERN_SWARM = Constants.PATTERN_SWARM
    PATTERN_SEQUENTIAL = Constants.PATTERN_SEQUENTIAL
    PATTERN_HIERARCHICAL = Constants.PATTERN_HIERARCHICAL
    PATTERN_AGENT_AS_TOOL = Constants.PATTERN_AGENT_AS_TOOL

    def __init__(self, logger: Optional[logging.Logger] = None,
                 llm: Optional[Any] = None,
                 structured_output_model_registry: Optional[OutputModelRegistry] = None,):
        """Initialize the orchestration builder.

        Args:
            logger: Optional logger instance
            llm: Optional LLM instance (required for agent-as-tool pattern)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.llm = llm
        self.structured_output_model_registry = structured_output_model_registry


    def build_orchestration(
            self,
            agent_map: Dict[str, Agent],
            context_map: Dict[str, List[str]],
            crew_config: Dict[str, Any] = None,
            system_prompt: Optional[str] = None
    ) -> Union[Agent, Any]:
        """Build orchestration system based on pattern and agents.

        Args:
            agent_map: Dictionary of agent_key -> Agent instance
            context_map: Dictionary of agent_key -> context dependencies
            crew_config: Dictionary of crew config
            system_prompt: System prompt

        Returns:
            Orchestration system (Graph, Swarm, or single Agent)

        Raises:
            ValueError: If pattern is unknown
        """
        if not agent_map:
            raise ValueError("Cannot build orchestration with no agents")

        pattern = crew_config.get('pattern', crew_config.get('process', 'sequential'))
        entry_agent_key = crew_config.get('entry_agent')
        structured_output_model = crew_config.get('structured_output_model')
        pattern_lower = pattern.lower()

        # Agent as Tool pattern - construct a supervisor agent with other agents as tools
        if pattern_lower == self.PATTERN_AGENT_AS_TOOL:
            return self._build_agent_as_tool(agent_map,
                                             entry_agent_key,
                                             system_prompt,
                                             structured_output_model)

        # Single agent - no orchestration needed
        if len(agent_map) == 1:
            agent = list(agent_map.values())[0]
            self.logger.info("Single agent system - no orchestration needed")
            return agent

        # Multi-agent orchestration
        if pattern_lower == self.PATTERN_SWARM:
            return self._build_swarm(agent_map, entry_agent_key)
        elif pattern_lower in [self.PATTERN_GRAPH, 
                               self.PATTERN_SEQUENTIAL, 
                               self.PATTERN_HIERARCHICAL]:
            return self._build_graph(agent_map, context_map)
        else:
            raise ValueError(
                f"Unknown orchestration pattern: {pattern}. "
                f"Use 'graph', 'swarm', 'sequential', 'hierarchical' or 'agent-as-tool'"
            )

    def _build_agent_as_tool(
            self,
            agent_map: Dict[str, Any],
            entry_agent_key: Optional[str],
            system_prompt: Optional[str]=None,
            structured_output_model: Optional[str]=None
    ) -> Agent:
        """Build an Agent-as-Tool orchestration system.

        In this pattern, one agent (the supervisor) is equipped with other agents
        as tools. The supervisor can then invoke these sub-agents to perform tasks.

        Args:
            agent_map: Dictionary of agents (or tools wrapping agents)
            entry_agent_key: Optional key for the supervisor agent. If not provided,
                             a default supervisor is created.

        Returns:
            The supervisor Agent instance configured with sub-agent tools.
        """
        # Collect all tools
        sub_agent_tools = list(agent_map.values())
        
        # Create a supervisor agent
        # We need an LLM. If not provided, we can't create a functional agent.
        if not hasattr(self, 'llm') or self.llm is None:
             raise ValueError("OrchestrationBuilder needs an LLM to create a supervisor for 'agent-as-tool' pattern.")
             
        supervisor = Agent(
            name="Supervisor",
            model=self.llm,
            system_prompt=system_prompt or "You are a supervisor agent. Use the available tools to answer the user's request.",
            tools=sub_agent_tools,
            structured_output_model=self.structured_output_model_registry.get_model(structured_output_model)
        )
        
        self.logger.info(f"Created Supervisor agent with {len(sub_agent_tools)} agent-tools")
        return supervisor

    def _build_swarm(
            self,
            agent_map: Dict[str, Agent],
            entry_agent_key: Optional[str]
    ) -> Swarm:
        """Build a Swarm orchestration system.

        Args:
            agent_map: Dictionary of agents
            entry_agent_key: Optional entry point agent key

        Returns:
            Configured Swarm instance
        """
        # Get entry point agent
        entry_agent = None
        if entry_agent_key and entry_agent_key in agent_map:
            entry_agent = agent_map[entry_agent_key]
            self.logger.debug(f"Using entry agent: {entry_agent_key}")

        # Create swarm with all agents
        agent_list = list(agent_map.values())

        swarm = Swarm(
            nodes=agent_list,
            entry_point=entry_agent
        )

        self.logger.info(
            f"Created Swarm with {len(agent_list)} agents"
            f"{f' (entry: {entry_agent_key})' if entry_agent_key else ''}"
        )

        return swarm

    def _build_graph(
            self,
            agent_map: Dict[str, Agent],
            context_map: Dict[str, List[str]]
    ) -> Any:
        """Build a Graph orchestration system.

        Args:
            agent_map: Dictionary of agents
            context_map: Dictionary of context dependencies

        Returns:
            Configured Graph instance
        """
        builder = GraphBuilder()

        # Add all agents as nodes
        for agent_key, agent in agent_map.items():
            builder.add_node(agent, agent_key)
            self.logger.debug(f"Added graph node: {agent_key}")

        if len(context_map) == 0:
            raise ValueError("for graph agent, context map cannot be empty and required to create the graph")

        # Add edges based on context dependencies
        edge_count = 0
        for agent_key, context_keys in context_map.items():
            for context_key in context_keys:
                if context_key in agent_map:
                    builder.add_edge(context_key, agent_key)
                    self.logger.debug(f"Added edge: {context_key} -> {agent_key}")
                    edge_count += 1
                else:
                    self.logger.warning(
                        f"Context dependency '{context_key}' not found for agent '{agent_key}'"
                    )

        # Build the graph
        graph = builder.build()

        self.logger.info(
            f"Created Graph with {len(agent_map)} nodes and {edge_count} edges"
        )

        return graph

    def validate_graph_config(
            self,
            agent_map: Dict[str, Agent],
            context_map: Dict[str, List[str]]
    ) -> tuple[bool, List[str]]:
        """Validate graph configuration for cycles and missing dependencies.

        Args:
            agent_map: Dictionary of agents
            context_map: Dictionary of context dependencies

        Returns:
            Tuple of (is_valid, warning_messages)
        """
        warnings = []

        # Check for missing context dependencies
        for agent_key, context_keys in context_map.items():
            for context_key in context_keys:
                if context_key not in agent_map:
                    warnings.append(
                        f"Agent '{agent_key}' depends on unknown agent '{context_key}'"
                    )

        # Check for isolated nodes (no incoming or outgoing edges)
        all_referenced = set()
        for context_keys in context_map.values():
            all_referenced.update(context_keys)

        for agent_key in agent_map.keys():
            has_outgoing = len(context_map.get(agent_key, [])) > 0
            has_incoming = agent_key in all_referenced

            if not has_outgoing and not has_incoming:
                warnings.append(
                    f"Agent '{agent_key}' is isolated (no connections)"
                )

        is_valid = len(warnings) == 0
        return is_valid, warnings

    def get_orchestration_info(
            self,
            orchestration: Any,
            pattern: str,
            agent_count: int
    ) -> Dict[str, Any]:
        """Get information about the orchestration system.

        Args:
            orchestration: The orchestration system instance
            pattern: Pattern used
            agent_count: Number of agents

        Returns:
            Dictionary with orchestration information
        """
        info = {
            'pattern': pattern,
            'agent_count': agent_count,
            'type': type(orchestration).__name__,
            'is_single_agent': agent_count == 1
        }

        # Add pattern-specific info
        if hasattr(orchestration, 'nodes'):
            info['node_count'] = len(orchestration.nodes)

        if hasattr(orchestration, 'entry_point'):
            info['has_entry_point'] = orchestration.entry_point is not None

        return info

    @staticmethod
    def get_supported_patterns() -> List[str]:
        """Get list of supported orchestration patterns.

        Returns:
            List of pattern names
        """
        return [
            OrchestrationBuilder.PATTERN_GRAPH,
            OrchestrationBuilder.PATTERN_SWARM,
            OrchestrationBuilder.PATTERN_SEQUENTIAL,
            OrchestrationBuilder.PATTERN_HIERARCHICAL,
            OrchestrationBuilder.PATTERN_AGENT_AS_TOOL
        ]

    def __repr__(self) -> str:
        """String representation of the builder."""
        return "OrchestrationBuilder()"
