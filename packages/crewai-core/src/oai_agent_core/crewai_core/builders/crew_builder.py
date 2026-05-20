"""Agent Builder for CrewAI multi-agent workflows.

This module handles the construction of CrewAI agents, tasks, and crews
from YAML configuration.
"""

import asyncio
import re
from typing import Dict, List, Any, Tuple, Optional

from crewai import Agent, Crew, Task, Process

from oai_agent_core.components.output_parser.output_model_registry import OutputModelRegistry
from oai_agent_core.crewai_core.components.registry.tool_registry import CrewAIToolRegistry as ToolRegistry
from oai_agent_core.crewai_core.builders.task_builder import TaskBuilder


class CrewBuilder:
    """Builds CrewAI agents, tasks, and crews from configuration.

    This class is responsible for:
    - Creating Agent instances from configuration
    - Building complete Crew instances
    - Validating configuration integrity

    Attributes:
        config: Agent configuration dictionary
        tool_registry: Registry for managing tools
        logger: Logger instance
        llm: Language model instance
    """

    def __init__(self,
                 config: Dict[str, Any],
                 tool_registry: ToolRegistry,
                 llm: Any,
                 logger: Any,
                 document_loader: Optional[Any] = None,
                 vector_store: Optional[Any] = None,
                 structured_output_model_registry: Optional[OutputModelRegistry] = None
                 ):
        """Initialize the agent builder.

        Args:
            config: YAML configuration dictionary
            tool_registry: Initialized tool registry
            llm: Language model instance
            logger: Logger instance
            document_loader: Optional document loader instance
            vector_store: Optional vector store instance
            structured_output_model_registry: Optional registry for Pydantic output models.
        """
        self.config = config
        self.tool_registry = tool_registry
        self.llm = llm
        self.logger = logger
        self.document_loader = document_loader
        self.vector_store = vector_store
        self.structured_output_model_registry = structured_output_model_registry
        
        # Lock to prevent race conditions during concurrent agent creation
        self._tool_lock = asyncio.Lock()
        
        # Initialize TaskBuilder
        self.task_builder = TaskBuilder(config, logger, structured_output_model_registry)

    def build_crew(self, session_id: str, step_callback: Any = None) -> Crew:
        """Build a complete CrewAI crew from configuration.

        Args:
            session_id: Session identifier for tracking
            step_callback: Optional callback for step execution

        Returns:
            Configured Crew instance

        Raises:
            ValueError: If configuration is invalid
        """
        return self._build_crew_sync(session_id, step_callback)

    def _build_crew_sync(self, session_id: str, step_callback: Any = None) -> Crew:
        """Synchronous implementation of build_crew."""
        agent_map, agent_list = self._build_agents_sync()
        task_map, task_list, agent_level_tasks = self.task_builder.build_tasks(agent_map)

        if not task_list:
            raise ValueError(
                "No tasks defined. Please add tasks either at agent level or crew level."
            )

        crew = self._create_crew(agent_list, task_list, step_callback)
        return crew

    async def build_crew_async(self, session_id: str, step_callback: Any = None) -> Crew:
        """Asynchronously build a complete CrewAI crew from configuration.

        Args:
            session_id: Session identifier for tracking
            step_callback: Optional callback for step execution

        Returns:
            Configured Crew instance
        """
        # Phase 1: Build agents in parallel
        agent_map, agent_list = await self._build_agents_async()

        # Phase 2: Build tasks with dependencies (tasks depend on agents, so this is sequential after agents)
        task_map, task_list, agent_level_tasks = self.task_builder.build_tasks(agent_map)

        if not task_list:
            raise ValueError(
                "No tasks defined. Please add tasks either at agent level or crew level."
            )

        # Phase 3: Create the crew
        crew = self._create_crew(agent_list, task_list, step_callback)

        return crew

    def _prepare_tools(self, agent_key: str, agent_data: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
        """Prepare tools for an agent (Regular + MCP + KB).
        
        Returns:
            Tuple of (all_tools, mcp_clients)
        """
        # Prepare tools
        tool_list = agent_data.get('tools', [])
        agent_tools = self.tool_registry.get_tools_for_agent(tool_list)
        if agent_tools is None:
            agent_tools = []

        # Prepare MCP servers
        mcp_configs = agent_data['mcps'] if 'mcps' in agent_data else self.tool_registry.get_mcp_configs()
        # Note: load_mcp_tools_from_config is sync in CrewAIToolRegistry
        mcps = self.tool_registry.load_mcp_tools_from_config(mcp_configs)

        # Handle Knowledge Bases
        kb_configs = agent_data.get('knowledge_base', [])
        if kb_configs:
            try:
                from oai_agent_core.crewai_core.components.knowledge.knowledge_base_factory import KnowledgeBaseFactory
                kb_factory = KnowledgeBaseFactory(
                    knowledge_base_config=kb_configs,
                    logger=self.logger,
                    project_root=self.tool_registry.project_root,
                    llm=self.llm,
                    document_loader=self.document_loader,
                    vector_store=self.vector_store
                )
                kb_tool = kb_factory.create_tool()
                agent_tools.append(kb_tool)
                self.logger.debug(f"Added knowledge base tool to agent '{agent_key}'")
            except ImportError:
                self.logger.warning("Could not initialize knowledge base: missing dependencies")
            except Exception as e:
                self.logger.error(f"Failed to initialize knowledge base for agent '{agent_key}': {e}")
                
        return agent_tools, mcps

    async def _prepare_tools_async(self, agent_key: str, agent_data: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
        """Prepare tools for an agent asynchronously (Regular + MCP + KB).
        
        Returns:
            Tuple of (all_tools, mcp_clients)
        """
        # Prepare tools
        tool_list = agent_data.get('tools', [])
        agent_tools = self.tool_registry.get_tools_for_agent(tool_list)
        if agent_tools is None:
            agent_tools = []

        # Prepare MCP servers
        mcp_configs = agent_data['mcps'] if 'mcps' in agent_data else self.tool_registry.get_mcp_configs()
        # Wrap sync call in to_thread
        mcps = await asyncio.to_thread(self.tool_registry.load_mcp_tools_from_config, mcp_configs)

        # Handle Knowledge Bases
        kb_configs = agent_data.get('knowledge_base', [])
        if kb_configs:
            try:
                from oai_agent_core.crewai_core.components.knowledge.knowledge_base_factory import KnowledgeBaseFactory
                
                # Use to_thread for KB initialization
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
                for kb_config in kb_configs:
                    name = kb_config.get('name', 'default_knowledge_base')
                    description = kb_config.get('description', 'Search the knowledge base.')
                    kb_tool = kb_factory.create_tool(name=name, description=description)
                    agent_tools.append(kb_tool)
                    kb_tool = kb_factory.create_load_tool(name=name)
                    agent_tools.append(kb_tool)

                self.logger.debug(f"Added knowledge base tool to agent '{agent_key}'")
            except ImportError:
                self.logger.warning("Could not initialize knowledge base: missing dependencies")
            except Exception as e:
                self.logger.error(f"Failed to initialize knowledge base for agent '{agent_key}': {e}")
                
        return agent_tools, mcps

    def _build_agents_sync(self) -> Tuple[Dict[str, Agent], List[Agent]]:
        """Build all agents from configuration synchronously."""
        agent_map = {}
        agent_list = []

        for agent_config in self.config.get('agent_list', []):
            agent_key = list(agent_config.keys())[0]
            agent_data = agent_config[agent_key]
            
            # Use consolidated tool loading
            agent_tools, mcps = self._prepare_tools(agent_key, agent_data)
            
            agent = Agent(
                config=agent_data,
                verbose=agent_data.get('verbose', True),
                llm=self.llm,
                allow_delegation=agent_data.get('allow_delegation', False),
                tools=agent_tools if agent_tools else None,
                mcps=mcps if mcps else None
            )
            
            agent_map[agent_key] = agent
            agent_list.append(agent)

        return agent_map, agent_list

    async def _build_agents_async(self) -> Tuple[Dict[str, Agent], List[Agent]]:
        """Build all agents from configuration asynchronously in parallel."""
        agent_map = {}
        agent_list = []
        tasks = []
        keys = []
        configs = []

        for agent_config in self.config.get('agent_list', []):
            agent_key = list(agent_config.keys())[0]
            agent_data = agent_config[agent_key]
            keys.append(agent_key)
            configs.append(agent_data)
            tasks.append(self._prepare_tools_async(agent_key, agent_data))

        # Run tool preparation in parallel
        results = await asyncio.gather(*tasks)

        for i, (agent_tools, mcps) in enumerate(results):
            agent_key = keys[i]
            agent_data = configs[i]
            
            agent = Agent(
                config=agent_data,
                verbose=agent_data.get('verbose', True),
                llm=self.llm,
                allow_delegation=agent_data.get('allow_delegation', False),
                tools=agent_tools if agent_tools else None,
                mcps=mcps if mcps else None
            )
            
            agent_map[agent_key] = agent
            agent_list.append(agent)

        return agent_map, agent_list

    def _create_crew(self,
                     agent_list: List[Agent],
                     task_list: List[Task],
                     step_callback: Any = None) -> Crew:
        """Create a CrewAI Crew instance.

        Args:
            agent_list: List of Agent instances
            task_list: List of Task instances
            step_callback: Optional callback for step execution

        Returns:
            Configured Crew instance
        """
        crew_config = self.config.get('crew_config', {})

        # Determine process type
        process_type = crew_config.get('process', 'sequential')
        if isinstance(process_type, str):
            process_type = (
                Process.sequential
                if process_type.lower() == 'sequential'
                else Process.hierarchical
            )

        # Create crew
        crew = Crew(
            agents=agent_list,
            tasks=task_list,
            verbose=crew_config.get('verbose', True),
            process=process_type,
            memory=crew_config.get('memory', False),
            cache=crew_config.get('cache', True),
            max_rpm=crew_config.get('max_rpm', None),
            share_crew=crew_config.get('share_crew', False),
            step_callback=step_callback
        )

        return crew

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration and return diagnostics.

        Returns:
            Dictionary containing validation results and metadata
        """

        diagnostics = {
            'agent_count': 0,
            'task_count': 0,
            'agent_level_tasks': 0,
            'crew_level_tasks': 0,
            'tool_count': 0,
            'agents': [],
            'tasks': [],
            'tools': {},
            'missing_context': [],
            'missing_tools': [],
            'input_variables': set()
        }

        # Analyze tools
        tools_config = self.config.get('tools', {})
        diagnostics['tool_count'] = len(tools_config)

        for tool_name, tool_config in tools_config.items():
            if isinstance(tool_config, dict):
                diagnostics['tools'][tool_name] = {
                    'module': tool_config.get('module'),
                    'class': tool_config.get('class')
                }

        # Analyze agents
        for agent_config in self.config.get('agent_list', []):
            agent_key = list(agent_config.keys())[0]
            agent_data = agent_config[agent_key]
            diagnostics['agent_count'] += 1

            agent_tools = agent_data.get('tools', [])
            agent_info = {'name': agent_key, 'tools': agent_tools}

            # Validate tool references
            for tool_name in agent_tools:
                if tool_name not in tools_config:
                    diagnostics['missing_tools'].append(
                        f"Agent '{agent_key}' references unknown tool '{tool_name}'"
                    )

            diagnostics['agents'].append(agent_info)

            # Analyze agent-level tasks
            if 'tasks' in agent_data:
                for task_config in agent_data['tasks']:
                    task_key = list(task_config.keys())[0]
                    task_data = task_config[task_key]
                    diagnostics['agent_level_tasks'] += 1
                    diagnostics['tasks'].append(f"{task_key} (agent: {agent_key})")
                    
                    # Extract input variables
                    self._extract_input_variables(
                        task_data,
                        diagnostics['input_variables']
                    )

        # Analyze crew-level tasks
        for task_config in self.config.get('task_list', []):
            task_key = list(task_config.keys())[0]
            task_data = task_config[task_key]
            diagnostics['crew_level_tasks'] += 1

            agent_key = task_data.get('agent', 'unknown')
            diagnostics['tasks'].append(f"{task_key} (agent: {agent_key})")

            # Extract input variables
            self._extract_input_variables(task_data, diagnostics['input_variables'])

        diagnostics['task_count'] = (
                diagnostics['agent_level_tasks'] + diagnostics['crew_level_tasks']
        )
        diagnostics['input_variables'] = sorted(list(diagnostics['input_variables']))

        return diagnostics

    def _extract_input_variables(self,
                                 task_data: Dict[str, Any],
                                 variable_set: set):
        """Extract input variables from task descriptions.

        Args:
            task_data: Task configuration dictionary
            variable_set: Set to add variables to
        """
        desc = task_data.get('description', '')
        output = task_data.get('expected_output', '')

        variable_set.update(re.findall(r'\{(\w+)\}', desc))
        variable_set.update(re.findall(r'\{(\w+)\}', output))
