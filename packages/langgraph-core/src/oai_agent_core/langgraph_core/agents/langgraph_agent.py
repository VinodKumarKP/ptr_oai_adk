"""Unified LangChain agent implementation for single and multi-agent workflows."""

import asyncio
from typing import List, AsyncGenerator, Optional, Dict, Any

from langchain_core.runnables import RunnableConfig
from oai_agent_core.core.base_agent import BaseAgent
from oai_agent_core.core.constants import Constants
from oai_agent_core.processing.message_formatter import MessageFormatter

from oai_agent_core.langgraph_core.builders.agent_builder import AgentBuilder
from oai_agent_core.langgraph_core.components import LangChainToolRegistry
from oai_agent_core.langgraph_core.components.configuration.model_config import \
    LangChainModelConfigurationManager as ModelConfigurationManager
from oai_agent_core.langgraph_core.processing import ResultExtractor

class LangGraphAgent(BaseAgent):
    """Unified LangChain agent that handles both single and multi-agent workflows.

    This class automatically detects the configuration and switches between:
    - Single agent mode: Uses LangChain's create_agent
    - Multi-agent mode: Uses LangGraph's create_supervisor

    Configuration structure:
        tools: Global tool configurations (loaded first)
        mcps: Global MCP configurations (loaded first)
        agent_list: List of agent configurations (single item = single agent mode)
        system_prompt: System prompt for the agent(s)

    Attributes:
        tool_registry: Tool registry for managing tools
        agent_builder: LangChain agent builder instance
        agent: The created agent (single or multi-agent system)
        base_agent_list: List of base agents (for multi-agent mode)
        is_multi_agent: Flag indicating multi-agent mode
    """

    def __init__(
            self,
            agent_name: str,
            agent_config: Optional[Dict[str, Any]] = None,
            llm: Any = None,
            tools: List[Any] = None,
            session_id: str = "default",
            user_id: str = "default",
            config_root: Optional[str] = None,
            document_loader: Optional[Any] = None,
            vector_store: Optional[Any] = None,
            **kwargs
    ):
        """Initialize the unified LangChain agent.

        Args:
            agent_name: Unique identifier for the agent
            agent_config: Configuration dictionary
            llm: Language model instance
            tools: Pre-configured tools list (optional, for backward compatibility)
            session_id: Session identifier
            user_id: User identifier
            config_root: Root directory for configurations
            document_loader: Optional document loader instance
            vector_store: Optional vector store instance
            **kwargs: Additional arguments
        """
        self.crew_config = None
        self.agent_type = Constants.LANGGRAPH
        # Initialize utility managers
        self.model_manager = ModelConfigurationManager()
        super().__init__(
            llm=llm,
            model_manager=self.model_manager,
            agent_name=agent_name,
            agent_config=agent_config,
            session_id=session_id,
            config_root=config_root,
            agent_type=self.agent_type,
            user_id=user_id,
            document_loader=document_loader,
            vector_store=vector_store,
            **kwargs
        )

        self.pre_loaded_tools = tools or []  # Store for backward compatibility
        self.user_id = user_id
        self.config = {
            "configurable": {
                "thread_id": session_id,
                "user_id": user_id
            }
        }

        # Multi-agent specific attributes
        self.base_agent_list = []
        self.is_multi_agent = False

        # Initialize tool registry
        self.tool_registry = LangChainToolRegistry(
            project_root=config_root,
            logger=self.logger,
            enable_lazy_loading=self.agent_config.get('crew_config', {}).get('enable_lazy_loading', False)
        )

        # Initialize agent builder (will be created after tools are loaded)
        self.agent_builder = None
        self.result_extractor = ResultExtractor(logger=self.logger)
        self.message_formatter = MessageFormatter(logger=self.logger)

        self._set_langfuse_config()

    def _set_langfuse_config(self) -> None:
        """Configure Langfuse observability if enabled."""
        self.langfuse_manager.initialize_callback_handler()
        if self.langfuse_manager.callback_handler is not None:
            updated_config = {
                "callbacks": [self.langfuse_manager.callback_handler],
                "metadata": {
                    "langfuse_session_id": self.session_id,
                    "name": self.agent_name,
                    "type": self.agent_type
                },
                "run_name": f"{self.agent_name}_{self.session_id}"
            }
            self.config.update(updated_config)

    async def initialize(self) -> None:
        """Initialize the agent system based on configuration.

        Process:
        1. Load global tools from 'tools' section into registry
        2. Load global MCP tools from 'mcps' or 'servers' section into registry
        3. Determine single vs multi-agent mode based on agent_list
        4. Create agent(s) using builder (which pulls from registry)
        """
        from oai_agent_core.langgraph_core.components import KnowledgeBaseFactory
        
        # Step 1 & 2: Load global tools, MCP config, and KB concurrently
        await self._load_tools_and_kb_and_memory(kb_factory_class=KnowledgeBaseFactory)

        # Step 3: Create agent builder with populated registry
        self.agent_builder = AgentBuilder(
            llm=self.llm,
            model_manager=self.model_manager,
            tool_registry=self.tool_registry,
            config_root=self.config_root,
            logger=self.logger,
            document_loader=self.document_loader,
            vector_store=self.vector_store,
            skill_registry=self.skill_registry,
            structured_output_model_registry=self.output_model_registry
        )

        # Step 4: Determine mode and create agent(s)
        agent_list_config = self.agent_config.get('agent_list', [])
        self.crew_config = self.agent_config.get('crew_config', {})

        if not agent_list_config:
            # No agent_list - create single agent from main config
            self.is_multi_agent = False
            await self._initialize_single_agent()
        elif len(agent_list_config) == 1:
            # Single agent in list - use single agent mode
            self.is_multi_agent = False
            await self._initialize_single_agent_from_list()
        else:
            # Multiple agents - use multi-agent mode
            self.is_multi_agent = True
            await self._initialize_multi_agent()

        self._initialized = True
        self.logger.info(
            f"✅ Initialized {'multi-agent' if self.is_multi_agent else 'single-agent'} "
            f"system: {self.agent_name}"
        )

    async def _initialize_single_agent(self) -> None:
        """Initialize a single LangChain agent from main configuration.

        The agent will use tools from the registry based on its 'tools' configuration.
        """
        self.agent = await self.agent_builder.create_single_agent(
            agent_name=self.agent_name,
            agent_config=self.agent_config,
            pre_loaded_tools=self.pre_loaded_tools
        )

    async def _initialize_single_agent_from_list(self) -> None:
        """Initialize a single agent from the first item in agent_list.

        The agent will use tools from the registry based on its 'tools' configuration.
        """
        from oai_agent_core.components.configuration.model_config import ConfigManager

        agent_list_data = self.agent_config.get('agent_list')

        if not agent_list_data:
            raise ValueError("agent_list configuration is empty or missing")

        # 1. Normalize input: If it's a list, extract the first item.
        #    This allows us to handle both {'agent': val} and [{'agent': val}] uniformly.
        current_item = agent_list_data[0] if isinstance(agent_list_data, list) else agent_list_data

        if isinstance(current_item, str):
            # Case: ['agent_name']
            agent_name = current_item
            raw_config = None
        elif isinstance(current_item, dict):
            # Case: {'agent_name': {...}} or {'agent_name': None}
            agent_name = next(iter(current_item))
            raw_config = current_item[agent_name]
        else:
            raise ValueError(f"Invalid item type in agent_list: {type(current_item)}")

        # 3. Resolve final configuration (Load from manager if None)
        if raw_config is None:
            config_manager = ConfigManager(config_root=self.config_root)
            agent_config = config_manager.load_agent_config(agent_name=agent_name)
        else:
            agent_config = raw_config

        # 4. Create the agent
        self.agent = await self.agent_builder.create_single_agent(
            agent_name=agent_name,
            agent_config=agent_config,
            pre_loaded_tools=self.pre_loaded_tools
        )

    async def _initialize_multi_agent(self) -> None:
        """Initialize a multi-agent system with supervisor.

        Each agent will use tools from the registry based on its 'tools' configuration.
        """
        agent_list_config = self.agent_config['agent_list']
        system_prompt = self.agent_config.get('system_prompt', '')

        self.agent, self.base_agent_list = await self.agent_builder.create_multi_agent_system(
            agent_configs=agent_list_config,
            system_prompt=system_prompt,
            session_id=self.session_id,
            crew_config=self.crew_config
        )

    def _prepare_message(
            self,
            user_message: str,
            config: Optional[Dict[str, Any]]
    ) -> str:
        """Prepare and format user message with inputs.

        Args:
            user_message: Raw user message
            config: Optional configuration with inputs

        Returns:
            Formatted message string
        """
        # Get inputs from config or extract defaults
        if config and 'inputs' in config:
            inputs = config['inputs']
        else:
            # Extract template variables from configuration
            variables = self.message_formatter.extract_variables_from_config(
                self.agent_config
            )
            inputs = self.message_formatter.create_default_inputs(
                user_message,
                variables
            )

        # Format the message
        user_message = self._guardrail_input_message(user_message)
        formatted = self.message_formatter.format_message(user_message, inputs)
        return formatted if formatted is not None else user_message

    async def astream(
            self,
            user_message: str,
            config: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Returns an async streaming response from the agent.

        Args:
            user_message: The message from the user.
            config: Optional configuration dictionary.

        Yields:
            Dictionary containing the stream chunk.
        """
        await self._ensure_initialized()
        self._set_langfuse_config()

        # Determine the actual original message to use (config overrides argument)
        actual_original_message = config.get('original_message', user_message) if config else user_message

        # Format message with template variables
        formatted_message = self._prepare_message(user_message, config)
        formatted_message = self._guardrail_input_message(formatted_message)

        # Augment prompt with global knowledge base and memory
        formatted_message = self._augment_message(formatted_message, original_query=user_message)

        yielded_message_count = 0
        final_response_content = ""

        try:
            async for chunk in self.agent.astream(
                    {"messages": [{"role": "user", "content": formatted_message}]},
                    stream_mode="values",
                    config=self._generate_runnable_config(config)
            ):
                verbose = config.get('verbose', False) if config else False
                if verbose:
                    if hasattr(chunk, 'get') and 'messages' in chunk:
                        messages = chunk['messages']
                        # Yield only new messages that haven't been yielded yet
                        for i in range(yielded_message_count, len(messages)):
                            message_chunk = {'messages': [messages[i]]}
                            formatted_chunk = self.result_extractor.format_stream_chunk(
                                message_chunk,
                                session_id=self.session_id,
                                model_id=getattr(self.llm, 'model', 'unknown'),
                                model_provider=self.agent_config.get('cloud_provider', 'langchain'),
                                is_final=False,
                                include_raw=config.get('include_raw', False) if config else False,
                                input_message=formatted_message if config and config.get('include_input_message', False) else None,
                                original_message=actual_original_message if config and config.get('include_original_message', False) else None
                            )
                            yield formatted_chunk

                            # Track last message content for memory
                            if hasattr(messages[i], 'content'):
                                final_response_content = messages[i].content
                            elif isinstance(messages[i], dict) and 'content' in messages[i]:
                                final_response_content = messages[i]['content']

                        yielded_message_count = len(messages)
                    else:
                        formatted_chunk = self.result_extractor.format_stream_chunk(
                            chunk,
                            session_id=self.session_id,
                            model_id=getattr(self.llm, 'model', 'unknown'),
                            model_provider=self.agent_config.get('cloud_provider', 'langchain'),
                            is_final=False,
                            include_raw=config.get('include_raw', False) if config else False,
                            input_message=formatted_message if config and config.get('include_input_message',
                                                                                     False) else None,
                            original_message=actual_original_message if config and config.get('include_original_message',
                                                                                   False) else None
                        )
                        yield formatted_chunk
                else:
                    formatted_chunk = self.result_extractor.format_stream_chunk(
                        chunk,
                        session_id=self.session_id,
                        model_id=getattr(self.llm, 'model', 'unknown'),
                        model_provider=self.agent_config.get('cloud_provider', 'langchain'),
                        is_final=False,
                        include_raw=config.get('include_raw', False) if config else False,
                        input_message=formatted_message if config and config.get('include_input_message',
                                                                                 False) else None,
                        original_message=actual_original_message if config and config.get('include_original_message',
                                                                               False) else None
                    )
                    formatted_chunk['content']['text'] = self._guardrail_output_message(formatted_chunk['content']['text'])
                    yield formatted_chunk

                    # Track content for memory if available in chunk
                    # Note: In non-verbose mode, chunk structure depends on stream_mode="values"
                    if isinstance(chunk, dict) and 'messages' in chunk and chunk['messages']:
                        last_msg = chunk['messages'][-1]
                        if hasattr(last_msg, 'content'):
                            final_response_content = last_msg.content
                        elif isinstance(last_msg, dict) and 'content' in last_msg:
                            final_response_content = last_msg['content']

            # Save turn to memory store
            if self.memory_store and final_response_content:
                self.memory_store.add_turn(
                    session_id=self.session_id,
                    user_id=self.user_id,
                    user_message=actual_original_message,
                    agent_response=final_response_content
                )

        except Exception as e:
            self.logger.error(f"Error in astream: {e}", exc_info=True)
            yield {
                "content": f"Error: {str(e)}",
                "type": "error",
                "final": True,
                "session_id": self.session_id
            }

    async def ainvoke(
            self,
            user_message: str,
            config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Async invocation with formatted response.

        Args:
            user_message: The message from the user.
            config: Optional configuration dictionary.

        Returns:
            Dictionary containing the formatted response.
        """
        await self._ensure_initialized()
        self._set_langfuse_config()

        # Determine the actual original message to use (config overrides argument)
        actual_original_message = config.get('original_message', user_message) if config else user_message

        # Format message with template variables
        formatted_message = self._prepare_message(user_message, config)
        formatted_message = self._guardrail_input_message(formatted_message)

        # Augment prompt with global knowledge base and memory
        formatted_message = self._augment_message(formatted_message, original_query=user_message)

        try:
            result = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": formatted_message}]},
                config=self._generate_runnable_config(config)
            )

            response = self.result_extractor.format_response(
                result,
                session_id=self.session_id,
                model_id=getattr(self.llm, 'model_name') or getattr(self.llm, 'model', 'unknown'),
                model_provider=self.agent_config.get('cloud_provider', 'langchain'),
                include_raw=config.get('include_raw', False) if config else False,
                input_message=formatted_message if config and config.get('include_input_message', False) else None,
                original_message=actual_original_message if config and config.get('include_original_message', False) else None,
                final=True
            )

            # Save turn to memory store
            if self.memory_store:
                self.memory_store.add_turn(
                    session_id=self.session_id,
                    user_id=self.user_id,
                    user_message=actual_original_message,
                    agent_response=response
                )
            response['content']['text'] = self._guardrail_output_message(response['content']['text'])
            return response

        except Exception as e:
            self.logger.error(f"Error in ainvoke: {e}", exc_info=True)
            return {
                "content": {"text": f"Error: {str(e)}", "type": "error"},
                "session_id": self.session_id,
                "model_id": getattr(self.llm, 'model_name', 'unknown'),
                "model_provider": "langchain",
                "final": True,
                "error": str(e)
            }

    def invoke(
            self,
            user_message: str,
            config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synchronous invocation with formatted response.

        Args:
            user_message: The message from the user.
            config: Optional configuration dictionary.

        Returns:
            Dictionary containing the formatted response.
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, use the sync version directly
            return self._sync_invoke(user_message, config)
        else:
            # Otherwise run async version
            return asyncio.run(self.ainvoke(user_message, config))

    def _sync_invoke(
            self,
            user_message: str,
            config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synchronous invocation helper.

        Args:
            user_message: The message from the user.
            config: Optional configuration dictionary.

        Returns:
            Dictionary containing the formatted response.
        """
        self._set_langfuse_config()

        # Determine the actual original message to use (config overrides argument)
        actual_original_message = config.get('original_message', user_message) if config else user_message

        # Format message with template variables
        formatted_message = self._prepare_message(user_message, config)
        formatted_message = self._guardrail_input_message(formatted_message)

        # Augment prompt with global knowledge base and memory
        formatted_message = self._augment_message(formatted_message, original_query=user_message)

        try:
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": formatted_message}]},
                config=self._generate_runnable_config(config)
            )

            response = self.result_extractor.format_response(
                result,
                session_id=self.session_id,
                model_id=getattr(self.llm, 'model_name') or getattr(self.llm, 'model', 'unknown'),
                model_provider=self.agent_config.get('cloud_provider', 'langchain'),
                include_raw=config.get('include_raw', False) if config else False,
                input_message=formatted_message if config and config.get('include_input_message', False) else None,
                original_message=actual_original_message if config and config.get('include_original_message', False) else None,
                final=True
            )

            # Save turn to memory store
            if self.memory_store:
                self.memory_store.add_turn(
                    session_id=self.session_id,
                    user_id=self.user_id,
                    user_message=actual_original_message,
                    agent_response=response
                )

            return response

        except Exception as e:
            self.logger.error(f"Error in invoke: {e}", exc_info=True)
            return {
                "content": [{"text": f"Error: {str(e)}", "type": "error"}],
                "session_id": self.session_id,
                "model_id": getattr(self.llm, 'model_name', 'unknown'),
                "model_provider": "langchain",
                "final": True,
                "error": str(e)
            }

    def _generate_runnable_config(
            self,
            config: Optional[Dict[str, Any]] = None
    ) -> RunnableConfig:
        """Generate RunnableConfig for agent execution.

        Args:
            config: Optional configuration dictionary to merge with defaults.

        Returns:
            RunnableConfig instance.
        """
        runnable_config = {'configurable': {}}

        for k, v in self.config.items():
            runnable_config[k] = v

        for k, v in self.config['configurable'].items():
            runnable_config['configurable'][k] = v

        if config is not None:
            config['thread_id'] = config.get('session_id', self.session_id)
            for k, v in config.items():
                runnable_config['configurable'][k] = v

        return RunnableConfig(**runnable_config)

    def validate_tasks(self) -> Dict[str, Any]:
        """Validate configuration and return diagnostics.

        Returns:
            Dictionary with validation results
        """
        agent_list_config = self.agent_config.get('agent_list', [])

        diagnostics = {
            'agent_count': len(agent_list_config) if agent_list_config else 1,
            'tool_count': len(self.agent_config.get('tools', {})),
            'mcp_count': len(self.agent_config.get('mcps', self.agent_config.get('servers', {}))),
            'is_multi_agent': self.is_multi_agent,
            'tools': list(self.agent_config.get('tools', {}).keys()),
            'mcps': list(self.agent_config.get('mcps', self.agent_config.get('servers', {})).keys()),
            'registry_tool_count': len(self.tool_registry.tools) if hasattr(self.tool_registry, 'tools') else 0
        }

        # Extract template variables
        variables = self.message_formatter.extract_variables_from_config(
            self.agent_config
        )
        diagnostics['input_variables'] = list(variables)

        if agent_list_config:
            diagnostics['agents'] = []
            for agent_config in agent_list_config:
                if isinstance(agent_config, str):
                    diagnostics['agents'].append(agent_config)
                elif isinstance(agent_config, dict):
                    diagnostics['agents'].append(list(agent_config.keys())[0])
        else:
            diagnostics['agents'] = [self.agent_name]

        # Validate configuration using builder
        if self.agent_builder:
            if self.is_multi_agent:
                for agent_config in agent_list_config:
                    if isinstance(agent_config, dict):
                        agent_name = list(agent_config.keys())[0]
                        agent_data = agent_config[agent_name]
                        is_valid, errors = self.agent_builder.validate_agent_config(
                            agent_name, agent_data
                        )
                        if not is_valid:
                            diagnostics.setdefault('validation_errors', []).extend(errors)
            else:
                is_valid, errors = self.agent_builder.validate_agent_config(
                    self.agent_name, self.agent_config
                )
                if not is_valid:
                    diagnostics['validation_errors'] = errors

        return diagnostics

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the configured agent system."""
        info = {
            'agent_name': self.agent_name,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'initialized': self._initialized,
            'agent_type': self.agent_type,
            'is_multi_agent': self.is_multi_agent,
            'model_id': getattr(self.llm, 'model_name', 'unknown'),
            'model_provider': 'langchain',
            'config_summary': self.validate_tasks()
        }

        if self.is_multi_agent:
            info['agent_count'] = len(self.base_agent_list)
            info['sub_agents'] = [
                agent.agent_name for agent in self.base_agent_list
            ]

        return info

    async def cleanup(self):
        """Cleanup resources, especially for remote agents."""
        if self.is_multi_agent:
            for base_agent in self.base_agent_list:
                if base_agent.agent_type == Constants.REMOTE and base_agent.agent:
                    try:
                        self.logger.info(f"Stopping remote agent {base_agent.agent_name}")
                        await base_agent.agent.stop()
                    except Exception:
                        pass

    def __repr__(self) -> str:
        """String representation of the agent."""
        mode = "multi-agent" if self.is_multi_agent else "single-agent"
        return (
            f"UnifiedLangChainAgent(name='{self.agent_name}', "
            f"mode='{mode}', "
            f"session='{self.session_id}', "
            f"initialized={self._initialized})"
        )

    def __del__(self):
        """Cleanup when agent is destroyed."""
        try:
            if self.langfuse_manager.is_enabled:
                self.langfuse_manager.flush()
        except Exception:
            pass

    async def stream(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Not implemented — use :meth:`astream` for streaming responses."""
        raise NotImplementedError("Use astream() for streaming responses from LangGraphAgent.")
