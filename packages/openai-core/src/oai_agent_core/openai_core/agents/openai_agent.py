import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

from oai_agent_core.core.base_agent import BaseAgent
from oai_agent_core.core.constants import Constants
from oai_agent_core.processing.message_formatter import MessageFormatter
from oai_agent_core.processing.output_serializer import OutputSerializer

from oai_agent_core.openai_core.builders.agent_builder import AgentBuilder
from oai_agent_core.openai_core.components.configuration.model_config import \
    OpenAIModelConfigurationManager as ModelConfigurationManager
from oai_agent_core.openai_core.components.tools.tool_registry import OpenAIToolRegistry
from oai_agent_core.openai_core.processing.result_extractor import ResultExtractor


class OpenAIAgent(BaseAgent):
    """Agent implementation using OpenAI's models via the 'agents' library."""

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
        """Initialize the OpenAI agent.

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
        self.pre_loaded_tools = tools
        self.crew_config = None
        self.agent_type = Constants.OPENAI
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

        # Multi-agent specific attributes
        self.base_agent_list = []
        self.is_multi_agent = False

        # Initialize tool registry
        self.tool_registry = OpenAIToolRegistry(
            project_root=config_root,
            logger=self.logger,
            enable_lazy_loading=self.agent_config.get('crew_config', {}).get('enable_lazy_loading', False)
        )

        self.message_formatter = MessageFormatter(logger=self.logger)
        self.output_serializer = OutputSerializer(logger=self.logger)
        self.result_extractor = ResultExtractor(logger=self.logger)

        self.agent_builder = None

    async def initialize(self):
        """Initialize the agent system.

        Loads tools, MCP servers, and builds the agent or multi-agent system
        based on the configuration.
        """
        from oai_agent_core.openai_core.components.knowledge.knowledge_base_factory import KnowledgeBaseFactory

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
        """Initialize a single agent from the main configuration."""
        self.agent = await self.agent_builder.create_single_agent(
            agent_name=self.agent_name,
            agent_config=self.agent_config,
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

    @asynccontextmanager
    async def _mcp_context(self):
        """Context manager for handling MCP server connections.

        Ensures MCP servers are connected before execution and properly closed after.
        Assigns active servers to the agent(s).
        """

        if self.tool_registry.enable_lazy_loading:
            yield
        else:
            from agents.mcp import MCPServerManager

            # Load all required MCP clients
            mcp_clients = self.tool_registry.get_mcp_clients()
            mcp_servers = []
            for mcp_list in mcp_clients.values():
                mcp_servers.extend(mcp_list)

            async with MCPServerManager(mcp_servers, connect_timeout_seconds=1200):
                # Assign active servers to agents based on their specific config

                def _assign_servers(agent_obj):
                    if hasattr(agent_obj, 'mcp_servers'):
                        agent_obj.mcp_servers = mcp_clients.get(agent_obj.name, [])
                        agent_obj.mcp_config = {}

                # Assign to sub-agents
                if self.is_multi_agent and self.base_agent_list:
                    for sub_agent in self.base_agent_list:
                        _assign_servers(sub_agent)
                else:
                    _assign_servers(self.agent)

                yield

    async def process_request(
            self,
            user_message: str,
            config: Optional[Dict[str, Any]] = None,
            async_mode: bool = True
    ) -> Dict[str, Any]:
        """Execute the agent system with given inputs.

        Args:
            user_message: User input message
            config: Optional configuration with custom inputs
            async_mode: Whether to use async execution

        Returns:
            Dictionary containing session_id and result
        """
        # Ensure system is initialized
        await self._ensure_initialized()

        # Format message with template variables
        formatted_message = self._prepare_message(user_message, config)
        formatted_message = self._guardrail_input_message(formatted_message)

        # Augment prompt with global knowledge base and memory
        formatted_message = self._augment_message(formatted_message, original_query=user_message)

        self.langfuse_manager.initialize_client()

        try:
            # Execute with or without tracing
            if self.langfuse_manager.is_enabled:
                result = await self._execute_with_tracing(formatted_message, async_mode)
            else:
                result = await self._execute_without_tracing(formatted_message, async_mode)

            return {
                "session_id": self.session_id,
                "result": result,
                'final': True,
                'input_message': formatted_message
            }

        except Exception as e:
            # Log error to observability platform
            if self.langfuse_manager.is_enabled:
                self.langfuse_manager.log_error(
                    user_message=user_message,
                    error=e,
                    metadata={'session_id': self.session_id}
                )

            self.logger.error(f"Error executing agent system: {e}", exc_info=True)
            raise

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
        formatted = self.message_formatter.format_message(user_message, inputs)
        return formatted if formatted is not None else user_message

    async def _execute_without_tracing(self, message: str, async_mode: bool) -> Any:
        """Execute agent system without Langfuse tracing.

        Args:
            message: Formatted message
            async_mode: Whether to use async execution

        Returns:
            Execution result
        """
        from agents import Runner
        async with self._mcp_context():
            result = await Runner.run(self.agent, message)
            return result

    async def _execute_with_tracing(self, message: str, async_mode: bool) -> Any:
        """Execute agent system with Langfuse tracing.

        Args:
            message: Formatted message
            async_mode: Whether to use async execution

        Returns:
            Execution result
        """
        with self.langfuse_manager.trace_generation(
                input_data=message,
                user_id=self.user_id,
                session_id=self.session_id,
                metadata={
                    'agent_name': self.agent_name,
                    'pattern': self.agent_config.get('crew_config', {}).get('pattern', 'single')
                }
        ) as span:
            # Execute the system
            result = await self._execute_without_tracing(message, async_mode)

            # Extract output for tracing
            output = result.final_output

            # Update trace with results
            self.langfuse_manager.update_trace(
                span=span,
                input_data=message,
                output_data=output,
                user_id=self.user_id,
                session_id=self.session_id,
                tags=[self.agent_name, 'openai']
            )

        return result

    async def ainvoke(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Asynchronously invoke the agent and get the full response.

        Args:
            user_message: The input message from the user.
            config: Optional configuration overrides for this request.

        Returns:
            The agent's response formatted by ResultExtractor.
        """
        result_dict = await self.process_request(user_message, config)
        result = result_dict['result']

        user_message = config.get('original_message') if config else user_message

        response = self.result_extractor.format_response(
            result,
            session_id=self.session_id,
            model_id=getattr(self.llm, 'model', 'unknown'),
            model_provider=self.agent_config.get('cloud_provider', 'openai'),
            include_raw=config.get('include_raw', False) if config else False,
            input_message=result_dict.get('input_message', None) if config and config.get('include_input_message',
                                                                                          False) else None,
            original_message=user_message if config and config.get('include_original_message', False) else None
        )

        if self.memory_store:
            self.memory_store.add_turn(session_id=self.session_id,
                                       user_id=self.user_id,
                                       user_message=user_message,
                                       agent_response=result.final_output)

        response['content']['text'] = self._guardrail_output_message(response['content']['text'])
        return response

    def invoke(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Synchronously invoke the agent and get the full response.

        Args:
            user_message: The input message from the user.
            config: Optional configuration overrides for this request.

        Returns:
            The agent's response.
        """
        return asyncio.run(self.ainvoke(user_message, config))

    async def astream(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Asynchronously stream the agent's response.

        Args:
            user_message: The input message from the user.
            config: Optional configuration overrides for this request.

        Yields:
            Chunks of the response formatted by ResultExtractor.
        """
        await self._ensure_initialized()

        formatted_message = self._prepare_message(user_message, config)
        formatted_message = self._guardrail_input_message(formatted_message)
        formatted_message = self._augment_message(formatted_message, original_query=user_message)

        self.langfuse_manager.initialize_client()

        try:
            if self.langfuse_manager.is_enabled:
                async for chunk in self._astream_with_tracing(formatted_message, user_message, config):
                    yield chunk
            else:
                async for chunk in self._astream_without_tracing(formatted_message, user_message, config):
                    yield chunk

            # Note: Memory store update for streaming is tricky without full response accumulation.
            # StrandsAgent does it by accumulating in _astream_without_tracing or caller.
            # Here we yield chunks directly.

        except Exception as e:
            self.logger.error(f"Error in astream: {e}", exc_info=True)
            if self.langfuse_manager.is_enabled:
                self.langfuse_manager.log_error(
                    user_message=user_message,
                    error=e,
                    metadata={'session_id': self.session_id}
                )
            yield f"Error: {str(e)}"

    async def _astream_without_tracing(self, message: str, original_message: str,
                                       config: Optional[Dict[str, Any]] = None):
        """Stream agent response without Langfuse tracing.

        Args:
            message: Formatted message.
            original_message: Original message.
            config: Configuration object (unused but kept for interface consistency).

        Yields:
            Formatted response chunks.
        """
        from agents import Runner

        # Determine the actual original message to use (config overrides argument)
        actual_original_message = config.get('original_message') if config else original_message

        async with self._mcp_context():
            result = Runner.run_streamed(self.agent, message)

            async for event in result.stream_events():
                final = False
                content: Optional[Dict[str, Any]] = None
                # We'll ignore the raw responses event deltas
                if event.type == "raw_response_event":
                    continue
                # When the agent updates, print that
                elif event.type == "agent_updated_stream_event":
                    content = {'content': f"{event.new_agent.name}"}

                # When items are generated, print them
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        content = {'content': f"Executing {event.item.raw_item.name}",
                                   'tool_calls': [{
                                       'args': event.item.raw_item.arguments,
                                       'id': event.item.raw_item.call_id,
                                       'name': event.item.raw_item.name,
                                       'type': 'tool_call'
                                   }
                                   ]}
                    elif event.item.type == "tool_call_output_item":
                        content = {
                            'content': f"{event.item.output}"
                        }
                    elif event.item.type == "message_output_item":
                        if hasattr(event.item.raw_item, 'status') and event.item.raw_item.status == 'completed':
                            final = True
                        from agents import ItemHelpers
                        text_content = ItemHelpers.text_message_output(event.item)
                        text_content = self._guardrail_output_message(text_content)
                        content = {
                            'content': f"{text_content}"
                        }
                    else:
                        pass  # Ignore other event types

                if content:
                    # Fix: Ensure usage is a dictionary, not a Usage object
                    usage_obj = result.context_wrapper.usage
                    usage_dict = {
                        "input_tokens": usage_obj.input_tokens,
                        "output_tokens": usage_obj.output_tokens,
                        "total_tokens": usage_obj.total_tokens
                    } if hasattr(usage_obj, 'input_tokens') else {}

                    content['usage'] = usage_dict
                    content['type'] = event.item.type if hasattr(event, 'item') and hasattr(event.item,
                                                                                                  'type') else event.type
                    content['raw_result'] = event

                    yield self.result_extractor.format_response(
                        content,
                        session_id=self.session_id,
                        model_id=getattr(self.llm, 'model', 'unknown'),
                        model_provider=self.agent_config.get('cloud_provider', 'langchain'),
                        include_raw=config.get('include_raw', False) if config else False,
                        input_message=message if config and config.get('include_input_message', False) else None,
                        original_message=actual_original_message if config and config.get('include_original_message',
                                                                                          False) else None,
                        final=final
                    )

            # Add final response to memory after streaming is complete
            if self.memory_store:
                # Use the final output from the result object if available
                # This ensures we store exactly what the agent considers its final answer
                final_output = getattr(result, 'final_output', None)
                if final_output:
                    self.memory_store.add_turn(
                        session_id=self.session_id,
                        user_id=self.user_id,
                        user_message=actual_original_message,
                        agent_response=self.result_extractor.extract_text(final_output)
                    )

    async def _astream_with_tracing(self, message: str, original_message: str, config: Any):
        """Stream agent response with Langfuse tracing.

        Args:
            message: Formatted message.
            original_message: Original message.
            config: Configuration object.

        Yields:
            Formatted response chunks.
        """
        with self.langfuse_manager.trace_generation(
                input_data=message,
                user_id=self.user_id,
                session_id=self.session_id,
                metadata={
                    'agent_name': self.agent_name,
                    'pattern': self.agent_config.get('crew_config', {}).get('pattern', 'single')
                },
                name_suffix='-stream'
        ) as span:
            collected_output = ""

            async for chunk in self._astream_without_tracing(message, original_message, config):
                # Collect output for tracing
                if isinstance(chunk, dict):
                    # Extract text from formatted response
                    content_list = chunk.get('content', [])
                    if isinstance(content_list, list):
                        for item in content_list:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                collected_output += item.get('text', '')
                elif isinstance(chunk, str):
                    collected_output += chunk

                yield chunk

            # Update trace with collected output
            if span and collected_output:
                self.langfuse_manager.update_trace(
                    span=span,
                    input_data=message,
                    output_data=collected_output,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    tags=[self.agent_name, 'openai', 'stream']
                )

    async def stream(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Asynchronously stream the agent's response (alias for astream).

        Args:
            user_message: The input message from the user.
            config: Optional configuration overrides for this request.

        Yields:
            Chunks of the response.
        """
        async for chunk in self.astream(user_message, config):
            yield chunk
