"""AWS Strands Agent implementation for YAML-configurable multi-agent workflows.

This module provides a configurable Strands agent system using the official
AWS Strands SDK, enabling building complex multi-agent workflows through
YAML configuration files.
"""

import asyncio
from typing import Optional, Any, Dict, AsyncGenerator

from oai_agent_core.core.base_agent import BaseAgent
from oai_agent_core.components.observability.tracing import traced, trace_span, traced_stream
from oai_agent_core.core.constants import Constants
from oai_agent_core.processing.message_formatter import MessageFormatter
from oai_agent_core.processing.output_serializer import OutputSerializer

from oai_agent_core.aws_strands_core.builders.agent_builder import AgentBuilder
from oai_agent_core.aws_strands_core.builders.orchestration_builder import OrchestrationBuilder
from oai_agent_core.aws_strands_core.components.configuration.model_config import \
    StrandsModelConfigurationManager as ModelConfigurationManager
from oai_agent_core.aws_strands_core.components.registry.tool_registry import AWSStrandsToolRegistry as ToolRegistry
from oai_agent_core.aws_strands_core.processing.result_extractor import ResultExtractor


class StrandsAgent(BaseAgent):
    """A configurable AWS Strands agent system for multi-agent workflows.

    This class enables building sophisticated multi-agent workflows through
    YAML configuration files using the official AWS Strands SDK.

    Supports:
    - Multiple agents with different roles and capabilities
    - Multi-agent patterns: Swarm, Graph, Agents-as-Tools
    - Tool integration (Strands tools, custom tools, MCP)
    - Sequential and parallel execution
    - Session management and observability

    Attributes:
        agent_map: Dictionary mapping agent keys to Strands Agent instances
        context_map: Dictionary mapping agent keys to their dependencies
        multi_agent_system: Graph or Swarm instance for orchestration
        model_manager: Model configuration manager
        tool_registry: Tool registry
        message_formatter: Message formatter for template variables
        output_serializer: Output serializer for session tracking
        result_extractor: Result extractor for response formatting
        langfuse_manager: Optional Langfuse observability manager
    """

    def __init__(
            self,
            agent_name: str,
            agent_config: Optional[Dict[str, Any]] = None,
            llm=None,
            session_id: str = "default",
            user_id: str = "default",
            config_root: Optional[str] = None,
            region_name: str = "us-west-2",
            document_loader: Optional[Any] = None,
            vector_store: Optional[Any] = None,
            **kwargs
    ):
        """Initialize the Strands agent.

        Args:
            agent_name: Unique identifier for the agent system
            agent_config: YAML configuration dictionary
            llm: Optional LLM instance
            session_id: Session identifier for output tracking
            user_id: User identifier for tracing
            config_root: Root directory for configuration files
            region_name: Deprecated parameter (kept for compatibility)
            document_loader: Optional document loader instance
            vector_store: Optional vector store instance
            **kwargs: Additional arguments passed to BaseAgent
        """
        # Initialize base agent
        self.agent_type = Constants.AWS_STRANDS
        # Build the properly-configured model manager before calling super() so the
        # parent receives the final instance (avoids constructing it twice).
        self.model_manager = ModelConfigurationManager()  # temporary; replaced below

        super().__init__(
            llm=llm,
            model_manager=self.model_manager,
            agent_name=agent_name,
            agent_config=agent_config,
            session_id=session_id,
            config_root=config_root,
            user_id=user_id,
            agent_type=self.agent_type,
            document_loader=document_loader,
            vector_store=vector_store,
            **kwargs
        )

        # Initialize core attributes
        self.region_name = region_name
        self.agent_map = {}
        self.context_map = {}
        self.multi_agent_system = None

        # Replace model manager with one that has the full config and logger.
        self.model_manager = ModelConfigurationManager(
            default_config=self.agent_config.get('model', {}),
            logger=self.logger
        )
        # Respect an injected LLM; only construct from config when none is provided.
        if llm is None:
            self.llm = self.model_manager.create_model()

        self.tool_registry = ToolRegistry(
            logger=self.logger,
            project_root=config_root,
            enable_lazy_loading=self.agent_config.get('crew_config', {}).get('enable_lazy_loading', False))
        self.message_formatter = MessageFormatter(logger=self.logger)
        self.output_serializer = OutputSerializer(logger=self.logger)
        self.result_extractor = ResultExtractor(logger=self.logger)

        # # Initialize observability
        # self.langfuse_manager = LangfuseObservabilityManager(
        #     agent_name=self.agent_name,
        #     logger=self.logger
        # )

        # Initialize builders (created during initialization)
        self.agent_builder = None
        self.orchestration_builder = None
        self._node_text_buffers = {}
        self._tool_use_text_buffers = {}
        self.global_kb_factory = None

    async def initialize(self, session_id: str = None) -> Any:
        """Initialize the Strands multi-agent system from configuration.

        Creates agents, tools, and orchestration pattern (Swarm or Graph)
        based on YAML configuration.

        Args:
            session_id: Optional session identifier

        Returns:
            Configured Strands multi-agent system (Graph, Swarm, or single Agent)
        """
        if session_id:
            self.session_id = session_id

        from oai_agent_core.aws_strands_core.components.knowledge.knowledge_base_factory import KnowledgeBaseFactory

        # Load tools, MCP config, and KB concurrently
        await self._load_tools_and_kb_and_memory(kb_factory_class=KnowledgeBaseFactory)

        # Create agent builder
        self.agent_builder = AgentBuilder(
            llm=self.llm,
            model_manager=self.model_manager,
            tool_registry=self.tool_registry,
            logger=self.logger,
            document_loader=self.document_loader,
            vector_store=self.vector_store,
            skill_registry=self.skill_registry,
            structured_output_model_registry=self.output_model_registry
        )

        crew_config = self.agent_config.get('crew_config', {})
        pattern = crew_config.get('pattern', crew_config.get('process', 'sequential'))
        entry_agent = crew_config.get('entry_agent')
        system_prompt = self.agent_config.get('system_prompt', None)

        # Create agents with callback handlers
        agent_configs = self.agent_config.get('agent_list', [])
        self.agent_map = await self.agent_builder.create_agents_from_config(
            agent_configs,
            callback_factory=self._create_callback_handler,
            architecture=pattern
        )

        # Extract context map
        self.context_map = self.agent_builder.extract_context_map(agent_configs)

        # Create orchestration system
        self.orchestration_builder = OrchestrationBuilder(logger=self.logger,
                                                          llm=self.llm,
                                                          structured_output_model_registry=self.output_model_registry)

        self.multi_agent_system = self.orchestration_builder.build_orchestration(
            agent_map=self.agent_map,
            context_map=self.context_map,
            crew_config=crew_config,
            system_prompt=system_prompt
        )

        self._initialized = True
        self.logger.info(f"Strands agent system initialized: {pattern} pattern")

        return self.multi_agent_system

    def _create_callback_handler(self, agent_key: str):
        """Create a callback handler for agent output tracking.

        Args:
            agent_key: Agent identifier

        Returns:
            Callback function
        """

        def callback(**kwargs):
            if "data" in kwargs:
                self.output_serializer.write_to_session(
                    content={
                        'agent': agent_key,
                        'type': 'text_delta',
                        'content': kwargs["data"]
                    },
                    session_id=self.session_id
                )
            elif "current_tool_use" in kwargs:
                tool_use = kwargs["current_tool_use"]
                self.output_serializer.write_to_session(
                    content={
                        'agent': agent_key,
                        'type': 'tool_use',
                        'tool_name': tool_use.get('name'),
                        'tool_use_id': tool_use.get('toolUseId')
                    },
                    session_id=self.session_id
                )

        return callback

    async def process_request(
            self,
            user_message: str,
            config: Optional[Dict[str, Any]] = None,
            async_mode: bool = True
    ) -> Dict[str, Any]:
        """Execute the multi-agent system with given inputs.

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

            self.logger.error(f"Error executing multi-agent system: {e}", exc_info=True)
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
        """Execute multi-agent system without Langfuse tracing.

        Args:
            message: Formatted message
            async_mode: Whether to use async execution

        Returns:
            Execution result
        """
        if async_mode:
            result = await self.multi_agent_system.invoke_async(message)
        else:
            result = self.multi_agent_system(message)

        return result

    async def _execute_with_tracing(self, message: str, async_mode: bool) -> Any:
        """Execute multi-agent system with Langfuse tracing.

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
            output = self.result_extractor.extract_text(result)

            # Update trace with results
            self.langfuse_manager.update_trace(
                span=span,
                input_data=message,
                output_data=output,
                user_id=self.user_id,
                session_id=self.session_id,
                tags=[self.agent_name, 'strands']
            )

        return result

    @traced("agent.ainvoke")
    async def ainvoke(
            self,
            user_message: str,
            config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Asynchronous execution of the multi-agent system.

        Args:
            user_message: User input message
            config: Optional configuration

        Returns:
            Formatted response dictionary
        """
        with trace_span("agent.llm.invoke", agent_name=self.agent_name):
            result = await self.process_request(user_message, config)
        session_id = result.get('session_id')
        final_result = result.get('result')

        with trace_span("agent.format_response", level="debug", agent_name=self.agent_name):
            response = self.result_extractor.format_response(
                result=final_result,
                session_id=session_id,
                model_id=self.model_manager.default_config['model_id'],
                model_provider=self.model_manager.get_model_info()['provider'],
                include_raw=config.get('include_raw', False) if config else False,
                input_message=result.get('input_message', None) if config and config.get('include_input_message',
                                                                                         False) else None,
                original_message=user_message if config and config.get('include_original_message', False) else None
            )

        if self.memory_store:
            self.memory_store.add_turn(session_id=self.session_id,
                                       user_id=self.user_id,
                                       user_message=user_message,
                                       agent_response=response)
        if 'content' in response:
            response['content']['text'] = self._guardrail_output_message(response['content'].get('text'))
        return response

    def invoke(
            self,
            user_message: str,
            config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synchronous execution of the multi-agent system.

        Args:
            user_message: User input message
            config: Optional configuration

        Returns:
            Formatted response dictionary
        """
        return asyncio.run(self.ainvoke(user_message, config))

    @traced("agent.astream")
    async def astream(
            self,
            user_message: str,
            config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from agent system.

        Args:
            user_message: User input message
            config: Optional configuration

        Yields:
            Response chunks as dictionaries
        """
        await self._ensure_initialized()

        verbose = config.get('verbose', False) if config else False

        # Format message
        formatted_message = self._prepare_message(user_message, config)
        formatted_message = self._guardrail_input_message(formatted_message)

        # Augment prompt with global knowledge base and memory
        formatted_message = self._augment_message(formatted_message, original_query=user_message)

        self.langfuse_manager.initialize_client()
        try:
            # Stream with or without tracing
            chunk = None
            if self.langfuse_manager.is_enabled:
                async for chunk in self._astream_with_tracing(formatted_message,
                                                              user_message,
                                                              config,
                                                              verbose):
                    yield chunk
            else:
                async for chunk in self._astream_without_tracing(formatted_message,
                                                                 user_message,
                                                                 config,
                                                                 verbose):
                    yield chunk

            if self.memory_store:
                self.memory_store.add_turn(session_id=self.session_id,
                                           user_id=self.user_id,
                                           user_message=user_message,
                                           agent_response=chunk)

        except Exception as e:
            self.logger.error(f"Error in astream: {e}", exc_info=True)

            if self.langfuse_manager.is_enabled:
                self.langfuse_manager.log_error(
                    user_message=user_message,
                    error=e,
                    metadata={'session_id': self.session_id}
                )

            yield {"content": f"Error: {str(e)}", "type": "error", "final": True}

    async def _astream_without_tracing(
            self,
            formatted_message: str,
            original_message: str,
            config: Optional[Dict[str, Any]] = None,
            verbose: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream execution without Langfuse tracing.

        Args:
            formatted_message: Formatted message

        Yields:
            Response chunks
        """
        # Determine the actual original message to use (config overrides argument)
        actual_original_message = config.get('original_message', original_message) if config else original_message

        if len(self.agent_map) == 1 or self.agent_config.get('crew_config', {}).get('pattern',
                                                                                    None) == Constants.PATTERN_AGENT_AS_TOOL:
            # Single agent - use direct streaming
            if len(self.agent_map) == 1:
                agent = list(self.agent_map.values())[0]
            else:
                agent = self.multi_agent_system
            # Always skip these junk events
            junk_events = {'contentBlockStart', 'contentBlockStop', 'messageStart', 'messageStop', 'start_event_loop',
                           'start', 'init_event_loop', 'metadata'}

            # Skip these only in non-verbose mode
            verbose_events = {'start_event_loop', 'start', 'init_event_loop'}

            # Skip these only in non-verbose mode (useful for debugging)
            debug_events = {'metadata', 'current_tool_use'}

            if hasattr(agent, 'stream_async'):
                content = ""
                token_usage = {}
                result = {}

                async for chunk in traced_stream(
                        "agent.llm.invoke",
                        agent.stream_async(formatted_message, stream_complete_only=True),
                        agent_name=self.agent_name):
                    result = chunk
                    if 'data' in chunk:
                        content += chunk['data']
                    else:
                        # Debug: log what we're processing
                        if verbose:
                            self.logger.debug(f"Processing chunk: {chunk.keys()}")

                        # Always skip junk events
                        if ('event' in chunk and
                                (any(event in chunk['event'] for event in junk_events) or
                                 'contentBlockDelta' in chunk['event'])):
                            if verbose:
                                self.logger.debug(f"Skipping junk event: {chunk.get('event', {})}")
                            continue

                        if any(event in chunk for event in junk_events):
                            continue

                        # Skip verbose events only in non-verbose mode
                        if not verbose and (
                                'current_tool_use' in chunk or
                                any(event in chunk for event in verbose_events) or
                                ('event' in chunk and any(event in chunk['event'] for event in verbose_events))
                        ):
                            if verbose:
                                self.logger.debug(f"Skipping verbose event in non-verbose mode")
                            continue

                        # In verbose mode, yield additional debug info
                        if verbose and ('current_tool_use' in chunk):
                            continue

                        # Extract metadata if present
                        if 'event' in chunk and 'metadata' in chunk['event']:
                            token_usage = chunk['event']['metadata'].get('usage', {})

                        if 'type' in chunk and chunk['type'] == 'tool_use_stream':
                            continue

                        if 'message' in chunk and 'content' in chunk['message']:
                            text = None
                            tool_calls = []
                            for tool_use_content in chunk['message']['content']:
                                if 'text' in tool_use_content:
                                    text = tool_use_content['text']
                                if 'toolUse' in tool_use_content:
                                    tool_calls = [
                                        {
                                            'args': tool_use_content['toolUse']['input'],
                                            'id': tool_use_content['toolUse']['toolUseId'],
                                            'name': tool_use_content['toolUse']['name'],
                                            'type': 'tool_call'
                                        }
                                    ]
                                if 'toolResult' in tool_use_content:
                                    text = tool_use_content['toolResult']['content'][0]['text']
                                    tool_calls = {
                                        'id': tool_use_content['toolResult']['toolUseId']
                                    }
                            if tool_calls and text:
                                yield {"content": {
                                    "text": text,
                                    'type': 'AIMessage'
                                },
                                    "tool_call": tool_calls
                                }

                # Yield final response
                result = result['result'] if 'result' in result else content
                response = self.result_extractor.format_response(
                    result=result,
                    session_id=self.session_id,
                    model_id=self.model_manager.default_config['model_id'],
                    model_provider=self.model_manager.get_model_info()['provider'],
                    include_raw=config.get('include_raw', False) if config else False,
                    input_message=formatted_message if config and config.get('include_input_message', False) else None,
                    original_message=actual_original_message if config and config.get('include_original_message',
                                                                                      False) else None
                )
                if 'content' in response:
                    response['content']['text'] = self._guardrail_output_message(response['content'].get('text'))
                yield response
            else:
                # Fallback to invoke
                result = await agent.invoke_async(formatted_message)
                yield self.result_extractor.format_response(
                    result=result,
                    session_id=self.session_id,
                    model_id=self.model_manager.default_config['model_id'],
                    model_provider=self.model_manager.get_model_info()['provider'],
                    include_raw=config.get('include_raw', False) if config else False,
                    input_message=formatted_message if config and config.get('include_input_message', False) else None,
                    original_message=actual_original_message if config and config.get('include_original_message',
                                                                                      False) else None
                )
        else:
            # Multi-agent system - use streaming
            if hasattr(self.multi_agent_system, 'stream_async'):
                # Define which events to surface
                allowed_events = [
                    "multiagent_node_start",
                    "multiagent_node_complete",
                    "multiagent_handoff",
                    "multiagent_result"
                ]

                if verbose:
                    allowed_events.append("multiagent_node_stream")

                async for event in traced_stream(
                        "agent.llm.invoke",
                        self.multi_agent_system.stream_async(
                            formatted_message,
                            stream_complete_only=True
                        ),
                        agent_name=self.agent_name):
                    event_type = event.get("type", "")
                    if event_type in allowed_events:
                        formatted = self._format_stream_event(event,
                                                              input_message=formatted_message,
                                                              original_message=actual_original_message,
                                                              config=config,
                                                              verbose=verbose)

                        if formatted:
                            yield formatted
            else:
                # Fallback to invoke
                result = await self.ainvoke(formatted_message, None)
                yield result

    def _format_stream_event(self,
                             event: Dict[str, Any],
                             input_message: str,
                             original_message: str,
                             config: Optional[Dict[str, Any]] = None,
                             verbose: bool = False) -> Optional[Dict[str, Any]]:
        """Format streaming event, yielding only complete strings for node streams."""
        event_type = event.get("type", "")
        agent_name = event.get("node_id", "Agent")

        if event_type == "multiagent_node_start":
            # Reset buffer for the new agent task
            self._node_text_buffers[agent_name] = ""
            if verbose:
                return self.result_extractor.format_streaming_chunk(
                    content=f"\n[Agent: {agent_name} starting...]\n",
                    chunk_type="status",
                    agent=agent_name
                )

        elif event_type == "multiagent_node_stream":
            inner_event = event.get("event", {})
            # AWS Strands often nests the standard agent event inside another 'event' key
            if "event" in inner_event:
                inner_event = inner_event["event"]

            if 'message' in inner_event and 'content' in inner_event['message']:
                text = None
                tool_calls = []
                for tool_use_content in inner_event['message']['content']:
                    if 'text' in tool_use_content:
                        text = tool_use_content['text']
                    if 'toolUse' in tool_use_content:
                        tool_calls = [
                            {
                                'args': tool_use_content['toolUse']['input'],
                                'id': tool_use_content['toolUse']['toolUseId'],
                                'name': f"{agent_name} - {tool_use_content['toolUse']['name']}",
                                'type': 'tool_call'
                            }
                        ]
                    if 'toolResult' in tool_use_content:
                        text = tool_use_content['toolResult']['content'][0]['text']
                        tool_calls = {
                            'id': tool_use_content['toolResult']['toolUseId'],
                            'type': 'tool_result'
                        }
                if tool_calls and text:
                    return {"content": {
                        "text": text,
                        'type': 'AIMessage'
                    },
                        "tool_call": tool_calls
                    }

            # 1. Accumulate Text Deltas
            if "contentBlockDelta" in inner_event:
                delta = inner_event["contentBlockDelta"].get("delta", {})
                text = delta.get("text", "")
                self._node_text_buffers[agent_name] = self._node_text_buffers.get(agent_name, "") + text
                return None  # DO NOT YIELD YET

            # 2. Check for Completion (Stop Events)
            # contentBlockStop or messageStop indicates the string is complete
            if "contentBlockStop" in inner_event or "messageStop" in inner_event:
                full_text = self._node_text_buffers.get(agent_name, "")
                if full_text:
                    self._node_text_buffers[agent_name] = ""  # Clear buffer after yielding
                    return self.result_extractor.format_streaming_chunk(
                        content=full_text,
                        chunk_type="agent_text",
                        agent=agent_name
                    )

        elif event_type == "multiagent_node_complete":
            # If verbose, we show the node is done.
            # Note: Text should have already been yielded by the 'Stop' event above.
            if verbose:
                return self.result_extractor.format_streaming_chunk(
                    content=f"\n[Agent: {agent_name} complete]\n",
                    chunk_type="status",
                    agent=agent_name
                )

        elif event_type == "multiagent_handoff":
            from_agents = ", ".join(event.get("from_node_ids", []))
            to_agents = ", ".join(event.get("to_node_ids", []))
            return self.result_extractor.format_streaming_chunk(
                content=f"\n>>> Handoff: {from_agents} → {to_agents} >>>\n",
                chunk_type="handoff"
            )

        elif event_type == "multiagent_result":
            # Final orchestrated result
            response = self.result_extractor.format_response(
                result=event.get("result"),
                session_id=self.session_id,
                model_id=self.model_manager.default_config.get('model_id'),
                model_provider=self.model_manager.get_model_info().get('provider'),
                include_raw=config.get('include_raw', False) if config else False,
                input_message=input_message if config and config.get('include_input_message', False) else None,
                original_message=original_message if config and config.get('include_original_message',
                                                                           False) else None
            )
            if 'content' in response:
                response['content']['text'] = self._guardrail_output_message(response['content'].get('text'))
            return response

        return None

    async def _astream_with_tracing(
            self,
            formatted_message: str,
            original_message: str,
            config: Optional[Dict[str, Any]] = None,
            verbose: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream execution with Langfuse tracing.

        Args:
            formatted_message: Formatted message

        Yields:
            Response chunks
        """
        with self.langfuse_manager.trace_generation(
                input_data=formatted_message,
                user_id=self.user_id,
                session_id=self.session_id,
                metadata={
                    'agent_name': self.agent_name,
                    'pattern': self.agent_config.get('crew_config', {}).get('pattern', 'single')
                },
                name_suffix='-stream'
        ) as span:
            collected_output = ""

            async for chunk in self._astream_without_tracing(formatted_message,
                                                             original_message,
                                                             config, verbose):
                # Collect output for tracing
                if "content" in chunk and isinstance(chunk["content"], list):
                    for item in chunk["content"]:
                        if isinstance(item, dict) and 'text' in item:
                            collected_output += item['text']
                elif "content" in chunk and chunk['content'].get("final", False):
                    collected_output = str(chunk["content"])

                yield chunk

            # Update trace with collected output
            if span and collected_output:
                self.langfuse_manager.update_trace(
                    span=span,
                    input_data=formatted_message,
                    output_data=collected_output,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    tags=[self.agent_name, 'strands', 'stream']
                )

    def validate_tasks(self) -> Dict[str, Any]:
        """Validate configuration and return diagnostics.

        Returns:
            Dictionary with validation results
        """
        diagnostics = {
            'agent_count': len(self.agent_config.get('agent_list', [])),
            'tool_count': len(self.agent_config.get('tools', {})),
            'agents': [
                list(ac.keys())[0]
                for ac in self.agent_config.get('agent_list', [])
            ],
            'tools': list(self.agent_config.get('tools', {}).keys()),
            'pattern': self.agent_config.get('crew_config', {}).get('pattern', 'sequential'),
        }

        # Extract template variables
        variables = self.message_formatter.extract_variables_from_config(
            self.agent_config
        )
        diagnostics['input_variables'] = list(variables)

        return diagnostics

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the configured agent system.

        Returns:
            Dictionary with agent system information
        """
        info = {
            'agent_name': self.agent_name,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'initialized': self._initialized,
            'has_observability': self.langfuse_manager.is_enabled,
            'region': self.region_name,
            'config_summary': self.validate_tasks()
        }

        # Add model info
        info['model_info'] = self.model_manager.get_model_info()

        # Add orchestration info if initialized
        if self._initialized and self.orchestration_builder:
            pattern = self.agent_config.get('crew_config', {}).get('pattern', 'sequential')
            info['orchestration'] = self.orchestration_builder.get_orchestration_info(
                orchestration=self.multi_agent_system,
                pattern=pattern,
                agent_count=len(self.agent_map)
            )

        return info

    def __repr__(self) -> str:
        """String representation of the agent."""
        return (
            f"StrandsAgent(name='{self.agent_name}', "
            f"session='{self.session_id}', "
            f"initialized={self._initialized})"
        )

    def __del__(self):
        """Cleanup when agent is destroyed."""
        try:
            if self.langfuse_manager.is_enabled:
                self.langfuse_manager.flush()
            self.close()
        except Exception:
            pass

    async def stream(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Not implemented — use :meth:`astream` for streaming responses."""
        raise NotImplementedError("Use astream() for streaming responses from StrandsAgent.")

    def close(self):
        try:
            for _, mcp_client in self.tool_registry.mcp_clients.items():
                for client in mcp_client:
                    client.stop(None, None, None)
        except Exception:
            pass

    async def __aenter__(self):
        # Setup: start the client
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Teardown: ensure the client stops regardless of errors
        # This solves your "3 required arguments" error automatically
        self.close()
