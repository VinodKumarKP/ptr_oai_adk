"""CrewAI Agent implementation for multi-agent workflows.

This module provides a configurable CrewAI agent system that enables building
complex multi-agent workflows through YAML configuration files. Supports both
Crew and Flow execution modes.
"""

import asyncio
import uuid
from typing import Optional, List, Any, Dict, AsyncGenerator

from crewai import LLM, Crew, Flow
from oai_agent_core.core.base_agent import BaseAgent
from oai_agent_core.components.observability.tracing import traced
from oai_agent_core.core.constants import Constants
from oai_agent_core.processing.message_formatter import MessageFormatter
from oai_agent_core.processing.output_serializer import OutputSerializer

from oai_agent_core.crewai_core.builders import CrewBuilder, FlowBuilder
from oai_agent_core.crewai_core.components import CrewAIModelConfigurationManager as ModelConfigurationManager
from oai_agent_core.crewai_core.components import CrewAIToolRegistry as ToolRegistry
from oai_agent_core.crewai_core.processing import ResultExtractor


class CrewAIAgent(BaseAgent):
    """A configurable CrewAI agent system for multi-agent workflows.

    This class enables building sophisticated multi-agent workflows through
    YAML configuration files. It supports:
    - Multiple agents with different roles and capabilities (Crew mode)
    - Flow-based workflows defined in Python scripts (Flow mode)
    - Task dependencies and context sharing
    - Tool integration (CrewAI tools, custom tools, MCP servers)
    - Sequential and hierarchical execution patterns
    - Langfuse integration for observability

    Attributes:
        execution_mode: Either 'crew' or 'flow'
        agent_builder: Handles construction of agents and crews (crew mode)
        flow_builder: Handles construction of flows (flow mode)
        result_extractor: Handles extraction and formatting of results
        crew: CrewAI crew instance for multi-agent coordination (crew mode)
        flow: CrewAI flow instance for workflow coordination (flow mode)
    """

    def __init__(self, agent_name: str,
                 agent_config: Optional[Dict[str, Any]] = None,
                 llm: LLM = None,
                 tools: List[Any] = None,
                 session_id: str = "default",
                 user_id: str = "default",
                 config_root: Optional[str] = None,
                 document_loader: Optional[Any] = None,
                 vector_store: Optional[Any] = None,
                 **kwargs):
        """Initialize the CrewAI agent.

        Args:
            agent_name: Unique identifier for the agent
            agent_config: YAML configuration dictionary
            llm: Optional LLM instance (defaults to Claude Haiku)
            tools: Legacy parameter (use config-based tools instead)
            session_id: Session identifier for output tracking
            user_id: User identifier for tracing
            config_root: Root directory for configuration files
            document_loader: Optional document loader instance
            vector_store: Optional vector store instance
            **kwargs: Additional arguments passed to BaseAgent
        """
        # Initialize utility managers
        self.agent_type = Constants.CREWAI
        self.model_manager = ModelConfigurationManager()

        # Initialize base agent
        super().__init__(llm=llm,
                         model_manager=self.model_manager,
                         agent_name=agent_name,
                         agent_config=agent_config,
                         session_id=session_id,
                         config_root=config_root,
                         agent_type=self.agent_type,
                         document_loader=document_loader,
                         vector_store=vector_store,
                         **kwargs)

        # Initialize CrewAI-specific attributes
        self.crew = None
        self.flow = None
        self.user_id = user_id
        self.global_kb_factory = None

        # Determine execution mode (crew or flow)
        self.execution_mode = self.agent_config.get('mode', 'crew').lower()

        if self.execution_mode not in ['crew', 'flow']:
            self.logger.warning(
                f"Invalid mode '{self.execution_mode}'. Defaulting to 'crew'."
            )
            self.execution_mode = 'crew'

        # Initialize utility components (shared by both modes)
        self.tool_registry = ToolRegistry(logger=self.logger, project_root=config_root)
        self.output_serializer = OutputSerializer(logger=self.logger)
        self.message_formatter = MessageFormatter(logger=self.logger)

        # Initialize mode-specific builder
        if self.execution_mode == 'flow':
            self.logger.info(f"Initializing agent in FLOW mode")
            self.flow_builder = FlowBuilder(
                config=self.agent_config,
                tool_registry=self.tool_registry,
                llm=self.llm,
                logger=self.logger,
                project_root=config_root
            )
            self.builder = self.flow_builder
        else:
            self.logger.info(f"Initializing agent in CREW mode")
            self.agent_builder = CrewBuilder(
                config=self.agent_config,
                tool_registry=self.tool_registry,
                llm=self.llm,
                logger=self.logger,
                document_loader=self.document_loader,
                vector_store=self.vector_store,
                structured_output_model_registry=self.output_model_registry
            )
            self.builder = self.agent_builder

        # Initialize result extractor (shared)
        self.result_extractor = ResultExtractor(
            output_serializer=self.output_serializer,
            llm=self.llm,
            logger=self.logger
        )

    async def initialize(self, session_id: str = None) -> Crew | Flow:
        """Initialize the CrewAI crew or flow from configuration.

        This method creates the execution context based on the mode:
        - Crew mode: Creates agents, tasks, and their relationships
        - Flow mode: Loads Python script and instantiates flow class

        Args:
            session_id: Optional session identifier for output tracking

        Returns:
            Configured CrewAI Crew or Flow instance ready for execution

        Raises:
            ValueError: If configuration is invalid or cannot be loaded
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())

        from oai_agent_core.crewai_core.components import KnowledgeBaseFactory

        # Load tools, MCP config, and KB concurrently
        await self._load_tools_and_kb_and_memory(kb_factory_class=KnowledgeBaseFactory)

        # Initialize based on execution mode
        if self.execution_mode == 'flow':
            return await self._initialize_flow(session_id)
        else:
            return await self._initialize_crew(session_id)

    async def _initialize_crew(self, session_id: str) -> Crew:
        """Initialize crew mode execution.

        Args:
            session_id: Session identifier for output tracking

        Returns:
            Configured Crew instance
        """
        step_callback = lambda x: self.output_serializer.write_to_session(x, session_id)

        # Use async build_crew to load agents in parallel
        crew = await self.agent_builder.build_crew_async(
            session_id=session_id,
            step_callback=step_callback
        )

        self.crew = crew
        self._initialized = True
        return crew

    async def _initialize_flow(self, session_id: str) -> Flow:
        """Initialize flow mode execution.

        Args:
            session_id: Session identifier for output tracking

        Returns:
            Configured Flow instance
        """
        flow = self.flow_builder.build_flow(
            session_id=session_id,
            inputs={}  # Inputs will be set during kickoff
        )

        self.flow = flow
        self._initialized = True
        return flow

    async def process_request(self, user_message: str,
                              config: Optional[Dict[str, Any]] = None,
                              async_mode: bool = True) -> Dict[str, Any]:
        """Execute the crew or flow with the given inputs.

        This is the main execution method that processes user requests through
        the configured CrewAI workflow (crew or flow mode).

        Args:
            user_message: User input message or dictionary of inputs
            config: Optional configuration with custom inputs
            async_mode: Whether to use async execution (default: True)

        Returns:
            Dictionary containing session_id, result, and final flag

        Raises:
            ValueError: If execution fails due to configuration issues
        """
        # Ensure Langfuse is set up for this execution
        self.langfuse_manager.initialize_client()

        input_message = self._guardrail_input_message(user_message)

        # Determine input variables for execution
        inputs = self.message_formatter.get_inputs(input_message, self.agent_config)

        if self.global_kb_factory or self.memory_store:
            # If inputs is just a string (unlikely for CrewAI but possible), treat it as the message
            if isinstance(inputs, str):
                inputs = self._augment_message(inputs, original_query=input_message)
            elif isinstance(inputs, dict):
                context_str = ""

                # 1. Knowledge Base
                if self.global_kb_factory:
                    try:
                        kb_result = self.global_kb_factory.search_custom_knowledge_base(input_message)
                        context_str += f"\n\nRelevant Context from Knowledge Base:\n{kb_result}"
                    except Exception as e:
                        self.logger.warning(f"Failed to search knowledge base: {e}")

                # 2. Memory
                if self.memory_store:
                    mem_context = self._get_conversation_context(current_message=input_message)
                    if mem_context:
                        context_str = f"{mem_context}\n\n{context_str}" if context_str else mem_context

                if context_str:
                    # Inject into inputs
                    injected = False
                    # Priority keys to inject context
                    priority_keys = ['topic', 'task', 'query', 'input', 'content', 'description']

                    for key in priority_keys:
                        if key in inputs and isinstance(inputs[key], str):
                            inputs[key] = f"{inputs[key]}\n\n{context_str}"
                            injected = True
                            break

                    if not injected:
                        # Fallback: append to first string value
                        for key, value in inputs.items():
                            if isinstance(value, str):
                                inputs[key] = f"{value}\n\n{context_str}"
                                injected = True
                                break

                    if not injected:
                        # Fallback: add as new key
                        inputs['context'] = context_str

        try:
            # Execute with or without tracing based on Langfuse availability
            if self.langfuse_manager.is_enabled:
                result = await self._execute_with_tracing(async_mode, inputs, input_message)
            else:
                result = await self._execute_without_tracing(async_mode, inputs)

            formatted_result = self.result_extractor.format_execution_result(
                session_id=self.session_id,
                result=result,
                final=True,
                input_message=input_message
            )

            # Save turn to memory store
            if self.memory_store:
                # Extract the actual text content from the result
                agent_response = formatted_result.get('result', '')
                # If result is complex, try to get a string representation
                if not isinstance(agent_response, str):
                    agent_response = str(agent_response)

                self.memory_store.add_turn(
                    session_id=self.session_id,
                    user_id=self.user_id,
                    user_message=user_message,
                    agent_response=agent_response
                )

            return formatted_result

        except Exception as e:
            # Log error to observability platform
            if self.langfuse_manager.is_enabled:
                self.langfuse_manager.log_error(
                    user_message=user_message,
                    error=e,
                    metadata={
                        'session_id': self.session_id,
                        'mode': self.execution_mode
                    }
                )

            self.logger.error(f"Error executing {self.execution_mode}: {e}")

            # Provide helpful error message
            error_details = self.result_extractor.extract_error_details(e, inputs)

            if "No task outputs available" in str(e):
                raise ValueError(
                    f"CrewAI {self.execution_mode} execution failed. "
                    f"{error_details.get('error_message')}\n"
                    f"Suggestions: {error_details.get('suggestions', [])}\n"
                    f"Inputs provided: {inputs}"
                )
            raise

    async def _execute_without_tracing(self, async_mode: bool, inputs: Dict[str, Any]) -> Any:
        """Execute crew/flow without Langfuse tracing.

        Args:
            async_mode: Whether to use async execution
            inputs: Input variables for execution

        Returns:
            CrewAI execution result
        """
        # Initialize the execution context (crew or flow — both use kickoff_async/kickoff)
        execution_context = await self.initialize(session_id=self.session_id)

        if async_mode:
            result = await execution_context.kickoff_async(inputs=inputs)
        else:
            result = execution_context.kickoff(inputs=inputs)

        return result

    async def _execute_with_tracing(self, async_mode: bool, inputs: Dict[str, Any], user_message: str) -> Any:
        """Execute crew/flow with Langfuse tracing enabled.

        Creates a Langfuse trace to monitor execution performance and capture
        metadata for observability and debugging.

        Args:
            async_mode: Whether to use async execution
            inputs: Input variables for execution
            user_message: Original user message for tracing

        Returns:
            CrewAI execution result
        """
        with self.langfuse_manager.trace_generation(
                input_data=user_message,
                user_id=self.user_id,
                session_id=self.session_id,
                metadata={
                    'agent_name': self.agent_name,
                    'mode': self.execution_mode,
                    'pattern': self.agent_config.get('crew_config', {}).get('pattern', 'single')
                    if self.execution_mode == 'crew' else 'flow'
                },
                name_suffix=f"crewai-{self.execution_mode}-{self.agent_name}"
        ) as span:
            # Update trace with user context
            self.langfuse_manager.update_trace(
                span=span,
                input_data=user_message,
                output_data='',
                user_id=self.user_id,
                session_id=self.session_id,
                tags=[self.agent_name, 'crewai', self.execution_mode],
            )

            # Execute the crew/flow (both use the same kickoff API)
            execution_context = await self.initialize(session_id=self.session_id)

            if async_mode:
                result = await execution_context.kickoff_async(inputs=inputs)
            else:
                result = execution_context.kickoff(inputs=inputs)

            # Update span with execution results
            response = self.result_extractor.get_response(
                session_id=self.session_id,
                final_result=result
            )

            self.langfuse_manager.update_trace(
                span=span,
                input_data=user_message,
                output_data=response,
                user_id=self.user_id,
                session_id=self.session_id,
                tags=[self.agent_name, 'crewai', self.execution_mode],
                metadata={
                    'agent_name': self.agent_name,
                    'mode': self.execution_mode,
                    'inputs': inputs,
                    'output': str(result)
                }
            )

        return result

    def validate_tasks(self) -> Dict[str, Any]:
        """Validate configuration and return diagnostic information.

        Performs comprehensive validation of the agent configuration,
        checking for missing references, invalid dependencies, and
        extracting metadata for debugging and optimization.

        Returns:
            Dictionary containing validation results and configuration summary
        """
        if self.execution_mode == 'flow':
            return self.flow_builder.validate_flow_configuration()
        else:
            return self.agent_builder.validate_configuration()

    @traced("agent.ainvoke")
    async def ainvoke(self, user_message: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Asynchronous execution of the crew/flow.

        Main entry point for executing the CrewAI workflow asynchronously.
        Processes the user input and returns the complete response.

        Args:
            user_message: User input message or dictionary of inputs
            config: Optional configuration with custom inputs

        Returns:
            Dictionary containing the formatted response with content,
            token usage, and model information
        """
        result = await self.process_request(user_message, config)
        session_id = result.get('session_id')
        final_result = result.get('result')
        input_message = result.get('input_message')
        original_message = config.get('original_message', user_message) if config else user_message

        response = self.result_extractor.get_response(
            session_id=session_id,
            final_result=final_result
        )

        response = self.result_extractor.format_response(
            response,
            session_id=self.session_id,
            model_id=getattr(self.llm, 'model', 'unknown'),
            model_provider=self.agent_config.get('cloud_provider', 'crewai'),
            include_raw=config.get('include_raw', False) if config else False,
            input_message=input_message if config and config.get('include_input_message', False) else None,
            original_message=original_message if config and config.get('include_original_message', False) else None
        )

        response['content']['text'] = self._guardrail_output_message(response['content']['text'])
        return response

    def invoke(self, user_message: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronous execution of the crew/flow.

        Wrapper around ainvoke for synchronous usage. Runs the async
        execution in an event loop and returns the result.

        Args:
            user_message: User input message or dictionary of inputs
            config: Optional configuration with custom inputs

        Returns:
            Dictionary containing the complete response
        """
        result = asyncio.run(self.ainvoke(user_message, config))
        return result

    @traced("agent.astream")
    async def astream(
            self,
            user_message: str,
            config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute crew/flow and stream output from session storage.

        Args:
            user_message: User input message
            config: Optional configuration

        Yields:
            Dictionary chunks with step outputs
        """
        result = await self.process_request(user_message, config)
        session_id = result.get('session_id')
        final_result = result.get('result')
        input_message = result.get('input_message')
        original_message = config.get('original_message', user_message) if config else user_message

        # Stream using ResultExtractor
        async for chunk in self.result_extractor.stream_response(session_id, final_result):
            response = self.result_extractor.format_response(
                chunk,
                session_id=self.session_id,
                model_id=getattr(self.llm, 'model', 'unknown'),
                model_provider=self.agent_config.get('cloud_provider', 'crewai'),
                include_raw=config.get('include_raw', False) if config else False,
                input_message=input_message if config and config.get('include_input_message', False) else None,
                original_message=original_message if config and config.get('include_original_message', False) else None
            )
            response['content']['text'] = self._guardrail_output_message(response['content']['text'])
            yield response

    async def stream(
            self,
            user_message: str,
            config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Stream execution - CrewAI has limited streaming support.

        Falls back to regular invoke with logged warning.

        Args:
            user_message: User input message
            config: Optional configuration

        Returns:
            Complete response (non-streaming)
        """
        self.logger.warning(
            f"CrewAI {self.execution_mode} mode does not support true streaming. "
            f"Using regular invoke instead. Consider using astream() for step-by-step output."
        )
        return await self.ainvoke(user_message, config)

    def __repr__(self) -> str:
        """String representation of the agent."""
        return (
            f"CrewAIAgent(name='{self.agent_name}', "
            f"mode='{self.execution_mode}', "
            f"session='{self.session_id}', "
            f"initialized={self._initialized})"
        )

    def __del__(self):
        """Cleanup when agent is destroyed."""
        try:
            # Flush Langfuse events on cleanup
            if self.langfuse_manager.is_enabled:
                self.langfuse_manager.flush()
        except Exception:
            pass  # Ignore errors during cleanup

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the configured agent.

        Returns:
            Dictionary with agent configuration details
        """
        info = {
            'agent_name': self.agent_name,
            'execution_mode': self.execution_mode,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'initialized': self._initialized,
            'has_langfuse': self.langfuse_manager.is_enabled,
            'model': {
                'name': self.llm.model,
                'provider': getattr(self.llm, 'provider', 'unknown')
            },
            'config_summary': self.validate_tasks()
        }

        # Add mode-specific info
        if self.execution_mode == 'flow':
            info['flow_metadata'] = self.flow_builder.get_flow_metadata()

        return info
