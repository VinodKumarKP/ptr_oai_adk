"""AnthropicAgent — claude-agent-sdk backed agent for OAI ADK.

Uses claude-agent-sdk for the agentic loop, sub-agents, MCP, and extended
thinking.  Provider routing (Bedrock / Vertex AI / direct Anthropic API) is
handled by a LiteLLM proxy — set ``model.litellm_proxy_url`` in the YAML.

Supported patterns (``crew_config.pattern``):
    single            – one Claude agent with tools, KB, memory, skills
    supervisor        – supervisor delegates to sub-agents via the built-in Agent tool
    agent-as-tool     – sub-agents described as tools; main agent routes automatically
    extended-thinking – single agent with Claude reasoning chain

YAML example::

    type: anthropic
    model:
      model_id: anthropic/claude-opus-4-5-20251101
      litellm_proxy_url: "${LITELLM_PROXY_URL}"   # omit for direct API
      api_key: "${ANTHROPIC_API_KEY}"
      max_tokens: 8096

    anthropic_config:
      prompt_caching: true
      thinking_budget_tokens: 16000   # only for extended-thinking pattern
      betas: []

    crew_config:
      pattern: supervisor
      entry_agent: coordinator
      enable_lazy_loading: true
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, Optional

from oai_agent_core.core.base_agent import BaseAgent
from oai_agent_core.core.constants import Constants
from oai_agent_core.processing.message_formatter import MessageFormatter
from oai_agent_core.processing.output_serializer import OutputSerializer

from oai_agent_core.anthropic_core.builders.agent_builder import AgentBuilder
from oai_agent_core.anthropic_core.builders.orchestration_builder import OrchestrationBuilder
from oai_agent_core.anthropic_core.components.configuration.model_config import AnthropicModelConfigurationManager
from oai_agent_core.anthropic_core.components.registry.tool_registry import AnthropicToolRegistry
from oai_agent_core.anthropic_core.processing.result_extractor import ResultExtractor

_AGENT_TYPE = "anthropic"


class AnthropicAgent(BaseAgent):
    """YAML-configurable Claude agent using claude-agent-sdk + LiteLLM proxy.

    The agent is fully compatible with the OAI ADK contract — it inherits
    ``BaseAgent`` and implements ``initialize``, ``ainvoke``, ``invoke``,
    ``astream``, and ``stream``.
    """

    def __init__(
        self,
        agent_name: str,
        agent_config: Optional[Dict[str, Any]] = None,
        llm=None,
        session_id: str = "default",
        user_id: str = "default",
        config_root: Optional[str] = None,
        document_loader=None,
        vector_store=None,
        **kwargs,
    ):
        self.agent_type = _AGENT_TYPE
        self.model_manager = AnthropicModelConfigurationManager()

        super().__init__(
            llm=llm if llm else {},
            model_manager=self.model_manager,
            agent_name=agent_name,
            agent_config=agent_config,
            session_id=session_id,
            config_root=config_root,
            user_id=user_id,
            agent_type=self.agent_type,
            document_loader=document_loader,
            vector_store=vector_store,
            **kwargs,
        )

        # Fully-wired model manager with config + logger
        self.model_manager = AnthropicModelConfigurationManager(
            default_config=self.agent_config.get("model", {}),
            logger=self.logger,
        )
        # Dict of LiteLLM-proxy-aware kwargs forwarded to ClaudeAgentOptions
        self.model_kwargs: Dict[str, Any] = self.model_manager.create_model()

        self.tool_registry = AnthropicToolRegistry(
            logger=self.logger,
            project_root=config_root,
            enable_lazy_loading=self.agent_config.get("crew_config", {}).get("enable_lazy_loading", False),
        )

        self.message_formatter = MessageFormatter(logger=self.logger)
        self.output_serializer = OutputSerializer(logger=self.logger)
        self.result_extractor = ResultExtractor(logger=self.logger)

        self.agent_builder: Optional[AgentBuilder] = None
        self.orchestration_builder: Optional[OrchestrationBuilder] = None
        # AgentDefinition objects keyed by agent name (sub-agents for multi-agent patterns)
        self.agent_definitions: Dict[str, Any] = {}
        self._entry_system_prompt: str = ""

    # ── Initialization ────────────────────────────────────────────────────────

    async def initialize(self, session_id: Optional[str] = None) -> None:
        """Resolve YAML config into claude-agent-sdk structures.

        Steps:
        1. Load KB, memory, skills via BaseAgent helpers.
        2. Register global MCP server configs in the tool registry.
        3. Register custom Python tools (wrapped as FastMCP server).
        4. Build AgentDefinition objects for all sub-agents.
        5. Wire up the OrchestrationBuilder.
        """
        if session_id:
            self.session_id = session_id

        from oai_agent_core.anthropic_core.components.knowledge.knowledge_base_factory import KnowledgeBaseFactory

        # KB / memory / skills loading via BaseAgent
        await self._load_tools_and_kb_and_memory(kb_factory_class=KnowledgeBaseFactory)

        # Register global MCP server configs (YAML mcps: block)
        global_mcps = self.agent_config.get("mcps", {})
        if global_mcps:
            self.tool_registry.load_mcp_configs(global_mcps)

        # Load custom Python tools from YAML tools: block.
        # load_tools_from_config handles {module, base_path} correctly
        # and populates tool_registry.custom_tools via _load_tools_from_module.
        tools_config = self.agent_config.get("tools", {})
        if tools_config:
            self.tool_registry.load_tools_from_config(tools_config)

        # Wrap custom Python tools as a native claude-agent-sdk in-process
        # MCP server (McpSdkServerConfig) so the CLI can call them.
        self.tool_registry.build_sdk_mcp_server()

        self.agent_builder = AgentBuilder(
            model_manager=self.model_manager,
            tool_registry=self.tool_registry,
            llm=self.model_kwargs,
            config_root=self.config_root,
            logger=self.logger,
            document_loader=self.document_loader,
            vector_store=self.vector_store,
            skill_registry=self.skill_registry,
            structured_output_model_registry=self.output_model_registry,
        )

        agent_list = self.agent_config.get("agent_list", [])

        # For multi-agent patterns: build AgentDefinition for each sub-agent
        crew_config = self.agent_config.get("crew_config", {})
        pattern = crew_config.get("pattern", "single")

        if pattern in (Constants.PATTERN_SUPERVISOR, Constants.PATTERN_AGENT_AS_TOOL):
            self.agent_definitions = await self.agent_builder.build_all_agent_definitions(agent_list)

        # Resolve the top-level system prompt
        self._entry_system_prompt = self.agent_builder.resolve_entry_system_prompt(
            agent_configs=agent_list,
            global_system_prompt=self.agent_config.get("system_prompt"),
            entry_agent=crew_config.get("entry_agent"),
        )

        self.orchestration_builder = OrchestrationBuilder(
            tool_registry=self.tool_registry,
            logger=self.logger,
        )

        self._initialized = True
        self.logger.info(
            "AnthropicAgent '%s' initialized — pattern: %s  proxy: %s",
            self.agent_name,
            pattern,
            self.model_kwargs.get("api_base_url", "direct"),
        )

    # ── Execution ─────────────────────────────────────────────────────────────

    async def process_request(
        self,
        user_message: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the agent and return the raw orchestration result."""
        await self._ensure_initialized()

        formatted = self._prepare_message(user_message, config)
        formatted = self._guardrail_input_message(formatted)
        # Inject KB / memory context into the message
        formatted = self._augment_message(formatted, original_query=user_message)

        crew_config = self.agent_config.get("crew_config", {})

        result = await self.orchestration_builder.invoke(
            pattern=crew_config.get("pattern", "single"),
            agent_definitions=self.agent_definitions,
            user_message=formatted,
            model_kwargs=self.model_kwargs,
            system_prompt=self._entry_system_prompt,
            entry_agent=crew_config.get("entry_agent"),
            anthropic_config=self.agent_config.get("anthropic_config", {}),
        )

        return {
            "session_id": self.session_id,
            "result": result,
            "final": True,
            "input_message": formatted,
        }

    async def ainvoke(
        self,
        user_message: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async invoke — returns a formatted ADK response dict."""
        raw = await self.process_request(user_message, config)
        include_raw = config.get("include_raw", False) if config else False

        response = self.result_extractor.format_response(
            result=raw["result"],
            session_id=self.session_id,
            model_id=self.model_kwargs.get("model", ""),
            model_provider=self.model_manager.get_model_info().get("provider", "anthropic"),
            include_raw=include_raw,
            input_message=raw["input_message"] if config and config.get("include_input_message") else None,
            original_message=user_message if config and config.get("include_original_message") else None,
        )

        if self.memory_store:
            self.memory_store.add_turn(
                session_id=self.session_id,
                user_id=self.user_id,
                user_message=user_message,
                agent_response=response,
            )

        if "content" in response:
            response["content"]["text"] = self._guardrail_output_message(
                response["content"].get("text")
            )

        return response

    def invoke(
        self,
        user_message: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Sync wrapper around ainvoke."""
        return asyncio.run(self.ainvoke(user_message, config))

    async def astream(
        self,
        user_message: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response chunks from claude-agent-sdk."""
        await self._ensure_initialized()

        formatted = self._prepare_message(user_message, config)
        formatted = self._guardrail_input_message(formatted)
        formatted = self._augment_message(formatted, original_query=user_message)

        crew_config = self.agent_config.get("crew_config", {})
        last_chunk = None

        try:
            async for chunk in self.orchestration_builder.stream(
                pattern=crew_config.get("pattern", "single"),
                agent_definitions=self.agent_definitions,
                user_message=formatted,
                model_kwargs=self.model_kwargs,
                system_prompt=self._entry_system_prompt,
                entry_agent=crew_config.get("entry_agent"),
                anthropic_config=self.agent_config.get("anthropic_config", {}),
            ):
                # Apply output guardrail on final chunk
                if chunk.get("final") and "content" in chunk:
                    chunk["content"]["text"] = self._guardrail_output_message(
                        chunk["content"].get("text")
                    )
                last_chunk = chunk
                yield chunk

            if self.memory_store and last_chunk:
                self.memory_store.add_turn(
                    session_id=self.session_id,
                    user_id=self.user_id,
                    user_message=user_message,
                    agent_response=last_chunk,
                )

        except Exception as exc:
            self.logger.error("astream error: %s", exc, exc_info=True)
            yield {"content": {"text": f"Error: {exc}"}, "type": "error", "final": True}

    async def stream(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        raise NotImplementedError("Use astream() for streaming from AnthropicAgent.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prepare_message(self, user_message: str, config: Optional[Dict]) -> str:
        if config and "inputs" in config:
            inputs = config["inputs"]
        else:
            variables = self.message_formatter.extract_variables_from_config(self.agent_config)
            inputs = self.message_formatter.create_default_inputs(user_message, variables)
        formatted = self.message_formatter.format_message(user_message, inputs)
        return formatted if formatted is not None else user_message

    def get_agent_info(self) -> Dict[str, Any]:
        crew_config = self.agent_config.get("crew_config", {})
        return {
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "initialized": self._initialized,
            "pattern": crew_config.get("pattern", "single"),
            "sub_agents": list(self.agent_definitions.keys()),
            "model": self.model_kwargs.get("model", ""),
            "via_proxy": bool(self.model_kwargs.get("api_base_url")),
            "mcp_servers": list(self.tool_registry.mcp_server_configs.keys()),
        }

    def validate_tasks(self) -> Dict[str, Any]:
        return {
            "agent_count": len(self.agent_config.get("agent_list", [])),
            "agents": [list(a.keys())[0] for a in self.agent_config.get("agent_list", [])],
            "pattern": self.agent_config.get("crew_config", {}).get("pattern", "single"),
            "mcp_servers": list(self.tool_registry.mcp_server_configs.keys()),
        }

    async def close(self) -> None:
        pass  # claude-agent-sdk manages its own connection lifecycle

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def __repr__(self) -> str:
        return (
            f"AnthropicAgent(name='{self.agent_name}', "
            f"session='{self.session_id}', "
            f"initialized={self._initialized})"
        )
