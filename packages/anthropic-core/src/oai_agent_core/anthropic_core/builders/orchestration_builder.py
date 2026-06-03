"""Orchestration builder — calls claude_agent_sdk.query() per pattern.

Why claude-agent-sdk instead of a manual tool loop
---------------------------------------------------
- Built-in agentic loop — no manual tool-call iteration.
- Sub-agents as first-class citizens via ``AgentDefinition`` + the ``Agent`` tool.
- MCP servers configured once and available to all agents automatically.
- Extended thinking, hooks, and permissions are single-option flags.
- LiteLLM proxy support: set ``api_base_url`` → routes to Bedrock / Vertex / etc.

Patterns
--------
single            ClaudeAgentOptions with tools and MCP from YAML.
supervisor        Supervisor prompt + sub-agents registered via agents= dict.
                  The supervisor uses the built-in "Agent" tool to delegate.
agent-as-tool     Same as supervisor — in claude-agent-sdk, the Agent tool IS
                  the agent-as-tool pattern.  Sub-agent descriptions guide routing.
extended-thinking Single agent with thinking={"type":"enabled","budget_tokens":N}.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from oai_agent_core.core.constants import Constants

from oai_agent_core.anthropic_core.components.registry.tool_registry import AnthropicToolRegistry


class OrchestrationBuilder:
    """Builds ``ClaudeAgentOptions`` and drives ``claude_agent_sdk.query()``."""

    def __init__(
        self,
        tool_registry: AnthropicToolRegistry,
        logger: Optional[logging.Logger] = None,
        langfuse_manager: Optional[Any] = None,
        output_model_registry: Optional[Any] = None,
    ):
        self.tool_registry = tool_registry
        self.logger = logger or logging.getLogger(__name__)
        self.langfuse_manager = langfuse_manager          # LangfuseObservabilityManager
        self.output_model_registry = output_model_registry  # OutputModelRegistry

    # ── Public entry points ───────────────────────────────────────────────────

    async def invoke(
        self,
        pattern: str,
        agent_definitions: Dict[str, Any],
        user_message: str,
        model_kwargs: Dict[str, Any],
        system_prompt: str,
        entry_agent: Optional[str] = None,
        anthropic_config: Optional[Dict[str, Any]] = None,
        structured_output_model: Optional[str] = None,
        session_id: str = "default",
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Run the agent to completion and return the final result dict."""
        from claude_agent_sdk import query

        options = self._build_options(
            pattern=pattern,
            agent_definitions=agent_definitions,
            model_kwargs=model_kwargs,
            system_prompt=system_prompt,
            anthropic_config=anthropic_config or {},
            structured_output_model=structured_output_model,
            session_id=session_id,
            user_id=user_id,
            entry_agent=entry_agent,
        )

        collected_text = ""
        result_message = None

        async for message in query(prompt=user_message, options=options):
            result_message = message
            # Accumulate any text content surfaced during execution
            text = self._extract_message_text(message)
            if text:
                collected_text = text   # take the latest (final) text

        return {
            "text": collected_text,
            "raw": result_message,
        }

    async def stream(
        self,
        pattern: str,
        agent_definitions: Dict[str, Any],
        user_message: str,
        model_kwargs: Dict[str, Any],
        system_prompt: str,
        entry_agent: Optional[str] = None,
        anthropic_config: Optional[Dict[str, Any]] = None,
        structured_output_model: Optional[str] = None,
        session_id: str = "default",
        user_id: str = "default",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream messages from claude-agent-sdk as ADK chunk dicts."""
        from claude_agent_sdk import query

        options = self._build_options(
            pattern=pattern,
            agent_definitions=agent_definitions,
            model_kwargs=model_kwargs,
            system_prompt=system_prompt,
            anthropic_config=anthropic_config or {},
            structured_output_model=structured_output_model,
            session_id=session_id,
            user_id=user_id,
            entry_agent=entry_agent,
        )

        async for message in query(prompt=user_message, options=options):
            text = self._extract_message_text(message)
            is_final = self._is_final_message(message)

            if text:
                yield {
                    "content": {"text": text, "type": "AIMessage"},
                    "final": is_final,
                    "raw": message,
                }
            elif is_final:
                # Yield a final marker even with empty text
                yield {"content": {"text": "", "type": "AIMessage"}, "final": True, "raw": message}

    # ── ClaudeAgentOptions builder ────────────────────────────────────────────

    def _build_options(
        self,
        pattern: str,
        agent_definitions: Dict[str, Any],
        model_kwargs: Dict[str, Any],
        system_prompt: str,
        anthropic_config: Dict[str, Any],
        structured_output_model: Optional[str] = None,
        session_id: str = "default",
        user_id: str = "default",
        entry_agent: Optional[str] = None,
    ) -> "ClaudeAgentOptions":
        from claude_agent_sdk import ClaudeAgentOptions

        pattern = (pattern or "single").lower()

        # ── Base options ──────────────────────────────────────────────────────
        options_kwargs: Dict[str, Any] = {"system_prompt": system_prompt}

        if model_kwargs.get("model"):
            options_kwargs["model"] = model_kwargs["model"]
        if model_kwargs.get("env"):
            options_kwargs["env"] = model_kwargs["env"]

        mcp_configs = self.tool_registry.get_mcp_server_configs()
        if mcp_configs:
            options_kwargs["mcp_servers"] = mcp_configs

        allowed_tools = self.tool_registry.get_tool_names_for_allowed_list()

        # ── Extended thinking ─────────────────────────────────────────────────
        if pattern == "extended-thinking":
            budget = anthropic_config.get("thinking_budget_tokens", 8000)
            options_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            self.logger.info("Extended thinking enabled: budget=%d tokens", budget)

        # ── Supervisor / agent-as-tool ────────────────────────────────────────
        if pattern in (Constants.PATTERN_SUPERVISOR, Constants.PATTERN_AGENT_AS_TOOL):
            if agent_definitions:
                # For agent-as-tool: exclude entry agent from callable agents
                # (entry agent is the main agent that calls others)
                agents_to_register = dict(agent_definitions)

                if pattern == Constants.PATTERN_AGENT_AS_TOOL and entry_agent:
                    agents_to_register.pop(entry_agent, None)
                    self.logger.debug(
                        "Agent-as-tool: entry_agent '%s' removed from callable agents dict",
                        entry_agent
                    )

                if agents_to_register:
                    options_kwargs["agents"] = agents_to_register
                    allowed_tools = list(set(allowed_tools + ["Agent"]))
                    self.logger.info(
                        "Pattern '%s': %d callable agent(s) registered.", pattern, len(agents_to_register)
                    )
                else:
                    self.logger.warning(
                        "Pattern '%s': no callable agents after filtering. entry_agent=%s",
                        pattern,
                        entry_agent
                    )

        # ── Structured output ─────────────────────────────────────────────────
        # Resolve the named Pydantic model → JSON schema → output_format
        if structured_output_model and self.output_model_registry:
            model_cls = self.output_model_registry.get_model(structured_output_model)
            if model_cls and hasattr(model_cls, "model_json_schema"):
                schema = model_cls.model_json_schema()
                options_kwargs["output_format"] = {"type": "json_schema", "schema": schema}
                self.logger.info(
                    "Structured output enabled: model=%s", structured_output_model
                )

        # ── Langfuse observability hooks ──────────────────────────────────────
        if self.langfuse_manager and self.langfuse_manager.is_enabled:
            hooks = self._build_langfuse_hooks(session_id=session_id, user_id=user_id)
            if hooks:
                options_kwargs["hooks"] = hooks

        # ── Beta features ─────────────────────────────────────────────────────
        betas: List[str] = anthropic_config.get("betas", [])
        if betas:
            options_kwargs["betas"] = betas

        if allowed_tools:
            options_kwargs["allowed_tools"] = allowed_tools

        self.logger.debug(
            "ClaudeAgentOptions: pattern=%s model=%s tools=%s observability=%s",
            pattern,
            options_kwargs.get("model", "default"),
            options_kwargs.get("allowed_tools", []),
            bool(options_kwargs.get("hooks")),
        )

        return ClaudeAgentOptions(**options_kwargs)

    # ── Langfuse hooks ────────────────────────────────────────────────────────

    def _build_langfuse_hooks(
        self, session_id: str, user_id: str
    ) -> Optional[Dict[str, List["HookMatcher"]]]:
        """Build claude-agent-sdk hooks that write tool-call spans to Langfuse.

        Hooks created:
          PreToolUse  — open a Langfuse span for the tool call
          PostToolUse — close the span with result and duration
          Stop        — flush cost / usage to the top-level trace
        """
        try:
            from claude_agent_sdk import HookMatcher
            from claude_agent_sdk.types import SyncHookJSONOutput
        except ImportError:
            return None

        mgr = self.langfuse_manager
        _span_store: Dict[str, Any] = {}   # tool_use_id → (span, start_time)

        # ── PreToolUse ──────────────────────────────────────────────────────
        async def _pre_tool_use(hook_input, tool_use_id, ctx) -> SyncHookJSONOutput:
            import time
            try:
                mgr.initialize_client()
                if mgr.is_enabled and mgr.client:
                    span = mgr.client.start_as_current_observation(
                        as_type="span",
                        name=f"tool:{hook_input.tool_name}",
                        input=hook_input.tool_input,
                        metadata={
                            "session_id": session_id,
                            "user_id": user_id,
                            "tool_use_id": tool_use_id,
                        },
                    )
                    _span_store[tool_use_id or ""] = (span, time.monotonic())
            except Exception as exc:
                self.logger.debug("Langfuse PreToolUse hook error: %s", exc)
            return {"continue_": True}

        # ── PostToolUse ─────────────────────────────────────────────────────
        async def _post_tool_use(hook_input, tool_use_id, ctx) -> SyncHookJSONOutput:
            import time
            try:
                entry = _span_store.pop(tool_use_id or "", None)
                if entry:
                    span, t0 = entry
                    duration_ms = int((time.monotonic() - t0) * 1000)
                    span.__exit__(None, None, None)
                    self.logger.debug(
                        "Tool '%s' completed in %dms", hook_input.tool_name, duration_ms
                    )
            except Exception as exc:
                self.logger.debug("Langfuse PostToolUse hook error: %s", exc)
            return {"continue_": True}

        # ── Stop ────────────────────────────────────────────────────────────
        async def _stop(hook_input, tool_use_id, ctx) -> SyncHookJSONOutput:
            try:
                mgr.flush()
            except Exception as exc:
                self.logger.debug("Langfuse Stop hook error: %s", exc)
            return {"continue_": True}

        return {
            "PreToolUse":  [HookMatcher(hooks=[_pre_tool_use])],
            "PostToolUse": [HookMatcher(hooks=[_post_tool_use])],
            "Stop":        [HookMatcher(hooks=[_stop])],
        }

    # ── Message helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_message_text(message: Any) -> str:
        """Pull text from any claude-agent-sdk message type."""
        # ResultMessage has a .result attribute
        if hasattr(message, "result") and message.result:
            return str(message.result)
        # AssistantMessage has .content list
        if hasattr(message, "content"):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                return "\n".join(parts)
        return ""

    @staticmethod
    def _is_final_message(message: Any) -> bool:
        """Return True for the terminal ResultMessage."""
        # claude-agent-sdk yields a ResultMessage as the last event
        return type(message).__name__ in ("ResultMessage", "Result")

    @staticmethod
    def get_supported_patterns() -> List[str]:
        return ["single", "supervisor", "agent-as-tool", "extended-thinking"]
