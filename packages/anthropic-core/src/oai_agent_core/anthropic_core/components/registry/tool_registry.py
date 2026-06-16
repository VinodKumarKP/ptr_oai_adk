"""Tool registry for claude-agent-sdk.

claude-agent-sdk handles MCP natively — we just collect MCP server configs
from the YAML and pass them to ``ClaudeAgentOptions.mcp_servers``.

Custom Python tools from the YAML ``tools:`` block are converted to an
in-process FastMCP server so they are available to claude-agent-sdk as
a standard MCP server under the key ``"__custom_tools__"``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

from oai_agent_core.core.base_tool_registry import BaseToolRegistry
from oai_agent_core.utils.dynamic_class_loader import DynamicClassLoader


class AnthropicToolRegistry(BaseToolRegistry):
    """Registry that converts YAML tool/MCP config to claude-agent-sdk format.

    Attributes:
        mcp_server_configs: name → MCP server config dict (stdio or remote)
        custom_tools:       name → callable (Python tools wrapped via FastMCP)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Collected MCP configs ready for ClaudeAgentOptions.mcp_servers
        self.mcp_server_configs: Dict[str, Dict[str, Any]] = {}
        # Custom Python callables — wrapped as a FastMCP server at build time
        self.custom_tools: Dict[str, Callable] = {}
        # Cache the FastMCP server instance once started
        self._fastmcp_server = None

    # ── MCP config collection ─────────────────────────────────────────────────

    def register_mcp_config(self, name: str, cfg: Dict[str, Any]) -> None:
        """Store one MCP server config in McpServerConfig format for ClaudeAgentOptions."""
        if "command" in cfg:
            # McpStdioServerConfig — type field is optional for stdio
            entry: Dict[str, Any] = {"command": cfg["command"]}
            if cfg.get("args"):
                entry["args"] = cfg["args"]
            if cfg.get("env"):
                entry["env"] = cfg["env"]
            self.mcp_server_configs[name] = entry
        elif "url" in cfg:
            # McpHttpServerConfig or McpSSEServerConfig
            # Default to "http"; caller can override with cfg["type"] = "sse"
            entry = {
                "type": cfg.get("type", "http"),
                "url": cfg["url"],
            }
            if cfg.get("headers"):
                entry["headers"] = cfg["headers"]
            self.mcp_server_configs[name] = entry
        else:
            self.logger.warning("MCP '%s' has neither 'command' nor 'url' — skipping.", name)

    def load_mcp_configs(self, mcp_block: Any) -> None:
        """Bulk-register all MCPs from the YAML ``mcps:`` block.

        Accepts:
          dict  — {server_name: config_dict, ...}  (global mcps: block)
          list  — [server_name, ...]  (agent-level reference to global MCPs;
                  configs are already registered — this is a no-op here)
        """
        if isinstance(mcp_block, list):
            # Agent-level list of names — nothing to register; wildcards are
            # added by the caller (AgentBuilder.build_agent_definition).
            return
        for name, cfg in mcp_block.items():
            self.register_mcp_config(name, cfg)

    def get_mcp_server_configs(self) -> Dict[str, Any]:
        """Return all registered MCP configs for ClaudeAgentOptions.mcp_servers.

        Includes both external MCP servers (stdio/http) and the in-process
        SDK MCP server created by build_sdk_mcp_server() — which is already
        stored under the key "custom_tools" in mcp_server_configs.
        """
        return dict(self.mcp_server_configs)

    # ── Custom Python tool registration ──────────────────────────────────────

    def register_tool_from_config(self, tool_name: str, tool_config: Dict[str, Any]) -> None:
        """Load a Python tool from YAML config and store it for SDK MCP server wrapping."""
        module_path = tool_config.get("module")
        class_name = tool_config.get("class")
        func_name = tool_config.get("function")

        try:
            # DynamicClassLoader has only static methods — do not instantiate it
            if func_name:
                module = DynamicClassLoader.dynamic_import_module(module_path)
                func = getattr(module, func_name, None)
            elif class_name:
                cls = DynamicClassLoader.dynamic_import(module_path, class_name)
                instance = cls()
                func = getattr(instance, "run", None) or getattr(instance, "__call__", None)
            else:
                return

            if func and callable(func):
                self.custom_tools[tool_name] = func
                self.logger.debug("Custom tool '%s' registered for SDK MCP wrapping.", tool_name)
        except Exception as exc:
            self.logger.warning("Could not load tool '%s': %s", tool_name, exc)

    # ── Module-level tool loading (overrides base class) ─────────────────────

    def _load_tools_from_module(
        self,
        module_name: str,
        function_list=None,
        function_params=None,
        tool_key: str = None,
    ) -> None:
        """Load all public functions from *module_name* into ``custom_tools``.

        Overrides the base-class implementation so that loaded callables land
        in ``self.custom_tools`` (executor map) instead of ``self.tools``
        (LiteLLM-format dict).  ``build_sdk_mcp_server()`` later wraps them
        as a native claude-agent-sdk in-process MCP server.
        """
        import sys
        import os

        # Resolve base_path — already inserted into sys.path by base class
        # _load_module_tool before calling us.  Re-insert just in case.
        if tool_key and tool_key not in sys.modules:
            pass  # sys.path already set by _resolve_base_path in base class

        try:
            module = DynamicClassLoader.dynamic_import_module(module_name)
        except ImportError as exc:
            self.logger.error("Cannot import module '%s': %s", module_name, exc)
            return

        # Collect all public callables (skip classes and private names)
        functions = self.get_module_functions(module, function_list or [])
        if not functions:
            self.logger.warning(
                "No functions found in module '%s'. Ensure they are public callables.",
                module_name,
            )
            return

        for func in functions:
            name = self._get_function_name(func)

            # Unwrap framework decorators (@tool from Strands, etc.)
            raw = getattr(func, "__wrapped__", None) or func
            if callable(raw):
                self.custom_tools[name] = raw
                self.logger.debug(
                    "Loaded custom tool '%s' from module '%s'.", name, module_name
                )

        self.logger.info(
            "Loaded %d tool(s) from '%s': %s",
            len(functions),
            module_name,
            list(self.custom_tools.keys()),
        )

    def build_sdk_mcp_server(self) -> Optional[Any]:
        """Create a native claude-agent-sdk in-process MCP server from custom_tools.

        Returns the ``McpSdkServerConfig`` or None if there are no custom tools.
        Once called, the server config is accessible via ``get_mcp_server_configs()``.
        """
        if not self.custom_tools:
            return None

        try:
            from claude_agent_sdk import tool as sdk_tool, create_sdk_mcp_server
            import json
        except ImportError:
            self.logger.warning(
                "claude-agent-sdk not installed — custom Python tools unavailable."
            )
            return None

        sdk_tools = []
        for tool_name, func in self.custom_tools.items():
            doc = inspect.getdoc(func) or tool_name
            description = doc.split("\n")[0]

            # Build input schema from function signature
            sig = inspect.signature(func)
            properties: Dict[str, Any] = {}
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue
                ann = param.annotation
                if ann is inspect.Parameter.empty:
                    ann = str
                properties[pname] = ann

            # Capture loop variable in closure
            _func = func
            _name = tool_name

            @sdk_tool(_name, description, properties)
            async def _handler(args, _f=_func):
                try:
                    import asyncio
                    if asyncio.iscoroutinefunction(_f):
                        result = await _f(**args)
                    else:
                        result = await asyncio.to_thread(_f, **args)
                    # Serialize result to string for MCP text content
                    if isinstance(result, str):
                        text = result
                    else:
                        text = json.dumps(result, default=str)
                    return {"content": [{"type": "text", "text": text}]}
                except Exception as exc:
                    return {
                        "content": [{"type": "text", "text": f"Error: {exc}"}],
                        "is_error": True,
                    }

            sdk_tools.append(_handler)
            self.logger.debug("Wrapped '%s' as SDK MCP tool.", tool_name)

        server_config = create_sdk_mcp_server("custom_tools", tools=sdk_tools)
        self.mcp_server_configs["custom_tools"] = server_config
        self._fastmcp_server = server_config

        self.logger.info(
            "Built SDK MCP server 'custom_tools' with %d tool(s).", len(sdk_tools)
        )
        return server_config

    # ── Tool system prompt (fallback for KB / skills) ─────────────────────────

    def get_tool_names_for_allowed_list(self) -> List[str]:
        """Return MCP server wildcard entries for ClaudeAgentOptions.allowed_tools.

        claude-agent-sdk names MCP tools as ``mcp__<server>__<tool>``.
        Wildcard ``mcp__<server>__*`` auto-allows every tool on a server.
        SDK MCP servers (type="sdk") are included automatically.
        """
        wildcards = []
        for name, cfg in self.mcp_server_configs.items():
            # SDK MCP server (McpSdkServerConfig) or stdio/http config dict
            wildcards.append(f"mcp__{name}__*")
        return wildcards

    # ── BaseToolRegistry abstract method implementations ──────────────────────

    def load_mcp_tools_from_config(
        self, mcp_config: Dict[str, Any], agent_name: Optional[str] = None
    ) -> Any:
        """Register MCP server configs from a config dict.

        In the claude-agent-sdk model, MCP servers are configured once in
        ``ClaudeAgentOptions.mcp_servers`` rather than loaded as individual
        tool objects.  This method stores the configs so they can be injected
        into ``ClaudeAgentOptions`` at invocation time.
        """
        self.load_mcp_configs(mcp_config)
        self.logger.debug(
            "load_mcp_tools_from_config: registered %d MCP server(s) for agent '%s'.",
            len(mcp_config),
            agent_name or "global",
        )
        return list(mcp_config.keys())

    def get_input_parameter_schema(self, tool_list: str) -> str:
        """Return a JSON-schema description of the named tools.

        ``tool_list`` is a comma-separated string of tool names.  We return a
        compact schema block suitable for injecting into a system prompt when
        the lazy-loading pattern is in use.
        """
        import json

        names = [t.strip() for t in tool_list.split(",") if t.strip()]
        schemas: Dict[str, Any] = {}
        for name in names:
            # MCP tools: report the server they belong to
            server = name.split("__")[1] if "__" in name else None
            schemas[name] = {
                "type": "mcp_tool",
                "server": server,
                "description": f"Tool '{name}' exposed by MCP server '{server}'." if server else name,
            }
        return json.dumps(schemas, indent=2)

    def _get_function_tool_type(self):
        return dict

    def _get_framework_tool_decorator(self):
        return lambda f: f

    def _is_framework_builtin_tool(self, module_name: str) -> bool:
        return False

    def _load_framework_builtin_tool(self, tool_name: str, module_name: str) -> None:
        pass

    def _is_framework_tool_type(self, obj: Any) -> bool:
        return isinstance(obj, dict)

    def _wrap_function_with_defaults(self, func, defaults):
        return func

    async def execute_tool(self, tool_name: str, arguments: Any) -> Any:
        """Execute a custom Python tool by name.

        MCP tools are executed by claude-agent-sdk itself.  This method is only
        called for custom Python tools registered via ``register_tool_from_config``.
        """
        import json

        func = self.custom_tools.get(tool_name)
        if not func:
            return f"Error: tool '{tool_name}' not found in custom tool registry."

        args = arguments if isinstance(arguments, dict) else {}
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                args = {"input": arguments}

        try:
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return await func(**args)
            return await asyncio.to_thread(func, **args)
        except Exception as exc:
            self.logger.error("Tool '%s' raised: %s", tool_name, exc, exc_info=True)
            return f"Error executing tool '{tool_name}': {exc}"


