"""Tool Registry for OpenAI Agents."""

import functools
import os
import shutil
import sys
from typing import Dict, Any, Callable, Optional, List

from agents.mcp import MCPServerStreamableHttp, MCPServerSse, MCPServerStdio
from oai_agent_core.core.base_tool_registry import BaseToolRegistry


class OpenAIToolRegistry(BaseToolRegistry):
    """Registry for managing tools for OpenAI agents.

    Handles registration, loading, and management of both standard tools
    and MCP (Model Context Protocol) tools.
    """

    def _get_function_tool_type(self):
        """Get the function tool type class.

        Returns:
            The FunctionTool class from the agents library.
        """
        from agents import FunctionTool
        return FunctionTool

    def _wrap_function_with_defaults(self, func: Callable, default_params: Dict[str, Any]) -> Callable:
        """Wrap a function to include default parameter values.

        Args:
            func: The function to wrap.
            default_params: Dictionary of parameter names and their default values.

        Returns:
            Wrapped function with new defaults.
        """
        # For openai-agents library tools, we use partial application or a wrapper
        # This is a simplified implementation

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Update kwargs with defaults if not present
            for key, value in default_params.items():
                if key not in kwargs:
                    kwargs[key] = value
            return func(*args, **kwargs)

        return wrapper

    def _get_framework_tool_decorator(self):
        """Get the framework-specific tool decorator.

        Returns:
            The tool decorator function (agents.function_tool).
        """
        from agents import function_tool
        return function_tool

    def _is_framework_builtin_tool(self, module_name: str) -> bool:
        """Check if the module is a framework built-in tool.

        Args:
            module_name: Name of the module.

        Returns:
            True if it's a framework built-in tool.
        """
        # Add logic to identify built-in tools if needed
        return False

    def _load_framework_builtin_tool(self, tool_name: str, module_name: str) -> None:
        """Load a framework-specific built-in tool.

        Args:
            tool_name: Name of the tool.
            module_name: Module name of the framework tool.
        """
        pass

    def _is_framework_tool_type(self, obj: Any) -> bool:
        """Check if an object is a framework-specific tool type.

        Args:
            obj: Object to check.

        Returns:
            True if it's a framework tool type (FunctionTool).
        """
        # Check if it's a LangChain tool
        from agents import FunctionTool
        return isinstance(obj, FunctionTool)

    async def load_mcp_tools_from_config(self, mcp_configs: Dict[str, Any],
                                         agent_name: Optional[str] = None) -> Any:
        """Load MCP (Model Context Protocol) tools from configuration.

        Args:
            mcp_configs: MCP server configurations.
            agent_name: Agent name to associate the tools with.

        Returns:
            List of MCP tools (clients).
        """
        # Placeholder for MCP implementation
        mcp_list = self._get_mcp_name_list_from_mcp_config(mcp_configs)

        for tool_name in mcp_list:
            mcp = self._get_mcp_config(mcp_configs, tool_name)

            try:
                client = None
                if 'command' in mcp:
                    # STDIO MCP client
                    # Resolve command to absolute path: check system PATH first,
                    # then fall back to the venv's bin directory so that MCP
                    # server executables installed into the venv are found even
                    # when the IDE doesn't add the venv's bin/ to PATH.
                    raw_cmd = mcp['command']
                    resolved_cmd = shutil.which(raw_cmd)
                    if resolved_cmd is None:
                        venv_bin = os.path.dirname(sys.executable)
                        candidate = os.path.join(venv_bin, raw_cmd)
                        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                            resolved_cmd = candidate
                    if resolved_cmd is None:
                        resolved_cmd = raw_cmd  # best-effort fallback
                    self.logger.info(f"Resolved MCP command '{raw_cmd}' -> '{resolved_cmd}'")
                    params = {
                        "name": tool_name,
                        "params": {
                            "command": resolved_cmd,
                            "args": mcp.get('args', []),
                            "env": mcp.get('env', {})
                        },
                        "cache_tools_list": True
                    }
                    client = MCPServerStdio(**params, client_session_timeout_seconds=1200)
                    self.logger.info(f"Creating STDIO MCP client for '{tool_name}'")

                elif 'url' in mcp:
                    url = mcp.get('url', '')
                    headers = self._sanitize_headers(mcp.get('headers', {}))
                    if 'sse' in url:
                        # SSE MCP client
                        params = {
                            "name": tool_name,
                            "params": {
                                "url": mcp['url'],
                                "headers": headers
                            },
                            "cache_tools_list": True
                        }
                        client = MCPServerSse(**params, client_session_timeout_seconds=1200)
                        self.logger.info(f"Creating SSE MCP client for '{tool_name}' at {url}")
                    elif 'mcp' in url:
                        # HTTP MCP client
                        headers = self._sanitize_headers(mcp.get('headers', {}))
                        params = {
                            "name": tool_name,
                            "params": {
                                "url": mcp['url'],
                                "headers": headers
                            },
                            "cache_tools_list": True
                        }
                        client = MCPServerStreamableHttp(**params, client_session_timeout_seconds=1200)
                        self.logger.info(f"Creating HTTP MCP client for '{tool_name}' at {url}")

                if client:
                    # Store the client - Strands will manage the context
                    if agent_name not in self.mcp_clients:
                        self.mcp_clients[agent_name] = []
                    self.mcp_clients[agent_name].append(client)
                    self.tools[tool_name] = client
                    self.logger.info(f"✅ Loaded MCP client: {tool_name}")

                    if self.enable_lazy_loading:
                        if tool_name not in self.available_mcp_tools:
                            self.available_mcp_tools[tool_name] = {}
                        async with client:
                            list_of_tools = await client.list_tools()
                            for tool in list_of_tools:
                                self.available_mcp_tools[tool.name] = tool_name
                                self.available_mcp_tools[tool_name][tool.name] = {
                                    'type': 'mcp',
                                    'mcp_client': client,
                                    'input_schema': tool.inputSchema
                                }

                else:
                    self.logger.warning(f"⚠️  No valid MCP configuration for '{tool_name}'")

            except Exception as e:
                self.logger.error(f"❌ Failed to load MCP tool '{tool_name}': {e}", exc_info=True)

    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format the schema into a concise string."""
        if not schema:
            return "No parameters"

        # Handle case where schema is just properties (legacy/simplified)
        properties = schema.get('properties', schema)
        # If properties is not a dict (e.g. it's the schema itself and has no properties key but is an object)
        if not isinstance(properties, dict):
            properties = schema

        required = schema.get('required', [])

        formatted = []
        for name, details in properties.items():
            if not isinstance(details, dict):
                continue
            type_ = details.get('type', 'any')
            desc = details.get('description', '')
            is_required = "required" if name in required else "optional"
            default = f", default={details['default']}" if 'default' in details else ""

            line = f"- {name} ({type_}, {is_required}{default}): {desc}"
            formatted.append(line)

        return "\n".join(formatted)

    def get_input_parameter_schema(self, tool_list: str) -> str:
        """
        Get the input parameter schema for a list of tools.

        Args:
            tool_list: Comma-separated list of tool names.

        Returns:
            String containing input parameter schemas for the tools.
        """
        self.logger.debug(f"Fetching input schema of tools: {tool_list}")
        schemas = []
        for tool_name in tool_list.split(','):
            tool_name = tool_name.strip()
            if tool_name in self.available_mcp_tools:
                mcp_client = self.available_mcp_tools[tool_name]
                schema = self.available_mcp_tools[mcp_client][tool_name]['input_schema']
                formatted_schema = self._format_schema(schema)
                schemas.append(f"{tool_name}:\n{formatted_schema}")
            elif tool_name in self.tools:
                schema = self.tools[tool_name].params_json_schema.get('properties', {})
                formatted_schema = self._format_schema(schema)
                schemas.append(f"{tool_name}:\n{formatted_schema}")
        return "\n\n".join(schemas)

    async def execute_tool(self, tool_name: str, arguments: Any) -> Any:
        """
        Execute a tool with the given arguments.
        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments for the tool (dict or json string).
        Returns:
            Result of the tool execution.
        """
        self.logger.debug(f"Executing tool: {tool_name} with arguments: {arguments}")
        import json
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass  # Maybe it's not JSON, but let the tool handle it or fail later

        if tool_name in self.available_mcp_tools:
            mcp_client = self.available_mcp_tools[tool_name]
            async with self.available_mcp_tools[mcp_client][tool_name]['mcp_client'] as client:
                return await client.call_tool(tool_name, arguments)
        elif tool_name in self.tools:
            import agents.tool as tool_module
            from agents.tool_context import ToolContext

            # Ensure arguments is a string for invoke_function_tool if it expects JSON string
            # But invoke_function_tool might expect dict or string depending on implementation
            # Based on previous code: arguments=json.dumps(arguments)

            args_str = json.dumps(arguments) if not isinstance(arguments, str) else arguments

            ctx = ToolContext(None, tool_name=tool_name, tool_call_id=f"{tool_name}_call", tool_arguments="{}")
            return await tool_module.invoke_function_tool(
                function_tool=self.tools[tool_name],
                arguments=args_str,
                context=ctx
            )
        else:
            raise ValueError(f"Tool '{tool_name}' not found")
