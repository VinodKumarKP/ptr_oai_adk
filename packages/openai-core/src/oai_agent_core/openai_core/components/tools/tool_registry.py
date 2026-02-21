"""Tool Registry for OpenAI Agents."""

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
        # For OpenAI/LangChain tools, we can use partial application or a wrapper
        # This is a simplified implementation
        import functools

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
                    params = {
                        "name": tool_name,
                        "params": {
                            "command": mcp['command'],
                            "args": mcp.get('args', []),
                            "env": mcp.get('env', {})
                        },
                        "cache_tools_list": True
                    }
                    client = MCPServerStdio(**params)
                    self.logger.info(f"Creating STDIO MCP client for '{tool_name}'")

                elif 'url' in mcp:
                    url = mcp.get('url', '')
                    headers = mcp.get('headers')
                    if 'sse' in url:
                        # SSE MCP client
                        params = {
                            "name": tool_name,
                            "params": {
                                "url": mcp['url'],
                                "headers": mcp.get('headers', {})
                            },
                            "cache_tools_list": True
                        }
                        client = MCPServerSse(**params)
                        self.logger.info(f"Creating SSE MCP client for '{tool_name}' at {url}")
                    elif 'mcp' in url:
                        # HTTP MCP client
                        params = {
                            "name": tool_name,
                            "params": {
                                "url": mcp['url'],
                                "headers": mcp.get('headers', {})
                            },
                            "cache_tools_list": True
                        }
                        client = MCPServerStreamableHttp(**params)
                        self.logger.info(f"Creating HTTP MCP client for '{tool_name}' at {url}")

                if client:
                    # Store the client - Strands will manage the context
                    if agent_name not in self.mcp_clients:
                        self.mcp_clients[agent_name] = []
                    self.mcp_clients[agent_name].append(client)
                    self.tools[tool_name] = client
                    self.logger.info(f"✅ Loaded MCP client: {tool_name}")
                else:
                    self.logger.warning(f"⚠️  No valid MCP configuration for '{tool_name}'")

            except Exception as e:
                self.logger.error(f"❌ Failed to load MCP tool '{tool_name}': {e}", exc_info=True)
