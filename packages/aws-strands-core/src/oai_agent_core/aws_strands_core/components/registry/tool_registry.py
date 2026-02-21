"""AWS Strands-specific tool registry implementation."""

import inspect
from typing import Dict, Any, Callable, List

from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from oai_agent_core.core.base_tool_registry import BaseToolRegistry
from strands.tools.mcp.mcp_client import MCPClient

from oai_agent_core.utils.dynamic_class_loader import DynamicClassLoader


class AWSStrandsToolRegistry(BaseToolRegistry):
    """AWS Strands-specific tool registry implementation.

    This registry manages tools for AWS Strands agents, including:
    - Strands built-in tools
    - Custom Python tools
    - MCP (Model Context Protocol) clients

    Attributes:
        tools: Dictionary mapping tool names to tool instances
        mcp_clients: List of active MCP client instances
        custom_modules: Dictionary mapping module names to their functions
        logger: Logger instance for debugging
        project_root: Project root directory path
    """

    def __init__(self, logger=None, project_root=None):
        """Initialize the AWS Strands tool registry.

        Args:
            logger: Optional logger instance
            project_root: Optional project root directory path
        """
        super().__init__(logger, project_root)

    async def load_mcp_tools_from_config(self, mcp_configs: Dict[str, Any], agent_name: str = None) -> List[Any]:
        """Load MCP tools defined in configuration.

        MCP clients are created and stored - they will be used with context managers
        when agents need them.

        Args:
            mcp_configs: Dictionary of MCP tool configurations from YAML

        Returns:
            List of loaded MCP clients

        Example config:
            mcp_tools:
              filesystem:
                command: npx
                args: ['-y', '@modelcontextprotocol/server-filesystem', '/allowed/path']
              search:
                url: https://example.com/mcp-sse
        """
        mcp_list = self._get_mcp_name_list_from_mcp_config(mcp_configs)
        loaded_clients = []

        for tool_name in mcp_list:
            mcp = self._get_mcp_config(mcp_configs, tool_name)

            try:
                client = None

                if 'command' in mcp:
                    # STDIO MCP client
                    params = StdioServerParameters(
                        command=mcp['command'],
                        args=mcp.get('args', []),
                        env=mcp.get('env', {})
                    )
                    client = MCPClient(lambda p=params: stdio_client(p))
                    self.logger.info(f"Creating STDIO MCP client for '{tool_name}'")

                elif 'url' in mcp:
                    url = mcp.get('url', '')
                    headers = mcp.get('headers')
                    if 'sse' in url:
                        # SSE MCP client
                        client = MCPClient(lambda u=url, h=headers: sse_client(url=u, headers=h))
                        self.logger.info(f"Creating SSE MCP client for '{tool_name}' at {url}")
                    elif 'mcp' in url:
                        # HTTP MCP client
                        client = MCPClient(lambda u=url, h=headers: streamable_http_client(url=u, headers=h))
                        self.logger.info(f"Creating HTTP MCP client for '{tool_name}' at {url}")

                if client:
                    # Store the client - Strands will manage the context
                    if tool_name not in self.mcp_clients:
                        self.mcp_clients[tool_name] = []
                    self.mcp_clients[tool_name].append(client)
                    self.tools[tool_name] = client
                    loaded_clients.append(client)
                    self.logger.info(f"✅ Loaded MCP client: {tool_name}")
                else:
                    self.logger.warning(f"⚠️  No valid MCP configuration for '{tool_name}'")

            except Exception as e:
                self.logger.error(f"❌ Failed to load MCP tool '{tool_name}': {e}", exc_info=True)
        
        return loaded_clients

    def _wrap_function_with_defaults(self, func: Callable, default_params: Dict[str, Any]) -> Callable:
        """Wrap a function to include default parameter values from config.

        Works with Strands @tool decorated functions by creating a proper wrapper.

        Args:
            func: The function to wrap (may be a DecoratedFunctionTool)
            default_params: Dictionary of parameter names and their default values

        Returns:
            Wrapped function with new defaults

        Example:
            >>> wrapped = registry._wrap_function_with_defaults(
            ...     my_function,
            ...     {'timeout': 30, 'retries': 3}
            ... )
        """
        from strands.tools import tool

        # Get the underlying function
        original_func = self._get_underlying_function(func)

        # Get the function's signature and type hints
        sig = inspect.signature(original_func)
        type_hints = inspect.get_annotations(original_func)

        # Validate parameters
        validated_params = self._validate_function_params(original_func, default_params)

        # Create wrapper function
        def wrapper(*args, **kwargs):
            # Merge default_params with provided kwargs (provided kwargs take precedence)
            merged_kwargs = {**validated_params, **kwargs}
            return original_func(*args, **merged_kwargs)

        # Preserve metadata
        self._preserve_function_metadata(wrapper, original_func, sig, validated_params, type_hints)

        # Re-apply the @tool decorator to the wrapper
        decorated_wrapper = tool(wrapper)

        return decorated_wrapper

    @staticmethod
    def _get_underlying_function(func: Callable) -> Callable:
        """Extract the underlying function from a decorated function.

        Args:
            func: Potentially decorated function

        Returns:
            The underlying unwrapped function
        """
        if hasattr(func, '__wrapped__'):
            return func.__wrapped__
        elif hasattr(func, 'func'):
            return func.func
        return func

    def _validate_function_params(self, func: Callable, default_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that default params exist in the function signature.

        Args:
            func: Function to validate against
            default_params: Parameters to validate

        Returns:
            Validated parameters (invalid ones removed)
        """
        sig = inspect.signature(func)
        func_params = set(sig.parameters.keys())
        config_params = set(default_params.keys())
        invalid_params = config_params - func_params

        if invalid_params:
            self.logger.warning(
                f"⚠️  Invalid parameters for {func.__name__}: {invalid_params}. "
                f"Valid parameters are: {func_params}"
            )
            return {k: v for k, v in default_params.items() if k in func_params}

        return default_params

    @staticmethod
    def _preserve_function_metadata(
        wrapper: Callable,
        original_func: Callable,
        sig: inspect.Signature,
        default_params: Dict[str, Any],
        type_hints: Dict[str, Any]
    ) -> None:
        """Preserve function metadata on the wrapper.

        Args:
            wrapper: Wrapper function to update
            original_func: Original function to copy metadata from
            sig: Function signature
            default_params: Default parameters to apply
            type_hints: Type annotations
        """
        wrapper.__name__ = original_func.__name__
        wrapper.__doc__ = original_func.__doc__
        if hasattr(original_func, '__module__'):
            wrapper.__module__ = original_func.__module__

        # Update signature with new defaults
        new_params = []
        for param_name, param in sig.parameters.items():
            if param_name in default_params:
                # Update the default value
                new_param = param.replace(default=default_params[param_name])
                new_params.append(new_param)
            else:
                new_params.append(param)

        wrapper.__signature__ = sig.replace(parameters=new_params)
        wrapper.__annotations__ = type_hints.copy()

    def _get_framework_tool_decorator(self):
        """Get the Strands tool decorator.

        Returns:
            The Strands @tool decorator function
        """
        from strands.tools import tool
        return tool

    def _is_framework_builtin_tool(self, module_name: str) -> bool:
        """Check if the module is a Strands built-in tool.

        Args:
            module_name: Name of the module to check

        Returns:
            True if it's a Strands built-in tool module
        """
        return module_name == 'strands_tools'

    def _load_framework_builtin_tool(self, tool_name: str, module_name: str) -> None:
        """Load a Strands-specific built-in tool.

        Args:
            tool_name: Name of the tool
            module_name: Module name (should be 'strands_tools')
        """
        try:
            tool_module = DynamicClassLoader.dynamic_import_tool(
                tool_name,
                base_module='strands_tools'
            )
            self.tools[tool_name] = tool_module
            self.logger.info(f"✅ Loaded Strands tool: {tool_name}")

        except ImportError as e:
            self.logger.warning(
                f"❌ Failed to import strands_tools.{tool_name}: {e}\n"
                f"   This tool may not be available in your strands_tools version.\n"
                f"   Run 'python discover_tools.py' to see available tools."
            )

    def _is_framework_tool_type(self, obj: Any) -> bool:
        """Check if an object is a Strands tool type.

        AWS Strands doesn't have a specific tool base class to check against,
        so we rely on other module membership checks.

        Args:
            obj: Object to check

        Returns:
            False (AWS Strands tools don't have a specific type to check)
        """
        return False

    def clear(self) -> None:
        """Clear all registered tools and MCP clients.

        This removes all tools and MCP clients from the registry,
        resetting it to an empty state.
        """
        super().clear()
        self.mcp_clients.clear()
        self.logger.debug("Tool registry and MCP clients cleared")

    def __repr__(self) -> str:
        """String representation of the registry.

        Returns:
            String describing the registry state
        """
        return (
            f"AWSStrandsToolRegistry("
            f"tools={len(self.tools)}, "
            f"mcp_clients={len(self.mcp_clients)}, "
            f"modules={len(self.custom_modules)})"
        )