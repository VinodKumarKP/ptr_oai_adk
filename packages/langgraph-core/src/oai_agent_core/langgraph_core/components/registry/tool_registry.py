"""LangChain-specific tool registry implementation."""

import inspect
from typing import Dict, Any, Callable

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection, SSEConnection, StreamableHttpConnection

from oai_agent_core.core.base_tool_registry import BaseToolRegistry
from oai_agent_core.utils.dynamic_class_loader import DynamicClassLoader


class LangChainToolRegistry(BaseToolRegistry):
    """LangChain-specific tool registry implementation."""

    async def load_mcp_tools_from_config(self, mcp_config: Dict[str, Any]) -> list[Any]:
        """Load MCP tools defined in configuration.

        Args:
            mcp_config: Dictionary of MCP tool configurations

        Returns:
            List of MCP tools
        """
        try:
            mcp_list = self._get_mcp_name_list_from_mcp_config(mcp_config)

            mcp_configs = {}
            for tool_name in mcp_list:
                mcp = self._get_mcp_config(mcp_config, tool_name)
                mcp = {k: v for k, v in mcp.items() if k in {'command', 'args', 'env', 'transport', 'url', 'headers', 'environment'}}

                if 'command' in mcp:
                    mcp['transport'] = 'stdio'
                    mcp_configs[tool_name] = StdioConnection(**mcp)
                    self.mcp_configs[tool_name] =  mcp_configs[tool_name]
                    self.logger.info(f"Creating STDIO MCP client for '{tool_name}'")
                elif 'url' in mcp:
                    url = mcp.get('url', '')
                    if 'sse' in url:
                        mcp['transport'] = 'sse'
                        mcp_configs[tool_name] = SSEConnection(**mcp)
                        self.mcp_configs[tool_name] = mcp_configs[tool_name]
                        self.logger.info(f"Creating SSE MCP client for '{tool_name}' at {url}")
                    elif 'mcp' in url:
                        mcp['transport'] = 'streamable_http'
                        mcp_configs[tool_name] = StreamableHttpConnection(**mcp)
                        self.mcp_configs[tool_name] = mcp_configs[tool_name]
                        self.logger.info(f"Creating HTTP MCP client for '{tool_name}' at {url}")
                    else:
                        raise ValueError("Unsupported url. It should either end with mcp and sse")

            client = MultiServerMCPClient(mcp_configs)
            tools = await client.get_tools()
            return tools
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP client: {e}")
            raise

    def _wrap_function_with_defaults(self, func: Callable, default_params: Dict[str, Any]) -> Callable:
        """Wrap a function to include default parameter values from config.

        Works with LangChain @tool decorated functions.

        Args:
            func: The function to wrap (may be a StructuredTool)
            default_params: Dictionary of parameter names and their default values

        Returns:
            Wrapped function with new defaults
        """
        from langchain.tools import tool

        # Get the underlying function
        original_func = self._get_underlying_function(func)

        # Get the function's signature and type hints
        sig = inspect.signature(original_func)
        type_hints = inspect.get_annotations(original_func)

        # Validate parameters
        validated_params = self._validate_function_params(original_func, default_params)

        # Create wrapper function
        def wrapper(*args, **kwargs):
            merged_kwargs = {**validated_params, **kwargs}
            return original_func(*args, **merged_kwargs)

        # Preserve metadata
        self._preserve_function_metadata(wrapper, original_func, sig, validated_params, type_hints)

        # Re-apply the @tool decorator
        return tool(wrapper)

    @staticmethod
    def _get_underlying_function(func: Callable) -> Callable:
        """Extract the underlying function from a decorated function."""
        if hasattr(func, '__wrapped__'):
            return func.__wrapped__
        elif hasattr(func, 'func'):
            return func.func
        return func

    def _validate_function_params(self, func: Callable, default_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that default params exist in the function signature."""
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
        """Preserve function metadata on the wrapper."""
        wrapper.__name__ = original_func.__name__
        wrapper.__doc__ = original_func.__doc__
        if hasattr(original_func, '__module__'):
            wrapper.__module__ = original_func.__module__

        # Update signature with new defaults
        new_params = []
        for param_name, param in sig.parameters.items():
            if param_name in default_params:
                new_param = param.replace(default=default_params[param_name])
                new_params.append(new_param)
            else:
                new_params.append(param)

        wrapper.__signature__ = sig.replace(parameters=new_params)
        wrapper.__annotations__ = type_hints.copy()

    def _get_framework_tool_decorator(self):
        """Get the LangChain tool decorator."""
        from langchain.tools import tool
        return tool

    def _is_framework_builtin_tool(self, module_name: str) -> bool:
        """Check if the module is a LangChain built-in tool."""
        return module_name == 'strands_tools'

    def _load_framework_builtin_tool(self, tool_name: str, module_name: str) -> None:
        """Load a LangChain-specific built-in tool.

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
        """Check if an object is a LangChain StructuredTool type."""
        return isinstance(obj, StructuredTool)
