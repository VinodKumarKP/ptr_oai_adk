"""CrewAI-specific tool registry implementation."""

import inspect
from typing import Dict, Any, Callable

from crewai.mcp import MCPServerStdio, MCPServerHTTP, MCPServerSSE
from crewai.tools.base_tool import Tool
from oai_agent_core.core.base_tool_registry import BaseToolRegistry

from oai_agent_core.utils.dynamic_class_loader import DynamicClassLoader


class CrewAIToolRegistry(BaseToolRegistry):
    """CrewAI-specific tool registry implementation."""

    def load_mcp_tools_from_config(self, mcp_configs: Dict[str, Any]) -> list[Any]:
        """Load MCP (Model Context Protocol) tools from configuration.

        Args:
            mcp_configs: List of MCP server configurations

        Returns:
            List of MCP server instances
        """
        mcp_list = self._get_mcp_name_list_from_mcp_config(mcp_configs)

        for tool_name in mcp_list:
            mcp = self._get_mcp_config(mcp_configs, tool_name)
            try:
                if 'command' in mcp:
                    self.mcp_clients.append(MCPServerStdio(**mcp))
                elif 'url' in mcp:
                    url = mcp.get('url', '')
                    if 'sse' in url:
                        self.mcp_clients.append(MCPServerSSE(**mcp))
                    elif 'mcp' in url:
                        self.mcp_clients.append(MCPServerHTTP(**mcp))
            except Exception as e:
                self.logger.error(f"Failed to load MCP server: {e}")

        return self.mcp_clients

    def _wrap_function_with_defaults(self, func: Callable, default_params: Dict[str, Any]) -> Callable:
        """Wrap a function to include default parameter values from config.

        Works with CrewAI @tool decorated functions.

        Args:
            func: The function to wrap (may be a DecoratedFunctionTool)
            default_params: Dictionary of parameter names and their default values

        Returns:
            Wrapped function with new defaults
        """
        from crewai.tools import tool

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
        """Get the CrewAI tool decorator."""
        from crewai.tools import tool
        return tool

    def _is_framework_builtin_tool(self, module_name: str) -> bool:
        """Check if the module is a CrewAI built-in tool."""
        return module_name == 'strands_tools'

    def _load_framework_builtin_tool(self, tool_name: str, module_name: str) -> None:
        """Load a CrewAI-specific built-in tool.

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
        """Check if an object is a CrewAI Tool type."""
        return isinstance(obj, Tool)
