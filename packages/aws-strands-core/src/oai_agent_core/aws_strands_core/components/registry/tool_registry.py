"""AWS Strands-specific tool registry implementation."""

import asyncio
import inspect
import json
import yaml
from typing import Dict, Any, Callable, List

import httpx

# NOTE: `mcp.client.*` and strands' `MCPClient` are imported lazily inside
# load_mcp_tools_from_config (below). Importing them at module scope pulls ~1.2s
# of `mcp` startup cost that every strands agent paid even when it declared no
# MCP tools.

from oai_agent_core.core.base_tool_registry import BaseToolRegistry
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-server cache of enumerated MCP tools and single-flight locks,
        # so a server referenced by several agents is only spawned once.
        self._mcp_tools_cache: Dict[str, list] = {}
        self._mcp_server_locks: Dict[str, asyncio.Lock] = {}

    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format the schema into a concise string."""
        if not schema:
            return "No parameters"

        # Handle case where schema is just properties (legacy/simplified)
        properties = schema.get('properties', schema)
        # If properties is not a dict (e.g. it's the schema itself and has no properties key but is an object)
        if not isinstance(properties, dict):
            properties = schema

        # If schema is nested inside 'json' key (common in some frameworks)
        if 'json' in schema and isinstance(schema['json'], dict):
            properties = schema['json'].get('properties', schema['json'])

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
        self.logger.info(f"Fetching input schema of tools: {tool_list}")
        schemas = []
        for tool_name in tool_list.split(','):
            tool_name = tool_name.strip()
            if tool_name in self.available_mcp_tools:
                mcp_client = self.available_mcp_tools[tool_name]
                schema = self.available_mcp_tools[mcp_client][tool_name]['input_schema']
                formatted_schema = self._format_schema(schema)
                schemas.append(f"{tool_name}:\n{formatted_schema}")
            elif tool_name in self.tools:
                schema = self.tools[tool_name].tool_spec.get('inputSchema', {})
                formatted_schema = self._format_schema(schema)
                schemas.append(f"{tool_name}:\n{formatted_schema}")
        return "\n\n".join(schemas)

    async def execute_multiple_tools(self, arguments: str):
        """
        Execute multiple tools in parallel.
        Args:
            arguments: YAML string containing arguments for each tool, keyed by tool name.
        """
        self.logger.info(f"Executing multiple tools with arguments: {arguments}")
        arguments = yaml.safe_load(arguments)

        tool_names = []
        tasks = []

        for tool_name in arguments.keys():
            tool_name_stripped = tool_name.strip()
            tool_names.append(tool_name_stripped)
            tasks.append(self.execute_tool(tool_name_stripped, arguments[tool_name]))

        results = await asyncio.gather(*tasks)

        return dict(zip(tool_names, results))

    async def execute_tool(self, tool_name: str, arguments: Any) -> Any:
        """
        Execute a tool with the given arguments.
        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments for the tool (dict or json string).
        Returns:
            Result of the tool execution.
        """
        self.logger.info(f"Executing tool:{tool_name} with arguments: {arguments}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass  # Maybe it's not JSON, but let the tool handle it or fail later

        if tool_name in self.available_mcp_tools:
            mcp_client = self.available_mcp_tools[tool_name]
            with self.available_mcp_tools[mcp_client][tool_name]['mcp_client'] as client:
                return client.call_tool_sync(tool_use_id=None,
                                             name=tool_name, arguments=arguments)
        elif tool_name in self.tools:
            return self.tools[tool_name](
                **arguments
            )
        else:
            raise ValueError(f"Tool '{tool_name}' not found")

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
        # Lazy import: only pay the mcp/strands-MCP startup cost when an agent
        # actually declares MCP tools.
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client, StdioServerParameters
        from mcp.client.streamable_http import streamable_http_client
        from strands.tools.mcp.mcp_client import MCPClient

        mcp_list = self._get_mcp_name_list_from_mcp_config(mcp_configs)
        loaded_clients = []
        lazy_registrations = []

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
                    headers = self._sanitize_headers(mcp.get('headers', {}))
                    if 'sse' in url:
                        # SSE MCP client
                        client = MCPClient(lambda u=url, h=headers: sse_client(url=u, headers=h))
                        self.logger.info(f"Creating SSE MCP client for '{tool_name}' at {url}")
                    elif 'mcp' in url:
                        # HTTP MCP client
                        custom_http_client = httpx.AsyncClient(
                            headers=headers,
                            timeout=httpx.Timeout(30.0, read=300.0)
                        )
                        client = MCPClient(lambda u=url: streamable_http_client(url=u,
                                                                                http_client=custom_http_client,
                                                                                terminate_on_close=True))
                        self.logger.info(f"Creating HTTP MCP client for '{tool_name}' at {url}")

                if client:
                    # Store the client - Strands will manage the context
                    if tool_name not in self.mcp_clients:
                        self.mcp_clients[tool_name] = []
                    self.mcp_clients[tool_name].append(client)
                    self.tools[tool_name] = client
                    loaded_clients.append(client)
                    self.logger.info(f"✅ Loaded MCP client: {tool_name}")

                    if self.enable_lazy_loading:
                        lazy_registrations.append((tool_name, client))

                else:
                    self.logger.warning(f"⚠️  No valid MCP configuration for '{tool_name}'")

            except Exception as e:
                self.logger.error(f"❌ Failed to load MCP tool '{tool_name}': {e}", exc_info=True)

        # Enumerate servers concurrently for lazy loading; per-server results
        # are cached so repeated references don't respawn the server.
        if lazy_registrations:
            results = await asyncio.gather(
                *(self._register_lazy_mcp_tools(server_name, client)
                  for server_name, client in lazy_registrations),
                return_exceptions=True
            )
            for (server_name, _), result in zip(lazy_registrations, results):
                if isinstance(result, Exception):
                    self.logger.error(
                        f"❌ Failed to enumerate MCP server '{server_name}': {result}",
                        exc_info=result
                    )

        return loaded_clients

    async def _register_lazy_mcp_tools(self, server_name: str, client: Any) -> None:
        """Enumerate a server's tools once and register them for lazy loading.

        A per-server lock makes concurrent requests single-flight: the first
        caller spawns the server and enumerates its tools; concurrent and
        later callers reuse the cached result.

        Args:
            server_name: Name of the MCP server.
            client: MCP client for the server.
        """
        lock = self._mcp_server_locks.setdefault(server_name, asyncio.Lock())
        async with lock:
            if server_name in self._mcp_tools_cache:
                return

            def _enumerate():
                with client:
                    return client.list_tools_sync()

            # list_tools_sync spawns the server and blocks; run it off the
            # event loop so distinct servers can enumerate concurrently.
            list_of_tools = await asyncio.to_thread(_enumerate)
            self._mcp_tools_cache[server_name] = list_of_tools

            if server_name not in self.available_mcp_tools:
                self.available_mcp_tools[server_name] = {}
            for tool in list_of_tools:
                self.available_mcp_tools[tool.tool_name] = server_name
                self.available_mcp_tools[server_name][tool.tool_name] = {
                    'type': 'mcp',
                    'mcp_client': client,
                    'input_schema': tool.tool_spec['inputSchema']
                }

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
        from strands.tools.decorator import DecoratedFunctionTool
        return isinstance(obj, DecoratedFunctionTool)

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