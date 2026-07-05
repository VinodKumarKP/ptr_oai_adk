"""LangChain-specific tool registry implementation."""

import asyncio
import inspect
import json
import yaml
from typing import Dict, Any, Callable

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection, SSEConnection, StreamableHttpConnection

from oai_agent_core.core.base_tool_registry import BaseToolRegistry


class LangChainToolRegistry(BaseToolRegistry):
    """LangChain-specific tool registry implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-server cache of enumerated MCP tools and single-flight locks,
        # so a server referenced by several agents is only spawned once.
        self._mcp_tools_cache: Dict[str, list] = {}
        self._mcp_server_locks: Dict[str, asyncio.Lock] = {}

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

        if isinstance(arguments, list):
            arguments = arguments[0] if arguments else {}

        if arguments is None:
            arguments = {}

        if tool_name in self.available_mcp_tools:
            mcp_client = self.available_mcp_tools[tool_name]
            return await self.available_mcp_tools[mcp_client][tool_name]['tool'].ainvoke(input=arguments)
        elif tool_name in self.tools:
            tool = self.tools[tool_name]
            # StructuredTool exposes the underlying callable via .func; other
            # tools/callables may not, so fall back to .invoke() or direct call.
            func = getattr(tool, 'func', None)
            if callable(func):
                return func(**arguments)
            if hasattr(tool, 'invoke'):
                return tool.invoke(arguments)
            if callable(tool):
                return tool(**arguments)
            raise TypeError(f"Tool '{tool_name}' is not callable")
        else:
            raise ValueError(f"Tool '{tool_name}' not found")

    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format the schema into a concise string."""
        if not schema or not isinstance(schema, dict):
            return "No parameters"

        properties = schema

        formatted = []
        for name, details in properties.items():
            if not isinstance(details, dict):
                continue

            type_ = details.get('type')
            if not type_ and 'anyOf' in details:
                types = [t['type'] for t in details['anyOf'] if t.get('type') and t.get('type') != 'null']
                if types:
                    type_ = " | ".join(types)

            if not type_:
                type_ = 'any'

            desc = details.get('description', '')
            if not desc and 'title' in details:
                desc = details.get('title', '')

            is_required = "optional" if 'default' in details else "required"
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
                schema = self.tools[tool_name].args
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

    async def load_mcp_tools_from_config(self, mcp_config: Dict[str, Any]) -> list[Any]:
        """Load MCP tools defined in configuration.

        Tool enumeration is cached per server: the first request spawns the
        server process to list its tools; later requests (e.g. other agents
        or subagents referencing the same server) reuse the cached result.
        Distinct servers are enumerated concurrently.

        Args:
            mcp_config: Dictionary of MCP tool configurations

        Returns:
            List of MCP tools
        """
        try:
            mcp_list = self._get_mcp_name_list_from_mcp_config(mcp_config)

            connections = {}
            for tool_name in mcp_list:
                mcp = self._get_mcp_config(mcp_config, tool_name)
                mcp = {k: v for k, v in mcp.items() if
                       k in {'command', 'args', 'env', 'transport', 'url', 'headers', 'environment'}}

                if 'command' in mcp:
                    mcp['transport'] = 'stdio'
                    connections[tool_name] = StdioConnection(**mcp)
                    self.logger.info(f"Creating STDIO MCP client for '{tool_name}'")
                elif 'url' in mcp:
                    url = mcp.get('url', '')
                    mcp['headers'] = self._sanitize_headers(mcp.get('headers', {}))
                    if 'sse' in url:
                        mcp['transport'] = 'sse'
                        connections[tool_name] = SSEConnection(**mcp)
                        self.logger.info(f"Creating SSE MCP client for '{tool_name}' at {url}")
                    elif 'mcp' in url:
                        mcp['transport'] = 'streamable_http'
                        connections[tool_name] = StreamableHttpConnection(**mcp)
                        self.logger.info(f"Creating HTTP MCP client for '{tool_name}' at {url}")
                    else:
                        raise ValueError("Unsupported url. It should either end with mcp and sse")

                if tool_name in connections:
                    self.mcp_configs[tool_name] = connections[tool_name]

            # Enumerate distinct servers concurrently; per-server results
            # are cached so repeated references don't respawn the server.
            tool_lists = await asyncio.gather(*(
                self._enumerate_mcp_server(server_name, connection)
                for server_name, connection in connections.items()
            ))

            if self.enable_lazy_loading:
                for (server_name, connection), list_of_tools in zip(connections.items(), tool_lists):
                    if server_name not in self.available_mcp_tools:
                        self.available_mcp_tools[server_name] = {}
                    for tool in list_of_tools:
                        self.available_mcp_tools[tool.name] = server_name
                        self.available_mcp_tools[server_name][tool.name] = {
                            'tool': tool,
                            'type': 'mcp',
                            'mcp_client': connection,
                            'input_schema': tool.tool_call_schema['properties']
                        }
                return []

            return [tool for tools in tool_lists for tool in tools]
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP client: {e}")
            raise

    async def _enumerate_mcp_server(self, server_name: str, connection: Any) -> list[Any]:
        """Enumerate a single MCP server's tools, caching the result.

        A per-server lock makes concurrent requests single-flight: the first
        caller spawns the server and enumerates its tools; concurrent and
        later callers reuse the cached list.

        Args:
            server_name: Name of the MCP server.
            connection: Connection configuration for the server.

        Returns:
            List of tools exposed by the server.
        """
        lock = self._mcp_server_locks.setdefault(server_name, asyncio.Lock())
        async with lock:
            if server_name not in self._mcp_tools_cache:
                client = MultiServerMCPClient({server_name: connection})
                self._mcp_tools_cache[server_name] = await client.get_tools()
                self.logger.debug(
                    f"Enumerated {len(self._mcp_tools_cache[server_name])} "
                    f"tools from MCP server '{server_name}'"
                )
            return self._mcp_tools_cache[server_name]

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
        """Check if the module is a LangChain built-in tool.

        LangChain community tools (e.g. from langchain_community) use the standard
        class-based loader path — there is no dedicated built-in module shortcut.
        """
        return False

    def _load_framework_builtin_tool(self, tool_name: str, module_name: str) -> None:
        """Load a LangChain-specific built-in tool.

        LangChain has no special built-in tool shortcut — this method is a no-op.
        Use the standard class-based tool configuration to load community tools.

        Args:
            tool_name: Name of the tool
            module_name: Module name (unused)
        """
        self.logger.warning(
            f"_load_framework_builtin_tool called for '{tool_name}' (module: '{module_name}'), "
            "but LangChain has no built-in tool shortcut. "
            "Configure the tool via the standard class-based loader instead."
        )

    def _is_framework_tool_type(self, obj: Any) -> bool:
        """Check if an object is a LangChain StructuredTool type."""
        return isinstance(obj, StructuredTool)

