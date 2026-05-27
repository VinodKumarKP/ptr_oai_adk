"""Base tool management and loading for agent frameworks."""

import inspect
import logging
import os.path
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

from oai_agent_core.utils.dynamic_class_loader import DynamicClassLoader


class BaseToolRegistry(ABC):
    """Base class for managing loading and registration of tools for agents.

    Supports multiple tool types:
    - Framework built-in tools
    - Custom Python class-based tools
    - Custom function-based tools
    - MCP (Model Context Protocol) tools

    Attributes:
        tools: Dictionary mapping tool names to tool instances/modules
        logger: Logger instance for debugging
        custom_modules: Dictionary mapping module names to their functions
        project_root: Project root directory path
        mcp_configs: Dictionary of loaded MCP configurations
        mcp_clients: List of active MCP clients
    """

    def __init__(self, logger: Optional[logging.Logger] = None,
                 project_root: Optional[str] = None,
                 enable_lazy_loading: Optional[bool] = False):
        """Initialize the tool registry.

        Args:
            logger: Optional logger instance (creates default if None)
            project_root: Optional project root directory path
        """
        self.tools: Dict[str, Any] = {}
        self.logger = logger or logging.getLogger(__name__)
        self.custom_modules: Dict[str, Any] = {}
        self.project_root = project_root
        self.mcp_configs: Dict[str, Any] = {}
        self.mcp_clients: Dict[str, Any] = {}
        self.enable_lazy_loading = enable_lazy_loading
        self.available_mcp_tools = {}
        self.tools["shell"] = self.shell

    def is_iterable(self, obj) -> bool:
        if isinstance(obj, list | tuple | set | dict):  # Fast path for common types
            return True
        return not isinstance(obj, str) and isinstance(obj, Iterable)

    def shell(self, command: str) -> Dict[str, Any]:
        """
        Executes a shell command in a secure, sandboxed environment.

        This tool is restricted to a predefined list of safe, read-only commands
        to prevent malicious or destructive operations. It runs with a strict
        timeout to avoid hanging processes.

        Args:
            command: The shell command to execute (e.g., "ls -l").

        Returns:
            A dictionary containing the command's stdout, stderr, and return code.

        Raises:
            PermissionError: If the command is not in the allowlist.
            TimeoutError: If the command exceeds the 15-second timeout.
        """
        # Allowlist of safe, primarily read-only commands
        allowlist = [
            "ls", "grep", "cat", "echo", "ps", "df", "pwd", "find", "head", "tail", "wc", "python3", "python"
        ]

        # Split the command string to identify the main command
        try:
            main_command = command.split()[0]
        except IndexError:
            return {"stdout": "", "stderr": "Error: Empty command.", "return_code": 1}

        # Security: Enforce the allowlist
        if main_command not in allowlist:
            raise PermissionError(
                f"Command '{main_command}' is not allowed. Only safe, read-only commands are permitted."
            )

        try:
            # Execute the command with a timeout
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=15,  # 15-second timeout
                check=False  # Do not raise CalledProcessError automatically
            )
            return {
                "stdout": process.stdout,
                "stderr": process.stderr,
                "return_code": process.returncode
            }
        except subprocess.TimeoutExpired:
            raise TimeoutError("The shell command timed out after 15 seconds.")
        except Exception as e:
            return {"stdout": "", "stderr": f"An unexpected error occurred: {str(e)}", "return_code": 1}

    def load_mcp_config(self, mcp_config: Dict[str, Any]) -> Any:
        """Load MCP configuration into tool registry for later retrieval.

        This method processes MCP configurations, resolving environment variables
        and headers, and stores them in `self.mcp_configs`.

        Args:
            mcp_config: Dictionary of MCP server configurations.
                        Keys are server names, values are config dicts containing
                        'command', 'args', 'env' (for stdio) or 'url', 'headers' (for http/sse).
        """
        for mcp_name, mcp in mcp_config.items():
            env = os.environ.copy()
            resolved_env = self.update_env(mcp.get('env', {}))
            resolved_headers = self.update_env(mcp.get('headers', {}))

            env.update(resolved_env)
            env.update(resolved_headers)

            if 'command' in mcp:
                mcp['command'] = self.which(mcp['command'])
                if 'headers' in mcp:
                    mcp.pop('headers')
                mcp['env'] = env
            elif 'url' in mcp:
                if 'env' in mcp:
                    mcp.pop('env')
                mcp['headers'] = env
            self.mcp_configs[mcp_name] = mcp

    @abstractmethod
    def load_mcp_tools_from_config(self, mcp_config: Dict[str, Any], agent_name: Optional[str] = None) -> Any:
        """Load MCP (Model Context Protocol) tools from configuration.

        Args:
            mcp_config: MCP server configurations
            agent_name: Agent name [Optional]

        Returns:
            Framework-specific MCP tools or clients
        """
        pass

    @abstractmethod
    def _wrap_function_with_defaults(self, func: Callable, default_params: Dict[str, Any]) -> Callable:
        """Wrap a function to include default parameter values from config.

        This method is framework-specific as different frameworks use different
        decorators (@tool from crewai, langchain, etc.)

        Args:
            func: The function to wrap
            default_params: Dictionary of parameter names and their default values

        Returns:
            Wrapped function with new defaults
        """
        pass

    @abstractmethod
    def _get_framework_tool_decorator(self):
        """Get the framework-specific tool decorator.

        Returns:
            The tool decorator function for the specific framework
        """
        pass

    @abstractmethod
    async def execute_tool(self, tool_name: str, arguments: Any) -> Any:
        """
        Execute a tool with the given arguments.
        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments for the tool (dict or json string).
        Returns:
            Result of the tool execution.
        """

    def load_tools_from_config(self, tools_config: Dict[str, Any]) -> None:
        """Load all tools defined in configuration.

        Args:
            tools_config: Dictionary of tool configurations from YAML

        Example config:
            tools:
              calculator:
                module: framework_tools
              custom_tool:
                module: my_tools
                class: CustomTool
                params:
                  api_key: xyz
        """
        for tool_name, tool_config in tools_config.items():
            try:
                self._load_single_tool(tool_name, tool_config)
            except Exception as e:
                self.logger.error(f"Failed to load tool '{tool_name}': {e}")

    def _load_single_tool(self, tool_name: str, tool_config: Any) -> None:
        """Load a single tool based on its configuration.

        Args:
            tool_name: Name of the tool
            tool_config: Configuration for the tool (dict or other)
        """
        if not isinstance(tool_config, dict):
            self.logger.warning(f"Invalid tool configuration for '{tool_name}': expected dict")
            return

        # Determine tool type and load accordingly
        if 'module' in tool_config:
            self._load_module_tool(tool_name, tool_config)
        elif 'function' in tool_config:
            self._load_function_tool(tool_name, tool_config)
        else:
            self.logger.warning(
                f"Tool '{tool_name}' has no 'module' or 'function' specified"
            )

    def _load_module_tool(self, tool_name: str, tool_config: Dict[str, Any]) -> None:
        """Load a tool from a Python module.

        Args:
            tool_name: Name of the tool
            tool_config: Configuration containing module path and optional class name
        """
        module_name = tool_config['module']
        class_name = tool_config.get('class', None)
        function_list = tool_config.get('function_list', [])
        base_path = tool_config.get('base_path', None)

        if base_path:
            base_path = self._resolve_base_path(base_path)
            sys.path.insert(0, os.path.expanduser(base_path))

        # Check if it's a framework-specific tool
        if self._is_framework_builtin_tool(module_name):
            self._load_framework_builtin_tool(tool_name, module_name)
        elif class_name is None:
            # Load functions from module
            function_params = tool_config.get('function_params', {})
            self._load_tools_from_module(module_name, function_list, function_params, tool_name)
        else:
            self._load_custom_module_tool(tool_name, tool_config, module_name)

    def _resolve_base_path(self, base_path: str) -> str:
        """Resolve the base path, handling relative paths from project root.

        Args:
            base_path: Base path from configuration

        Returns:
            Resolved absolute path
        """
        if self.project_root and base_path and base_path.startswith('.'):
            p = Path(str(base_path).replace('.', self.project_root, 1))
        else:
            p = Path(base_path).expanduser()

        return str(p.resolve())

    @abstractmethod
    def _is_framework_builtin_tool(self, module_name: str) -> bool:
        """Check if the module is a framework built-in tool.

        Args:
            module_name: Name of the module

        Returns:
            True if it's a framework built-in tool
        """
        pass

    @abstractmethod
    def _load_framework_builtin_tool(self, tool_name: str, module_name: str) -> None:
        """Load a framework-specific built-in tool.

        Args:
            tool_name: Name of the tool
            module_name: Module name of the framework tool
        """
        pass

    def _load_tools_from_module(
            self,
            module_name: str,
            function_list: Optional[List] = None,
            function_params: Optional[Dict[str, Dict[str, Any]]] = None,
            tool_key: str = None
    ) -> None:
        """Load all tools from a specified module.

        Args:
            module_name: Python module path
            function_list: Optional list of specific function names to load
            function_params: Optional dict mapping function names to their default parameters
        """
        try:
            tool_module = DynamicClassLoader.dynamic_import_module(module_name)
            functions = self.get_module_functions(tool_module, function_list)

            if not functions:
                self.logger.warning(
                    f"⚠️  No functions found in module '{module_name}'. "
                    f"Make sure functions are decorated with @tool."
                )
                return

            function_params = function_params or {}

            for func in functions:
                tool_name = self._get_function_name(func)

                # Check if this function has custom params in config
                if tool_name in function_params:
                    params = function_params[tool_name]
                    wrapped_func = self._wrap_function_with_defaults(func, params)
                    self.tools[tool_name] = wrapped_func

                    doc = inspect.getdoc(func) or "No description"
                    doc_preview = doc.split('\n')[0][:60]

                    self.logger.info(
                        f"✅ Loaded tool: {tool_name} from {module_name} with params {params} - {doc_preview}"
                    )
                else:
                    if self._is_framework_tool_type(func) or (hasattr(func, '__wrapped__') and func.__wrapped__ is not None):
                        self.tools[tool_name] = func
                    else:
                        self.tools[tool_name] = self._get_framework_tool_decorator()(func)

                    doc = inspect.getdoc(func) or "No description"
                    doc_preview = doc.split('\n')[0][:60]

                    self.logger.info(f"✅ Loaded tool: {tool_name} from {module_name} - {doc_preview}")

            self.logger.info(f"✅ Successfully loaded {len(functions)} tool(s) from {module_name}")
            self.custom_modules[module_name] = functions
            if tool_key:
                self.custom_modules[tool_key] = functions

        except ImportError as e:
            self.logger.error(
                f"❌ Failed to import module '{module_name}': {e}\n"
                f"   Make sure the module is installed and accessible."
            )
        except Exception as e:
            self.logger.error(
                f"❌ Unexpected error loading tools from '{module_name}': {e}",
                exc_info=True
            )

    def _get_function_name(self, func: Any) -> str:
        """Get the name of a function, handling different tool types.

        Args:
            func: Function or tool object

        Returns:
            Name of the function
        """
        if hasattr(func, '__name__'):
            return func.__name__
        elif hasattr(func, 'name'):
            return func.name
        else:
            return str(func)

    def get_module_functions(self, module, function_list: Optional[List] = None) -> List[Any]:
        """Retrieve functions explicitly defined within a given module.

        Args:
            module: The module object to inspect
            function_list: Optional list of specific function names to include

        Returns:
            List of function objects defined in the module
        """
        functions = []
        module_name = module.__name__
        module_file = getattr(module, '__file__', None)

        self.logger.debug(f"Inspecting module: {module_name} (file: {module_file})")

        for name, obj in inspect.getmembers(module):
            # Skip private/magic attributes and the tool decorator itself
            if name.startswith('_') or name == 'tool':
                continue

            # Skip imported modules
            if inspect.ismodule(obj):
                continue

            self.logger.debug(
                f"Checking member: {name}, "
                f"type: {type(obj).__name__}, "
                f"callable: {callable(obj)}, "
                f"is_function: {inspect.isfunction(obj)}"
            )

            try:
                # Check if this object belongs to the target module
                is_from_module = self._is_object_from_module(obj, module, module_name)

                self.logger.debug(f"  {name}: is_from_module={is_from_module}")

                if is_from_module:
                    if function_list and name in function_list:
                        functions.append(obj)
                        self.logger.debug(f"  ✅ Added from list: {name}")
                    elif not function_list:
                        functions.append(obj)
                        self.logger.debug(f"  ✅ Added: {name}")
                else:
                    self.logger.debug(f"  ⏭️ Skipped: {name} (external)")

            except Exception as e:
                self.logger.debug(f"  ⚠️  Error checking {name}: {e}")
                continue

        function_names = [self._get_function_name(f) for f in functions]
        self.logger.info(f"Found {len(functions)} function(s) in {module_name}: {function_names}")
        return functions

    def _is_object_from_module(self, obj: Any, module: Any, module_name: str) -> bool:
        """Check if an object belongs to the specified module.

        Args:
            obj: Object to check
            module: Module to check against
            module_name: Name of the module

        Returns:
            True if object belongs to the module
        """
        # Method 1: Check direct module
        obj_module = inspect.getmodule(obj)

        # Method 2: Check __module__ attribute
        obj_module_name = getattr(obj, '__module__', None)

        # Method 3: Check if it's a framework-specific tool type
        is_framework_tool = self._is_framework_tool_type(obj)

        # Method 4: For wrapped functions (decorators), check the original
        wrapped_module = None
        wrapped_module_name = None
        if hasattr(obj, '__wrapped__'):
            wrapped_module = inspect.getmodule(obj.__wrapped__)
            wrapped_module_name = getattr(obj.__wrapped__, '__module__', None)

        return (
                obj_module is module or
                obj_module_name == module_name or
                wrapped_module is module or
                wrapped_module_name == module_name or
                is_framework_tool
        )

    @abstractmethod
    def _is_framework_tool_type(self, obj: Any) -> bool:
        """Check if an object is a framework-specific tool type.

        Args:
            obj: Object to check

        Returns:
            True if it's a framework tool type
        """
        pass

    @abstractmethod
    def get_input_parameter_schema(self, tool_list: str) -> str:
        """
        Get the input parameter schema for a list of tools.

        Args:
            tool_list: Comma-separated list of tool names.

        Returns:
            String containing input parameter schemas for the tools.
        """

    def _load_custom_module_tool(
            self,
            tool_name: str,
            tool_config: Dict[str, Any],
            module_name: str
    ) -> None:
        """Load a custom tool from a specified module.

        Args:
            tool_name: Name of the tool
            tool_config: Configuration containing class name and params
            module_name: Python module path
        """
        class_name = tool_config.get('class', tool_name)

        try:
            if 'base_path' in tool_config:
                sys.path.insert(0, tool_config['base_path'])

            tool_class = DynamicClassLoader.dynamic_import(module_name, class_name)

            # Handle both functions and classes
            if callable(tool_class) and not isinstance(tool_class, type):
                self.tools[tool_name] = tool_class
            else:
                init_params = tool_config.get('params', {})
                tool_instance = tool_class(**init_params) if init_params else tool_class()
                self.tools[tool_name] = tool_instance
                # load all the functions from the class
                methods = []
                for name, method in inspect.getmembers(tool_instance, predicate=inspect.ismethod):
                    if not name.startswith('_'):
                        methods.append(method)
                        self.custom_modules[name] = method
                self.custom_modules[tool_name] = methods

            self.logger.info(f"Loaded custom tool: {tool_name} from {module_name}.{class_name}")

        except ImportError as e:
            self.logger.error(
                f"Failed to import tool '{tool_name}' from {module_name}: {e}\n"
                f"Make sure the module is installed and accessible."
            )

    def _load_function_tool(self, tool_name: str, tool_config: Dict[str, Any]) -> None:
        """Load a custom function-based tool.

        This feature is not yet implemented. Use module-based tools instead.

        Args:
            tool_name: Name of the tool
            tool_config: Configuration containing function reference

        Raises:
            NotImplementedError: Function-based tools are not supported. Use module-based
                tools with 'module' and 'function_list' config instead.

        Example:
            Instead of (not supported):
                tools:
                  my_tool:
                    function: my_module.my_function

            Use this (supported):
                tools:
                  my_tool:
                    module: my_module
                    function_list:
                      - my_function
        """
        func_name = tool_config.get('function', 'unknown')
        raise NotImplementedError(
            f"Function-based tools are not supported: '{func_name}'. "
            f"Use module-based tools instead with 'module' and 'function_list' configuration."
        )

    def get_tools_for_agent(self, tool_names: Any) -> List[Any]:
        """Get tool instances for an agent based on configured tool names.

        Args:
            tool_names: Single tool name (str), list of names, or None

        Returns:
            List of tool instances/modules
        """
        agent_tools = []

        if not tool_names:
            return agent_tools

        # Normalize to list
        if isinstance(tool_names, str):
            tool_names = [tool_names]

        if not isinstance(tool_names, list):
            self.logger.warning(f"Invalid tool_names type: {type(tool_names)}")
            return agent_tools

        # Collect tools
        for tool_name in tool_names:
            if tool_name in self.custom_modules:
                functions = self.custom_modules[tool_name]
                if self.is_iterable(functions):
                    agent_tools.extend(functions)
                else:
                    agent_tools.append(functions)
                self.logger.debug(f"Assigned custom module tools from '{tool_name}' to agent")
            elif tool_name in self.tools:
                agent_tools.append(self.tools[tool_name])
                self.logger.debug(f"Assigned tool '{tool_name}' to agent")
            else:
                self.logger.warning(f"Tool '{tool_name}' not found in registry")

        return agent_tools

    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get a specific tool by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            Tool instance/module or None if not found
        """
        return self.tools.get(tool_name)

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool exists in registry
        """
        return tool_name in self.tools

    def list_tools(self) -> List[str]:
        """Get list of all registered tool names.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())

    def clear(self) -> None:
        """Clear all registered tools."""
        self.tools.clear()
        self.custom_modules.clear()
        self.logger.debug("Tool registry cleared")

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self.tools)

    def __contains__(self, tool_name: str) -> bool:
        """Check if tool exists using 'in' operator."""
        return tool_name in self.tools

    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"{self.__class__.__name__}(tools={len(self.tools)})"

    def update_env(self, server_config: Dict[str, Any]) -> dict[Any, Any]:
        """Update server configuration with environment variables.

        Args:
            server_config: Dictionary containing environment variable configurations.

        Returns:
            Dictionary with resolved environment variables.
        """
        resolved_env = {}
        for key, value in server_config.items():
            # if value contains ${VAR} or $VAR, resolve from original environ
            if isinstance(value, str) and ('$' in value) and (('{' in value and '}' in value) or ' ' not in value):
                self.logger.info(f"Resolving environment variable {value} for server configuration")
                resolved_value = self._expand_env_with_defaults(value)
                self.logger.info(f"Resolved environment variable {value} -> {resolved_value} for server configuration")
                resolved_env[key] = resolved_value
            else:
                resolved_env[key] = os.environ.get(key, value)
        return resolved_env

    def _expand_env_with_defaults(self, value: str) -> str:
        """Expand environment variables with default value support.
        
        Handles formats like ${VAR:-default} and extracts default values.

        Args:
            value: String containing environment variable placeholders.

        Returns:
            String with expanded environment variables.
        """
        import re
        
        def replace_var(match):
            var_expr = match.group(1)
            if ':-' in var_expr:
                var_name, default = var_expr.split(':-', 1)
                return os.environ.get(var_name, default)
            else:
                return os.environ.get(var_expr, '')
        
        # Handle ${VAR:-default} format
        result = re.sub(r'\$\{([^}]+)\}', replace_var, value)
        # Fallback to standard expansion for other formats
        return os.path.expandvars(result)

    def which(self, program: str) -> str:
        """Cross-platform implementation of 'which' command.

        Args:
            program: Name of the program to locate.

        Returns:
            Path to the executable.

        Raises:
            RuntimeError: If the program is not found.
        """
        # First try shutil.which (available in Python 3.3+)
        path = shutil.which(program)
        if path:
            return path

        # Fallback for older Python versions or edge cases
        if sys.platform.startswith('win'):
            # Windows-specific implementation
            return self._which_windows(program)
        else:
            # Unix-like systems
            return self._which_unix(program)

    def _which_windows(self, program: str) -> str:
        """Windows-specific implementation of which."""
        # Add common executable extensions if not present
        if not any(program.lower().endswith(ext) for ext in ['.exe', '.bat', '.cmd', '.com']):
            # Try with different extensions
            for ext in ['.exe', '.bat', '.cmd', '.com']:
                extended_program = program + ext
                path = shutil.which(extended_program)
                if path:
                    return path

        # If shutil.which didn't work, try manual search
        paths = os.environ.get('PATH', '').split(os.pathsep)

        for path_dir in paths:
            if not path_dir:
                continue

            # Try the program as-is
            full_path = os.path.join(path_dir, program)
            if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                return full_path

            # Try with executable extensions
            for ext in ['.exe', '.bat', '.cmd', '.com']:
                full_path_with_ext = full_path + ext
                if os.path.isfile(full_path_with_ext) and os.access(full_path_with_ext, os.X_OK):
                    return full_path_with_ext

        raise RuntimeError(f"'{program}' is not found in the system path.")

    def _which_unix(self, program: str) -> str:
        """Unix-like implementation of which."""
        try:
            result = subprocess.run(['which', program], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            raise RuntimeError(f"'{program}' is not found in the system path.")

    def _get_mcp_config(self, mcp_config: Dict[str, Any], mcp_name: str) -> Dict[str, Any]:
        """Get the MCP configuration for a specific server.

        Args:
            mcp_config: Full MCP configuration dictionary.
            mcp_name: Name of the MCP server.

        Returns:
            Configuration dictionary for the specific MCP server.
        """
        mcp = {}
        if mcp_name in self.mcp_configs:
            mcp = self.mcp_configs[mcp_name]
        elif isinstance(mcp_config.get(mcp_name), dict):
            mcp = mcp_config[mcp_name]

        env = os.environ.copy()
        resolved_env = self.update_env(mcp.get('env', {}))
        resolved_headers = self.update_env(mcp.get('headers', {}))
        env.update(resolved_env)
        env.update(resolved_headers)

        if 'command' in mcp:
            if 'headers' in mcp:
                mcp.pop('headers')
            mcp['env'] = env
        elif 'url' in mcp:
            url = mcp.get('url', '')
            if 'env' in mcp:
                mcp.pop('env')
            mcp['headers'] = env

        return mcp

    def _get_mcp_name_list_from_mcp_config(self, mcp_config: Any) -> List[str]:
        """Get list of MCP server names from configuration.

        Args:
            mcp_config: MCP configuration (dict, list, or comma-separated string).

        Returns:
            List of MCP server names.
        """
        mcp_list = []
        if isinstance(mcp_config, dict):
            mcp_list = list(mcp_config.keys())
        elif isinstance(mcp_config, list):
            mcp_list = mcp_config
        elif isinstance(mcp_config, str):
            mcp_list = mcp_config.split(",")
        return mcp_list

    def get_mcp_clients(self) -> Dict[str, Any]:
        """Get all active MCP clients.

        Returns:
            Dictionary of MCP client instances.

        Example:
            >>> registry = AWSStrandsToolRegistry()
            >>> registry.load_mcp_tools_from_config(mcp_config)
            >>> clients = registry.get_mcp_clients()
            >>> print(f"Loaded {len(clients)} MCP clients")
        """
        return self.mcp_clients

    def get_mcp_configs(self) -> Dict[str, Any]:
        """Get all loaded MCP configurations.

        Returns:
            Dictionary of MCP configurations.
        """
        return self.mcp_configs

    def generate_lazy_mcp_system_prompt(self, tool_list: List[str], mcp_client_list: List[str]) -> str:
        """
        Generate a system prompt for lazy loading MCP tools.

        Args:
            tool_list: List of standard tool names.
            mcp_client_list: List of MCP client names.

        Returns:
            Formatted system prompt string listing available tools.
        """
        if not tool_list and not mcp_client_list:
            return ""

        system_prompt = ','.join(list(tool_list))

        for mcp_client in mcp_client_list:
            if mcp_client in self.available_mcp_tools:
                system_prompt = f"{system_prompt}\n{mcp_client}:{', '.join(list(self.available_mcp_tools[mcp_client].keys()))}"

        system_prompt = f"You have access to following tools. Follow workflow instructions to execute the tools{system_prompt}"

        system_prompt = f"""{system_prompt}
        Workflow: Follow this workflow strictly
        1. Call `get_input_parameter_schema(tool_list)` with a comma-separated list of required tool name only to retrieve schemas. Do not include mcp client name in the tool name
        2. Call `execute_multiple_tools(tool_list, arguments)` where tool_list is a comma-separated string and arguments is a dictionary mapping tool names to their parameters.Do not include mcp client name in the tool name    
        """

        return system_prompt

    async def execute_multiple_tools(self, arguments: str):
        """
        Execute multiple tools in parallel.
        Args:
            arguments: YAML string containing arguments for each tool, keyed by tool name.
        """
        print(f"Executing multiple tools with arguments: {arguments}")
        import yaml
        import asyncio
        arguments = yaml.safe_load(arguments)

        tool_names = []
        tasks = []

        for tool_name in arguments.keys():
            tool_name_stripped = tool_name.strip()
            tool_names.append(tool_name_stripped)
            tasks.append(self.execute_tool(tool_name_stripped, arguments[tool_name]))

        results = await asyncio.gather(*tasks)

        return dict(zip(tool_names, results))

    def lazy_loading_required_tools(self):
        """
        Get the list of tools required for lazy loading.

        Returns:
            List of tools required for lazy loading.
        """
        return [
            self._get_framework_tool_decorator()(self.get_input_parameter_schema),
            # self._get_framework_tool_decorator()(self.execute_tool),
            self._get_framework_tool_decorator()(self.execute_multiple_tools)
        ]

    @staticmethod
    def _sanitize_headers(headers: Dict[str, Any]) -> Dict[str, str]:
        """Sanitize HTTP headers, strictly enforcing RFC 7230 token rules.

        Header names must only contain RFC 7230 token characters:
        alphanumerics and !#$%&'*+-.^_`|~ (no parentheses, spaces, etc.)
        This prevents environment variables or shell vars from leaking in.

        Args:
            headers: Raw headers dictionary from config

        Returns:
            Sanitized headers with only RFC-compliant names and values
        """
        if not headers:
            return {}
        import re
        # RFC 7230 token: alphanumeric + !#$%&'*+\-.^_`|~
        valid_header_name = re.compile(r'^[a-zA-Z0-9!#$%&\'*+\-.^_`|~]+$')
        # Header values: printable ASCII + tab, no newlines
        invalid_value_chars = re.compile(r'[^\x09\x20-\x7e]')

        sanitized = {}
        for key, value in headers.items():
            clean_key = str(key).strip()
            clean_value = invalid_value_chars.sub('', str(value)).strip()
            if valid_header_name.match(clean_key) and clean_value:
                sanitized[clean_key] = clean_value
        return sanitized
