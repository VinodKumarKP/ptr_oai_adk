"""Tool Loading Strategies for BaseToolRegistry

This module implements the Strategy pattern for loading different types of tools:
- Framework-specific built-in tools (CrewAI, LangChain, etc.)
- Custom Python class-based tools
- Python function-based tools
- MCP (Model Context Protocol) tools

Each strategy encapsulates the logic for loading a specific tool type,
making it easy to test, extend, and modify individual loading behaviors.

## Architecture

    ToolLoadingStrategy (Abstract Base)
        ├── FrameworkToolStrategy
        ├── PythonClassStrategy
        ├── FunctionStrategy
        └── MCPStrategy
    
    ToolLoadingContext (Orchestrator)
        └── Uses strategies to load different tool types

## Design Benefits

✅ Open/Closed Principle: Easy to add new tool types without modifying existing code
✅ Single Responsibility: Each strategy handles one tool type
✅ Testability: Each strategy can be unit tested independently
✅ Clarity: Clear separation of concerns
✅ Extensibility: Custom tool types can be added as new strategies
"""

import logging
import os
import sys
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

from oai_agent_core.core.exceptions import (
    ToolLoadingError,
    ToolConfigurationError,
    MCPLoadingError,
)
from oai_agent_core.utils.dynamic_class_loader import DynamicClassLoader


class ToolLoadingStrategy(ABC):
    """Abstract base class for tool loading strategies.
    
    Each strategy knows how to load a specific type of tool from configuration.
    Strategies are used by ToolLoadingContext to dispatch to the appropriate
    loading logic based on tool configuration.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the strategy.
        
        Args:
            logger: Optional logger instance for this strategy
        """
        self.logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    def can_handle(self, tool_config: Dict[str, Any]) -> bool:
        """Check if this strategy can handle the given tool configuration.
        
        Args:
            tool_config: The tool configuration dictionary
            
        Returns:
            True if this strategy can handle the tool, False otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    def load(self, tool_name: str, tool_config: Dict[str, Any], 
             registry: Any) -> Any:
        """Load a tool using this strategy.
        
        Args:
            tool_name: Name of the tool being loaded
            tool_config: Configuration for the tool
            registry: The tool registry instance (for shared state)
            
        Returns:
            The loaded tool instance/module
            
        Raises:
            ToolLoadingError: If tool loading fails
            ToolConfigurationError: If configuration is invalid
        """
        raise NotImplementedError


class FrameworkToolStrategy(ToolLoadingStrategy):
    """Strategy for loading framework-specific built-in tools.
    
    Handles tools that come with agent frameworks like CrewAI, LangChain, etc.
    These are typically imported from framework modules and may need
    framework-specific initialization.
    
    Configuration Example:
        tools:
          calculator:
            module: framework_tools
            # No class_name - indicates framework-specific tool
    """
    
    def can_handle(self, tool_config: Dict[str, Any]) -> bool:
        """Check if this is a framework-specific tool.
        
        Framework tools have a 'module' key and the module is a framework tool.
        """
        if 'module' not in tool_config:
            return False
        
        module_name = tool_config.get('module', '')
        # Check if it's a framework tool (subclasses should override
        # _is_framework_builtin_tool logic here)
        return module_name in [
            'crewai_tools',
            'langchain_tools',
            'aws_strands_tools',
            'anthropic_tools',
            'openai_tools',
            'framework_tools'
        ]
    
    def load(self, tool_name: str, tool_config: Dict[str, Any], 
             registry: Any) -> Any:
        """Load a framework-specific tool.
        
        Args:
            tool_name: Name of the tool
            tool_config: Configuration dict with 'module' key
            registry: Tool registry instance
            
        Returns:
            Loaded framework tool
        """
        module_name = tool_config.get('module')
        
        try:
            # Import the framework tool module
            tool = DynamicClassLoader.dynamic_import(module_name, tool_name)
            
            self.logger.debug(f"Loaded framework tool '{tool_name}' from {module_name}")
            return tool
        except Exception as e:
            raise ToolLoadingError(
                f"Failed to load framework tool '{tool_name}' from {module_name}: {e}"
            )


class PythonClassStrategy(ToolLoadingStrategy):
    """Strategy for loading custom Python class-based tools.
    
    Handles tools defined as Python classes with optional initialization
    parameters. Classes are instantiated with provided parameters.
    
    Configuration Example:
        tools:
          custom_tool:
            module: my_tools
            class: CustomTool
            params:
              api_key: xyz
              timeout: 30
    """
    
    def can_handle(self, tool_config: Dict[str, Any]) -> bool:
        """Check if this is a Python class-based tool.
        
        Class-based tools have both 'module' and 'class' keys.
        """
        return 'module' in tool_config and 'class' in tool_config
    
    def load(self, tool_name: str, tool_config: Dict[str, Any], 
             registry: Any) -> Any:
        """Load a custom Python class-based tool.
        
        Args:
            tool_name: Name of the tool
            tool_config: Configuration with 'module' and 'class' keys
            registry: Tool registry instance
            
        Returns:
            Instantiated tool class
        """
        module_name = tool_config.get('module')
        class_name = tool_config.get('class')
        params = tool_config.get('params', {})
        base_path = tool_config.get('base_path')
        
        # Handle base_path if provided
        if base_path:
            base_path = self._resolve_base_path(base_path, registry.project_root)
            sys.path.insert(0, os.path.expanduser(base_path))
        
        try:
            # Load the class using static method
            tool_class = DynamicClassLoader.dynamic_import(module_name, class_name)
            
            # Instantiate with parameters
            tool_instance = tool_class(**params)
            
            self.logger.debug(
                f"Loaded custom class tool '{tool_name}' from {module_name}.{class_name}"
            )
            return tool_instance
        except TypeError as e:
            raise ToolConfigurationError(
                f"Invalid parameters for {class_name}: {e}"
            )
        except Exception as e:
            raise ToolLoadingError(
                f"Failed to load class tool '{tool_name}' from {module_name}.{class_name}: {e}"
            )
    
    @staticmethod
    def _resolve_base_path(base_path: str, project_root: Optional[str]) -> str:
        """Resolve relative base paths from project root.
        
        Args:
            base_path: Base path from configuration
            project_root: Project root directory
            
        Returns:
            Resolved absolute path
        """
        if project_root and base_path.startswith('.'):
            return os.path.join(project_root, base_path)
        return base_path


class FunctionStrategy(ToolLoadingStrategy):
    """Strategy for loading function-based tools.
    
    Handles tools defined as individual functions, either discovered from
    a module or provided directly. Functions can be decorated with @tool
    or used directly.
    
    Configuration Examples:
        tools:
          # Discover functions from module
          my_functions:
            module: my_tools
            function_list: [func1, func2]
            
          # Single function
          single_func:
            function: path.to.my_function
    """
    
    def can_handle(self, tool_config: Dict[str, Any]) -> bool:
        """Check if this is a function-based tool.
        
        Function tools have 'module' with no 'class', or have 'function' key.
        """
        has_module_no_class = 'module' in tool_config and 'class' not in tool_config
        has_function = 'function' in tool_config
        return has_module_no_class or has_function
    
    def load(self, tool_name: str, tool_config: Dict[str, Any], 
             registry: Any) -> Any:
        """Load function-based tools.
        
        Args:
            tool_name: Name of the tool
            tool_config: Configuration with 'module' or 'function' key
            registry: Tool registry instance
            
        Returns:
            Loaded function or module with functions
        """
        if 'function' in tool_config:
            return self._load_single_function(tool_name, tool_config, registry)
        else:
            return self._load_module_functions(tool_name, tool_config, registry)
    
    def _load_single_function(self, tool_name: str, tool_config: Dict[str, Any], 
                              registry: Any) -> Callable:
        """Load a single function by path.
        
        Args:
            tool_name: Tool name
            tool_config: Config with 'function' key (e.g., 'module.submodule.func')
            registry: Tool registry
            
        Returns:
            Loaded function
        """
        func_path = tool_config.get('function')
        
        try:
            # Parse path like "module.submodule.function_name"
            parts = func_path.rsplit('.', 1)
            if len(parts) != 2:
                raise ToolConfigurationError(f"Invalid function path: {func_path}")
            
            module_name, func_name = parts
            func = DynamicClassLoader.dynamic_import(module_name, func_name)
            
            self.logger.debug(f"Loaded function tool '{tool_name}' from {func_path}")
            return func
        except Exception as e:
            raise ToolLoadingError(f"Failed to load function '{tool_name}': {e}")
    
    def _load_module_functions(self, tool_name: str, tool_config: Dict[str, Any], 
                               registry: Any) -> Dict[str, Callable]:
        """Load multiple functions from a module.
        
        Args:
            tool_name: Tool group name
            tool_config: Config with 'module' and optional 'function_list'
            registry: Tool registry
            
        Returns:
            Dictionary of function_name -> function
        """
        module_name = tool_config.get('module')
        function_list = tool_config.get('function_list', [])
        function_params = tool_config.get('function_params', {})
        base_path = tool_config.get('base_path')
        
        if base_path:
            base_path = PythonClassStrategy._resolve_base_path(
                base_path, registry.project_root
            )
            sys.path.insert(0, os.path.expanduser(base_path))
        
        try:
            # Import the module
            functions_module = DynamicClassLoader.dynamic_import_module(module_name)
            
            # Get the functions as a dictionary
            functions = {}
            for func_name in function_list:
                func = getattr(functions_module, func_name, None)
                if func is None:
                    self.logger.warning(
                        f"Function '{func_name}' not found in module {module_name}"
                    )
                else:
                    functions[func_name] = func
            
            self.logger.debug(
                f"Loaded {len(functions)} functions from {module_name}"
            )
            return functions
        except Exception as e:
            raise ToolLoadingError(
                f"Failed to load functions from {module_name}: {e}"
            )


class MCPStrategy(ToolLoadingStrategy):
    """Strategy for loading MCP (Model Context Protocol) tools.
    
    Handles Model Context Protocol servers which provide tools via
    stdio or HTTP/SSE transports. MCP servers are loaded asynchronously.
    
    Configuration Example:
        mcp:
          my_mcp_server:
            command: python
            args: ["-m", "my_mcp_server"]
            env:
              API_KEY: ${API_KEY}
            # OR
            url: http://localhost:8000
            headers:
              Authorization: Bearer token
    """
    
    def can_handle(self, mcp_config: Dict[str, Any]) -> bool:
        """Check if this is an MCP configuration.
        
        MCP configs have 'command'+'args' (stdio) or 'url' (HTTP/SSE).
        """
        has_stdio = 'command' in mcp_config and 'args' in mcp_config
        has_http = 'url' in mcp_config
        return has_stdio or has_http
    
    def load(self, mcp_name: str, mcp_config: Dict[str, Any], 
             registry: Any) -> Dict[str, Any]:
        """Load MCP server configuration.
        
        Validates MCP config and prepares it for server instantiation.
        Handles env variable resolution and header/env cleanup based on transport type.
        
        Args:
            mcp_name: Name of the MCP server
            mcp_config: MCP configuration (stdio or HTTP)
            registry: Tool registry instance (for update_env method)
            
        Returns:
            Validated MCP configuration with cleaned env/headers
        """
        try:
            # Make a copy to avoid modifying the original
            resolved_config = dict(mcp_config)
            
            # Validate configuration
            if not self.can_handle(resolved_config):
                raise ToolConfigurationError(
                    f"MCP '{mcp_name}' must have either 'command'+'args' (stdio) "
                    f"or 'url' (HTTP/SSE)"
                )
            
            # Handle environment variables and headers
            env = os.environ.copy()
            resolved_env = {}
            resolved_headers = {}
            
            # Use registry's update_env if available
            if hasattr(registry, 'update_env'):
                resolved_env = registry.update_env(resolved_config.get('env', {}))
                resolved_headers = registry.update_env(resolved_config.get('headers', {}))
            else:
                resolved_env = self._resolve_env_vars(resolved_config.get('env', {}))
                resolved_headers = self._resolve_env_vars(resolved_config.get('headers', {}))
            
            env.update(resolved_env)
            env.update(resolved_headers)
            
            # Handle stdio (command-based) transport
            if 'command' in resolved_config:
                cmd = resolved_config['command']
                # Use registry's which method if available (for testing/flexibility)
                if hasattr(registry, 'which'):
                    resolved_config['command'] = registry.which(cmd)
                else:
                    resolved_config['command'] = self._which(cmd)
                if not resolved_config['command']:
                    raise ToolConfigurationError(f"MCP command not found: {cmd}")
                # For stdio: remove headers key, keep env
                if 'headers' in resolved_config:
                    resolved_config.pop('headers')
                resolved_config['env'] = env
            
            # Handle HTTP/SSE (URL-based) transport
            elif 'url' in resolved_config:
                # For HTTP: remove env key, keep headers
                if 'env' in resolved_config:
                    resolved_config.pop('env')
                resolved_config['headers'] = env
            
            self.logger.debug(f"Loaded MCP configuration for '{mcp_name}'")
            return resolved_config
        except ToolConfigurationError:
            raise
        except Exception as e:
            raise MCPLoadingError(
                f"Failed to load MCP configuration '{mcp_name}': {e}"
            )
    
    @staticmethod
    def _resolve_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve environment variables in config values.
        
        Replaces ${VAR_NAME} with environment variable values.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with env vars resolved
        """
        resolved = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                var_name = value[2:-1]
                resolved[key] = os.environ.get(var_name, value)
            elif isinstance(value, dict):
                resolved[key] = MCPStrategy._resolve_env_vars(value)
            elif isinstance(value, list):
                resolved[key] = [
                    MCPStrategy._resolve_env_vars({'v': v}).get('v')
                    if isinstance(v, str) and v.startswith('${')
                    else v
                    for v in value
                ]
            else:
                resolved[key] = value
        return resolved
    
    @staticmethod
    def _which(command: str) -> Optional[str]:
        """Find command in PATH.
        
        Args:
            command: Command name to find
            
        Returns:
            Full path to command or None if not found
        """
        try:
            result = subprocess.run(
                ['which', command],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None


class ToolLoadingContext:
    """Orchestrator for tool loading using strategies.
    
    The ToolLoadingContext dispatches to the appropriate strategy based on
    the tool configuration. This provides a clean separation between
    configuration analysis and tool loading logic.
    
    ## Usage
    
        context = ToolLoadingContext(
            strategies=[
                FrameworkToolStrategy(),
                PythonClassStrategy(),
                FunctionStrategy(),
                MCPStrategy(),
            ],
            logger=logger
        )
        
        # Load a tool
        tool = context.load_tool('my_tool', tool_config, registry)
        
        # Load MCP
        mcp_config = context.load_mcp('my_mcp', mcp_config_dict, registry)
    """
    
    def __init__(self, strategies: List[ToolLoadingStrategy], 
                 logger: Optional[logging.Logger] = None):
        """Initialize the context with strategies.
        
        Args:
            strategies: List of tool loading strategies
            logger: Optional logger instance
        """
        self.strategies = strategies
        self.logger = logger or logging.getLogger(__name__)
    
    def load_tool(self, tool_name: str, tool_config: Dict[str, Any], 
                  registry: Any) -> Any:
        """Load a tool using the appropriate strategy.
        
        Tries each strategy in order until one can handle the configuration.
        
        Args:
            tool_name: Name of the tool
            tool_config: Tool configuration
            registry: Tool registry instance
            
        Returns:
            Loaded tool
            
        Raises:
            ToolConfigurationError: If no strategy can handle the config
            ToolLoadingError: If tool loading fails
        """
        for strategy in self.strategies:
            if strategy.can_handle(tool_config):
                return strategy.load(tool_name, tool_config, registry)
        
        raise ToolConfigurationError(
            f"No strategy found to handle tool '{tool_name}' with config: {tool_config}"
        )
    
    def load_mcp(self, mcp_name: str, mcp_config: Dict[str, Any], 
                 registry: Any) -> Dict[str, Any]:
        """Load MCP configuration using MCPStrategy.
        
        Args:
            mcp_name: Name of the MCP server
            mcp_config: MCP configuration
            registry: Tool registry instance
            
        Returns:
            Validated MCP configuration
        """
        # Find MCPStrategy in strategies list
        mcp_strategy = None
        for strategy in self.strategies:
            if isinstance(strategy, MCPStrategy):
                mcp_strategy = strategy
                break
        
        if not mcp_strategy:
            raise MCPLoadingError("MCPStrategy not registered in context")
        
        return mcp_strategy.load(mcp_name, mcp_config, registry)
