"""Tool registry and tool loading service."""

import logging
from typing import Dict, Any, Optional

from oai_agent_core.utils.exceptions import ToolLoadingError


class ToolService:
    """Service for tool registry and tool loading.

    Handles:
    - Creating tool registries
    - Loading tools from configuration
    - Managing tool registration
    - Providing access to loaded tools
    """

    def __init__(self, project_root: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the tool service.

        Args:
            project_root: Project root directory
            logger: Optional logger instance
        """
        self.project_root = project_root
        self.logger = logger or logging.getLogger(__name__)
        self._registries: Dict[str, Any] = {}

    def create_registry(self, framework: str = 'base') -> Any:
        """Create a tool registry for a specific framework.

        Args:
            framework: Framework name (base, crewai, langchain, etc.)

        Returns:
            Tool registry instance

        Raises:
            ToolLoadingError: If registry creation fails
        """
        try:
            if framework.lower() == 'base':
                from oai_agent_core.core.base_tool_registry import BaseToolRegistry
                # BaseToolRegistry is abstract, so return None
                # Subclasses must provide concrete implementation
                return None
            else:
                # Framework-specific registries should be created by subclasses
                raise NotImplementedError(f"Framework '{framework}' not supported")
        except Exception as e:
            raise ToolLoadingError(f"Failed to create tool registry: {e}")

    def load_tools(self, registry: Any, tools_config: Dict[str, Any]) -> int:
        """Load tools into the registry.

        Args:
            registry: Tool registry instance
            tools_config: Tools configuration

        Returns:
            Number of tools loaded
        """
        if not registry:
            self.logger.warning("Tool registry not initialized")
            return 0

        try:
            registry.load_tools_from_config(tools_config)

            # Get tool count - support both get_tools() method and tools attribute
            if hasattr(registry, 'get_tools') and callable(registry.get_tools):
                tool_count = len(registry.get_tools())
            elif hasattr(registry, 'tools'):
                tool_count = len(registry.tools)
            else:
                tool_count = 0

            self.logger.info(f"Loaded {tool_count} tools into registry")
            return tool_count
        except Exception as e:
            self.logger.error(f"Failed to load tools: {e}")
            return 0

    def load_mcp_tools(self, registry: Any, mcp_config: Dict[str, Any]) -> int:
        """Load MCP tools into the registry.

        Args:
            registry: Tool registry instance
            mcp_config: MCP configuration

        Returns:
            Number of MCP servers loaded
        """
        if not registry:
            self.logger.warning("Tool registry not initialized")
            return 0

        try:
            registry.load_mcp_config(mcp_config)
            mcp_count = len(registry.mcp_configs)
            self.logger.info(f"Loaded {mcp_count} MCP servers into registry")
            return mcp_count
        except Exception as e:
            self.logger.error(f"Failed to load MCP tools: {e}")
            return 0

    def get_tool(self, registry: Any, tool_name: str) -> Optional[Any]:
        """Get a specific tool from the registry.

        Args:
            registry: Tool registry instance
            tool_name: Name of the tool

        Returns:
            Tool instance or None if not found
        """
        if not registry:
            return None
        return registry.get_tool(tool_name)
