from inspect import ismethod, isfunction, getmembers
from typing import List
from fastmcp import FastMCP


class MCPRegistry:
    """
    Handles registration of MCP components (tools, prompts, resources).
    """

    def __init__(self, mcp: FastMCP):
        """
        Initialize the registry.
        
        Args:
            mcp: FastMCP instance
        """
        self.mcp = mcp

    def register_tools(self, object_list: List[object]):
        """
        Register public methods from a list of objects as tools.
        
        Args:
            object_list: List of objects containing methods to register as tools
        """
        for obj in object_list:
            self._register_object_methods(obj)

    def _register_object_methods(self, tool_object: object):
        """
        Register public methods of a single object as tools.
        
        Args:
            tool_object: Object containing methods to register
        """
        for method_name, method in getmembers(tool_object):
            if ismethod(method) or isfunction(method):
                # Skip private methods
                if method_name.startswith('_'):
                    continue
                self.mcp.tool()(method)

    # Future expansion for prompts and resources
    # def register_prompts(self, ...):
    #     pass

    # def register_resources(self, ...):
    #     pass
