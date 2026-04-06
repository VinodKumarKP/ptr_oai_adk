from oai_mcp_registry.services.registry import MCPRegistry

registry_instance = MCPRegistry()

def get_registry() -> MCPRegistry:
    """Dependency to get the registry instance."""
    return registry_instance
