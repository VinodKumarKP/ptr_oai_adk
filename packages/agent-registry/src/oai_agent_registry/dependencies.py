from oai_agent_registry.services.registry import AgentRegistry

# This is a simple in-memory instance.
# For a more robust application, you might want to manage this differently.
registry_instance = AgentRegistry()

def get_registry() -> AgentRegistry:
    """Dependency to get the registry instance."""
    return registry_instance
