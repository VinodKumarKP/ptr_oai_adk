from typing import Dict, Optional
from pydantic import BaseModel, Field

class ServerConfig(BaseModel):
    """Represents a single upstream MCP server."""
    endpoint: Optional[str] = None
    port: Optional[int] = None
    description: str = "A proxied MCP server"

class RegistryConfig(BaseModel):
    """Configuration for the registry server."""
    host: str = "0.0.0.0"
    port: int = 8081
    enable_auto_discovery: bool = False
    start_port: int = 8000
    end_port: int = 8100

class AppConfig(BaseModel):
    """Represents the JSON configuration file."""
    servers: Dict[str, ServerConfig]
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
