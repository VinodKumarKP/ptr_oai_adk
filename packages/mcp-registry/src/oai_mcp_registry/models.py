from typing import Dict, Optional, Literal, List
from pydantic import BaseModel, Field

class ServerConfig(BaseModel):
    """Represents a single upstream MCP server."""
    endpoint: Optional[str] = None
    enabled: bool = True
    port: Optional[int] = None
    description: str = "A proxied MCP server"
    registered_via: Literal["config", "dynamic", "registry"] = "dynamic"
    source: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)
    current_version: Optional[str] = None
    available_versions: Optional[List[str]] = Field(default_factory=list)
    deployment_mode: Literal["docker", "kubernetes", "python_package"] = "docker"

class ServerRegistration(BaseModel):
    """Payload for registering an MCP server."""
    name: str
    description: str = "A proxied MCP server"
    endpoint: Optional[str] = None
    port: Optional[int] = None
    active: bool = True
    registered_via: Literal["config", "dynamic", "registry"] = "dynamic"
    source: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)
    current_version: Optional[str] = None
    available_versions: Optional[List[str]] = Field(default_factory=list)
    deployment_mode: Literal["docker", "kubernetes", "python_package", "unknown"] = "unknown"

class ServerDeregistration(BaseModel):
    """Payload for deregistering an MCP server."""
    name: str

class RegistryConfig(BaseModel):
    """Configuration for the registry server."""
    host: str = "0.0.0.0"
    port: int = 8081
    enable_auto_discovery: bool = False
    start_port: int = 8000
    end_port: int = 8100
    enable_cors: bool = True

class AppConfig(BaseModel):
    """Represents the JSON configuration file."""
    servers: Dict[str, ServerConfig]
    registry: RegistryConfig = Field(default_factory=RegistryConfig)

class McpServerLifecycleAction(BaseModel):
    """Payload for executing a lifecycle action on an agent."""
    action: Literal["start", "stop", "redeploy", "refresh", "restart", "rebuild", "upgrade", "update", "downgrade", "delete"]
    version: Optional[str] = None
    stream_output: Optional[bool] = False


class ServerAction(BaseModel):
    """Record of a single server action/lifecycle event."""
    id: int
    server_name: str
    action: str  # "start", "stop", "restart", "update", etc.
    version: Optional[str] = None
    created_at: str  # ISO 8601 timestamp


class ServerActionHistory(BaseModel):
    """Complete action history for an MCP server."""
    server_name: str
    total_count: int
    actions: List[ServerAction]
