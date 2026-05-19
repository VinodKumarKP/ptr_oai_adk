from typing import Dict, Optional, Literal, List
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Bulk / Discovery models
# ---------------------------------------------------------------------------

class MCPServerDiscoveryItem(BaseModel):
    """A single MCP server discovered from a Git repository."""
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)
    port: Optional[int] = None
    source: Optional[str] = None
    status: str = "available"       # "available" | "already_registered" | "invalid"
    config_file: Optional[str] = None
    error: Optional[str] = None


class MCPServerDiscoveryResult(BaseModel):
    """Result of scanning a Git repository for MCP server config YAMLs."""
    git_repository_url: str
    total_found: int
    available_to_register: int
    already_registered: int
    invalid: int
    servers: List[MCPServerDiscoveryItem]


class BulkMCPServerRegistrationRequest(BaseModel):
    """Request body for bulk-registering MCP servers from a Git repository."""
    git_repository_url: str
    server_names: List[str]
    deployment_mode: Literal["docker", "kubernetes", "python_package"] = "docker"
    auth_token: Optional[str] = None    # GitHub PAT for private repos
    config_path: Optional[str] = None  # Custom path to server config YAMLs in the repo


class BulkMCPServerRegistrationResult(BaseModel):
    """Summary of a bulk MCP server registration operation."""
    total_registered: int
    successful: List[str]
    failed: List[Dict[str, str]]


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
    # Infra auto-start: when True, runs docker compose up on the infra compose
    # file before the database is initialized so Postgres/Valkey are ready.
    auto_start_infra: bool = False
    infra_compose_file: Optional[str] = None   # absolute path; None → bundled docker-compose.yaml
    infra_startup_timeout: int = 60            # seconds to wait for Postgres to accept connections

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
