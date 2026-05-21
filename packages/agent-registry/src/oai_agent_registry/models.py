from typing import Dict, Optional, Any, List, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Environment variable models
# ---------------------------------------------------------------------------

class EnvVarRequirement(BaseModel):
    """A single environment variable required by an agent at runtime."""
    name: str                        # e.g. "OPENAI_API_KEY"
    default: Optional[str] = None   # value from ${VAR:-default}, if any
    required: bool = True            # False when a default is available
    sensitive: bool = False          # True → mask value in API responses and UI


# ---------------------------------------------------------------------------
# Bulk / Discovery models
# ---------------------------------------------------------------------------

class AgentDiscoveryItem(BaseModel):
    """A single agent discovered from a Git repository."""
    name: str
    description: Optional[str] = None
    framework: Optional[str] = None       # mapped from YAML 'type' field
    agent_type: Optional[str] = None      # raw 'type' value from YAML
    tags: Optional[List[str]] = Field(default_factory=list)
    prompts: Optional[List[str]] = Field(default_factory=list)
    port: Optional[int] = None
    source: Optional[str] = None
    status: str = "available"             # "available" | "already_registered" | "invalid"
    config_file: Optional[str] = None    # filename of the YAML in the repo
    error: Optional[str] = None
    required_env: List[EnvVarRequirement] = Field(default_factory=list)  # env vars from YAML `env:` section


class AgentDiscoveryResult(BaseModel):
    """Result of scanning a Git repository for agent config YAMLs."""
    git_repository_url: str
    total_found: int
    available_to_register: int
    already_registered: int
    invalid: int
    agents: List[AgentDiscoveryItem]


class BulkAgentRegistrationRequest(BaseModel):
    """Request body for bulk-registering agents from a Git repository."""
    git_repository_url: str
    agent_names: List[str]
    framework: Optional[Literal["langgraph", "openai", "crewai", "strands"]] = None
    deployment_mode: Literal["docker", "kubernetes", "python_package"] = "docker"
    auth_token: Optional[str] = None     # GitHub PAT for private repos
    config_path: Optional[str] = None   # Custom path to agent config YAMLs in the repo
    env_vars: Optional[Dict[str, str]] = None  # user-supplied runtime env var values


class BulkAgentRegistrationResult(BaseModel):
    """Summary of a bulk agent registration operation."""
    total_registered: int
    successful: List[str]
    failed: List[Dict[str, str]]

class AgentConfig(BaseModel):
    """Configuration for a single agent."""
    name: str = None
    endpoint: Optional[str] = None
    enabled: bool = True
    timeout: int = 300
    description: str = None
    port: Optional[int] = None
    source: Optional[str] = None
    framework: Optional[Literal["langgraph", "openai", "crewai", "strands"]] = None
    prompts: Optional[List[str]] = Field(default_factory=list)
    tags: Optional[List[str]] = Field(default_factory=list)
    current_version: Optional[str] = None
    available_versions: Optional[List[str]] = Field(default_factory=list)
    registered_via: Literal["config", "dynamic", "registry"] = "dynamic"
    deployment_mode: Literal["docker", "kubernetes", "python_package"] = "docker"
    env_vars: Optional[Dict[str, str]] = None
    sensitive_vars: Optional[List[str]] = None  # names of env_vars that should be masked in responses


class AgentRegistration(BaseModel):
    """Payload for registering an agent."""
    name: str
    description: str = None
    endpoint: Optional[str] = None
    port: Optional[int] = None
    source: Optional[str] = None
    active: bool = True
    registered_via: Literal["config", "dynamic", "registry"] = "dynamic"
    framework: Optional[Literal["langgraph", "openai", "crewai", "strands"]] = None
    prompts: Optional[List[str]] = Field(default_factory=list)
    tags: Optional[List[str]] = Field(default_factory=list)
    current_version: Optional[str] = None
    available_versions: Optional[List[str]] = Field(default_factory=list)
    deployment_mode: Literal["docker", "kubernetes", "python_package", "unknown"] = "unknown"
    env_vars: Optional[Dict[str, str]] = None       # user-supplied runtime env var values
    sensitive_vars: Optional[List[str]] = None      # names of env_vars that should be masked


class AgentDeregistration(BaseModel):
    """Payload for deregistering an agent."""
    name: str


class AgentLifecycleAction(BaseModel):
    """Payload for executing a lifecycle action on an agent."""
    action: Literal["start", "stop", "redeploy", "refresh", "restart", "rebuild", "upgrade", "update", "downgrade", "delete"]
    version: Optional[str] = None
    stream_output: Optional[bool] = False


class RegistryConfig(BaseModel):
    """Configuration for the registry server."""
    host: str = "0.0.0.0"
    port: int = 8081
    default_timeout: int = 300
    enable_cors: bool = True
    log_requests: bool = True
    strip_prefix: bool = True
    enable_auto_discovery: bool = False
    start_port: int = 8000
    end_port: int = 8200
    auth_enabled: bool = False
    force_auth: bool = False
    api_key: Optional[str] = None
    deployment_mode: Literal["docker", "kubernetes", "python_package"] = "docker"
    max_version: int = 10
    # Infra auto-start: when True, runs docker compose up on the infra compose
    # file before the database is initialized so Postgres/Valkey are ready.
    auto_start_infra: bool = False
    infra_compose_file: Optional[str] = None   # absolute or project-relative path; None → bundled docker-compose.yaml
    infra_startup_timeout: int = 60            # seconds to wait for Postgres to accept connections


class AgentAction(BaseModel):
    """Record of a single agent action/lifecycle event."""
    id: int
    agent_name: str
    action: str  # "start", "stop", "rebuild", "update", etc.
    version: Optional[str] = None
    created_at: str  # ISO 8601 timestamp


class AgentActionHistory(BaseModel):
    """Complete action history for an agent."""
    agent_name: str
    total_count: int
    actions: List[AgentAction]


class Config(BaseModel):
    """Root configuration model."""
    agents: Dict[str, Any]
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
