from typing import Dict, Optional, Any, List, Literal
from pydantic import BaseModel, Field

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
    registered_via: Literal["config", "dynamic", "registry"] = "dynamic"


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


class AgentDeregistration(BaseModel):
    """Payload for deregistering an agent."""
    name: str


class AgentLifecycleAction(BaseModel):
    """Payload for executing a lifecycle action on an agent."""
    action: Literal["start", "stop", "redeploy"]


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

class Config(BaseModel):
    """Root configuration model."""
    agents: Dict[str, Any]
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
