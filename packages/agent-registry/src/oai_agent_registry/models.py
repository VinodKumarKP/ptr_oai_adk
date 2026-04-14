from typing import Dict, Optional, Any, List
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    """Configuration for a single agent."""
    name: Optional[str] = None
    endpoint: str
    enabled: bool = True
    timeout: int = 300
    description: Optional[str] = None
    agent_class: Optional[str] = None
    tools: List[str] = []


class AgentRegistration(BaseModel):
    """Payload for registering an agent."""
    name: str
    description: Optional[str] = None
    endpoint: str
    agent_class: Optional[str] = None
    tools: List[str] = []


class AgentDeregistration(BaseModel):
    """Payload for deregistering an agent."""
    name: str


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
    end_port: int = 8100
    auth_enabled: bool = False
    force_auth: bool = False
    api_key: Optional[str] = None

class Config(BaseModel):
    """Root configuration model."""
    agents: Dict[str, Any]
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
