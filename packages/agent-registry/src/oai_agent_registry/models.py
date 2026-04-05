from typing import Dict, Optional, Any
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    """Configuration for a single agent."""
    name: Optional[str] = None
    endpoint: str
    enabled: bool = True
    timeout: int = 300
    description: Optional[str] = None

class RegistryConfig(BaseModel):
    """Configuration for the registry server."""
    host: str = "0.0.0.0"
    port: int = 8081
    default_timeout: int = 300
    enable_cors: bool = True
    log_requests: bool = True
    strip_prefix: bool = True

class Config(BaseModel):
    """Root configuration model."""
    agents: Dict[str, Any]
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
