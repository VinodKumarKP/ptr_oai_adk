from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, model_validator

class ClientConfig(BaseModel):
    """
    Configuration for the AgentClient.
    
    You must provide either a 'url' for connecting to an existing server,
    or a 'command' and 'args' to start a new local server.
    """
    # Server connection options
    url: Optional[HttpUrl] = Field(None, description="The base URL of the agent server.")
    command: Optional[str] = Field(None, description="The command to start a local server process.")
    args: Optional[List[str]] = Field(None, description="A list of arguments for the command.")
    
    # Connection settings
    headers: Dict[str, str] = Field({}, description="Custom headers to send with each request.")
    timeout: int = Field(300, description="Default timeout for requests in seconds.")
    
    # Server management settings
    port: int = Field(8000, description="The port for the local server, if managed.")
    host: str = Field("localhost", description="The host for the local server, if managed.")
    health_endpoint: str = Field("/health", description="The health check endpoint for the server.")
    startup_timeout: int = Field(30, description="Timeout in seconds for waiting for the managed server to start.")
    
    # Logging settings
    log_level: str = Field("INFO", description="Logging level for the client (e.g., 'INFO', 'DEBUG').")

    @model_validator(mode='after')
    def check_server_config(cls, values: Any) -> Any:
        url = values.url
        command = values.command
        
        if url and command:
            raise ValueError("Provide either 'url' or 'command'/'args', not both.")
        if not url and not command:
            raise ValueError("You must provide either 'url' or 'command'/'args'.")
        
        return values
