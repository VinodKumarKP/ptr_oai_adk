import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict
from ruamel.yaml import YAML

from oai_mcp_server_core.core.exceptions import ConfigurationError


class MCPServerConfig(BaseModel):
    """
    Configuration model for MCP servers.
    """
    description: str
    port: Optional[int] = None

    model_config = ConfigDict(extra='allow')


def get_server_config(server_name: str, source_file: str = None) -> MCPServerConfig:
    """
    Get the server configuration from the directory servers_config using the server name.
    Supports .json, .yaml, and .yml files.
    
    Args:
        server_name: Name of the server
        source_file: Optional source file path to resolve config directory relative to
        
    Returns:
        MCPServerConfig: Parsed server configuration
        
    Raises:
        ConfigurationError: If configuration file is not found or invalid
    """
    file_root = source_file if source_file else __file__
    # Adjust path traversal based on where this file is located relative to project root
    # Assuming this file is in oai_mcp_server_core/config/settings.py
    # and servers_config is in the project root
    file_root = Path(file_root).parent.parent.parent
    config_dir = os.path.join(file_root, "servers_config")
    yaml_loader = YAML()

    # Check for JSON, YAML, and YML files in order
    for ext, loader in [
        (".yaml", lambda f: yaml_loader.load(f)),
        (".yml", lambda f: yaml_loader.load(f)),
        (".json", lambda f: json.load(f))
    ]:
        config_path = os.path.join(config_dir, f"{server_name}{ext}")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = loader(f)
                return MCPServerConfig(**config_data)
            except Exception as e:
                raise ConfigurationError(f"Failed to load configuration for {server_name}: {str(e)}")

    raise ConfigurationError(
        f"Server name configuration {server_name}.json/.yaml/.yml not found in servers_config directory {config_dir}"
    )
