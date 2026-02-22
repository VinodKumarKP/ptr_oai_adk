import pytest
import os
import json
from oai_mcp_server_core.config.settings import get_server_config, MCPServerConfig
from oai_mcp_server_core.core.exceptions import ConfigurationError

def test_get_server_config_yaml(tmp_path):
    # Create mock config structure
    config_dir = tmp_path / "servers_config"
    config_dir.mkdir()
    
    config_file = config_dir / "test_server.yaml"
    config_file.write_text("description: Test Server\nport: 8080")
    
    # Mock file root resolution
    # The function looks for ../../servers_config relative to source_file
    # So source_file should be at tmp_path/src/module/file.py
    src_file = tmp_path / "src" / "module" / "file.py"
    src_file.parent.mkdir(parents=True)
    
    config = get_server_config("test_server", source_file=str(src_file))
    
    assert config.description == "Test Server"
    assert config.port == 8080

def test_get_server_config_json(tmp_path):
    config_dir = tmp_path / "servers_config"
    config_dir.mkdir()
    
    config_file = config_dir / "test_server.json"
    config_file.write_text('{"description": "JSON Server", "port": 9090}')
    
    src_file = tmp_path / "src" / "module" / "file.py"
    src_file.parent.mkdir(parents=True)
    
    config = get_server_config("test_server", source_file=str(src_file))
    
    assert config.description == "JSON Server"
    assert config.port == 9090

def test_config_not_found(tmp_path):
    src_file = tmp_path / "src" / "module" / "file.py"
    src_file.parent.mkdir(parents=True)
    (tmp_path / "servers_config").mkdir()
    
    with pytest.raises(ConfigurationError):
        get_server_config("non_existent", source_file=str(src_file))
