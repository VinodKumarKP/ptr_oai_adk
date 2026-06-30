import os
import sys
from unittest.mock import patch, MagicMock

import pytest

from oai_agent_registry.cli import main
from oai_agent_registry.dependencies import registry_instance

@pytest.fixture
def mock_uvicorn():
    with patch("uvicorn.run") as mock_run:
        yield mock_run

@pytest.fixture
def mock_registry_instance():
    with patch("oai_agent_registry.cli.registry_instance") as mock_instance:
        mock_instance.registry_config = MagicMock()
        mock_instance.registry_config.host = "127.0.0.1"
        mock_instance.registry_config.port = 8000
        yield mock_instance

def test_main_default_args(mock_uvicorn, mock_registry_instance):
    test_args = ["agent-registry"]
    with patch.object(sys, "argv", test_args):
        main()
        
    mock_uvicorn.assert_called_once()
    args, kwargs = mock_uvicorn.call_args
    assert kwargs.get("workers") == 1
    assert kwargs.get("proxy_headers") is True
    
def test_main_with_config(mock_uvicorn, mock_registry_instance):
    test_args = ["agent-registry", "--config", "test_config.yaml"]
    with patch.object(sys, "argv", test_args):
        main()
        
    assert mock_registry_instance.config_path == "test_config.yaml"
    mock_registry_instance.load_configuration.assert_called_once()
    mock_registry_instance.initialize_proxies.assert_called_once()

def test_main_config_failure(mock_uvicorn, mock_registry_instance):
    test_args = ["agent-registry", "--config", "test_config.yaml"]
    mock_registry_instance.load_configuration.side_effect = Exception("Config error")
    with patch.object(sys, "argv", test_args), patch("sys.exit") as mock_exit:
        main()
        mock_exit.assert_called_once_with(1)

def test_main_override_args(mock_uvicorn, mock_registry_instance):
    test_args = [
        "agent-registry",
        "--enable-auto-discovery",
        "--start-port", "8000",
        "--end-port", "8010",
        "--auto-start-infra",
        "--infra-compose-file", "infra.yaml",
        "--infra-startup-timeout", "120"
    ]
    with patch.object(sys, "argv", test_args):
        main()
        
    assert mock_registry_instance.registry_config.enable_auto_discovery is True
    assert mock_registry_instance.registry_config.start_port == 8000
    assert mock_registry_instance.registry_config.end_port == 8010
    assert mock_registry_instance.registry_config.auto_start_infra is True
    assert mock_registry_instance.registry_config.infra_compose_file == "infra.yaml"
    assert mock_registry_instance.registry_config.infra_startup_timeout == 120

def test_main_env_vars_agent_registry_url(mock_uvicorn, mock_registry_instance):
    test_args = ["agent-registry"]
    with patch.object(sys, "argv", test_args), patch.dict(os.environ, {"AGENT_REGISTRY_URL": "http://localhost:9090"}):
        main()
        
        assert os.environ.get("AGENT_BASE_URL") == "http" + "://localhost"
        assert os.environ.get("AGENT_BASE_URL_PORT") == "9090"
    
def test_main_env_vars_mcp_base_url(mock_uvicorn, mock_registry_instance):
    test_args = ["agent-registry"]
    with patch.object(sys, "argv", test_args), patch.dict(os.environ, {"AGENT_BASE_URL": "http://example.com", "AGENT_BASE_URL_PORT": "8888"}):
        main()
        
        assert os.environ.get("AGENT_REGISTRY_URL") == "http://example.com:8888"

def test_main_host_port_args(mock_uvicorn, mock_registry_instance):
    test_args = ["agent-registry", "--host", "0.0.0.0", "--port", "5050"]
    with patch.object(sys, "argv", test_args):
        main()
        
    mock_uvicorn.assert_called_once()
    args, kwargs = mock_uvicorn.call_args
    assert kwargs.get("host") == "0.0.0.0"
    assert kwargs.get("port") == 5050
