import logging
import os

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

from oai_agent_server.main import AgentHTTPServer, ServerState, parse_args, main, _configure_structured_logging

@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.agent_name = "test_agent"
    agent.initialize = AsyncMock()
    return agent

def test_server_state():
    state = ServerState()
    assert state.active_requests == 0
    assert state.is_shutting_down is False

def test_agent_http_server_init(mock_agent):
    with patch('oai_agent_server.main.get_logger'), \
         patch('oai_agent_server.main.ConfigManager'), \
         patch('oai_agent_server.main.DatabaseLogger'):
        
        server = AgentHTTPServer(mock_agent, "test_agent")
        assert server.agent_name == "test_agent"
        assert server.app.title == "Agent HTTP Server - test_agent"

@pytest.mark.asyncio
async def test_agent_http_server_startup(mock_agent):
    with patch('oai_agent_server.main.get_logger'), \
         patch('oai_agent_server.main.ConfigManager'), \
         patch('oai_agent_server.main.DatabaseLogger') as MockDBLogger:
        
        server = AgentHTTPServer(mock_agent, "test_agent")
        server.db_logger.initialize = AsyncMock()
        
        await server.startup()
        mock_agent.initialize.assert_called_once()
        server.db_logger.initialize.assert_called_once()

def test_agent_http_server_run(mock_agent):
    with patch('oai_agent_server.main.get_logger'), \
         patch('oai_agent_server.main.ConfigManager'), \
         patch('oai_agent_server.main.DatabaseLogger'), \
         patch('uvicorn.run') as mock_uvicorn:
        
        server = AgentHTTPServer(mock_agent, "test_agent")
        server.run()
        mock_uvicorn.assert_called_once()

def test_parse_args():
    with patch('sys.argv', ['main.py', 'my_agent', '--port', '9000']):
        args = parse_args()
        assert args.agent_name == 'my_agent'
        assert args.port == 9000

def test_server_state_is_agent_ready_default_true():
    state = ServerState()
    assert state.is_agent_ready is True


class TestConfigureStructuredLogging:
    def setup_method(self):
        # Snapshot root handlers/filters so we can restore.
        root = logging.getLogger()
        self._handlers = list(root.handlers)
        self._filters = list(root.filters)
        self._level = root.level

    def teardown_method(self):
        root = logging.getLogger()
        root.handlers = self._handlers
        root.filters = self._filters
        root.level = self._level

    def test_text_mode_default(self, monkeypatch):
        monkeypatch.delenv("LOG_FORMAT", raising=False)
        _configure_structured_logging()
        root = logging.getLogger()
        assert any(f.__class__.__name__ == "RequestIdFilter" for f in root.filters)
        # At least one handler installed
        assert len(root.handlers) >= 1

    def test_json_mode(self, monkeypatch):
        monkeypatch.setenv("LOG_FORMAT", "json")
        try:
            _configure_structured_logging()
        except Exception:
            # If pythonjsonlogger not installed, code falls through to text.
            pass
        root = logging.getLogger()
        assert any(f.__class__.__name__ == "RequestIdFilter" for f in root.filters)


class TestCORSResolution:
    """Test CORS env-var-driven configuration via _setup_middleware()."""

    def _make_server(self, monkeypatch, allowed_origins_env=None):
        agent = MagicMock()
        agent.agent_name = "test_agent"
        if allowed_origins_env is not None:
            monkeypatch.setenv("ALLOWED_ORIGINS", allowed_origins_env)
        else:
            monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
        with patch('oai_agent_server.main.get_logger'), \
             patch('oai_agent_server.main.ConfigManager'), \
             patch('oai_agent_server.main.DatabaseLogger'):
            return AgentHTTPServer(agent, "test_agent")

    def test_default_origins(self, monkeypatch):
        server = self._make_server(monkeypatch, allowed_origins_env="")
        # Server should have been built successfully — CORS middleware added.
        assert server.app is not None

    def test_wildcard_origin(self, monkeypatch):
        server = self._make_server(monkeypatch, allowed_origins_env="*")
        assert server.app is not None

    def test_comma_separated_origins(self, monkeypatch):
        server = self._make_server(monkeypatch, allowed_origins_env="https://a.com,https://b.com")
        assert server.app is not None


class TestSchedulerConditional:
    def test_scheduler_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("ENABLE_SCHEDULER", "false")
        agent = MagicMock()
        agent.agent_name = "test_agent"
        with patch('oai_agent_server.main.get_logger'), \
             patch('oai_agent_server.main.ConfigManager'), \
             patch('oai_agent_server.main.DatabaseLogger'):
            server = AgentHTTPServer(agent, "test_agent")
        # Scheduler-related state remains None / not started.
        assert server.scheduler is None


class TestReadyEndpoint:
    def test_ready_returns_503_when_not_ready(self, monkeypatch):
        agent = MagicMock()
        agent.agent_name = "test_agent"
        with patch('oai_agent_server.main.get_logger'), \
             patch('oai_agent_server.main.ConfigManager'), \
             patch('oai_agent_server.main.DatabaseLogger'):
            server = AgentHTTPServer(agent, "test_agent")
        server.server_state.is_agent_ready = False
        server.app.state.server_state = server.server_state
        client = TestClient(server.app)
        r = client.get("/ready")
        assert r.status_code == 503
        assert r.json()["status"] == "not_ready"

    def test_health_always_ok(self, monkeypatch):
        agent = MagicMock()
        agent.agent_name = "test_agent"
        with patch('oai_agent_server.main.get_logger'), \
             patch('oai_agent_server.main.ConfigManager'), \
             patch('oai_agent_server.main.DatabaseLogger'):
            server = AgentHTTPServer(agent, "test_agent")
        client = TestClient(server.app)
        r = client.get("/health")
        assert r.status_code == 200


def test_main_entry_point(mock_agent):
    with patch('oai_agent_server.main.parse_args') as mock_parse, \
         patch('oai_agent_server.main.AgentHTTPServer') as MockServer:
        
        mock_parse.return_value.port = 8000
        mock_server_instance = MockServer.return_value
        mock_server_instance.agent_name = "test_agent"
        mock_server_instance.base_config_manager.load_agent_config.return_value = {}
        
        main(mock_server_instance)
        mock_server_instance.run.assert_called_once()
