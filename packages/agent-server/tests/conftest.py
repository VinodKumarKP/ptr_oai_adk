import os

import pytest
from unittest.mock import MagicMock, AsyncMock
import sys
import types

# Add src to sys.path to ensure local packages are discoverable
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Mock oai_agent_core package structure
if 'oai_agent_core' not in sys.modules:
    oai_agent_core = types.ModuleType('oai_agent_core')
    oai_agent_core.__path__ = []
    sys.modules['oai_agent_core'] = oai_agent_core
    
    # Core
    core = types.ModuleType('oai_agent_core.core')
    core.__path__ = []
    sys.modules['oai_agent_core.core'] = core
    sys.modules['oai_agent_core'].core = core
    
    # Base Agent
    mock_base_agent = types.ModuleType('oai_agent_core.core.base_agent')
    class MockBaseAgent:
        def __init__(self, **kwargs):
            self.agent_name = kwargs.get('agent_name', 'mock_agent')
            self.session_id = kwargs.get('session_id', 'default')
            self.agent_config = kwargs.get('agent_config', {})
            self.agent_type = kwargs.get('agent_type', 'mock_framework')
            self.logger = MagicMock()
            self._initialized = False
            
        async def initialize(self):
            self._initialized = True
            
        async def ainvoke(self, user_message, config=None):
            return {"content": f"Response to {user_message}"}
            
        async def astream(self, user_message, config=None):
            yield {"content": f"Chunk for {user_message}"}
            
    mock_base_agent.BaseAgent = MockBaseAgent
    sys.modules['oai_agent_core.core.base_agent'] = mock_base_agent
    core.BaseAgent = MockBaseAgent
    
    # Components
    components = types.ModuleType('oai_agent_core.components')
    components.__path__ = []
    sys.modules['oai_agent_core.components'] = components
    sys.modules['oai_agent_core'].components = components
    
    # Config Manager
    config_pkg = types.ModuleType('oai_agent_core.components.configuration')
    config_pkg.__path__ = []
    sys.modules['oai_agent_core.components.configuration'] = config_pkg
    
    model_config = types.ModuleType('oai_agent_core.components.configuration.model_config')
    class MockConfigManager:
        def __init__(self, config_root=None):
            pass
        def load_agent_config(self, agent_name, abort_if_not_found=True):
            return {"agent_name": agent_name, "port": 8000}
            
    model_config.ConfigManager = MockConfigManager
    sys.modules['oai_agent_core.components.configuration.model_config'] = model_config
    
    # Utils
    utils = types.ModuleType('oai_agent_core.utils')
    utils.__path__ = []
    sys.modules['oai_agent_core.utils'] = utils
    sys.modules['oai_agent_core'].utils = utils
    
    logger_mod = types.ModuleType('oai_agent_core.utils.logger')
    logger_mod.get_logger = MagicMock(return_value=MagicMock())
    sys.modules['oai_agent_core.utils.logger'] = logger_mod

    dotenv_loader_mod = types.ModuleType('oai_agent_core.utils.dotenv_loader')
    dotenv_loader_mod.load_dotenv = MagicMock(return_value=0)
    sys.modules['oai_agent_core.utils.dotenv_loader'] = dotenv_loader_mod

    # Deployment helper: mirror the real (tiny, env-driven) implementation so
    # tests can exercise AgentCore behavior by setting DEPLOYMENT_TARGET.
    deployment_mod = types.ModuleType('oai_agent_core.utils.deployment')

    def _mock_get_deployment_target():
        return os.environ.get("DEPLOYMENT_TARGET", "").strip().lower()

    def _mock_is_agentcore_runtime():
        return _mock_get_deployment_target() == "agentcore"

    def _mock_is_runtime_package_install_disabled():
        flag = os.environ.get("DISABLE_RUNTIME_PACKAGE_INSTALL", "").strip().lower()
        return _mock_is_agentcore_runtime() or flag in {"1", "true", "yes", "on"}

    deployment_mod.get_deployment_target = _mock_get_deployment_target
    deployment_mod.is_agentcore_runtime = _mock_is_agentcore_runtime
    deployment_mod.is_runtime_package_install_disabled = _mock_is_runtime_package_install_disabled
    sys.modules['oai_agent_core.utils.deployment'] = deployment_mod

# Mock redis
if 'redis' not in sys.modules:
    redis = MagicMock()
    sys.modules['redis'] = redis

# Mock asyncpg
if 'asyncpg' not in sys.modules:
    asyncpg = MagicMock()
    sys.modules['asyncpg'] = asyncpg
    sys.modules['asyncpg.pool'] = MagicMock()

# Mock a2a module
if 'a2a' not in sys.modules:
    a2a = types.ModuleType('a2a')
    a2a.__path__ = []
    sys.modules['a2a'] = a2a
    
    a2a_server = types.ModuleType('a2a.server')
    a2a_server.__path__ = []
    sys.modules['a2a.server'] = a2a_server
    
    a2a_routes = types.ModuleType('a2a.server.routes')
    a2a_routes.create_agent_card_routes = MagicMock(return_value=MagicMock())
    a2a_routes.create_jsonrpc_routes = MagicMock(return_value=MagicMock())
    a2a_routes.create_rest_routes = MagicMock(return_value=MagicMock())
    sys.modules['a2a.server.routes'] = a2a_routes


@pytest.fixture
def mock_agent():
    agent = MockBaseAgent(agent_name="test_agent")
    return agent

@pytest.fixture
def mock_db_logger():
    logger = MagicMock()
    logger.is_active = True
    logger.log_interaction = AsyncMock()
    logger.log_stream_chunks_batch = AsyncMock()
    logger.get_logs = AsyncMock(return_value=[])
    logger.get_session_logs = AsyncMock(return_value=[])
    logger.get_stats = AsyncMock(return_value={})
    logger.get_user_stats = AsyncMock(return_value=[])
    return logger
