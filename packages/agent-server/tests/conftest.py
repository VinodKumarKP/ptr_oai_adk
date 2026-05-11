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

# Mock redis
if 'redis' not in sys.modules:
    redis = MagicMock()
    sys.modules['redis'] = redis

# Mock asyncpg
if 'asyncpg' not in sys.modules:
    asyncpg = MagicMock()
    sys.modules['asyncpg'] = asyncpg
    sys.modules['asyncpg.pool'] = MagicMock()

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
