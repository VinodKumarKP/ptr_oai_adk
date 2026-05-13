import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from oai_agent_core.core.base_agent import BaseAgent

# Concrete implementation for testing abstract base class
class ConcreteAgent(BaseAgent):
    async def initialize(self):
        self._initialized = True

    async def astream(self, user_message, config=None):
        yield "response"

    async def ainvoke(self, user_message, config=None):
        return "response"

    def invoke(self, user_message, config=None):
        return "response"

    async def stream(self, user_message, config=None):
        yield "response"

@pytest.fixture
def mock_config_manager():
    with patch('oai_agent_core.core.base_agent.ConfigManager') as MockCM:
        instance = MockCM.return_value
        instance.load_agent_config.return_value = {'type': 'test_agent', 'model': {}}
        instance.get_config_value.side_effect = lambda config, key, default: config.get(key, default)
        instance.merge_configs.side_effect = lambda base, updates: {**base, **updates}
        yield instance

@pytest.fixture
def mock_model_manager():
    manager = MagicMock()
    manager.create_model.return_value = "mock_llm"
    return manager

@pytest.fixture
def mock_langfuse():
    with patch('oai_agent_core.core.base_agent.LangfuseObservabilityManager') as MockLF:
        yield MockLF.return_value

def test_init(mock_config_manager, mock_model_manager, mock_langfuse):
    agent = ConcreteAgent(
        agent_name="test_agent",
        agent_type="custom",
        model_manager=mock_model_manager
    )

    assert agent.agent_name == "test_agent"
    assert agent.llm == "mock_llm"
    assert agent.agent_type == "custom"
    assert agent.is_initialized is False
    mock_config_manager.load_agent_config.assert_called_with("test_agent")

def test_init_with_config(mock_config_manager, mock_model_manager):
    config = {'type': 'custom', 'model': {}}
    agent = ConcreteAgent(
        agent_name="test_agent",
        agent_config=config,
        model_manager=mock_model_manager
    )
    
    assert agent.agent_config == config
    mock_config_manager.load_agent_config.assert_not_called()

def test_get_config_value(mock_config_manager, mock_model_manager):
    agent = ConcreteAgent("test", model_manager=mock_model_manager)
    agent.agent_config = {'key': 'value'}
    
    val = agent.get_config_value('key')
    assert val == 'value'
    mock_config_manager.get_config_value.assert_called()

def test_set_config_value(mock_config_manager, mock_model_manager):
    agent = ConcreteAgent("test", model_manager=mock_model_manager)
    
    agent.set_config_value('key', 'new_val')
    mock_config_manager.set_config_value.assert_called()

def test_update_config(mock_config_manager, mock_model_manager):
    agent = ConcreteAgent("test", model_manager=mock_model_manager)
    agent.agent_config = {'a': 1}
    
    agent.update_config({'b': 2})
    
    # Since we mocked merge_configs to return simple merge
    assert agent.agent_config == {'a': 1, 'b': 2}

def test_validate_config_valid(mock_config_manager, mock_model_manager):
    agent = ConcreteAgent("test", model_manager=mock_model_manager)
    agent.agent_config = {'type': 'custom', 'model': {'name': 'gpt-4'}}
    agent.validate_config() # Should not raise

def test_validate_config_invalid(mock_config_manager, mock_model_manager):
    agent = ConcreteAgent("test", model_manager=mock_model_manager)
    agent.agent_config = {} # Missing type

    with pytest.raises(ValueError, match="Configuration validation failed"):
        agent.validate_config()

@pytest.mark.asyncio
async def test_ensure_initialized(mock_config_manager, mock_model_manager):
    agent = ConcreteAgent("test", model_manager=mock_model_manager)
    assert agent.is_initialized is False
    
    await agent._ensure_initialized()
    assert agent.is_initialized is True

def test_del(mock_config_manager, mock_model_manager, mock_langfuse):
    agent = ConcreteAgent("test", model_manager=mock_model_manager)
    mock_langfuse.is_enabled = True
    
    agent.__del__()
    mock_langfuse.flush.assert_called()
