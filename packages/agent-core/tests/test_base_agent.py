import pytest
from unittest.mock import MagicMock, patch
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

class TestBaseAgent:
    
    @pytest.fixture
    def mock_config_manager(self):
        with patch('oai_agent_core.core.base_agent.ConfigManager') as MockCM:
            instance = MockCM.return_value
            instance.load_agent_config.return_value = {'type': 'test_agent', 'model': {}}
            instance.get_config_value.side_effect = lambda config, key, default: config.get(key, default)
            instance.merge_configs.side_effect = lambda base, updates: {**base, **updates}
            yield instance

    @pytest.fixture
    def mock_model_manager(self):
        manager = MagicMock()
        manager.create_model.return_value = "mock_llm"
        return manager

    @pytest.fixture
    def agent(self, mock_config_manager, mock_model_manager):
        return ConcreteAgent(
            agent_name="test_agent",
            session_id="test_session",
            user_id="test_user",
            model_manager=mock_model_manager
        )
    
    def test_init(self, agent):
        assert agent.agent_name == "test_agent"
        assert agent.session_id == "test_session"
        assert agent.user_id == "test_user"
        assert agent.logger is not None
    
    def test_init_defaults(self, mock_config_manager, mock_model_manager):
        agent = ConcreteAgent(agent_name="test", model_manager=mock_model_manager)
        assert agent.session_id == "default"
        assert agent.user_id == "default"
    
    @pytest.mark.asyncio
    async def test_ainvoke(self, agent):
        result = await agent.ainvoke("test message")
        assert result == "response"
    
    def test_invoke(self, agent):
        result = agent.invoke("test message")
        assert result == "response"
    
    @pytest.mark.asyncio
    async def test_astream(self, agent):
        chunks = []
        async for chunk in agent.astream("test message"):
            chunks.append(chunk)
        assert chunks == ["response"]
    
    @pytest.mark.asyncio
    async def test_stream(self, agent):
        chunks = []
        async for chunk in agent.stream("test message"):
            chunks.append(chunk)
        assert chunks == ["response"]
    
    def test_get_agent_info(self, agent):
        # The base agent doesn't implement get_agent_info
        with pytest.raises(AttributeError):
            agent.get_agent_info()
    
    def test_validate_tasks_not_implemented(self, agent):
        # The base agent doesn't implement validate_tasks
        with pytest.raises(AttributeError):
            agent.validate_tasks()
    
    @pytest.mark.asyncio
    async def test_initialize_not_implemented(self, agent):
        await agent.initialize()
        assert agent.is_initialized
    
    def test_abstract_methods_not_implemented(self):
        # Test that we can't instantiate the base class directly
        with pytest.raises(TypeError):
            BaseAgent(agent_name="test")
