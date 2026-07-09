import pytest
from unittest.mock import MagicMock, patch
from oai_agent_core.langgraph_core.components.configuration.model_config import LangChainModelConfigurationManager

@pytest.fixture
def manager():
    with patch('oai_agent_core.langgraph_core.components.configuration.model_config.BaseModelConfigurationManager.__init__', return_value=None) as mock_init:
        manager = LangChainModelConfigurationManager()
        manager.logger = MagicMock()
        manager.default_config = {
            'model_id': 'gpt-4',
            'params': {
                'temperature': 0.7,
                'max_tokens': 1000
            }
        }
        # Mock _merge_with_defaults to return the input or default
        manager._merge_with_defaults = lambda config: config if config else manager.default_config
        # Mock _validate_config to do nothing
        manager._validate_config = lambda config: None
        return manager

def test_create_model_success(manager):
    with patch.dict('sys.modules', {'langchain_litellm': MagicMock()}):
        from langchain_litellm import ChatLiteLLM
        mock_model = MagicMock()
        ChatLiteLLM.return_value = mock_model
        
        model = manager.create_model()
        
        assert model == mock_model
        ChatLiteLLM.assert_called_once()
        call_kwargs = ChatLiteLLM.call_args[1]
        assert call_kwargs['model'] == 'gpt-4'
        assert call_kwargs['temperature'] == 0.7
        assert call_kwargs['max_tokens'] == 1000

def test_create_model_with_custom_config(manager):
    with patch.dict('sys.modules', {'langchain_litellm': MagicMock()}):
        from langchain_litellm import ChatLiteLLM
        
        custom_config = {
            'model_id': 'gpt-3.5-turbo',
            'params': {
                'temperature': 0.5,
                'max_tokens': 500,
                'top_p': 0.9
            }
        }
        
        manager.create_model(custom_config)
        
        call_kwargs = ChatLiteLLM.call_args[1]
        assert call_kwargs['model'] == 'gpt-3.5-turbo'
        assert call_kwargs['temperature'] == 0.5
        assert call_kwargs['max_tokens'] == 500
        assert call_kwargs['top_p'] == 0.9

def test_create_model_import_error(manager):
    # The _import_chat_litellm function now gracefully handles imports with lazy stubs.
    # To test ImportError, we verify that when the actual package is unavailable,
    # the create_model call fails appropriately with a helpful message.
    with patch('oai_agent_core.langgraph_core.components.configuration.model_config._import_chat_litellm') as mock_import:
        mock_import.side_effect = ImportError("No module named 'langchain_litellm'")
        with pytest.raises(ImportError, match="pip install langchain-litellm"):
            manager.create_model()

def test_create_model_generic_error(manager):
    with patch.dict('sys.modules', {'langchain_litellm': MagicMock()}):
        from langchain_litellm import ChatLiteLLM
        ChatLiteLLM.side_effect = Exception("Creation failed")
        
        with pytest.raises(Exception, match="Creation failed"):
            manager.create_model()
        
        manager.logger.error.assert_called_once()
