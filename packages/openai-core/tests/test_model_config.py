import pytest
import sys
from unittest.mock import MagicMock, patch
from oai_agent_core.openai_core.components.configuration.model_config import OpenAIModelConfigurationManager

@pytest.fixture
def manager():
    return OpenAIModelConfigurationManager()

@pytest.fixture
def mock_litellm_class():
    mock_cls = MagicMock()
    return mock_cls

@pytest.fixture
def mock_litellm_module(mock_litellm_class):
    mock_mod = MagicMock()
    mock_mod.LitellmModel = mock_litellm_class
    return mock_mod

def test_create_model_success(manager, mock_litellm_module, mock_litellm_class):
    config = {
        'model_id': 'gpt-4',
        'temperature': 0.7
    }
    
    mock_instance = MagicMock()
    mock_litellm_class.return_value = mock_instance
    
    # Patch the module in sys.modules to ensure the import inside the method gets our mock
    with patch.dict(sys.modules, {'agents.extensions.models.litellm_model': mock_litellm_module}):
        # Mock _validate_config to avoid validation errors
        with patch.object(manager, '_validate_config'):
            model = manager.create_model(config)
        
        assert model == mock_instance
        mock_litellm_class.assert_called_once()
        call_kwargs = mock_litellm_class.call_args[1]
        assert call_kwargs['model'] == 'gpt-4'

def test_create_model_default_config(manager, mock_litellm_module, mock_litellm_class):
    # Mock default config
    manager.default_config = {'model_id': 'default-model'}
    
    with patch.dict(sys.modules, {'agents.extensions.models.litellm_model': mock_litellm_module}):
        with patch.object(manager, '_validate_config'):
            manager.create_model(None)
        
        call_kwargs = mock_litellm_class.call_args[1]
        assert call_kwargs['model'] == 'default-model'

def test_create_model_creation_error(manager, mock_litellm_module, mock_litellm_class):
    config = {'model_id': 'gpt-4'}
    
    mock_litellm_class.side_effect = Exception("Creation failed")
    
    with patch.dict(sys.modules, {'agents.extensions.models.litellm_model': mock_litellm_module}):
        with patch.object(manager, '_validate_config'):
            with pytest.raises(Exception, match="Creation failed"):
                manager.create_model(config)

def test_create_model_validation(manager, mock_litellm_module):
    with patch.dict(sys.modules, {'agents.extensions.models.litellm_model': mock_litellm_module}):
        with patch.object(manager, '_validate_config') as mock_validate:
            manager.create_model({'model_id': 'gpt-4'})
            mock_validate.assert_called_once()
